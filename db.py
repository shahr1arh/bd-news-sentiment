"""
db.py — SQLite persistence layer
Stores headline batches so the dashboard survives server restarts
and accumulates historical sentiment data for trend analysis.
"""

import json
import logging
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent / "sentiment.db"


# ── Connection ─────────────────────────────────────────────────────────────────

@contextmanager
def _conn():
    """Thread-safe context manager that always closes the connection."""
    con = sqlite3.connect(DB_PATH, check_same_thread=False)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ── Schema ─────────────────────────────────────────────────────────────────────

def init_db() -> None:
    """Create tables if they don't exist yet, and enable WAL mode."""
    with _conn() as con:
        # WAL allows concurrent reads during a write, preventing "database is locked"
        # errors when the background scraper thread and Flask request handlers overlap.
        # NORMAL synchronous is safe with WAL and significantly faster than FULL.
        con.execute("PRAGMA journal_mode=WAL")
        con.execute("PRAGMA synchronous=NORMAL")
        con.executescript("""
            CREATE TABLE IF NOT EXISTS batches (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                scraped_at      REAL NOT NULL,
                total           INTEGER NOT NULL,
                pos_count       INTEGER NOT NULL,
                neg_count       INTEGER NOT NULL,
                neu_count       INTEGER NOT NULL,
                avg_compound    REAL NOT NULL,
                overall_sentiment TEXT NOT NULL,
                source_avg      TEXT NOT NULL,   -- JSON
                topic_counts    TEXT NOT NULL    -- JSON
            );

            CREATE TABLE IF NOT EXISTS headlines (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id    INTEGER NOT NULL REFERENCES batches(id) ON DELETE CASCADE,
                title       TEXT NOT NULL,
                link        TEXT NOT NULL,
                source      TEXT NOT NULL,
                sentiment   TEXT NOT NULL,
                compound    REAL NOT NULL,
                pos         REAL NOT NULL,
                neg         REAL NOT NULL,
                neu         REAL NOT NULL,
                topic       TEXT NOT NULL,
                scraped_at  TEXT NOT NULL,
                scraped_ts  REAL NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_headlines_batch ON headlines(batch_id);
            CREATE INDEX IF NOT EXISTS idx_batches_scraped ON batches(scraped_at);
        """)
    logger.info(f"Database ready at {DB_PATH}")


# ── Write ──────────────────────────────────────────────────────────────────────

def save_batch(headlines: list[dict], stats: dict) -> int:
    """Persist a full scrape batch. Returns the new batch id."""
    counts = stats.get("counts", {})
    with _conn() as con:
        cur = con.execute(
            """INSERT INTO batches
               (scraped_at, total, pos_count, neg_count, neu_count,
                avg_compound, overall_sentiment, source_avg, topic_counts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                time.time(),
                stats.get("total", 0),
                counts.get("Positive", 0),
                counts.get("Negative", 0),
                counts.get("Neutral", 0),
                stats.get("avg_compound", 0.0),
                stats.get("overall_sentiment", "Neutral"),
                json.dumps(stats.get("source_avg", {})),
                json.dumps(stats.get("topic_counts", {})),
            ),
        )
        batch_id = cur.lastrowid

        con.executemany(
            """INSERT INTO headlines
               (batch_id, title, link, source, sentiment, compound,
                pos, neg, neu, topic, scraped_at, scraped_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    batch_id,
                    h["title"], h["link"], h["source"],
                    h["sentiment"], h["compound"],
                    h["pos"], h["neg"], h["neu"],
                    h.get("topic", "General"),
                    h["scraped_at"], h.get("scraped_ts", time.time()),
                )
                for h in headlines
            ],
        )

    # Keep only the latest 50 batches (prune old data automatically)
    _prune_old_batches(keep=50)

    logger.info(f"Saved batch #{batch_id} ({len(headlines)} headlines).")
    return batch_id


def _prune_old_batches(keep: int = 50) -> None:
    with _conn() as con:
        con.execute(
            """DELETE FROM batches WHERE id NOT IN
               (SELECT id FROM batches ORDER BY scraped_at DESC LIMIT ?)""",
            (keep,),
        )


# ── Read ───────────────────────────────────────────────────────────────────────

def load_latest_batch() -> tuple[list[dict], dict] | tuple[None, None]:
    """Load the most recent batch from disk. Returns (headlines, stats) or (None, None)."""
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM batches ORDER BY scraped_at DESC LIMIT 1"
        ).fetchone()
        if not row:
            return None, None

        batch_id = row["id"]
        hl_rows = con.execute(
            "SELECT * FROM headlines WHERE batch_id = ?", (batch_id,)
        ).fetchall()

    headlines = [dict(r) for r in hl_rows]
    stats = {
        "total": row["total"],
        "counts": {
            "Positive": row["pos_count"],
            "Negative": row["neg_count"],
            "Neutral":  row["neu_count"],
        },
        "avg_compound":      row["avg_compound"],
        "overall_sentiment": row["overall_sentiment"],
        "source_avg":        json.loads(row["source_avg"]),
        "topic_counts":      json.loads(row["topic_counts"]),
    }
    return headlines, stats


def load_trend_history(limit: int = 48) -> list[dict]:
    """Return the last `limit` batch summaries for trend charts (oldest first)."""
    with _conn() as con:
        rows = con.execute(
            """SELECT scraped_at, total, pos_count, neg_count, neu_count, avg_compound, overall_sentiment
               FROM batches ORDER BY scraped_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()

    return [dict(r) for r in reversed(rows)]
