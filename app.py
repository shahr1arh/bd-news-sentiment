"""
app.py — Flask backend for BD News Sentiment Dashboard
Features: SQLite persistence, background auto-refresh, structured logging,
          /api/health, /api/trends, env-based config, waitress/gunicorn ready
"""

import logging
import os
import threading
import time

from flask import Flask, jsonify, render_template

from analyzer import analyze_headlines, summary_stats
from db import init_db, save_batch, load_latest_batch, load_trend_history
from scraper import get_all_headlines

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── App ────────────────────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
CACHE_TTL:    int  = int(os.environ.get("CACHE_TTL", 300))
AUTO_REFRESH: bool = os.environ.get("AUTO_REFRESH", "1") != "0"
PORT:         int  = int(os.environ.get("PORT", 5000))
DEBUG:        bool = os.environ.get("FLASK_DEBUG", "0") == "1"

# ── In-memory cache (warmed from SQLite on startup) ────────────────────────────
_cache: dict = {"data": None, "stats": None, "last_updated": None}
_lock = threading.Lock()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_payload() -> dict:
    with _lock:
        hl   = _cache["data"]  or []
        stats = _cache["stats"] or {}
        return {
            "headlines":    hl,
            "stats":        stats,
            "last_updated": _cache["last_updated"],
            "source_count": len({h["source"] for h in hl}),
        }


def _needs_refresh() -> bool:
    if _cache["data"] is None:
        return True
    return (time.time() - (_cache["last_updated"] or 0)) > CACHE_TTL


def refresh_data() -> bool:
    """Scrape → analyse → persist to SQLite → update in-memory cache."""
    logger.info("Starting data refresh…")
    try:
        headlines = get_all_headlines()
        if not headlines:
            logger.warning("No headlines returned by scrapers.")
            return False

        analyzed = analyze_headlines(headlines)
        stats    = summary_stats(analyzed)

        # Persist to SQLite (survives server restarts)
        save_batch(analyzed, stats)

        with _lock:
            _cache["data"]         = analyzed
            _cache["stats"]        = stats
            _cache["last_updated"] = time.time()

        logger.info(f"Refresh complete — {len(analyzed)} headlines, {stats['counts']}.")
        return True

    except Exception as exc:
        logger.error(f"refresh_data failed: {exc}", exc_info=True)
        return False


def _warm_cache_from_db() -> bool:
    """On startup, load the last batch from SQLite so the first request is instant."""
    headlines, stats = load_latest_batch()
    if headlines is None:
        return False
    with _lock:
        _cache["data"]         = headlines
        _cache["stats"]        = stats
        _cache["last_updated"] = time.time() - CACHE_TTL + 30  # trigger refresh soon
    logger.info(f"Cache warmed from DB ({len(headlines)} headlines).")
    return True


def _background_loop():
    while True:
        time.sleep(CACHE_TTL)
        logger.info("Background refresh triggered.")
        refresh_data()


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/headlines")
def api_headlines():
    if _needs_refresh():
        refresh_data()
    return jsonify(_build_payload())


@app.route("/api/refresh", methods=["POST"])
def api_refresh():
    if refresh_data():
        return jsonify({"status": "ok", **_build_payload()})
    return jsonify({"status": "error", "message": "Could not fetch headlines"}), 500


@app.route("/api/trends")
def api_trends():
    """Return historical batch summaries for the trend line chart."""
    history = load_trend_history(limit=48)
    return jsonify({"history": history})


@app.route("/api/health")
def api_health():
    with _lock:
        last = _cache["last_updated"]
        n    = len(_cache["data"]) if _cache["data"] else 0
    return jsonify({
        "status":           "ok",
        "headlines_cached": n,
        "last_updated":     last,
        "cache_age_s":      round(time.time() - last, 1) if last else None,
        "cache_ttl_s":      CACHE_TTL,
    })


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    init_db()

    # Try to warm from DB first (instant), then fetch fresh data
    warmed = _warm_cache_from_db()
    if not warmed:
        logger.info("No DB data found — doing cold fetch…")
        refresh_data()
    else:
        # Schedule a fresh fetch shortly after boot without blocking startup
        t = threading.Thread(target=refresh_data, daemon=True)
        t.start()

    if AUTO_REFRESH:
        bg = threading.Thread(target=_background_loop, daemon=True)
        bg.start()
        logger.info(f"Background refresh thread started (every {CACHE_TTL}s).")

    app.run(debug=DEBUG, port=PORT, use_reloader=False)
