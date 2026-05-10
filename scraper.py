"""
scraper.py — Parallel multi-source BD news headline scraper
Features: ThreadPoolExecutor, UA rotation, retry with backoff, 5 sources, dedup
"""

import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── User-Agent pool ────────────────────────────────────────────────────────────
_USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_4) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

_BASE_HEADERS = {
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "DNT": "1",
    "Connection": "keep-alive",
}

SOURCES = {
    "The Daily Star": {
        "url": "https://www.thedailystar.net/news/bangladesh",
        "selector": "h3.title a, h2.title a",
        "base": "https://www.thedailystar.net",
    },
    "Dhaka Tribune": {
        "url": "https://www.dhakatribune.com/bangladesh",
        "selector": "h3.title-container a, h2.entry-title a, h3.entry-title a",
        "base": "https://www.dhakatribune.com",
    },
    "bdnews24": {
        "url": "https://bdnews24.com/bangladesh",
        "selector": "h1.headline a, h2.headline a, .article-title a, h2.lnk a",
        "base": "https://bdnews24.com",
    },
    "New Age": {
        "url": "https://www.newagebd.net/",
        "selector": "h4 a, h3.nag-post-title a, .headline a",
        "base": "https://www.newagebd.net",
    },
    "The Business Standard": {
        "url": "https://www.tbsnews.net/bangladesh",
        "selector": "h3 a, h2.title a, .card-title a, .article-title a",
        "base": "https://www.tbsnews.net",
    },
}

MAX_PER_SOURCE = 25
MAX_RETRIES    = 2
RETRY_DELAY    = 1.5
TIMEOUT        = 10
MAX_WORKERS    = 5


def _random_ua() -> str:
    return random.choice(_USER_AGENTS)


def _normalize_url(href: str, base: str) -> str:
    href = href.strip()
    if href.startswith("http"):
        return href
    if href.startswith("//"):
        return "https:" + href
    return base.rstrip("/") + "/" + href.lstrip("/")


def _fetch_with_retry(url: str) -> requests.Response:
    """
    GET with per-request UA rotation and selective retry:
      - 4xx (client errors): fail immediately — retrying a 403/404 is pointless.
      - 5xx (server errors): retry with backoff — server may recover.
      - Network errors (timeout, DNS): retry with backoff.
    """
    for attempt in range(MAX_RETRIES + 1):
        headers = {**_BASE_HEADERS, "User-Agent": _random_ua()}

        # Network-level errors (no response received at all)
        try:
            resp = requests.get(url, headers=headers, timeout=TIMEOUT)
        except requests.exceptions.RequestException as exc:
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY * (attempt + 1)
                logger.warning(
                    f"[retry {attempt+1}/{MAX_RETRIES}] {url} — {exc}. Waiting {delay}s"
                )
                time.sleep(delay)
                continue
            raise

        # We have a response — decide whether to retry based on HTTP status
        if 400 <= resp.status_code < 500:
            # Client error: server explicitly rejected us (403, 404, 410…)
            # Retrying won't change anything — fail immediately.
            logger.warning(
                f"[{url}] HTTP {resp.status_code} — client error, skipping retries."
            )
            resp.raise_for_status()   # raises HTTPError, exits the loop

        if resp.status_code >= 500:
            # Server error: may be transient — worth retrying.
            if attempt < MAX_RETRIES:
                delay = RETRY_DELAY * (attempt + 1)
                logger.warning(
                    f"[retry {attempt+1}/{MAX_RETRIES}] {url} — "
                    f"HTTP {resp.status_code}. Waiting {delay}s"
                )
                time.sleep(delay)
                continue
            resp.raise_for_status()

        # 2xx / 3xx — success (requests follows redirects automatically)
        return resp


def _scrape_one(name: str, config: dict) -> list[dict]:
    """Scrape a single source. Runs inside a thread."""
    headlines = []
    try:
        resp = _fetch_with_retry(config["url"])
        soup = BeautifulSoup(resp.text, "lxml")
        elements = soup.select(config["selector"])
        seen: set[str] = set()

        for el in elements[:MAX_PER_SOURCE]:
            text = el.get_text(strip=True)
            href = el.get("href", "")
            if not text or len(text) < 15 or text in seen:
                continue
            seen.add(text)
            headlines.append({
                "title":      text,
                "link":       _normalize_url(href, config["base"]),
                "source":     name,
                "scraped_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "scraped_ts": datetime.now().timestamp(),
            })
        logger.info(f"[{name}] {len(headlines)} headlines scraped.")
    except Exception as exc:
        logger.error(f"[{name}] Scrape failed: {exc}")
    return headlines


def get_all_headlines() -> list[dict]:
    """Scrape all sources in parallel. Returns globally deduplicated headlines."""
    raw: list[dict] = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(_scrape_one, name, cfg): name
            for name, cfg in SOURCES.items()
        }
        for future in as_completed(futures):
            name = futures[future]
            try:
                raw.extend(future.result())
            except Exception as exc:
                logger.error(f"[{name}] Future raised: {exc}")

    seen:   set[str]  = set()
    unique: list[dict] = []
    for h in raw:
        key = h["title"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique.append(h)

    logger.info(f"Total unique headlines after dedup: {len(unique)}")
    return unique
