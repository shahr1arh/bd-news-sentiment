# 🇧🇩 Bangladesh News Sentiment Dashboard

A real-time NLP-powered web dashboard that scrapes English-language Bangladeshi news headlines and classifies them as **Positive**, **Negative**, or **Neutral** using VADER sentiment analysis — with historical trend tracking and a fully interactive UI.

---

## ✨ Features

- Scrapes live headlines from **5 sources**: The Daily Star, Dhaka Tribune, bdnews24, New Age, The Business Standard
- **Parallel scraping** via `ThreadPoolExecutor` — all 5 sources hit simultaneously
- **Rotating User-Agent pool** (5 agents) to reduce bot-detection risk
- Sentiment analysis with **VADER NLP** + 70-term Bangladesh-specific lexicon
- **Phrase-aware topic classifier** across 7 categories (Politics, Economy, Crime, Sports, Environment, Health, International) — uses multi-word phrase matching to avoid false positives like "bank robbery" being tagged Economy
- **SQLite persistence** — cache survives server restarts; instant warm boot
- **Historical trend chart** — avg sentiment over the last 48 refresh cycles
- Interactive dashboard:
  - Sentiment distribution **Doughnut chart**
  - Per-source average **Bar chart**
  - Sentiment trend **Line chart** (historical)
  - Live **headline search** with match highlighting (`Ctrl+K`)
  - Sentiment **filter chips** + dynamic **topic pills**
  - **Sort** by Latest / Most Positive / Most Negative / A→Z
  - **CSV export** (date-stamped filename)
  - Auto-refresh **countdown timer** in the header

---

## 🛠️ Tech Stack

| Layer      | Technology |
|---|---|
| Backend    | Python 3.11+, Flask 3 |
| Database   | SQLite (via stdlib `sqlite3`) |
| Scraping   | BeautifulSoup4 + lxml, Requests (parallel + retry/backoff) |
| NLP        | NLTK VADER + custom lexicon + phrase-aware rule classifier |
| Frontend   | HTML5, CSS3 (custom design system), Vanilla JS |
| Charts     | Chart.js 4 |
| Production (Linux/Cloud) | Gunicorn |
| Production (Windows)     | Waitress |

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run — Development
```bash
python app.py
```

### 3. Run — Production on **Linux / Cloud**
```bash
gunicorn app:app --workers 2 --bind 0.0.0.0:5000 --timeout 120
```

### 4. Run — Production on **Windows**
> Gunicorn is UNIX-only and **will not run on Windows**.
> Use Waitress instead:
```bash
waitress-serve --port=5000 app:app
```

### 5. Open in browser
```
http://localhost:5000
```

---

## ⚙️ Environment Variables

| Variable       | Default | Description |
|---|---|---|
| `CACHE_TTL`    | `300`   | Cache lifetime in seconds (5 min) |
| `AUTO_REFRESH` | `1`     | Background auto-refresh thread (`0` = disable) |
| `PORT`         | `5000`  | Listening port |
| `FLASK_DEBUG`  | `0`     | Flask debug mode |

---

## 📁 Project Structure

```
bd-news-sentiment/
├── app.py              # Flask backend, cache, background thread
├── scraper.py          # Parallel multi-source scraper (ThreadPoolExecutor)
├── analyzer.py         # VADER engine + custom lexicon + topic classifier
├── db.py               # SQLite persistence layer (headlines + batch history)
├── requirements.txt
├── Procfile            # For Render / Railway deployment
├── README.md
└── templates/
    └── index.html      # Full dashboard UI (search, filter, trend chart, CSV export)
```

---

## 📡 API Endpoints

| Method | Endpoint          | Description |
|---|---|---|
| GET    | `/`               | Dashboard UI |
| GET    | `/api/headlines`  | Latest cached headlines + aggregate stats |
| POST   | `/api/refresh`    | Force re-scrape; returns fresh data directly |
| GET    | `/api/trends`     | Historical batch summaries (last 48 cycles) |
| GET    | `/api/health`     | Cache age, headline count, TTL — for monitoring |

---

## 📊 How It Works

1. **`scraper.py`** — fires all 5 source scrapers in parallel via `ThreadPoolExecutor`. Each thread uses a randomly rotated User-Agent and retries up to 2× with backoff. Headlines are globally deduplicated before returning.
2. **`analyzer.py`** — runs VADER `polarity_scores()` on each title (augmented with 70+ BD-specific lexicon terms), then classifies topic using a phrase-aware scorer: multi-word matches (e.g. `"bank robbery"`) score 2.5× vs single words, keeping `"bank"` alone from false-triggering Economy.
3. **`db.py`** — persists every refresh batch to SQLite (headlines + aggregate stats). On startup, `app.py` warms the in-memory cache instantly from the DB and triggers a background fresh-fetch — so the first request is never a cold start.
4. **`app.py`** — serves a REST API with an in-memory cache backed by SQLite, a background daemon thread for auto-refresh, and a `/api/trends` endpoint exposing historical batch summaries.
5. **`index.html`** — fetches `/api/headlines` and `/api/trends` on load, renders three Chart.js visualisations, and handles search, topic/sentiment filtering, sorting, and CSV export entirely client-side.

---

## ☁️ Deploy to Render (Free)

1. Push to a GitHub repo
2. Create a new **Web Service** on [render.com](https://render.com)
3. Set **Build Command**: `pip install -r requirements.txt`
4. Set **Start Command**: `gunicorn app:app --workers 2 --bind 0.0.0.0:$PORT --timeout 120`
5. The `Procfile` in the repo root handles this automatically if you use Render's auto-detect

---

## 🔮 Future Improvements

- Bengali-language support using BanglaBERT
- Email / Slack alert when negative sentiment spikes above a threshold
- Docker container for one-command deployment anywhere
- Zero-shot ML topic classification to replace rule-based classifier
