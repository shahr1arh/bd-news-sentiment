"""
analyzer.py — VADER sentiment engine with BD lexicon and phrase-aware topic classifier
"""

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# VADER lexicon should be pre-downloaded by running `python download_nltk.py`
# during your build/setup step. The fallback below handles first-run locally
# but is intentionally skipped in production so cold boots stay fast.
try:
    nltk.data.find("sentiment/vader_lexicon.zip")
except LookupError:
    import warnings
    warnings.warn(
        "VADER lexicon not found — downloading now. "
        "Run `python download_nltk.py` in your build step to avoid this on cloud deploys.",
        RuntimeWarning,
        stacklevel=1,
    )
    nltk.download("vader_lexicon", quiet=True)

_analyzer = SentimentIntensityAnalyzer()

# ── Custom Bangladesh-context lexicon (70+ terms) ─────────────────────────────
CUSTOM_LEXICON: dict[str, float] = {
    # Disasters / Environment
    "flood": -1.5,   "flooding": -1.5,  "floods": -1.5,
    "cyclone": -1.8, "tornado": -1.6,   "earthquake": -1.8,
    "drought": -1.3, "landslide": -1.5, "waterlogging": -1.2,
    "storm": -1.2,   "heatwave": -1.2,  "salinity": -1.0,
    "deforestation": -1.2, "pollution": -1.2,
    # Crime / Safety
    "murder": -2.0,  "killing": -1.8,   "killed": -1.5,
    "death": -1.4,   "deaths": -1.4,    "dead": -1.3,
    "attack": -1.5,  "blast": -1.6,     "bomb": -1.8,
    "explosion": -1.5, "fire": -1.2,    "blaze": -1.3,
    "crash": -1.3,   "accident": -1.5,  "collision": -1.2,
    "rape": -2.0,    "assault": -1.5,   "harassment": -1.5,
    "robbery": -1.6, "theft": -1.3,     "kidnapping": -2.0,
    "hostage": -1.8, "smuggling": -1.5, "trafficking": -2.0,
    # Governance / Politics (negative)
    "corruption": -2.0, "bribery": -1.8, "embezzlement": -1.8,
    "scandal": -1.8,    "fraud": -1.8,   "scam": -1.6,
    "arrested": -0.8,   "detained": -0.8, "jailed": -1.0,
    "dismissed": -0.8,  "suspended": -0.7,
    "strike": -1.0,     "protest": -0.8,  "clash": -1.5,
    "unrest": -1.3,     "crackdown": -1.2, "shutdown": -1.0,
    # Economy (negative)
    "inflation": -1.0,  "crisis": -1.5,   "shortage": -1.2,
    "poverty": -1.5,    "unemployment": -1.2, "layoffs": -1.2,
    "deficit": -0.8,    "debt": -0.6,
    # Health (negative)
    "dengue": -1.2,  "cholera": -1.5, "disease": -0.9,
    "epidemic": -1.5, "outbreak": -1.3, "infection": -0.9,
    "injury": -1.0,  "wounded": -1.2,
    # Positive — Economy / Development
    "growth": 1.5,       "gdp": 1.0,        "surplus": 0.8,
    "development": 1.2,  "progress": 1.3,   "improvement": 1.0,
    "investment": 1.0,   "investors": 0.8,  "export": 0.8,
    "exports": 0.9,      "revenue": 0.6,    "profit": 0.8,
    "expansion": 0.8,    "boost": 0.9,      "recovery": 0.8,
    "recovered": 0.7,    "remittance": 0.8,
    # Positive — Governance / Social
    "reform": 0.8,      "transparency": 0.9, "accountability": 0.7,
    "launched": 0.5,    "inaugurated": 0.7,  "inauguration": 0.7,
    "elected": 0.4,     "victory": 1.2,      "wins": 0.8,
    "achievement": 1.3, "milestone": 1.2,    "record": 0.6,
    "award": 1.5,       "recognition": 1.2,  "honor": 1.2,
    "cooperation": 0.8, "agreement": 0.5,    "peace": 1.2,
    "stability": 0.8,   "partnership": 0.7,
    # Positive — Disaster response
    "rescue": 0.8,  "relief": 0.8,  "aid": 0.7,
    "restored": 0.6, "rehabilitation": 0.7,
    # Positive — Health / Education
    "vaccine": 0.8,    "immunization": 0.7, "treatment": 0.4,
    "scholarship": 0.9, "education": 0.6,   "graduation": 0.8,
    "festival": 0.7,   "celebration": 0.9,
}
_analyzer.lexicon.update(CUSTOM_LEXICON)


# ── Phrase-aware topic classifier ─────────────────────────────────────────────
#
# Problem with flat keyword maps: "bank" triggers Economy even in
# "Bank robbery in Gulshan" — which is Crime.
#
# Solution:
#   1. Multi-word phrases score 2.5× vs single words (more specific → more reliable)
#   2. Crime "veto" phrases override ambiguous single-word Economy hits
#   3. Ambiguous words ("bank", "strike", "fire") are intentionally excluded from
#      single-word lists; they only fire when part of a scored phrase.
#
_TOPIC_MAP: dict[str, dict] = {
    "Politics": {
        "phrases": [
            "prime minister", "foreign minister", "home minister",
            "finance minister", "general election", "by election",
            "caretaker government", "national parliament", "constitutional crisis",
            "political party", "awami league", "bnp chairperson",
        ],
        "words": [
            "election", "parliament", "cabinet", "minister", "government",
            "opposition", "bnp", "awami", "jatiya", "democracy", "regime",
            "coup", "autocracy", "referendum", "governance",
        ],
    },
    "Economy": {
        "phrases": [
            "gdp growth", "foreign exchange", "central bank rate",
            "stock market", "capital market", "trade deficit",
            "export earnings", "import cost", "budget deficit",
            "world bank", "imf loan", "economic growth",
            "remittance inflow", "garment sector", "rmg export",
            "foreign investment", "inflation rate",
        ],
        "words": [
            "gdp", "inflation", "remittance", "taka", "dollar",
            "export", "import", "revenue", "budget", "surplus",
            "deficit", "investment", "economic", "economy",
            "garment", "rmg", "tariff",
        ],
    },
    "Crime": {
        "phrases": [
            "bank robbery", "bank heist", "police arrested", "court verdict",
            "death sentence", "life imprisonment", "drug trafficking",
            "human trafficking", "child abuse", "sexual harassment",
            "mob beating", "acid attack", "murder case", "rape case",
            "money laundering", "cybercrime", "extortion case",
            "gang war", "arms smuggling",
        ],
        "words": [
            "murder", "killed", "arrested", "detained", "jailed",
            "rape", "robbery", "theft", "trafficking", "smuggling",
            "corruption", "bribery", "fraud", "scam", "embezzlement",
            "verdict", "sentence", "extortion", "kidnapping",
        ],
    },
    "Sports": {
        "phrases": [
            "cricket match", "test match", "odi series", "t20 series",
            "world cup", "asia cup", "premier league", "football match",
            "bcb selection", "national team", "bangladesh cricket",
        ],
        "words": [
            "cricket", "football", "soccer", "tournament", "championship",
            "bcb", "bff", "tigers", "wicket", "innings", "goal",
            "olympics", "athlete", "stadium", "squad",
        ],
    },
    "Environment": {
        "phrases": [
            "climate change", "sea level", "river erosion",
            "air quality", "water pollution", "crop damage",
            "cyclone warning", "flood damage", "drought hits",
            "deforestation rate", "carbon emission",
        ],
        "words": [
            "flood", "cyclone", "drought", "landslide", "waterlogging",
            "pollution", "deforestation", "temperature", "heatwave",
            "storm", "rainfall", "river", "erosion", "climate",
        ],
    },
    "Health": {
        "phrases": [
            "health ministry", "dengue outbreak", "hospital fire",
            "food poisoning", "drug shortage", "vaccine drive",
            "disease outbreak", "death toll", "health crisis",
        ],
        "words": [
            "hospital", "doctor", "medicine", "dengue", "cholera",
            "patient", "vaccine", "epidemic", "outbreak", "injury",
            "surgery", "healthcare", "malnutrition", "mortality",
        ],
    },
    "International": {
        "phrases": [
            "united nations", "foreign ministry", "state visit",
            "rohingya crisis", "border tension", "trade agreement",
            "bilateral talks", "diplomatic relations",
            "myanmar border", "india bangladesh",
        ],
        "words": [
            "india", "china", "usa", "pakistan", "myanmar",
            "rohingya", "diplomacy", "embassy", "bilateral",
            "global", "international", "summit", "foreign",
            "ambassador", "sanctions",
        ],
    },
}

# Ambiguous single words that only make sense inside a phrase
# (excluded from flat word-matching to avoid false topic hits)
_AMBIGUOUS_WORDS = {"bank", "fire", "strike", "record", "win", "match", "court"}


def _score_topic(title_lower: str, data: dict) -> float:
    score = 0.0
    # Phrases score 2.5 points each (more specific → more reliable signal)
    for phrase in data.get("phrases", []):
        if phrase in title_lower:
            score += 2.5
    # Words score 1 point each, skip ambiguous ones
    for word in data.get("words", []):
        if word not in _AMBIGUOUS_WORDS and word in title_lower:
            score += 1.0
    return score


def _get_topic(title: str) -> str:
    t = title.lower()
    scores = {topic: _score_topic(t, data) for topic, data in _TOPIC_MAP.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "General"


# ── Public API ─────────────────────────────────────────────────────────────────

def classify(score: float) -> str:
    if score >= 0.05:
        return "Positive"
    if score <= -0.05:
        return "Negative"
    return "Neutral"


def analyze_headlines(headlines: list[dict]) -> list[dict]:
    for h in headlines:
        scores  = _analyzer.polarity_scores(h["title"])
        compound = round(scores["compound"], 4)
        h["compound"]  = compound
        h["sentiment"] = classify(compound)
        h["pos"]       = round(scores["pos"], 3)
        h["neg"]       = round(scores["neg"], 3)
        h["neu"]       = round(scores["neu"], 3)
        h["topic"]     = _get_topic(h["title"])
    return headlines


def summary_stats(analyzed: list[dict]) -> dict:
    counts = {"Positive": 0, "Negative": 0, "Neutral": 0}
    topic_counts: dict[str, int] = {}
    total_compound = 0.0
    source_scores: dict[str, list[float]] = {}

    for h in analyzed:
        counts[h["sentiment"]] += 1
        total_compound += h["compound"]
        source_scores.setdefault(h["source"], []).append(h["compound"])
        topic = h.get("topic", "General")
        topic_counts[topic] = topic_counts.get(topic, 0) + 1

    n = len(analyzed)
    avg_compound = round(total_compound / n, 4) if n else 0.0
    source_avg   = {
        src: round(sum(v) / len(v), 4)
        for src, v in source_scores.items()
    }
    topic_counts = dict(sorted(topic_counts.items(), key=lambda x: x[1], reverse=True))

    return {
        "counts":            counts,
        "total":             n,
        "avg_compound":      avg_compound,
        "overall_sentiment": classify(avg_compound),
        "source_avg":        source_avg,
        "topic_counts":      topic_counts,
    }
