"""
fetch.py — RSS feed fetcher for AI-related sources.

Pulls articles from curated AI/ML RSS feeds, deduplicates against
previously processed entries, and returns today's fresh items.
"""

import json
import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser

logger = logging.getLogger(__name__)

RSS_FEEDS: list[dict[str, str]] = [
    {
        "name": "MIT Technology Review – AI",
        "url": "https://www.technologyreview.com/topic/artificial-intelligence/feed",
    },
    {
        "name": "OpenAI Blog",
        "url": "https://openai.com/blog/rss.xml",
    },
    {
        "name": "Google AI Blog",
        "url": "https://blog.google/technology/ai/rss/",
    },
    {
        "name": "The Batch (deeplearning.ai)",
        "url": "https://www.deeplearning.ai/the-batch/feed/",
    },
    {
        "name": "Hugging Face Blog",
        "url": "https://huggingface.co/blog/feed.xml",
    },
    {
        "name": "AI News (VentureBeat)",
        "url": "https://venturebeat.com/category/ai/feed/",
    },
    {
        "name": "Towards Data Science (Medium)",
        "url": "https://towardsdatascience.com/feed",
    },
    {
        "name": "arXiv cs.AI (recent)",
        "url": "https://rss.arxiv.org/rss/cs.AI",
    },
]

PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed.json"


def _article_id(url: str, title: str) -> str:
    raw = f"{url}|{title}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _load_processed() -> dict[str, Any]:
    if PROCESSED_PATH.exists():
        return json.loads(PROCESSED_PATH.read_text(encoding="utf-8"))
    return {"seen_ids": [], "last_run": None}


def _save_processed(data: dict[str, Any]) -> None:
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    PROCESSED_PATH.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _parse_feed(feed_meta: dict[str, str]) -> list[dict[str, str]]:
    """Parse a single RSS feed and return normalised article dicts."""
    articles: list[dict[str, str]] = []
    try:
        parsed = feedparser.parse(feed_meta["url"])
        for entry in parsed.entries[:15]:
            link = getattr(entry, "link", "")
            title = getattr(entry, "title", "")
            summary = getattr(entry, "summary", "")
            published = getattr(entry, "published", "")
            if not title:
                continue
            articles.append(
                {
                    "source": feed_meta["name"],
                    "title": title.strip(),
                    "url": link.strip(),
                    "summary": summary.strip()[:1000],
                    "published": published,
                }
            )
    except Exception:
        logger.warning("Failed to fetch feed: %s", feed_meta["name"], exc_info=True)
    return articles


def fetch_articles(max_per_source: int = 10) -> list[dict[str, str]]:
    """
    Fetch fresh AI articles from all RSS sources.

    Returns a list of article dicts that have not been seen before.
    Updates the processed.json ledger on disk.
    """
    processed = _load_processed()
    seen: set[str] = set(processed.get("seen_ids", []))

    fresh: list[dict[str, str]] = []

    for feed_meta in RSS_FEEDS:
        logger.info("Fetching %s …", feed_meta["name"])
        raw_articles = _parse_feed(feed_meta)

        count = 0
        for art in raw_articles:
            aid = _article_id(art["url"], art["title"])
            if aid in seen:
                continue
            seen.add(aid)
            fresh.append(art)
            count += 1
            if count >= max_per_source:
                break

        logger.info("  → %d new article(s)", count)

    processed["seen_ids"] = list(seen)
    processed["last_run"] = datetime.now(timezone.utc).isoformat()
    _save_processed(processed)

    logger.info("Total fresh articles: %d", len(fresh))
    return fresh


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    items = fetch_articles()
    for i, a in enumerate(items, 1):
        print(f"{i}. [{a['source']}] {a['title']}")
        print(f"   {a['url']}\n")
