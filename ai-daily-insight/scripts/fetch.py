"""
fetch.py — Multi-source fetcher for AI-related content.

Pulls from four categories of sources:
  1. Papers    — arXiv cs.AI / cs.LG / cs.CL
  2. Company   — OpenAI, Anthropic, Google DeepMind, Meta AI
  3. Open-source — GitHub Trending (AI/ML), HuggingFace trending models
  4. Industry  — TechCrunch AI, VentureBeat AI, Crunchbase News

Deduplicates against previously processed entries and returns fresh items.
"""

import json
import hashlib
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import feedparser
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# RSS feeds organised by category
# ─────────────────────────────────────────────────────────────────────
RSS_FEEDS: list[dict[str, str]] = [
    # ── 1. 论文 (Papers) ──────────────────────────────────────────
    {
        "name": "arXiv cs.AI",
        "url": "https://rss.arxiv.org/rss/cs.AI",
        "category": "papers",
    },
    {
        "name": "arXiv cs.LG",
        "url": "https://rss.arxiv.org/rss/cs.LG",
        "category": "papers",
    },
    {
        "name": "arXiv cs.CL",
        "url": "https://rss.arxiv.org/rss/cs.CL",
        "category": "papers",
    },

    # ── 2. 公司动态 (Company Updates) ─────────────────────────────
    {
        "name": "OpenAI Blog",
        "url": "https://openai.com/blog/rss.xml",
        "category": "company",
    },
    {
        "name": "Google DeepMind Blog",
        "url": "https://deepmind.google/blog/rss.xml",
        "category": "company",
    },

    # ── 4. 资本与行业 (Capital & Industry) ────────────────────────
    {
        "name": "TechCrunch AI",
        "url": "https://techcrunch.com/category/artificial-intelligence/feed/",
        "category": "industry",
    },
    {
        "name": "VentureBeat AI",
        "url": "https://venturebeat.com/category/ai/feed/",
        "category": "industry",
    },
    {
        "name": "Crunchbase News",
        "url": "https://news.crunchbase.com/feed/",
        "category": "industry",
    },
]

PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed.json"
HTTP_TIMEOUT = 20
HTTP_HEADERS = {
    "User-Agent": "AI-Daily-Insight/1.0 (https://github.com; bot)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


# ─────────────────────────────────────────────────────────────────────
# Dedup helpers
# ─────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────
# RSS parser
# ─────────────────────────────────────────────────────────────────────

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
                    "category": feed_meta.get("category", ""),
                    "title": title.strip(),
                    "url": link.strip(),
                    "summary": summary.strip()[:1000],
                    "published": published,
                }
            )
    except Exception:
        logger.warning("Failed to fetch RSS: %s", feed_meta["name"], exc_info=True)
    return articles


# ─────────────────────────────────────────────────────────────────────
# 3a. GitHub Trending (AI/ML repos)
# ─────────────────────────────────────────────────────────────────────

_GH_TRENDING_URL = "https://github.com/trending/python?since=daily"

AI_KEYWORDS = re.compile(
    r"(machine.?learning|deep.?learning|neural|transformer|llm|gpt|diffusion|"
    r"langchain|rag|agent|vision|nlp|reinforcement|generative|embedding|"
    r"fine.?tun|train|inference|model|ai\b|ml\b)",
    re.IGNORECASE,
)


def _fetch_github_trending() -> list[dict[str, str]]:
    """Scrape GitHub Trending for Python repos, filter to AI-related."""
    articles: list[dict[str, str]] = []
    try:
        resp = requests.get(
            _GH_TRENDING_URL, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        for row in soup.select("article.Box-row"):
            h2 = row.select_one("h2 a")
            if not h2:
                continue
            repo_path = h2.get("href", "").strip("/")
            repo_url = f"https://github.com/{repo_path}"
            repo_name = repo_path.replace("/", " / ")

            desc_tag = row.select_one("p")
            desc = desc_tag.get_text(strip=True) if desc_tag else ""

            stars_tag = row.select_one("span.d-inline-block.float-sm-right")
            stars = stars_tag.get_text(strip=True) if stars_tag else ""

            combined = f"{repo_name} {desc}"
            if not AI_KEYWORDS.search(combined):
                continue

            summary = desc
            if stars:
                summary = f"{desc}  ⭐ {stars}" if desc else f"⭐ {stars}"

            articles.append(
                {
                    "source": "GitHub Trending",
                    "category": "opensource",
                    "title": repo_name,
                    "url": repo_url,
                    "summary": summary[:1000],
                    "published": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                }
            )
    except Exception:
        logger.warning("Failed to fetch GitHub Trending", exc_info=True)

    logger.info("GitHub Trending: %d AI-related repo(s) found", len(articles))
    return articles


# ─────────────────────────────────────────────────────────────────────
# 3b. HuggingFace Trending Models
# ─────────────────────────────────────────────────────────────────────

_HF_API_URL = "https://huggingface.co/api/models"


def _fetch_huggingface_trending(limit: int = 15) -> list[dict[str, str]]:
    """Fetch trending models from the HuggingFace API."""
    articles: list[dict[str, str]] = []
    try:
        resp = requests.get(
            _HF_API_URL,
            params={"sort": "trendingScore", "direction": "-1", "limit": limit},
            headers=HTTP_HEADERS,
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        models = resp.json()

        for m in models:
            model_id: str = m.get("id", "")
            if not model_id:
                continue
            pipeline = m.get("pipeline_tag", "")
            downloads = m.get("downloads", 0)
            likes = m.get("likes", 0)

            summary_parts = []
            if pipeline:
                summary_parts.append(f"Pipeline: {pipeline}")
            summary_parts.append(f"Downloads: {downloads:,}")
            summary_parts.append(f"Likes: {likes:,}")

            articles.append(
                {
                    "source": "HuggingFace Trending",
                    "category": "opensource",
                    "title": model_id,
                    "url": f"https://huggingface.co/{model_id}",
                    "summary": "  |  ".join(summary_parts),
                    "published": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
                }
            )
    except Exception:
        logger.warning("Failed to fetch HuggingFace trending", exc_info=True)

    logger.info("HuggingFace Trending: %d model(s) found", len(articles))
    return articles


# ─────────────────────────────────────────────────────────────────────
# 2b. Anthropic Blog (no RSS — HTML scrape)
# ─────────────────────────────────────────────────────────────────────

_ANTHROPIC_NEWS_URL = "https://www.anthropic.com/news"


def _fetch_anthropic_blog() -> list[dict[str, str]]:
    """Scrape Anthropic's /news page for recent posts."""
    articles: list[dict[str, str]] = []
    try:
        resp = requests.get(
            _ANTHROPIC_NEWS_URL, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        seen_hrefs: set[str] = set()
        for a_tag in soup.find_all("a", href=True):
            href: str = a_tag["href"]
            if not href.startswith("/news/") or href == "/news/":
                continue
            if href in seen_hrefs:
                continue
            seen_hrefs.add(href)

            title = a_tag.get_text(strip=True)
            if not title or len(title) < 10:
                continue

            articles.append(
                {
                    "source": "Anthropic Blog",
                    "category": "company",
                    "title": title[:200],
                    "url": f"https://www.anthropic.com{href}",
                    "summary": "",
                    "published": "",
                }
            )
            if len(articles) >= 15:
                break
    except Exception:
        logger.warning("Failed to fetch Anthropic Blog", exc_info=True)

    logger.info("Anthropic Blog: %d post(s) found", len(articles))
    return articles


# ─────────────────────────────────────────────────────────────────────
# 2c. Meta AI Blog (no RSS — HTML scrape)
# ─────────────────────────────────────────────────────────────────────

_META_AI_BLOG_URL = "https://ai.meta.com/blog/"


def _fetch_meta_ai_blog() -> list[dict[str, str]]:
    """Scrape Meta AI's /blog/ page for recent posts."""
    articles: list[dict[str, str]] = []
    try:
        resp = requests.get(
            _META_AI_BLOG_URL, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT
        )
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        seen_hrefs: set[str] = set()
        for a_tag in soup.find_all("a", href=True):
            href: str = a_tag["href"]
            if "/blog/" not in href or href.rstrip("/") == "https://ai.meta.com/blog":
                continue

            title = a_tag.get_text(strip=True)
            if not title or len(title) < 10:
                continue

            url = href if href.startswith("http") else f"https://ai.meta.com{href}"
            if url in seen_hrefs:
                continue
            seen_hrefs.add(url)

            articles.append(
                {
                    "source": "Meta AI Blog",
                    "category": "company",
                    "title": title[:200],
                    "url": url,
                    "summary": "",
                    "published": "",
                }
            )
            if len(articles) >= 15:
                break
    except Exception:
        logger.warning("Failed to fetch Meta AI Blog", exc_info=True)

    logger.info("Meta AI Blog: %d post(s) found", len(articles))
    return articles


# ─────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────

def fetch_articles(max_per_source: int = 10) -> list[dict[str, str]]:
    """
    Fetch fresh AI content from all sources (RSS + custom).

    Returns a list of article dicts that have not been seen before.
    Updates the processed.json ledger on disk.
    """
    processed = _load_processed()
    seen: set[str] = set(processed.get("seen_ids", []))

    fresh: list[dict[str, str]] = []

    def _dedupe_and_collect(raw: list[dict[str, str]], label: str) -> None:
        count = 0
        for art in raw:
            aid = _article_id(art["url"], art["title"])
            if aid in seen:
                continue
            seen.add(aid)
            fresh.append(art)
            count += 1
            if count >= max_per_source:
                break
        logger.info("  → %s: %d new item(s)", label, count)

    # ── RSS feeds ──────────────────────────────────────────────────
    for feed_meta in RSS_FEEDS:
        logger.info("Fetching RSS: %s …", feed_meta["name"])
        raw = _parse_feed(feed_meta)
        _dedupe_and_collect(raw, feed_meta["name"])

    # ── Anthropic Blog (scrape) ────────────────────────────────────
    logger.info("Fetching Anthropic Blog …")
    anthropic_articles = _fetch_anthropic_blog()
    _dedupe_and_collect(anthropic_articles, "Anthropic Blog")

    # ── Meta AI Blog (scrape) ──────────────────────────────────────
    logger.info("Fetching Meta AI Blog …")
    meta_articles = _fetch_meta_ai_blog()
    _dedupe_and_collect(meta_articles, "Meta AI Blog")

    # ── GitHub Trending ────────────────────────────────────────────
    logger.info("Fetching GitHub Trending (AI/ML) …")
    gh_articles = _fetch_github_trending()
    _dedupe_and_collect(gh_articles, "GitHub Trending")

    # ── HuggingFace Trending ───────────────────────────────────────
    logger.info("Fetching HuggingFace Trending Models …")
    hf_articles = _fetch_huggingface_trending()
    _dedupe_and_collect(hf_articles, "HuggingFace Trending")

    # ── Persist ────────────────────────────────────────────────────
    processed["seen_ids"] = list(seen)
    processed["last_run"] = datetime.now(timezone.utc).isoformat()
    _save_processed(processed)

    logger.info("Total fresh items: %d", len(fresh))
    return fresh


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    items = fetch_articles()
    for i, a in enumerate(items, 1):
        print(f"{i}. [{a['source']}] {a['title']}")
        print(f"   {a['url']}\n")
