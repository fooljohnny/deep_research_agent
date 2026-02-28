"""
fetch.py — Multi-source fetcher for AI-related content.

Pulls from four categories of sources:
  1. Papers    — arXiv cs.AI / cs.LG / cs.CL
  2. Company   — OpenAI, Anthropic, Google DeepMind, Google AI, Meta AI
  3. Open-source — GitHub Trending (AI/ML), HuggingFace trending models
  4. Industry  — TechCrunch AI, VentureBeat AI, Crunchbase News

Deduplicates against previously processed entries and returns fresh items.
"""

import json
import hashlib
import logging
import re
import time
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
    {"name": "arXiv cs.AI",        "url": "https://rss.arxiv.org/rss/cs.AI",        "category": "papers"},
    {"name": "arXiv cs.LG",        "url": "https://rss.arxiv.org/rss/cs.LG",        "category": "papers"},
    {"name": "arXiv cs.CL",        "url": "https://rss.arxiv.org/rss/cs.CL",        "category": "papers"},
    {"name": "OpenAI Blog",        "url": "https://openai.com/blog/rss.xml",        "category": "company"},
    {"name": "Google DeepMind Blog", "url": "https://deepmind.google/blog/rss.xml",   "category": "company"},
    {"name": "Google AI Blog",       "url": "https://research.google/blog/rss/", "category": "company"},
    {"name": "TechCrunch AI",      "url": "https://techcrunch.com/category/artificial-intelligence/feed/", "category": "industry"},
    {"name": "VentureBeat AI",     "url": "https://venturebeat.com/category/ai/feed/","category": "industry"},
    {"name": "Crunchbase News",    "url": "https://news.crunchbase.com/feed/",       "category": "industry"},
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
            articles.append({
                "source": feed_meta["name"],
                "category": feed_meta.get("category", ""),
                "title": title.strip(),
                "url": link.strip(),
                "summary": summary.strip()[:1000],
                "published": published,
            })
    except Exception:
        logger.warning("Failed to fetch RSS: %s", feed_meta["name"], exc_info=True)
    return articles


# ─────────────────────────────────────────────────────────────────────
# GitHub Trending (AI/ML repos)
# ─────────────────────────────────────────────────────────────────────

_GH_TRENDING_URL = "https://github.com/trending/python?since=daily"

AI_KEYWORDS = re.compile(
    r"(machine.?learning|deep.?learning|neural|transformer|llm|gpt|diffusion|"
    r"langchain|rag|agent|vision|nlp|reinforcement|generative|embedding|"
    r"fine.?tun|train|inference|model|ai\b|ml\b)",
    re.IGNORECASE,
)


def _fetch_github_trending() -> list[dict[str, str]]:
    articles: list[dict[str, str]] = []
    try:
        resp = requests.get(_GH_TRENDING_URL, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
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
            if not AI_KEYWORDS.search(f"{repo_name} {desc}"):
                continue
            summary = f"{desc}  ⭐ {stars}" if stars else desc
            articles.append({
                "source": "GitHub Trending", "category": "opensource",
                "title": repo_name, "url": repo_url,
                "summary": summary[:1000],
                "published": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            })
    except Exception:
        logger.warning("Failed to fetch GitHub Trending", exc_info=True)
    return articles


# ─────────────────────────────────────────────────────────────────────
# HuggingFace Trending Models
# ─────────────────────────────────────────────────────────────────────

_HF_API_URL = "https://huggingface.co/api/models"


def _fetch_huggingface_trending(limit: int = 15) -> list[dict[str, str]]:
    articles: list[dict[str, str]] = []
    try:
        resp = requests.get(
            _HF_API_URL,
            params={"sort": "trendingScore", "direction": "-1", "limit": limit},
            headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        for m in resp.json():
            model_id = m.get("id", "")
            if not model_id:
                continue
            pipeline = m.get("pipeline_tag", "")
            downloads = m.get("downloads", 0)
            likes = m.get("likes", 0)
            parts = []
            if pipeline:
                parts.append(f"Pipeline: {pipeline}")
            parts.append(f"Downloads: {downloads:,}")
            parts.append(f"Likes: {likes:,}")
            articles.append({
                "source": "HuggingFace Trending", "category": "opensource",
                "title": model_id, "url": f"https://huggingface.co/{model_id}",
                "summary": "  |  ".join(parts),
                "published": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
            })
    except Exception:
        logger.warning("Failed to fetch HuggingFace trending", exc_info=True)
    return articles


# ─────────────────────────────────────────────────────────────────────
# Anthropic Blog (no RSS — HTML scrape)
# ─────────────────────────────────────────────────────────────────────

_ANTHROPIC_NEWS_URL = "https://www.anthropic.com/news"


def _fetch_anthropic_blog() -> list[dict[str, str]]:
    articles: list[dict[str, str]] = []
    try:
        resp = requests.get(_ANTHROPIC_NEWS_URL, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
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
            articles.append({
                "source": "Anthropic Blog", "category": "company",
                "title": title[:200], "url": f"https://www.anthropic.com{href}",
                "summary": "", "published": "",
            })
            if len(articles) >= 15:
                break
    except Exception:
        logger.warning("Failed to fetch Anthropic Blog", exc_info=True)
    return articles


# ─────────────────────────────────────────────────────────────────────
# Meta AI Blog (no RSS — HTML scrape)
# ─────────────────────────────────────────────────────────────────────

_META_AI_BLOG_URL = "https://ai.meta.com/blog/"


def _fetch_meta_ai_blog() -> list[dict[str, str]]:
    articles: list[dict[str, str]] = []
    try:
        resp = requests.get(_META_AI_BLOG_URL, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
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
            articles.append({
                "source": "Meta AI Blog", "category": "company",
                "title": title[:200], "url": url,
                "summary": "", "published": "",
            })
            if len(articles) >= 15:
                break
    except Exception:
        logger.warning("Failed to fetch Meta AI Blog", exc_info=True)
    return articles


# ─────────────────────────────────────────────────────────────────────
# Metrics snapshot (for trend analysis — no dedup)
# ─────────────────────────────────────────────────────────────────────

def fetch_metrics_snapshot() -> dict[str, Any]:
    """
    采集当日指标快照，用于 arXiv 能力曲线、GitHub star 增速、HF 下载量分析。

    返回结构化数据，不做去重（每日全量快照）。
    """
    from datetime import datetime, timezone

    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    arxiv_papers: list[dict[str, Any]] = []
    github_repos: list[dict[str, Any]] = []
    hf_models: list[dict[str, Any]] = []

    # arXiv — 从 RSS 获取
    for feed_meta in RSS_FEEDS:
        if "arXiv" not in feed_meta["name"]:
            continue
        try:
            parsed = feedparser.parse(feed_meta["url"])
            for entry in parsed.entries[:15]:
                title = getattr(entry, "title", "").strip()
                summary = getattr(entry, "summary", "").strip()[:2000]
                link = getattr(entry, "link", "").strip()
                if title:
                    arxiv_papers.append({
                        "title": title,
                        "abstract": summary,
                        "url": link,
                        "source": feed_meta["name"],
                    })
        except Exception:
            logger.warning("Failed to fetch arXiv for metrics: %s", feed_meta["name"], exc_info=True)

    # GitHub — 带 star 数
    try:
        resp = requests.get(_GH_TRENDING_URL, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for row in soup.select("article.Box-row"):
            h2 = row.select_one("h2 a")
            if not h2:
                continue
            repo_path = h2.get("href", "").strip("/")
            stars_tag = row.select_one("span.d-inline-block.float-sm-right")
            stars = stars_tag.get_text(strip=True) if stars_tag else ""
            desc_tag = row.select_one("p")
            desc = desc_tag.get_text(strip=True) if desc_tag else ""
            if not AI_KEYWORDS.search(f"{repo_path} {desc}"):
                continue
            github_repos.append({
                "repo": repo_path,
                "stars": stars,
                "url": f"https://github.com/{repo_path}",
                "description": desc[:500],
            })
    except Exception:
        logger.warning("Failed to fetch GitHub for metrics", exc_info=True)

    # HuggingFace — 带 downloads
    try:
        resp = requests.get(
            _HF_API_URL,
            params={"sort": "trendingScore", "direction": "-1", "limit": 20},
            headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        for m in resp.json():
            model_id = m.get("id", "")
            if not model_id:
                continue
            hf_models.append({
                "model_id": model_id,
                "downloads": m.get("downloads", 0),
                "likes": m.get("likes", 0),
                "pipeline_tag": m.get("pipeline_tag", ""),
                "url": f"https://huggingface.co/{model_id}",
            })
    except Exception:
        logger.warning("Failed to fetch HuggingFace for metrics", exc_info=True)

    return {
        "date": date,
        "arxiv_papers": arxiv_papers,
        "github_repos": github_repos,
        "hf_models": hf_models,
    }


# ─────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────

def fetch_articles(
    max_per_source: int = 10,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """
    Fetch fresh AI content from all sources.

    Returns:
      (articles, fetch_stats) — articles list and per-source stats list.
    """
    processed = _load_processed()
    seen: set[str] = set(processed.get("seen_ids", []))

    fresh: list[dict[str, str]] = []
    stats: list[dict[str, Any]] = []

    def _dedupe_and_collect(
        raw: list[dict[str, str]],
        label: str,
        method: str,
        elapsed: float,
        ok: bool,
    ) -> None:
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
        stats.append({
            "source": label,
            "method": method,
            "status": "ok" if ok else "fail",
            "fetched": len(raw),
            "new": count,
            "elapsed_s": round(elapsed, 2),
        })

    # ── RSS feeds ──────────────────────────────────────────────────
    for feed_meta in RSS_FEEDS:
        logger.info("Fetching RSS: %s …", feed_meta["name"])
        t0 = time.monotonic()
        raw = _parse_feed(feed_meta)
        elapsed = time.monotonic() - t0
        _dedupe_and_collect(raw, feed_meta["name"], "RSS", elapsed, len(raw) > 0)

    # ── Anthropic Blog (scrape) ────────────────────────────────────
    logger.info("Fetching Anthropic Blog …")
    t0 = time.monotonic()
    anthropic_articles = _fetch_anthropic_blog()
    elapsed = time.monotonic() - t0
    _dedupe_and_collect(anthropic_articles, "Anthropic Blog", "scrape", elapsed, len(anthropic_articles) > 0)

    # ── Meta AI Blog (scrape) ──────────────────────────────────────
    logger.info("Fetching Meta AI Blog …")
    t0 = time.monotonic()
    meta_articles = _fetch_meta_ai_blog()
    elapsed = time.monotonic() - t0
    _dedupe_and_collect(meta_articles, "Meta AI Blog", "scrape", elapsed, len(meta_articles) > 0)

    # ── GitHub Trending ────────────────────────────────────────────
    logger.info("Fetching GitHub Trending (AI/ML) …")
    t0 = time.monotonic()
    gh_articles = _fetch_github_trending()
    elapsed = time.monotonic() - t0
    _dedupe_and_collect(gh_articles, "GitHub Trending", "scrape", elapsed, len(gh_articles) > 0)

    # ── HuggingFace Trending ───────────────────────────────────────
    logger.info("Fetching HuggingFace Trending Models …")
    t0 = time.monotonic()
    hf_articles = _fetch_huggingface_trending()
    elapsed = time.monotonic() - t0
    _dedupe_and_collect(hf_articles, "HuggingFace Trending", "API", elapsed, len(hf_articles) > 0)

    # ── Persist ────────────────────────────────────────────────────
    processed["seen_ids"] = list(seen)
    processed["last_run"] = datetime.now(timezone.utc).isoformat()
    _save_processed(processed)

    logger.info("Total fresh items: %d", len(fresh))
    return fresh, stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    items, stats = fetch_articles()
    for i, a in enumerate(items, 1):
        print(f"{i}. [{a['source']}] {a['title']}")
        print(f"   {a['url']}\n")
    print("\n--- Fetch Stats ---")
    for s in stats:
        print(f"  {s['source']:25s}  {s['method']:6s}  {s['status']:4s}  "
              f"fetched={s['fetched']:3d}  new={s['new']:3d}  {s['elapsed_s']:.2f}s")
