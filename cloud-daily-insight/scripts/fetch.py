"""
fetch.py — Multi-source fetcher for cloud computing & SaaS content.

Pulls from cloud providers and SaaS vendors:
  1. Cloud Providers — AWS, Google Cloud, Azure, Alibaba Cloud, Tencent Cloud, 火山云(Volcengine)
  2. SaaS Vendors — Salesforce, ServiceNow, Adobe, Workday, Zoom, etc.
  3. Industry News — TechCrunch, VentureBeat, The Register, etc.

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
# RSS feeds: Cloud providers + SaaS + Industry
# ─────────────────────────────────────────────────────────────────────
RSS_FEEDS: list[dict[str, str]] = [
    # 头部云厂商
    {"name": "AWS Blog", "url": "https://aws.amazon.com/blogs/aws/feed/", "category": "cloud"},
    {"name": "Google Cloud Blog", "url": "https://cloudblog.withgoogle.com/rss", "category": "cloud"},
    {"name": "Microsoft Azure Blog", "url": "https://azure.microsoft.com/en-us/blog/feed/", "category": "cloud"},
    {"name": "Alibaba Cloud Blog", "url": "https://www.alibabacloud.com/blog/feed", "category": "cloud"},
    # Tencent Cloud 无官方 RSS，改用下方 scrape 抓取
    # 火山云/字节云 - 使用博客园
    {"name": "火山引擎博客", "url": "https://www.cnblogs.com/volcengine/rss", "category": "cloud"},
    # 头部 SaaS 厂商
    {"name": "Salesforce Blog", "url": "https://www.salesforce.com/blog/feed/", "category": "saas"},
    {"name": "ServiceNow Blog", "url": "https://www.servicenow.com/blogs/feed.xml", "category": "saas"},
    {"name": "Adobe Blog", "url": "https://blog.adobe.com/feed/", "category": "saas"},
    {"name": "Workday Blog", "url": "https://blog.workday.com/feed/", "category": "saas"},
    {"name": "Zoom Blog", "url": "https://blog.zoom.us/feed/", "category": "saas"},
    {"name": "Slack Blog", "url": "https://slack.com/blog/feed", "category": "saas"},
    {"name": "HubSpot Blog", "url": "https://blog.hubspot.com/feed", "category": "saas"},
    {"name": "Zendesk Blog", "url": "https://www.zendesk.com/blog/feed/", "category": "saas"},
    # 行业资讯
    {"name": "TechCrunch", "url": "https://techcrunch.com/feed/", "category": "industry"},
    {"name": "VentureBeat Enterprise", "url": "https://venturebeat.com/category/enterprise/feed/", "category": "industry"},
    {"name": "The Register", "url": "https://www.theregister.com/feed/", "category": "industry"},
    {"name": "Cloudflare Blog", "url": "https://blog.cloudflare.com/rss/", "category": "cloud"},
]

PROCESSED_PATH = Path(__file__).resolve().parent.parent / "data" / "processed.json"
HTTP_TIMEOUT = 20
HTTP_HEADERS = {
    "User-Agent": "Cloud-Daily-Insight/1.0 (https://github.com; bot)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Cloud/SaaS keywords for filtering when scraping generic feeds
CLOUD_KEYWORDS = re.compile(
    r"(cloud|aws|azure|gcp|google cloud|alibaba cloud|tencent cloud|volcengine|火山|"
    r"saas|iaas|paas|serverless|kubernetes|k8s|container|microservice|"
    r"salesforce|servicenow|workday|zoom|slack|hubspot|zendesk|"
    r"database|storage|compute|network|security|devops|mlops)",
    re.IGNORECASE,
)


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
                "category": feed_meta.get("category", "other"),
                "title": title.strip(),
                "url": link.strip(),
                "summary": summary.strip()[:1000],
                "published": published,
            })
    except Exception:
        logger.warning("Failed to fetch RSS: %s", feed_meta["name"], exc_info=True)
    return articles


# ─────────────────────────────────────────────────────────────────────
# Scrape fallbacks for sources without reliable RSS
# ─────────────────────────────────────────────────────────────────────

def _fetch_tencent_cloud_blog() -> list[dict[str, str]]:
    """腾讯云开发者社区 - 备用抓取"""
    articles: list[dict[str, str]] = []
    try:
        url = "https://cloud.tencent.com/developer/community"
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.select("a[href*='/developer/article/']")[:15]:
            href = a.get("href", "")
            title = a.get_text(strip=True)
            if title and len(title) > 5:
                full_url = href if href.startswith("http") else f"https://cloud.tencent.com{href}"
                articles.append({
                    "source": "Tencent Cloud Blog", "category": "cloud",
                    "title": title[:200], "url": full_url,
                    "summary": "", "published": "",
                })
    except Exception:
        logger.warning("Failed to fetch Tencent Cloud Blog", exc_info=True)
    return articles


def _fetch_volcengine_blog() -> list[dict[str, str]]:
    """火山引擎开发者社区 - 备用抓取"""
    articles: list[dict[str, str]] = []
    try:
        url = "https://developer.volcengine.com/articles"
        resp = requests.get(url, headers=HTTP_HEADERS, timeout=HTTP_TIMEOUT)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.select("a[href*='/articles/']")[:15]:
            href = a.get("href", "")
            title = a.get_text(strip=True)
            if title and len(title) > 5:
                full_url = href if href.startswith("http") else f"https://developer.volcengine.com{href}"
                articles.append({
                    "source": "火山引擎 Blog", "category": "cloud",
                    "title": title[:200], "url": full_url,
                    "summary": "", "published": "",
                })
    except Exception:
        logger.warning("Failed to fetch Volcengine Blog", exc_info=True)
    return articles


# ─────────────────────────────────────────────────────────────────────
# Metrics snapshot (for trend analysis — no dedup)
# ─────────────────────────────────────────────────────────────────────

def fetch_metrics_snapshot() -> dict[str, Any]:
    """
    采集当日指标快照，用于云服务采用趋势分析。

    返回结构化数据，不做去重（每日全量快照）。
    """
    date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    cloud_articles: list[dict[str, Any]] = []
    saas_articles: list[dict[str, Any]] = []

    for feed_meta in RSS_FEEDS:
        try:
            parsed = feedparser.parse(feed_meta["url"])
            cat = feed_meta.get("category", "other")
            for entry in parsed.entries[:10]:
                title = getattr(entry, "title", "").strip()
                summary = getattr(entry, "summary", "").strip()[:500]
                link = getattr(entry, "link", "").strip()
                if title:
                    item = {"title": title, "summary": summary, "url": link, "source": feed_meta["name"]}
                    if cat == "cloud":
                        cloud_articles.append(item)
                    elif cat == "saas":
                        saas_articles.append(item)
        except Exception:
            logger.warning("Failed to fetch for metrics: %s", feed_meta["name"], exc_info=True)

    return {
        "date": date,
        "cloud_articles": cloud_articles,
        "saas_articles": saas_articles,
    }


# ─────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────

def fetch_articles(
    max_per_source: int = 10,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    """
    Fetch fresh cloud/SaaS content from all sources.

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

    # ── Tencent Cloud（无官方 RSS，用 scrape）──────────────────────
    logger.info("Fetching Tencent Cloud Blog (scrape) …")
    t0 = time.monotonic()
    tc_articles = _fetch_tencent_cloud_blog()
    elapsed = time.monotonic() - t0
    _dedupe_and_collect(tc_articles, "Tencent Cloud Blog", "scrape", elapsed, len(tc_articles) > 0)

    # ── 火山引擎（补充 scrape，博客园 RSS 可能不全）──────────────────
    logger.info("Fetching 火山引擎 Blog (scrape) …")
    t0 = time.monotonic()
    ve_articles = _fetch_volcengine_blog()
    elapsed = time.monotonic() - t0
    _dedupe_and_collect(ve_articles, "火山引擎 Blog", "scrape", elapsed, len(ve_articles) > 0)

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
