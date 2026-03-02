"""
metrics.py — 每日指标采集与趋势分析（云计算版）。

采集并持久化云/SaaS 相关数据，用于洞察报告：
  1. 云厂商内容 — 话题演化趋势
  2. SaaS 厂商内容 — 话题演化趋势

数据存储在 data/metrics/ 目录，按日期分文件。
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

METRICS_DIR = Path(__file__).resolve().parent.parent / "data" / "metrics"
MIN_HISTORY_DAYS = 3


def save_daily_metrics(date: str, metrics: dict[str, Any]) -> Path:
    """保存当日指标快照。"""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    path = METRICS_DIR / f"{date}.json"
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Metrics saved: %s", path)
    return path


def load_metrics_history(before_date: str, days: int = 30) -> list[dict[str, Any]]:
    """加载指定日期之前的 N 天指标历史。"""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    target = datetime.strptime(before_date, "%Y-%m-%d")
    records: list[dict[str, Any]] = []

    for i in range(1, days + 1):
        d = (target - timedelta(days=i)).strftime("%Y-%m-%d")
        path = METRICS_DIR / f"{d}.json"
        if path.exists():
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                logger.warning("Corrupt metrics file: %s", path, exc_info=True)

    records.sort(key=lambda r: r.get("date", ""))
    return records


def _extract_cloud_texts(metrics: dict[str, Any]) -> list[str]:
    """从指标中提取云厂商内容文本。"""
    items = metrics.get("cloud_articles", [])
    return [f"{p.get('title', '')} {p.get('summary', '')}" for p in items if p.get("title")]


def _extract_saas_texts(metrics: dict[str, Any]) -> list[str]:
    """从指标中提取 SaaS 厂商内容文本。"""
    items = metrics.get("saas_articles", [])
    return [f"{p.get('title', '')} {p.get('summary', '')}" for p in items if p.get("title")]


def compute_topic_evolution(
    today_texts: list[str],
    history: list[dict[str, Any]],
    extract_fn,
    label: str,
) -> dict[str, Any]:
    """
    基于 TF-IDF 向量，计算话题演化曲线。
    通过对比今日内容与历史内容的语义相似度，得到延续/跃迁信号。
    """
    today_combined = " ".join(today_texts) if today_texts else ""
    if not today_combined.strip():
        return {
            "has_data": False,
            "message": f"今日无{label}数据",
            "novelty_score": None,
            "trend": "unknown",
        }

    history_combined = []
    for rec in history:
        texts = extract_fn(rec)
        history_combined.append(" ".join(texts) if texts else "")

    if len([t for t in history_combined if t.strip()]) < MIN_HISTORY_DAYS:
        return {
            "has_data": True,
            "message": f"历史数据不足（需至少 {MIN_HISTORY_DAYS} 天）",
            "novelty_score": None,
            "trend": "unknown",
            "history_days": len([t for t in history_combined if t.strip()]),
        }

    corpus = [c for c in history_combined if c.strip()] + [today_combined]
    if len(corpus) < 2:
        return {"has_data": True, "message": "语料不足", "novelty_score": None, "trend": "unknown"}

    try:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=5000,
        )
        matrix = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(matrix[-1:], matrix[:-1]).flatten()
        mean_sim = float(np.mean(sims))

        novelty = 1.0 - mean_sim
        if novelty > 0.4:
            trend = "跃迁"
        elif mean_sim > 0.5:
            trend = "延续"
        else:
            trend = "渐进"

        return {
            "has_data": True,
            "novelty_score": round(novelty, 3),
            "mean_similarity": round(mean_sim, 3),
            "trend": trend,
            "history_days": len(history),
            "article_count": len(today_texts),
        }
    except Exception:
        logger.warning("%s topic evolution computation failed", label, exc_info=True)
        return {"has_data": True, "message": "计算失败", "novelty_score": None, "trend": "unknown"}


def analyze_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    综合指标分析入口。

    接收当日指标快照，与历史对比，返回 cloud / saas 两部分分析结果。
    """
    date = metrics.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    history = load_metrics_history(before_date=date, days=30)

    cloud_texts = _extract_cloud_texts(metrics)
    saas_texts = _extract_saas_texts(metrics)

    cloud_report = compute_topic_evolution(
        cloud_texts, history, _extract_cloud_texts, "云厂商"
    )
    saas_report = compute_topic_evolution(
        saas_texts, history, _extract_saas_texts, "SaaS厂商"
    )

    report = {
        "date": date,
        "cloud_topic_evolution": cloud_report,
        "saas_topic_evolution": saas_report,
    }

    logger.info(
        "Metrics report: Cloud=%s, SaaS=%s",
        cloud_report.get("trend", "n/a"),
        saas_report.get("trend", "n/a"),
    )
    return report
