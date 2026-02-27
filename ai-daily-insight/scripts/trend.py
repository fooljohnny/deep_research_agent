"""
trend.py — Historical trend comparison via topic vectors.

Stores daily analysis as topic vectors (TF-IDF embeddings), compares
today's vector against the past 30 days, and detects:
  1. Suddenly strengthening topics  — absent recently, strong today
  2. Continuously strengthening trends — growing over multiple days
  3. Newly emerging topics — never seen in the 30-day window

This is the layer that turns raw daily analysis into *real insight*.
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

HISTORY_DIR = Path(__file__).resolve().parent.parent / "data" / "history"

DIMENSION_NAMES = ["technology", "infrastructure", "application", "capital", "risk"]

DIMENSION_LABELS = {
    "technology": "技术层",
    "infrastructure": "基础设施",
    "application": "应用层",
    "capital": "资本信号",
    "risk": "风险信号",
}

MIN_HISTORY_DAYS = 3


# ─────────────────────────────────────────────────────────────────────
# Text extraction from analysis JSON
# ─────────────────────────────────────────────────────────────────────

def _extract_dimension_text(dim_data: dict[str, Any]) -> str:
    """Flatten a single dimension's analysis into a text string."""
    parts: list[str] = []
    skip_keys = {"has_change", "intensity", "evidence"}
    for key, val in dim_data.items():
        if key in skip_keys:
            continue
        if isinstance(val, str) and val:
            parts.append(val)
    for ev in dim_data.get("evidence", []):
        if isinstance(ev, dict):
            parts.append(ev.get("title", ""))
    return " ".join(parts)


def _extract_all_texts(analysis: dict[str, Any]) -> dict[str, str]:
    """Extract per-dimension text + combined text from a full analysis."""
    dims = analysis.get("dimensions", {})
    texts: dict[str, str] = {}
    for dim in DIMENSION_NAMES:
        texts[dim] = _extract_dimension_text(dims.get(dim, {}))

    combined_parts = [analysis.get("core_insight", ""), analysis.get("title", "")]
    combined_parts.extend(texts.values())
    texts["_combined"] = " ".join(combined_parts)
    return texts


def _extract_keywords(analysis: dict[str, Any]) -> list[str]:
    """Extract keywords from the analysis if the LLM provided them."""
    return analysis.get("keywords", [])


# ─────────────────────────────────────────────────────────────────────
# History persistence
# ─────────────────────────────────────────────────────────────────────

def save_daily(date: str, analysis: dict[str, Any]) -> Path:
    """Store today's analysis texts and keywords for future trend comparison."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    texts = _extract_all_texts(analysis)
    keywords = _extract_keywords(analysis)

    record = {
        "date": date,
        "title": analysis.get("title", ""),
        "core_insight": analysis.get("core_insight", ""),
        "dimension_texts": texts,
        "keywords": keywords,
    }
    path = HISTORY_DIR / f"{date}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("History saved: %s", path)
    return path


def load_history(before_date: str, days: int = 30) -> list[dict[str, Any]]:
    """Load up to `days` historical records before the given date."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)

    target = datetime.strptime(before_date, "%Y-%m-%d")
    records: list[dict[str, Any]] = []

    for i in range(1, days + 1):
        d = (target - timedelta(days=i)).strftime("%Y-%m-%d")
        path = HISTORY_DIR / f"{d}.json"
        if path.exists():
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                logger.warning("Corrupt history file: %s", path, exc_info=True)

    records.sort(key=lambda r: r["date"])
    logger.info("Loaded %d historical record(s) (up to %d days back)", len(records), days)
    return records


# ─────────────────────────────────────────────────────────────────────
# TF-IDF topic vectors
# ─────────────────────────────────────────────────────────────────────

def _build_tfidf_vectors(
    today_texts: dict[str, str],
    history: list[dict[str, Any]],
    dimension: str,
) -> tuple[np.ndarray | None, np.ndarray | None, TfidfVectorizer | None]:
    """
    Build TF-IDF vectors for a dimension across today + history.

    Returns (today_vec, history_matrix, vectorizer).
    history_matrix rows are ordered chronologically.
    Returns (None, None, None) if the corpus is too sparse.
    """
    corpus: list[str] = []
    for rec in history:
        corpus.append(rec.get("dimension_texts", {}).get(dimension, ""))
    corpus.append(today_texts.get(dimension, ""))

    non_empty = sum(1 for doc in corpus if doc.strip())
    if non_empty < 2:
        return None, None, None

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=5000,
        min_df=1,
    )

    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return None, None, None

    today_vec = matrix[-1:]
    history_matrix = matrix[:-1]
    return today_vec, history_matrix, vectorizer


# ─────────────────────────────────────────────────────────────────────
# Trend detection algorithms
# ─────────────────────────────────────────────────────────────────────

def _detect_sudden_spikes(
    today_vec: np.ndarray,
    history_matrix: np.ndarray,
    dates: list[str],
    dim_label: str,
) -> list[dict[str, Any]]:
    """
    Detect topics suddenly strengthening:
    high similarity to last 3 days but low to earlier history.
    """
    if history_matrix.shape[0] < MIN_HISTORY_DAYS:
        return []

    sims = cosine_similarity(today_vec, history_matrix).flatten()

    recent_n = min(3, len(sims))
    older_n = len(sims) - recent_n
    if older_n < 1:
        return []

    recent_sim = float(np.mean(sims[-recent_n:]))
    older_sim = float(np.mean(sims[:older_n]))

    if recent_sim > older_sim + 0.15 and recent_sim > 0.3:
        return [{
            "type": "sudden_spike",
            "dimension": dim_label,
            "description": (
                f"{dim_label}近3天内容相似度({recent_sim:.2f})显著高于"
                f"更早期均值({older_sim:.2f})，表明该方向话题突然增强。"
            ),
            "recent_similarity": round(recent_sim, 3),
            "older_similarity": round(older_sim, 3),
            "confidence": round(min((recent_sim - older_sim) / 0.3, 1.0), 2),
        }]
    return []


def _detect_continuous_trends(
    today_vec: np.ndarray,
    history_matrix: np.ndarray,
    dates: list[str],
    dim_label: str,
) -> list[dict[str, Any]]:
    """
    Detect continuously strengthening trends:
    similarity increasing monotonically over the last 5-7 days.
    """
    if history_matrix.shape[0] < 4:
        return []

    sims = cosine_similarity(today_vec, history_matrix).flatten()

    window = min(7, len(sims))
    recent_sims = list(sims[-window:])
    recent_sims.append(1.0)  # today vs itself = max

    increasing_count = sum(
        1 for i in range(1, len(recent_sims)) if recent_sims[i] >= recent_sims[i - 1]
    )
    ratio = increasing_count / (len(recent_sims) - 1) if len(recent_sims) > 1 else 0

    if ratio >= 0.6 and len(recent_sims) >= 4:
        trend_days = len(recent_sims)
        return [{
            "type": "continuous_trend",
            "dimension": dim_label,
            "description": (
                f"{dim_label}的话题向量在最近{trend_days}天内呈现持续增强趋势"
                f"（{increasing_count}/{trend_days - 1}天递增），表明该方向正在持续升温。"
            ),
            "trend_days": trend_days,
            "monotonic_ratio": round(ratio, 2),
            "confidence": round(ratio, 2),
        }]
    return []


def _detect_emerging_topics(
    today_vec: np.ndarray,
    history_matrix: np.ndarray,
    dim_label: str,
) -> list[dict[str, Any]]:
    """
    Detect newly emerging topics:
    today's content has very low similarity to ALL historical days.
    """
    if history_matrix.shape[0] < MIN_HISTORY_DAYS:
        return []

    sims = cosine_similarity(today_vec, history_matrix).flatten()
    max_sim = float(np.max(sims))
    mean_sim = float(np.mean(sims))

    if max_sim < 0.25 and mean_sim < 0.15:
        novelty = round(1.0 - max_sim, 2)
        return [{
            "type": "emerging_topic",
            "dimension": dim_label,
            "description": (
                f"{dim_label}出现全新话题（与历史最大相似度仅{max_sim:.2f}），"
                f"这可能是一个新兴方向的早期信号。"
            ),
            "max_historical_similarity": round(max_sim, 3),
            "mean_historical_similarity": round(mean_sim, 3),
            "novelty_score": novelty,
        }]
    return []


def _compute_keyword_trends(
    today_keywords: list[str],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """Track keyword frequency over the history window."""
    if not today_keywords:
        return {"new_keywords": [], "rising_keywords": [], "fading_keywords": []}

    history_kw_sets: list[set[str]] = []
    kw_freq: dict[str, int] = {}
    for rec in history:
        kws = set(rec.get("keywords", []))
        history_kw_sets.append(kws)
        for k in kws:
            kw_freq[k] = kw_freq.get(k, 0) + 1

    today_set = set(today_keywords)
    all_history_kws = set(kw_freq.keys())

    new_keywords = sorted(today_set - all_history_kws)

    recent_days = history_kw_sets[-5:] if len(history_kw_sets) >= 5 else history_kw_sets
    older_days = history_kw_sets[:-5] if len(history_kw_sets) > 5 else []

    rising = []
    for kw in today_set & all_history_kws:
        recent_count = sum(1 for s in recent_days if kw in s)
        older_count = sum(1 for s in older_days if kw in s) if older_days else 0
        older_rate = older_count / max(len(older_days), 1)
        recent_rate = recent_count / max(len(recent_days), 1)
        if recent_rate > older_rate + 0.2:
            rising.append(kw)

    fading = []
    for kw in all_history_kws - today_set:
        if kw_freq.get(kw, 0) >= 3:
            recent_count = sum(1 for s in recent_days if kw in s)
            if recent_count == 0:
                fading.append(kw)

    return {
        "new_keywords": new_keywords[:10],
        "rising_keywords": sorted(rising)[:10],
        "fading_keywords": sorted(fading)[:10],
    }


def _overall_novelty(
    today_texts: dict[str, str],
    history: list[dict[str, Any]],
) -> float:
    """Compute overall novelty score of today vs full history."""
    if not history:
        return 1.0

    today_combined = today_texts.get("_combined", "")
    if not today_combined.strip():
        return 1.0

    history_combined = [
        rec.get("dimension_texts", {}).get("_combined", "") for rec in history
    ]
    corpus = history_combined + [today_combined]

    non_empty = sum(1 for doc in corpus if doc.strip())
    if non_empty < 2:
        return 1.0

    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), max_features=5000
    )
    try:
        matrix = vectorizer.fit_transform(corpus)
    except ValueError:
        return 1.0

    sims = cosine_similarity(matrix[-1:], matrix[:-1]).flatten()
    return round(1.0 - float(np.mean(sims)), 3)


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def analyze_trends(analysis: dict[str, Any]) -> dict[str, Any]:
    """
    Main entry point: compare today's analysis against historical data.

    Returns a trend report dict suitable for injection into Stage-2.
    """
    today = analysis.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    today_texts = _extract_all_texts(analysis)
    today_keywords = _extract_keywords(analysis)
    history = load_history(before_date=today, days=30)

    if len(history) < MIN_HISTORY_DAYS:
        logger.info(
            "Only %d day(s) of history (need %d). Trend analysis skipped for now.",
            len(history), MIN_HISTORY_DAYS,
        )
        return {
            "has_enough_history": False,
            "history_days": len(history),
            "message": f"趋势分析需要至少{MIN_HISTORY_DAYS}天历史数据，当前仅{len(history)}天。",
            "signals": [],
            "keyword_trends": {"new_keywords": [], "rising_keywords": [], "fading_keywords": []},
            "overall_novelty": 1.0,
        }

    dates = [rec["date"] for rec in history]
    signals: list[dict[str, Any]] = []

    for dim in DIMENSION_NAMES:
        label = DIMENSION_LABELS[dim]
        today_vec, hist_mat, _ = _build_tfidf_vectors(today_texts, history, dim)

        if today_vec is None or hist_mat is None or hist_mat.shape[0] == 0:
            continue

        signals.extend(_detect_sudden_spikes(today_vec, hist_mat, dates, label))
        signals.extend(_detect_continuous_trends(today_vec, hist_mat, dates, label))
        signals.extend(_detect_emerging_topics(today_vec, hist_mat, label))

    keyword_trends = _compute_keyword_trends(today_keywords, history)
    novelty = _overall_novelty(today_texts, history)

    signals.sort(key=lambda s: s.get("confidence", 0), reverse=True)

    report = {
        "has_enough_history": True,
        "history_days": len(history),
        "overall_novelty": novelty,
        "signals": signals,
        "keyword_trends": keyword_trends,
    }

    signal_summary = ", ".join(
        f"{s['dimension']}({s['type']})" for s in signals[:5]
    ) or "无显著趋势信号"
    logger.info("Trend report: %d signal(s) — %s", len(signals), signal_summary)
    logger.info("Overall novelty: %.3f", novelty)

    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    sample = {
        "date": "2026-02-26",
        "title": "Agent范式加速",
        "core_insight": "Agent从概念走向产业级部署。",
        "keywords": ["agent", "RAG", "inference", "multimodal"],
        "dimensions": {
            "technology": {
                "has_change": True,
                "intensity": "渐进改善",
                "model_capability": "开源模型推理能力逼近闭源前沿。",
                "new_paradigm": "Agent编排框架标准化。",
                "evidence": [{"title": "Test", "source": "arXiv", "url": "https://arxiv.org"}],
            },
            "infrastructure": {
                "has_change": False,
                "intensity": "无显著变化",
                "inference_cost": "",
                "compute_trend": "",
                "evidence": [],
            },
            "application": {
                "has_change": False,
                "intensity": "无显著变化",
                "new_industries": "",
                "displacement": "",
                "evidence": [],
            },
            "capital": {
                "has_change": False,
                "intensity": "弱信号",
                "funding_trend": "",
                "valuation": "",
                "strategic_moves": "",
                "evidence": [],
            },
            "risk": {
                "has_change": False,
                "intensity": "弱信号",
                "regulation": "",
                "ethics_safety": "",
                "supply_chain": "",
                "evidence": [],
            },
        },
    }

    save_daily("2026-02-26", sample)
    report = analyze_trends(sample)
    print(json.dumps(report, indent=2, ensure_ascii=False))
