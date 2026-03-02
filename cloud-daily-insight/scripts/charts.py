"""
charts.py — 自动生成趋势图（云计算版）。

根据指标历史数据生成：
  - 云厂商话题演化曲线
  - SaaS 厂商话题演化曲线
"""

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

CHARTS_DIR = Path(__file__).resolve().parent.parent / "content" / "charts"
METRICS_DIR = Path(__file__).resolve().parent.parent / "data" / "metrics"


def _ensure_charts_dir() -> Path:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    return CHARTS_DIR


def _load_metrics_history(days: int = 30) -> list[dict[str, Any]]:
    """加载最近 N 天的指标历史。"""
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    target = datetime.strptime(today, "%Y-%m-%d")
    records: list[dict[str, Any]] = []

    for i in range(days):
        d = (target - timedelta(days=i)).strftime("%Y-%m-%d")
        path = METRICS_DIR / f"{d}.json"
        if path.exists():
            try:
                records.append(json.loads(path.read_text(encoding="utf-8")))
            except Exception:
                logger.warning("Corrupt metrics file: %s", path, exc_info=True)

    records.sort(key=lambda r: r.get("date", ""))
    return records


def _compute_novelty_series(
    history: list[dict[str, Any]],
    key: str,
) -> list[tuple[str, float]]:
    """计算每日内容与前一日的语义新颖度。"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    series: list[tuple[str, float]] = []
    if len(history) < 2:
        return series

    for i in range(1, len(history)):
        curr = history[i]
        prev = history[i - 1]
        curr_items = curr.get(key, [])
        prev_items = prev.get(key, [])
        curr_combined = " ".join(
            f"{p.get('title', '')} {p.get('summary', '')}" for p in curr_items
        )
        prev_combined = " ".join(
            f"{p.get('title', '')} {p.get('summary', '')}" for p in prev_items
        )

        if not curr_combined.strip() or not prev_combined.strip():
            series.append((curr.get("date", ""), 0.0))
            continue

        try:
            vectorizer = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 4), max_features=3000
            )
            matrix = vectorizer.fit_transform([prev_combined, curr_combined])
            sim = cosine_similarity(matrix[1:2], matrix[0:1])[0, 0]
            novelty = 1.0 - float(sim)
            series.append((curr.get("date", ""), round(novelty, 3)))
        except Exception:
            series.append((curr.get("date", ""), 0.0))

    return series


def generate_trend_charts(date: str) -> list[Path]:
    """
    生成当日趋势图，保存到 content/charts/ 目录。

    返回生成的文件路径列表。
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        logger.warning("matplotlib not installed — skipping chart generation")
        return []

    _ensure_charts_dir()
    history = _load_metrics_history(days=30)
    if len(history) < 2:
        logger.info("Not enough metrics history for charts")
        return []

    paths: list[Path] = []

    # 1. 云厂商话题演化曲线
    cloud_series = _compute_novelty_series(history, "cloud_articles")
    if cloud_series:
        fig, ax = plt.subplots(figsize=(10, 4))
        x = [datetime.strptime(d, "%Y-%m-%d") for d, _ in cloud_series]
        y = [v for _, v in cloud_series]
        ax.plot(x, y, "o-", color="#2563eb", linewidth=2, markersize=6)
        ax.fill_between(x, y, alpha=0.3, color="#2563eb")
        ax.set_title("Cloud Providers — Topic Novelty Curve", fontsize=12)
        ax.set_ylabel("Novelty (1 - similarity to prev day)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(x) // 7)))
        plt.xticks(rotation=45)
        plt.tight_layout()
        p = CHARTS_DIR / f"{date}_cloud_novelty.png"
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close()
        paths.append(p)

    # 2. SaaS 厂商话题演化曲线
    saas_series = _compute_novelty_series(history, "saas_articles")
    if saas_series:
        fig, ax = plt.subplots(figsize=(10, 4))
        x = [datetime.strptime(d, "%Y-%m-%d") for d, _ in saas_series]
        y = [v for _, v in saas_series]
        ax.plot(x, y, "s-", color="#059669", linewidth=2, markersize=6)
        ax.fill_between(x, y, alpha=0.3, color="#059669")
        ax.set_title("SaaS Vendors — Topic Novelty Curve", fontsize=12)
        ax.set_ylabel("Novelty (1 - similarity to prev day)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(x) // 7)))
        plt.xticks(rotation=45)
        plt.tight_layout()
        p = CHARTS_DIR / f"{date}_saas_novelty.png"
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close()
        paths.append(p)

    for p in paths:
        logger.info("Chart saved: %s", p)
    return paths
