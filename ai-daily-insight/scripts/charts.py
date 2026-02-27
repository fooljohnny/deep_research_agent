"""
charts.py — 自动生成趋势图。

根据指标历史数据生成：
  - arXiv 模型能力提升曲线
  - GitHub star 增速趋势
  - HuggingFace 模型下载量变化
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


def _compute_arxiv_novelty_series(history: list[dict[str, Any]]) -> list[tuple[str, float]]:
    """计算每日 arXiv 摘要与前一日的语义新颖度（简化版：用摘要长度与关键词密度近似）。"""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np

    series: list[tuple[str, float]] = []
    if len(history) < 2:
        return series

    for i in range(1, len(history)):
        curr = history[i]
        prev = history[i - 1]
        curr_texts = [f"{p.get('title','')} {p.get('abstract','')}" for p in curr.get("arxiv_papers", [])]
        prev_texts = [f"{p.get('title','')} {p.get('abstract','')}" for p in prev.get("arxiv_papers", [])]
        curr_combined = " ".join(curr_texts) if curr_texts else ""
        prev_combined = " ".join(prev_texts) if prev_texts else ""

        if not curr_combined.strip() or not prev_combined.strip():
            series.append((curr.get("date", ""), 0.0))
            continue

        try:
            vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=3000)
            matrix = vectorizer.fit_transform([prev_combined, curr_combined])
            sim = cosine_similarity(matrix[1:2], matrix[0:1])[0, 0]
            novelty = 1.0 - float(sim)
            series.append((curr.get("date", ""), round(novelty, 3)))
        except Exception:
            series.append((curr.get("date", ""), 0.0))

    return series


def _compute_github_star_series(history: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """计算每日 GitHub 总 star 数（仅统计能解析的）。"""
    import re

    def parse_stars(s: str) -> int:
        if not s:
            return 0
        m = re.search(r"([\d,\.]+)\s*(k|K)?\s*stars?", str(s), re.IGNORECASE)
        if not m:
            return 0
        try:
            val = float(m.group(1).replace(",", ""))
            if m.group(2):
                val *= 1000
            return int(val)
        except ValueError:
            return 0

    series: list[tuple[str, int]] = []
    for rec in history:
        total = 0
        for r in rec.get("github_repos", []):
            total += parse_stars(str(r.get("stars", "")))
        series.append((rec.get("date", ""), total))
    return series


def _compute_hf_download_series(history: list[dict[str, Any]]) -> list[tuple[str, int]]:
    """计算每日 HuggingFace 总下载量。"""
    series: list[tuple[str, int]] = []
    for rec in history:
        total = sum(m.get("downloads", 0) for m in rec.get("hf_models", []))
        series.append((rec.get("date", ""), total))
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
    dates = [r.get("date", "") for r in history]
    date_objs = [datetime.strptime(d, "%Y-%m-%d") for d in dates if d]

    # 1. arXiv 能力提升曲线（新颖度）
    arxiv_series = _compute_arxiv_novelty_series(history)
    if arxiv_series:
        fig, ax = plt.subplots(figsize=(10, 4))
        x = [datetime.strptime(d, "%Y-%m-%d") for d, _ in arxiv_series]
        y = [v for _, v in arxiv_series]
        ax.plot(x, y, "o-", color="#2563eb", linewidth=2, markersize=6)
        ax.fill_between(x, y, alpha=0.3, color="#2563eb")
        ax.set_title("arXiv Abstract Novelty — Model Capability Curve", fontsize=12)
        ax.set_ylabel("Novelty (1 - similarity to prev day)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(x) // 7)))
        plt.xticks(rotation=45)
        plt.tight_layout()
        p = CHARTS_DIR / f"{date}_arxiv_capability.png"
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close()
        paths.append(p)

    # 2. GitHub star 增速
    gh_series = _compute_github_star_series(history)
    if gh_series:
        fig, ax = plt.subplots(figsize=(10, 4))
        x = [datetime.strptime(d, "%Y-%m-%d") for d, _ in gh_series]
        y = [v for _, v in gh_series]
        ax.bar(x, y, color="#238636", alpha=0.8, width=0.6)
        ax.set_title("GitHub Daily Star Total Trend", fontsize=12)
        ax.set_ylabel("Total Stars")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(x) // 7)))
        plt.xticks(rotation=45)
        plt.tight_layout()
        p = CHARTS_DIR / f"{date}_github_stars.png"
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close()
        paths.append(p)

    # 3. HuggingFace 下载量变化
    hf_series = _compute_hf_download_series(history)
    if hf_series:
        fig, ax = plt.subplots(figsize=(10, 4))
        x = [datetime.strptime(d, "%Y-%m-%d") for d, _ in hf_series]
        y = [v for _, v in hf_series]
        ax.plot(x, y, "s-", color="#ffd21e", linewidth=2, markersize=6)
        ax.fill_between(x, y, alpha=0.3, color="#ffd21e")
        ax.set_title("HuggingFace Model Daily Downloads", fontsize=12)
        ax.set_ylabel("Downloads")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(x) // 7)))
        plt.xticks(rotation=45)
        plt.tight_layout()
        p = CHARTS_DIR / f"{date}_hf_downloads.png"
        fig.savefig(p, dpi=100, bbox_inches="tight")
        plt.close()
        paths.append(p)

    for p in paths:
        logger.info("Chart saved: %s", p)
    return paths
