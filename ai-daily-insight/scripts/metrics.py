"""
metrics.py — 每日指标采集与趋势分析。

采集并持久化以下数据，用于洞察报告：
  1. arXiv 摘要 — 对比模型能力提升曲线（基于摘要语义演化）
  2. GitHub star — 增速分析
  3. HuggingFace 模型下载量 — 变化趋势

数据存储在 data/metrics/ 目录，按日期分文件。
"""

import json
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

METRICS_DIR = Path(__file__).resolve().parent.parent / "data" / "metrics"
MIN_HISTORY_DAYS = 3


# ─────────────────────────────────────────────────────────────────────
# Persistence
# ─────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────
# arXiv 摘要对比 — 模型能力提升曲线
# ─────────────────────────────────────────────────────────────────────

def _extract_arxiv_texts(metrics: dict[str, Any]) -> list[str]:
    """从指标中提取 arXiv 摘要文本。"""
    papers = metrics.get("arxiv_papers", [])
    return [f"{p.get('title', '')} {p.get('abstract', '')}" for p in papers if p.get("abstract") or p.get("title")]


def compute_arxiv_capability_curve(
    today_metrics: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    基于 arXiv 摘要的 TF-IDF 向量，计算「模型能力相关」话题的演化曲线。

    通过对比今日摘要与历史摘要的语义相似度，得到能力话题的延续/跃迁信号。
    """
    today_texts = _extract_arxiv_texts(today_metrics)
    if not today_texts:
        return {
            "has_data": False,
            "message": "今日无 arXiv 摘要数据",
            "capability_score": None,
            "trend": "unknown",
        }

    today_combined = " ".join(today_texts)
    history_combined = []
    for rec in history:
        texts = _extract_arxiv_texts(rec)
        history_combined.append(" ".join(texts) if texts else "")

    if len([t for t in history_combined if t.strip()]) < MIN_HISTORY_DAYS:
        return {
            "has_data": True,
            "message": f"历史数据不足（需至少 {MIN_HISTORY_DAYS} 天）",
            "capability_score": None,
            "trend": "unknown",
            "history_days": len([t for t in history_combined if t.strip()]),
        }

    corpus = [c for c in history_combined if c.strip()] + [today_combined]
    if len(corpus) < 2:
        return {"has_data": True, "message": "语料不足", "capability_score": None, "trend": "unknown"}

    try:
        vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(2, 4),
            max_features=5000,
        )
        matrix = vectorizer.fit_transform(corpus)
        sims = cosine_similarity(matrix[-1:], matrix[:-1]).flatten()
        mean_sim = float(np.mean(sims))
        max_sim = float(np.max(sims))

        # 能力提升曲线：与历史相似度低 = 新话题多 = 可能能力跃迁
        novelty = 1.0 - mean_sim
        if novelty > 0.4:
            trend = "跃迁"
        elif mean_sim > 0.5:
            trend = "延续"
        else:
            trend = "渐进"

        return {
            "has_data": True,
            "capability_score": round(novelty, 3),
            "mean_similarity": round(mean_sim, 3),
            "trend": trend,
            "history_days": len(history),
            "paper_count": len(today_texts),
        }
    except Exception:
        logger.warning("arXiv capability curve computation failed", exc_info=True)
        return {"has_data": True, "message": "计算失败", "capability_score": None, "trend": "unknown"}


# ─────────────────────────────────────────────────────────────────────
# GitHub star 增速分析
# ─────────────────────────────────────────────────────────────────────

def _parse_star_count(s: str) -> int | None:
    """解析 GitHub star 字符串，如 '1.2k stars today' 或 '123 stars today'。"""
    if not s:
        return None
    m = re.search(r"([\d,\.]+)\s*(k|K)?\s*stars?", s, re.IGNORECASE)
    if not m:
        return None
    num_str = m.group(1).replace(",", "")
    try:
        val = float(num_str)
        if m.group(2):
            val *= 1000
        return int(val)
    except ValueError:
        return None


def compute_github_star_growth(
    today_metrics: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    分析 GitHub 仓库 star 增速。

    对比同一 repo 在不同日期的 star 数，计算增速；对新出现的 repo 标记为「新晋」。
    """
    today_repos = {r["repo"]: r.get("stars") for r in today_metrics.get("github_repos", [])}
    if not today_repos:
        return {
            "has_data": False,
            "message": "今日无 GitHub 数据",
            "top_growth": [],
            "new_repos": [],
            "total_stars_today": 0,
        }

    # 解析 star 数
    repo_stars: dict[str, int] = {}
    for repo, stars in today_repos.items():
        if stars is not None:
            parsed = _parse_star_count(str(stars)) if isinstance(stars, str) else stars
            if isinstance(parsed, int):
                repo_stars[repo] = parsed

    total_stars = sum(repo_stars.values())

    # 历史对比
    hist_repo_stars: dict[str, list[int]] = {}
    for rec in history:
        for r in rec.get("github_repos", []):
            repo = r.get("repo", "")
            stars = r.get("stars")
            parsed = _parse_star_count(str(stars)) if isinstance(stars, str) else stars
            if repo and isinstance(parsed, int):
                hist_repo_stars.setdefault(repo, []).append(parsed)

    top_growth: list[dict[str, Any]] = []
    new_repos: list[str] = []

    for repo, today_s in repo_stars.items():
        hist_s = hist_repo_stars.get(repo, [])
        if not hist_s:
            new_repos.append(repo)
            continue
        prev_s = max(hist_s)
        if prev_s > 0:
            growth_pct = (today_s - prev_s) / prev_s
            top_growth.append({
                "repo": repo,
                "stars": today_s,
                "prev_stars": prev_s,
                "growth_pct": round(growth_pct, 3),
            })

    top_growth.sort(key=lambda x: x["growth_pct"], reverse=True)

    return {
        "has_data": True,
        "total_stars_today": total_stars,
        "repo_count": len(today_repos),
        "top_growth": top_growth[:10],
        "new_repos": new_repos[:10],
        "history_days": len(history),
    }


# ─────────────────────────────────────────────────────────────────────
# HuggingFace 模型下载量变化
# ─────────────────────────────────────────────────────────────────────

def compute_hf_download_changes(
    today_metrics: dict[str, Any],
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    分析 HuggingFace 模型下载量变化。

    对比同一模型在不同日期的 downloads，计算增量与增速。
    """
    today_models = {
        m["model_id"]: m.get("downloads", 0)
        for m in today_metrics.get("hf_models", [])
    }
    if not today_models:
        return {
            "has_data": False,
            "message": "今日无 HuggingFace 数据",
            "top_downloads": [],
            "top_growth": [],
            "total_downloads": 0,
        }

    total_downloads = sum(today_models.values())

    hist_model_downloads: dict[str, list[int]] = {}
    for rec in history:
        for m in rec.get("hf_models", []):
            mid = m.get("model_id", "")
            d = m.get("downloads", 0)
            if mid:
                hist_model_downloads.setdefault(mid, []).append(d)

    top_downloads = sorted(
        [{"model_id": k, "downloads": v} for k, v in today_models.items()],
        key=lambda x: x["downloads"],
        reverse=True,
    )[:10]

    top_growth: list[dict[str, Any]] = []
    for mid, today_d in today_models.items():
        hist_d = hist_model_downloads.get(mid, [])
        if not hist_d:
            continue
        prev_d = max(hist_d)
        if prev_d > 0:
            growth_pct = (today_d - prev_d) / prev_d
            top_growth.append({
                "model_id": mid,
                "downloads": today_d,
                "prev_downloads": prev_d,
                "growth_pct": round(growth_pct, 3),
            })

    top_growth.sort(key=lambda x: x["growth_pct"], reverse=True)

    return {
        "has_data": True,
        "total_downloads": total_downloads,
        "model_count": len(today_models),
        "top_downloads": top_downloads,
        "top_growth": top_growth[:10],
        "history_days": len(history),
    }


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────

def analyze_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    综合指标分析入口。

    接收当日指标快照，与历史对比，返回 arxiv / github / hf 三部分分析结果。
    """
    date = metrics.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))
    history = load_metrics_history(before_date=date, days=30)

    arxiv_report = compute_arxiv_capability_curve(metrics, history)
    github_report = compute_github_star_growth(metrics, history)
    hf_report = compute_hf_download_changes(metrics, history)

    report = {
        "date": date,
        "arxiv_capability": arxiv_report,
        "github_stars": github_report,
        "huggingface_downloads": hf_report,
    }

    logger.info(
        "Metrics report: arXiv=%s, GitHub=%s, HF=%s",
        arxiv_report.get("trend", "n/a"),
        github_report.get("repo_count", 0),
        hf_report.get("model_count", 0),
    )
    return report
