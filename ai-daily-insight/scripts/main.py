"""
main.py — Pipeline orchestrator.

Runs the four-step daily insight pipeline:
  1. fetch    – pull fresh content from RSS feeds, GitHub, HuggingFace
  2. analyze  – Stage-1 LLM → five-dimension structural change JSON
  3. trend    – compare today's topic vectors against 30-day history
  4. generate – Stage-2 LLM → Markdown insight blog post

Writes a detailed process log to logs/YYYY-MM-DD.log.
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fetch import fetch_articles, fetch_metrics_snapshot
from analyze import analyze_articles
from trend import analyze_trends, save_daily
from metrics import analyze_metrics, save_daily_metrics
from charts import generate_trend_charts
from generate import generate_post

logger = logging.getLogger("ai-daily-insight")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"


def _setup_logging(verbose: bool, today: str) -> Path:
    """Configure console + file logging. Returns the log file path."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{today}.log"

    root = logging.getLogger()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG if verbose else logging.INFO)
    console.setFormatter(fmt)
    root.addHandler(console)

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    return log_path


def _write_summary(
    log_path: Path,
    today: str,
    pipeline_elapsed: float,
    fetch_stats: list[dict[str, Any]],
    article_count: int,
    stage1_usage: dict[str, Any],
    trend_report: dict[str, Any],
    stage2_usage: dict[str, Any],
    blog_chars: int,
) -> None:
    """Append a structured summary block to the end of the log file."""
    total_fetched = sum(s["fetched"] for s in fetch_stats)
    total_new = sum(s["new"] for s in fetch_stats)
    total_fetch_time = sum(s["elapsed_s"] for s in fetch_stats)
    ok_count = sum(1 for s in fetch_stats if s["status"] == "ok")
    fail_count = sum(1 for s in fetch_stats if s["status"] == "fail")

    s1_total = stage1_usage.get("total_tokens", 0)
    s2_total = stage2_usage.get("total_tokens", 0)
    grand_total_tokens = s1_total + s2_total

    lines = [
        "",
        "=" * 72,
        f"  PIPELINE SUMMARY — {today}",
        "=" * 72,
        "",
        f"  Total elapsed:       {pipeline_elapsed:.1f}s",
        f"  Blog post:           content/{today}.md ({blog_chars:,} chars)",
        f"  Log file:            logs/{today}.log",
        "",
        "  ── Fetch ─────────────────────────────────────────────────",
        f"  Sources:             {len(fetch_stats)} ({ok_count} ok, {fail_count} fail)",
        f"  Articles fetched:    {total_fetched}",
        f"  New (after dedup):   {total_new}",
        f"  Fetch time:          {total_fetch_time:.1f}s",
        "",
    ]

    # Per-source table
    lines.append(f"  {'Source':<28s} {'Method':<7s} {'Status':<5s} {'Fetched':>7s} {'New':>5s} {'Time':>6s}")
    lines.append(f"  {'─' * 28} {'─' * 7} {'─' * 5} {'─' * 7} {'─' * 5} {'─' * 6}")
    for s in fetch_stats:
        lines.append(
            f"  {s['source']:<28s} {s['method']:<7s} {s['status']:<5s} "
            f"{s['fetched']:>7d} {s['new']:>5d} {s['elapsed_s']:>5.1f}s"
        )

    lines += [
        "",
        "  ── LLM Token Usage ───────────────────────────────────────",
        f"  Model:               {stage1_usage.get('model', 'n/a')}",
        f"  Stage-1 (analyze):   {stage1_usage.get('prompt_tokens', 0):>6,} prompt"
        f" + {stage1_usage.get('completion_tokens', 0):>6,} completion"
        f" = {s1_total:>7,} total",
        f"  Stage-2 (generate):  {stage2_usage.get('prompt_tokens', 0):>6,} prompt"
        f" + {stage2_usage.get('completion_tokens', 0):>6,} completion"
        f" = {s2_total:>7,} total",
        f"  Grand total tokens:  {grand_total_tokens:>7,}",
        "",
        "  ── Trend Analysis ────────────────────────────────────────",
        f"  History days:        {trend_report.get('history_days', 0)}",
        f"  Has enough history:  {trend_report.get('has_enough_history', False)}",
        f"  Trend signals:       {len(trend_report.get('signals', []))}",
        f"  Overall novelty:     {trend_report.get('overall_novelty', 'n/a')}",
    ]

    kw = trend_report.get("keyword_trends", {})
    if kw.get("new_keywords"):
        lines.append(f"  New keywords:        {', '.join(kw['new_keywords'])}")
    if kw.get("rising_keywords"):
        lines.append(f"  Rising keywords:     {', '.join(kw['rising_keywords'])}")
    if kw.get("fading_keywords"):
        lines.append(f"  Fading keywords:     {', '.join(kw['fading_keywords'])}")

    lines += ["", "=" * 72, ""]

    summary_text = "\n".join(lines)

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(summary_text)

    for line in lines:
        if line.strip():
            logger.info(line.strip())


def run_pipeline(dry_run: bool = False) -> None:
    pipeline_start = time.monotonic()
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    logger.info("=== AI Daily Insight Pipeline — %s ===", today)

    # ── Step 1: Fetch ──────────────────────────────────────────────
    logger.info("[1/5] Fetching content from all sources …")
    articles, fetch_stats = fetch_articles()
    logger.info("Collected %d fresh article(s).", len(articles))

    if not articles:
        logger.warning("No new articles found. Pipeline will produce a minimal post.")

    if dry_run:
        print(json.dumps(articles, indent=2, ensure_ascii=False))
        logger.info("Dry-run mode — stopping after fetch.")
        return

    # 采集指标快照（arXiv / GitHub / HuggingFace）
    logger.info("Fetching metrics snapshot for insight …")
    metrics_snapshot = fetch_metrics_snapshot()
    save_daily_metrics(today, metrics_snapshot)
    metrics_report = analyze_metrics(metrics_snapshot)

    # ── Step 2: Structural Analysis (Stage-1 Prompt) ────────────────
    logger.info("[2/5] Running structural change analysis …")
    analysis, stage1_usage = analyze_articles(articles)
    logger.info("Title: %s", analysis.get("title", ""))
    logger.info("Keywords: %s", ", ".join(analysis.get("keywords", [])))

    analysis_path = DATA_DIR / "latest_analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Step 3: Trend Comparison ───────────────────────────────────
    logger.info("[3/5] Comparing against historical trends …")
    today_date = analysis.get("date", today)
    trend_report = analyze_trends(analysis)
    save_daily(today_date, analysis)

    trend_path = DATA_DIR / "latest_trends.json"
    trend_path.write_text(
        json.dumps(trend_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # ── Step 4: Generate trend charts ──────────────────────────────
    logger.info("[4/5] Generating trend charts …")
    chart_paths: list[str] = []
    for p in generate_trend_charts(today):
        # 相对路径：content/YYYY-MM-DD.md 引用 charts/xxx.png
        content_dir = PROJECT_ROOT / "content"
        rel = p.relative_to(content_dir)
        chart_paths.append(str(rel))

    # ── Step 5: Generate Insight Post (Stage-2 Prompt) ──────────────
    logger.info("[5/5] Generating Markdown insight post …")
    markdown, stage2_usage = generate_post(
        analysis, trend_report,
        metrics_report=metrics_report,
        chart_paths=chart_paths if chart_paths else None,
    )

    pipeline_elapsed = time.monotonic() - pipeline_start
    logger.info("=== Pipeline complete (%.1fs) ===", pipeline_elapsed)

    # ── Write summary to log ───────────────────────────────────────
    log_path = LOGS_DIR / f"{today}.log"
    _write_summary(
        log_path=log_path,
        today=today,
        pipeline_elapsed=pipeline_elapsed,
        fetch_stats=fetch_stats,
        article_count=len(articles),
        stage1_usage=stage1_usage,
        trend_report=trend_report,
        stage2_usage=stage2_usage,
        blog_chars=len(markdown),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Daily Insight pipeline")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only fetch articles; skip LLM calls.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = _setup_logging(args.verbose, today)
    logger.info("Log file: %s", log_path)

    try:
        run_pipeline(dry_run=args.dry_run)
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
