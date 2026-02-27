"""
main.py — Pipeline orchestrator.

Runs the four-step daily insight pipeline:
  1. fetch    – pull fresh content from RSS feeds, GitHub, HuggingFace
  2. analyze  – Stage-1 LLM → five-dimension structural change JSON
  3. trend    – compare today's topic vectors against 30-day history
  4. generate – Stage-2 LLM → Markdown insight blog post

Designed to be called from GitHub Actions or locally.
"""

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from fetch import fetch_articles
from analyze import analyze_articles
from trend import analyze_trends, save_daily
from generate import generate_post

logger = logging.getLogger("ai-daily-insight")


def run_pipeline(dry_run: bool = False) -> None:
    logger.info("=== AI Daily Insight Pipeline ===")

    # ── Step 1: Fetch ──────────────────────────────────────────────
    logger.info("[1/4] Fetching content from all sources …")
    articles = fetch_articles()
    logger.info("Collected %d fresh article(s).", len(articles))

    if not articles:
        logger.warning("No new articles found. Pipeline will produce a minimal post.")

    if dry_run:
        print(json.dumps(articles, indent=2, ensure_ascii=False))
        logger.info("Dry-run mode — stopping after fetch.")
        return

    # ── Step 2: Structural Analysis (Stage-1 Prompt) ────────────────
    logger.info("[2/4] Running structural change analysis …")
    analysis = analyze_articles(articles)
    logger.info("Title: %s", analysis.get("title", ""))
    logger.info("Core insight: %s", analysis.get("core_insight", ""))
    logger.info("Keywords: %s", ", ".join(analysis.get("keywords", [])))

    data_dir = Path(__file__).resolve().parent.parent / "data"
    analysis_path = data_dir / "latest_analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Analysis saved to %s", analysis_path)

    # ── Step 3: Trend Comparison ───────────────────────────────────
    logger.info("[3/4] Comparing against historical trends …")
    today = analysis.get("date", datetime.now(timezone.utc).strftime("%Y-%m-%d"))

    trend_report = analyze_trends(analysis)

    save_daily(today, analysis)

    trend_path = data_dir / "latest_trends.json"
    trend_path.write_text(
        json.dumps(trend_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Trend report saved to %s", trend_path)

    if trend_report.get("has_enough_history"):
        n_signals = len(trend_report.get("signals", []))
        novelty = trend_report.get("overall_novelty", 0)
        logger.info("Trend signals: %d | Overall novelty: %.3f", n_signals, novelty)
    else:
        logger.info("Not enough history for trend analysis yet.")

    # ── Step 4: Generate Insight Post (Stage-2 Prompt) ──────────────
    logger.info("[4/4] Generating Markdown insight post …")
    markdown = generate_post(analysis, trend_report)
    logger.info("Blog post generated (%d characters).", len(markdown))

    logger.info("=== Pipeline complete ===")


def main() -> None:
    parser = argparse.ArgumentParser(description="AI Daily Insight pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only fetch articles; skip LLM calls.",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable debug logging.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        run_pipeline(dry_run=args.dry_run)
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
