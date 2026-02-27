"""
main.py — Pipeline orchestrator.

Runs the three-step daily insight pipeline:
  1. fetch    – pull fresh content from RSS feeds, GitHub, HuggingFace
  2. analyze  – Stage-1 LLM → five-dimension structural change JSON
  3. generate – Stage-2 LLM → Markdown insight blog post

Designed to be called from GitHub Actions or locally.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from fetch import fetch_articles
from analyze import analyze_articles
from generate import generate_post

logger = logging.getLogger("ai-daily-insight")


def run_pipeline(dry_run: bool = False) -> None:
    logger.info("=== AI Daily Insight Pipeline ===")

    # ── Step 1: Fetch ──────────────────────────────────────────────
    logger.info("[1/3] Fetching content from all sources …")
    articles = fetch_articles()
    logger.info("Collected %d fresh article(s).", len(articles))

    if not articles:
        logger.warning("No new articles found. Pipeline will produce a minimal post.")

    if dry_run:
        print(json.dumps(articles, indent=2, ensure_ascii=False))
        logger.info("Dry-run mode — stopping after fetch.")
        return

    # ── Step 2: Structural Analysis (Stage-1 Prompt) ────────────────
    logger.info("[2/3] Running structural change analysis …")
    analysis = analyze_articles(articles)
    logger.info("Title: %s", analysis.get("title", ""))
    logger.info("Core insight: %s", analysis.get("core_insight", ""))

    analysis_path = Path(__file__).resolve().parent.parent / "data" / "latest_analysis.json"
    analysis_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logger.info("Analysis saved to %s", analysis_path)

    # ── Step 3: Generate Insight Post (Stage-2 Prompt) ──────────────
    logger.info("[3/3] Generating Markdown insight post …")
    markdown = generate_post(analysis)
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
