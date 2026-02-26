"""
analyze.py — Stage-1 prompt: extract structured insights from raw articles.

Sends the fetched article list to an LLM and asks it to identify themes,
rank significance, and produce a structured JSON analysis that Stage-2
(generate.py) will turn into a polished Markdown blog post.
"""

import json
import logging
import os
from typing import Any

import openai

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are an expert AI industry analyst. Your job is to review a batch of recent
AI-related articles and produce a structured analysis in JSON.

Rules:
1. Identify 3-5 major themes or trends across the articles.
2. For each theme, cite the most relevant articles (by title).
3. Rate each theme's significance: "high", "medium", or "low".
4. Write a concise one-paragraph executive summary covering the day's highlights.
5. Suggest a catchy, informative blog title for today's insight post.

Return ONLY valid JSON with the following schema (no markdown fences):

{
  "title": "string – blog post title",
  "executive_summary": "string – one paragraph",
  "themes": [
    {
      "name": "string – theme name",
      "significance": "high | medium | low",
      "description": "string – 2-3 sentences",
      "related_articles": ["article title 1", "article title 2"]
    }
  ],
  "notable_articles": [
    {
      "title": "string",
      "source": "string",
      "url": "string",
      "why_notable": "string – one sentence"
    }
  ]
}
"""


def _build_user_prompt(articles: list[dict[str, str]]) -> str:
    lines = [f"Today's date: {_today()}.", "", "Articles to analyse:", ""]
    for i, a in enumerate(articles, 1):
        lines.append(f"{i}. [{a['source']}] {a['title']}")
        lines.append(f"   URL: {a['url']}")
        if a.get("summary"):
            lines.append(f"   Summary: {a['summary'][:300]}")
        lines.append("")
    return "\n".join(lines)


def _today() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def analyze_articles(articles: list[dict[str, str]]) -> dict[str, Any]:
    """
    Stage-1 analysis: send articles to LLM → structured JSON insights.
    """
    if not articles:
        logger.warning("No articles to analyse — returning empty analysis.")
        return {
            "title": f"AI Daily Insight – {_today()}",
            "executive_summary": "No new articles were collected today.",
            "themes": [],
            "notable_articles": [],
        }

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. "
            "Add it as a repository secret and pass it via the workflow."
        )

    client = openai.OpenAI(api_key=api_key)

    user_prompt = _build_user_prompt(articles)
    logger.info("Sending %d articles to LLM for Stage-1 analysis …", len(articles))

    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        temperature=0.4,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content
    analysis: dict[str, Any] = json.loads(raw)  # type: ignore[arg-type]
    logger.info("Stage-1 analysis complete: %s", analysis.get("title", ""))
    return analysis


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = [
        {
            "source": "Test Source",
            "title": "Sample AI Article",
            "url": "https://example.com",
            "summary": "A test summary about artificial intelligence.",
        }
    ]
    result = analyze_articles(sample)
    print(json.dumps(result, indent=2, ensure_ascii=False))
