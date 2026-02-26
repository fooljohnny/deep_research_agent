"""
generate.py — Stage-2 prompt: turn structured analysis into a Markdown blog.

Takes the JSON analysis from Stage-1 and asks the LLM to produce a
well-written, engaging Markdown blog post suitable for publication.
"""

import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

import openai

logger = logging.getLogger(__name__)

CONTENT_DIR = Path(__file__).resolve().parent.parent / "content"

SYSTEM_PROMPT = """\
You are an expert technology writer who publishes a daily AI insight blog.

Given a structured JSON analysis of today's AI news, write a polished
Markdown blog post.

Formatting rules:
- Start with a YAML front-matter block (title, date, tags).
- Use ## for section headings.
- Include an "Executive Summary" section at the top.
- Dedicate a section to each theme with analysis and context.
- End with a "Notable Reads" section that links to the most interesting
  original articles.
- Close with a brief "Looking Ahead" paragraph.
- Keep the tone professional yet accessible; avoid hype.
- Write in English.
- Total length: 800-1500 words.
"""


def _build_user_prompt(analysis: dict[str, Any]) -> str:
    return (
        "Here is today's structured analysis in JSON. "
        "Please convert it into a Markdown blog post following the rules above.\n\n"
        + json.dumps(analysis, indent=2, ensure_ascii=False)
    )


def generate_post(analysis: dict[str, Any]) -> str:
    """
    Stage-2 generation: structured JSON → Markdown blog post.

    Returns the Markdown string and writes it to content/<date>.md.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")

    client = openai.OpenAI(api_key=api_key)

    user_prompt = _build_user_prompt(analysis)
    logger.info("Sending analysis to LLM for Stage-2 generation …")

    response = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
        temperature=0.6,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    markdown: str = response.choices[0].message.content or ""

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONTENT_DIR / f"{today}.md"
    out_path.write_text(markdown, encoding="utf-8")
    logger.info("Blog post written to %s (%d chars)", out_path, len(markdown))

    return markdown


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_analysis = {
        "title": "AI Daily Insight – 2026-02-26",
        "executive_summary": "Today saw breakthroughs in multimodal AI ...",
        "themes": [
            {
                "name": "Multimodal Models",
                "significance": "high",
                "description": "Several labs announced new multimodal capabilities.",
                "related_articles": ["Sample Article"],
            }
        ],
        "notable_articles": [
            {
                "title": "Sample Article",
                "source": "Test",
                "url": "https://example.com",
                "why_notable": "Demonstrates a key trend.",
            }
        ],
    }
    md = generate_post(sample_analysis)
    print(md)
