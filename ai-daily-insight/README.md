# AI Daily Insight

Automated daily blog posts covering the latest developments in artificial intelligence.

## Architecture

```
GitHub Actions (cron 06:00 UTC)
        │
        ▼
   ┌─────────┐     ┌──────────────┐     ┌─────────────┐
   │ fetch.py │────▶│  analyze.py  │────▶│ generate.py │
   │ RSS pull │     │ Stage-1 LLM  │     │ Stage-2 LLM │
   └─────────┘     └──────────────┘     └─────────────┘
                                               │
                                               ▼
                                      content/YYYY-MM-DD.md
```

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `fetch.py` | Pull fresh articles from 8+ AI RSS feeds; deduplicate via `processed.json` |
| 2 | `analyze.py` | Stage-1 prompt — extract themes, significance, and executive summary as JSON |
| 3 | `generate.py` | Stage-2 prompt — convert structured analysis into a polished Markdown post |
| — | `main.py` | Orchestrator that runs steps 1→2→3 in sequence |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
export OPENAI_API_KEY="sk-..."

# 3. Run the pipeline
cd scripts
python main.py

# Dry-run (fetch only, no LLM calls)
python main.py --dry-run
```

## GitHub Actions Setup

1. Go to **Settings → Secrets and variables → Actions**.
2. Add a repository secret named `OPENAI_API_KEY`.
3. (Optional) Add a variable `OPENAI_MODEL` to override the default model (`gpt-4o`).
4. The workflow runs automatically at 06:00 UTC every day, or trigger it manually via **Actions → AI Daily Insight → Run workflow**.

## RSS Sources

| Source | Feed |
|--------|------|
| MIT Technology Review – AI | `technologyreview.com/…/feed` |
| OpenAI Blog | `openai.com/blog/rss.xml` |
| Google AI Blog | `blog.google/technology/ai/rss/` |
| The Batch (deeplearning.ai) | `deeplearning.ai/the-batch/feed/` |
| Hugging Face Blog | `huggingface.co/blog/feed.xml` |
| VentureBeat AI | `venturebeat.com/category/ai/feed/` |
| Towards Data Science | `towardsdatascience.com/feed` |
| arXiv cs.AI | `rss.arxiv.org/rss/cs.AI` |

## License

MIT
