# AI Daily Insight

Automated daily blog posts covering the latest developments in artificial intelligence.

Powered by **Groq** (default) for blazing-fast LLM inference, with OpenAI as an alternative.

## Architecture

```
GitHub Actions (cron 06:00 UTC)
        │
        ▼
   ┌─────────┐     ┌──────────────┐     ┌─────────────┐
   │ fetch.py │────▶│  analyze.py  │────▶│ generate.py │
   │ RSS/API  │     │ Stage-1 LLM  │     │ Stage-2 LLM │
   └─────────┘     └──────────────┘     └─────────────┘
        │                 │                     │
        │                 └──── llm_client.py ──┘
        │                  (Groq / OpenAI / custom)
        │                                       │
        │                                       ▼
  ┌─────┴──────┐                       content/YYYY-MM-DD.md
  │  Sources:  │
  │  RSS       │
  │  GitHub    │
  │  HF API   │
  └────────────┘
```

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `fetch.py` | Pull fresh content from 12+ sources across 4 categories |
| 2 | `analyze.py` | Stage-1 prompt — extract themes, significance, and executive summary as JSON |
| 3 | `generate.py` | Stage-2 prompt — convert structured analysis into a polished Markdown post |
| — | `main.py` | Orchestrator that runs steps 1→2→3 in sequence |
| — | `llm_client.py` | Shared LLM client factory — supports Groq, OpenAI, or any compatible API |

## Information Sources

### 1. Papers

| Source | Type | Feed |
|--------|------|------|
| arXiv cs.AI | RSS | `rss.arxiv.org/rss/cs.AI` |
| arXiv cs.LG | RSS | `rss.arxiv.org/rss/cs.LG` |
| arXiv cs.CL | RSS | `rss.arxiv.org/rss/cs.CL` |

### 2. Company Updates

| Source | Type | Feed |
|--------|------|------|
| OpenAI Blog | RSS | `openai.com/blog/rss.xml` |
| Anthropic Blog | RSS | `anthropic.com/rss.xml` |
| Google DeepMind Blog | RSS | `deepmind.google/blog/rss.xml` |
| Meta AI Blog | RSS | `ai.meta.com/blog/rss/` |

### 3. Open-Source Ecosystem

| Source | Type | How |
|--------|------|-----|
| GitHub Trending (AI/ML) | HTML scrape | Trending Python repos filtered by AI keywords |
| HuggingFace Trending | API | `huggingface.co/api/models?sort=trending` |

### 4. Capital & Industry

| Source | Type | Feed |
|--------|------|------|
| TechCrunch AI | RSS | `techcrunch.com/category/artificial-intelligence/feed/` |
| VentureBeat AI | RSS | `venturebeat.com/category/ai/feed/` |
| Crunchbase News | RSS | `news.crunchbase.com/feed/` |

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Groq API key (default provider)
export LLM_API_KEY="gsk_..."

# 3. Run the pipeline
cd scripts
python main.py

# Dry-run (fetch only, no LLM calls)
python main.py --dry-run
```

### Using OpenAI instead of Groq

```bash
export LLM_PROVIDER="openai"
export LLM_API_KEY="sk-..."
export LLM_MODEL="gpt-4o"       # optional, this is the default for openai
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_API_KEY` | **Yes** | — | API key for the LLM provider |
| `LLM_PROVIDER` | No | `groq` | Provider name: `groq`, `openai`, or `custom` |
| `LLM_MODEL` | No | `llama-3.3-70b-versatile` (Groq) / `gpt-4o` (OpenAI) | Model to use |
| `LLM_BASE_URL` | No | Auto-set per provider | Override the API endpoint |

## GitHub Actions Setup

1. Go to **Settings → Secrets and variables → Actions**.
2. Add a repository secret named **`LLM_API_KEY`** (your Groq API key).
3. (Optional) Add variables:
   - `LLM_PROVIDER` — set to `openai` if using OpenAI instead.
   - `LLM_MODEL` — override the default model.
4. The workflow runs automatically at 06:00 UTC every day, or trigger it manually via **Actions → AI Daily Insight → Run workflow**.

## Supported Groq Models

| Model | Context | Notes |
|-------|---------|-------|
| `llama-3.3-70b-versatile` | 128k | Default — great balance of quality and speed |
| `llama-3.1-8b-instant` | 128k | Faster, lower cost |
| `mixtral-8x7b-32768` | 32k | Strong multilingual support |
| `gemma2-9b-it` | 8k | Lightweight alternative |

Get your free API key at [console.groq.com](https://console.groq.com).

## License

MIT
