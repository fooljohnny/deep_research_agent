# AI Daily Insight

Automated daily AI industry **structural change** analysis — not news summaries.

Powered by **Groq** (default) for blazing-fast LLM inference, with OpenAI as an alternative.

## Architecture

```
GitHub Actions (cron 06:00 UTC)
        │
        ▼
  ┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌──────────────┐
  │ fetch.py  │──▶│ analyze.py  │──▶│ trend.py  │──▶│ generate.py  │
  │ RSS / API │   │ Stage-1 LLM │   │ TF-IDF    │   │ Stage-2 LLM  │
  │ Scrape    │   │ 结构分析     │   │ 趋势对比   │   │ Markdown生成  │
  └──────────┘    └─────────────┘    └──────────┘    └──────────────┘
       │                │                 │                  │
       │                │           data/history/            │
       │                └── llm_client.py ──┘                │
       │                                                     ▼
  12+ sources                                      content/YYYY-MM-DD.md
```

| Step | Script | Purpose |
|------|--------|---------|
| 1 | `fetch.py` | Pull fresh content from 12+ sources across 4 categories |
| 2 | `analyze.py` | Stage-1 — five-dimension structural change analysis + keyword extraction |
| 3 | `trend.py` | Compare today's topic vectors against 30-day history; detect trend signals |
| 4 | `metrics.py` | arXiv 能力曲线、GitHub star 增速、HuggingFace 下载量分析 |
| 5 | `charts.py` | 自动生成趋势图（arXiv / GitHub / HF） |
| 6 | `generate.py` | Stage-2 — generate Markdown insight post with trend + metrics + charts |
| — | `main.py` | Orchestrator: fetch → metrics → analyze → trend → charts → generate |
| — | `llm_client.py` | Shared LLM client factory (Groq / OpenAI / custom) |

## Trend Analysis (the "real insight" layer)

The `trend.py` module is what separates this project from a news aggregator.
Each day after Stage-1 analysis:

1. **Store** — Today's dimension texts and keywords are saved as a daily record in `data/history/`.
2. **Vectorize** — TF-IDF topic vectors are built across the 30-day corpus (character n-grams for Chinese+English support).
3. **Compare** — Today's topic vector is compared against the historical average via cosine similarity.
4. **Detect** three types of signals:

| Signal | Detection Method | Example |
|--------|-----------------|---------|
| **Sudden spike** | High similarity to last 3 days, low to older history | "基础设施话题近3天突然升温" |
| **Continuous trend** | Monotonically increasing similarity over 5-7 days | "资本信号已连续6天增强" |
| **Emerging topic** | Very low similarity to ALL history | "技术层出现全新话题（novelty 0.99）" |

Additionally, **keyword frequency tracking** identifies:
- **New keywords** — terms never seen in the 30-day window
- **Rising keywords** — terms with increasing daily frequency
- **Fading keywords** — previously common terms that have disappeared

The trend report is injected into the Stage-2 prompt, so the generated blog post
naturally includes historical context like "this trend has been building for 5 days"
or "this is a brand-new direction in the 30-day window."

## Data Insights (每日洞察新增)

每日洞察报告新增以下数据维度：

| 维度 | 说明 | 数据来源 |
|------|------|----------|
| **arXiv 摘要对比 — 模型能力提升曲线** | 基于 TF-IDF 语义相似度，对比今日论文摘要与历史，判断能力话题的延续/跃迁/渐进 | `data/metrics/` |
| **GitHub Star 增速分析** | 同一 repo 跨日 star 数对比，识别增速最快项目及新晋热门 | GitHub Trending |
| **HuggingFace 模型下载量变化** | 模型下载量跨日对比，反映开源模型采用热度 | HuggingFace API |
| **自动生成趋势图** | 生成 PNG 趋势图：arXiv 新颖度曲线、GitHub star 总量、HF 下载量 | `content/charts/` |

指标快照每日保存至 `data/metrics/YYYY-MM-DD.json`，用于历史对比与图表生成。

## Information Sources

### 1. Papers

| Source | Type | Feed |
|--------|------|------|
| arXiv cs.AI | RSS | `rss.arxiv.org/rss/cs.AI` |
| arXiv cs.LG | RSS | `rss.arxiv.org/rss/cs.LG` |
| arXiv cs.CL | RSS | `rss.arxiv.org/rss/cs.CL` |

### 2. Company Updates

| Source | Type | Method |
|--------|------|--------|
| OpenAI Blog | RSS | `openai.com/blog/rss.xml` |
| Anthropic Blog | Scrape | HTML parse from `/news` |
| Google DeepMind Blog | RSS | `deepmind.google/blog/rss.xml` |
| Google AI Blog | RSS | `research.google/blog/rss/` |
| Meta AI Blog | Scrape | HTML parse from `/blog/` |

### 3. Open-Source Ecosystem

| Source | Type | Method |
|--------|------|--------|
| GitHub Trending (AI/ML) | Scrape | Trending Python repos filtered by AI keywords |
| HuggingFace Trending | API | `huggingface.co/api/models?sort=trendingScore` |

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
export LLM_MODEL="gpt-4o"
```

### Using DeepSeek (official API)

Use provider `deepseek` so the client targets `https://api.deepseek.com/v1` automatically.

```bash
export LLM_PROVIDER="deepseek"
export LLM_API_KEY="sk-..."
export LLM_MODEL="deepseek-v3.1-terminus"   # or deepseek-chat; use the id from your DeepSeek console
```

Do **not** set `LLM_PROVIDER=openai` while using a DeepSeek model name — that still calls `api.openai.com` and you get **401** with a DeepSeek key.

### Using another OpenAI-compatible host (`custom`)

```bash
export LLM_PROVIDER="custom"
export LLM_BASE_URL="https://api.example.com/v1"
export LLM_API_KEY="..."
export LLM_MODEL="..."
```

On GitHub Actions: for DeepSeek, set variable `LLM_PROVIDER=deepseek`, variable `LLM_MODEL`, secret `LLM_API_KEY`. For `custom`, also set variable `LLM_BASE_URL`.

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `LLM_API_KEY` | **Yes** | — | API key for the LLM provider |
| `LLM_PROVIDER` | No | `groq` | `groq`, `openai`, `deepseek`, or `custom` |
| `LLM_MODEL` | No | Per provider (e.g. Groq llama, OpenAI gpt-4o, DeepSeek deepseek-chat) | Model id from the provider |
| `LLM_BASE_URL` | **Yes** for `custom` | Auto-set for `groq` / `openai` / `deepseek` | Override or required for `custom` only |

## GitHub Actions Setup

1. Go to **Settings → Secrets and variables → Actions**.
2. Add a repository secret **`LLM_API_KEY`** (Groq, OpenAI, DeepSeek, etc., depending on provider).
3. (Optional) Add variables:
   - `LLM_PROVIDER` — e.g. `deepseek`, `openai`, `groq`, or `custom`.
   - `LLM_MODEL` — e.g. `deepseek-v3.1-terminus` for DeepSeek.
   - `LLM_BASE_URL` — **only for `custom`** (e.g. a self-hosted gateway). Not needed for `deepseek`.
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
