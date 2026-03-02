# Cloud Daily Insight

每日自动化云计算与 SaaS 产业**结构性变化**洞察分析。

北京时间凌晨 5:00 定时运行，覆盖头部云厂商（AWS、Google Cloud、Azure、阿里云、腾讯云、火山云）及头部 SaaS 厂商（Salesforce、ServiceNow、Adobe、Workday、Zoom、Slack 等）的最新进展。

Powered by **Groq**（默认使用 `groq/compound`，无每日 token 上限）或 OpenAI 进行 LLM 推理。

## 架构

```
GitHub Actions (cron 21:00 UTC = 北京时间 05:00)
        │
        ▼
  ┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌──────────────┐
  │ fetch.py  │──▶│ analyze.py  │──▶│ trend.py  │──▶│ generate.py  │
  │ RSS / 抓取 │   │ Stage-1 LLM │   │ TF-IDF    │   │ Stage-2 LLM  │
  │ 云/SaaS源  │   │ 结构分析     │   │ 趋势对比   │   │ Markdown生成  │
  └──────────┘    └─────────────┘    └──────────┘    └──────────────┘
       │                │                 │                  │
       │                │           data/history/            │
       │                └── llm_client.py ──┘                │
       │                                                     ▼
  20+ 数据源                                      content/YYYY-MM-DD.md
```

| 步骤 | 脚本 | 用途 |
|------|------|------|
| 1 | `fetch.py` | 从 20+ 云/SaaS 源拉取内容 |
| 2 | `analyze.py` | Stage-1 — 五维度结构性变化分析 + 关键词提取 |
| 3 | `trend.py` | 今日话题向量与 30 天历史对比，检测趋势信号 |
| 4 | `metrics.py` | 云厂商/SaaS 话题演化分析 |
| 5 | `charts.py` | 自动生成趋势图 |
| 6 | `generate.py` | Stage-2 — 生成 Markdown 洞察报告 |
| — | `main.py` | 编排器 |
| — | `llm_client.py` | 共享 LLM 客户端（Groq / OpenAI） |

## 数据来源

### 1. 头部云厂商

| 来源 | 类型 | 说明 |
|------|------|------|
| AWS Blog | RSS | aws.amazon.com/blogs/aws/feed/ |
| Google Cloud Blog | RSS | cloud.google.com/blog/rss |
| Microsoft Azure Blog | RSS | azure.microsoft.com/blog/feed/ |
| Alibaba Cloud Blog | RSS | alibabacloud.com/blog/feed |
| Tencent Cloud Blog | RSS / 抓取 | 开发者社区 |
| 火山引擎 Blog | RSS / 抓取 | 博客园 + 开发者社区 |

### 2. 头部 SaaS 厂商

| 来源 | 类型 |
|------|------|
| Salesforce Blog | RSS |
| ServiceNow Blog | RSS |
| Adobe Blog | RSS |
| Workday Blog | RSS |
| Zoom Blog | RSS |
| Slack Blog | RSS |
| HubSpot Blog | RSS |
| Zendesk Blog | RSS |

### 3. 行业资讯

| 来源 | 类型 |
|------|------|
| TechCrunch | RSS |
| VentureBeat Enterprise | RSS |
| The Register | RSS |
| Cloudflare Blog | RSS |

## 快速开始

```bash
# 1. 安装依赖
cd cloud-daily-insight
pip install -r requirements.txt

# 2. 设置 Groq API Key（默认 provider）
export LLM_API_KEY="gsk_..."

# 3. 运行 pipeline
cd scripts
python main.py

# Dry-run（仅抓取，不调用 LLM）
python main.py --dry-run
```

### 使用 OpenAI

```bash
export LLM_PROVIDER="openai"
export LLM_API_KEY="sk-..."
export LLM_MODEL="gpt-4o"
```

## 环境变量

| 变量 | 必填 | 默认 | 说明 |
|------|------|------|------|
| `LLM_API_KEY` | **是** | — | LLM 的 API Key |
| `LLM_PROVIDER` | 否 | `groq` | `groq` / `openai` / `custom` |
| `LLM_MODEL` | 否 | `groq/compound` | 模型名称 |
| `STAGE_DELAY_SEC` | 否 | `65` | Stage-1 与 Stage-2 间隔秒数，缓解 TPM 限流 |
| `RETRY_DELAY_SEC` | 否 | `65` | 429 重试前等待秒数 |

## Groq 限流说明

`groq/compound` 是混合模型，由 **meta-llama/llama-4-scout-17b** 与 **openai/gpt-oss-120b** 组成，  
限流取两者中**更严格**者：

| 子模型 | On-Demand TPM | Developer TPM |
|--------|---------------|---------------|
| gpt-oss-120b | 8K | 250K |
| llama-4-scout-17b | 30K | 30K |

Pipeline 两阶段合计约 10K tokens，受 **gpt-oss-120b 的 8K TPM** 约束，需跨分钟执行。

- 查看你的实际限制：<https://console.groq.com/settings/limits>
- 若为 **Developer 计划**（compound 约 200K TPM），可将 `STAGE_DELAY_SEC=0` 加速
- 若仍遇 429，可增大 `STAGE_DELAY_SEC` 或 `RETRY_DELAY_SEC`

## GitHub Actions 配置

1. 进入 **Settings → Secrets and variables → Actions**
2. 添加 `LLM_API_KEY` 仓库 secret（如使用 AI Daily Insight 可复用同一 key）
3. 工作流每日北京时间 05:00 自动运行，也可手动触发：**Actions → Cloud Daily Insight → Run workflow**

## License

MIT
