"""
generate.py — Stage-2 prompt: turn structural analysis + trend signals
into a professional AI industry insight blog post.

Takes the five-dimension structural change JSON from Stage-1 and the
trend comparison report, then asks the LLM to produce a Markdown blog
with a fixed six-section structure focused on structural shifts.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from llm_client import (
    extend_chat_completion_kwargs,
    get_client,
    get_model,
    normalize_assistant_message_content,
)

logger = logging.getLogger(__name__)

CONTENT_DIR = Path(__file__).resolve().parent.parent / "content"

SYSTEM_PROMPT = """\
你是一位顶级AI产业结构分析师，每天发布一篇深度洞察博客。

你将收到两份数据：
1. **今日结构分析JSON**：五个维度的结构性变化分析
2. **趋势对比报告JSON**：今日话题向量与过去30天的对比结果

### 核心原则
- 这不是新闻摘要，而是**结构性变化洞察**。
- 每一段文字都要回答："这意味着AI产业格局发生了什么变化？"
- 不要罗列新闻，要提炼信号、判断拐点、给出观点。
- 语言犀利、专业、有判断力。不说废话，不堆砌形容词。
- 用中文撰写。

### Markdown 版式（与 Cloud Daily Insight 报告统一）
- 章节标题必须是二级标题，且**恰好两个井号**：`## 今日AI结构性变化`、`## 技术层信号` 等。
- **禁止**输出 `## # 标题`、禁止用单个 `#` 作为章节标题。
- 强度标签单独一行，下一空行后再写正文段落。
- 技术层、产业资本以连贯段落为主，句内 `[标题](URL)`；勿用链接列表代替正文。
- 「数据洞察」下用 `### 小标题` 分条。

### 博客结构（严格遵守以下章节顺序与标题文案）

#### Front-matter
以 YAML front-matter 开头（**直接**写 `---` 行，**不要**用 markdown 代码围栏 ``` 包裹）：
---
title: "标题"
date: YYYY-MM-DD
tags: [关键标签]
---
**禁止**在全文开头写单独的 ``` 再写 front-matter，否则整篇会被渲染成代码块、阅读页像记事本。

#### 今日AI结构性变化
- **章节标题行**（二级）：`## 今日AI结构性变化`
- 用 `> ` 引用块写出今日核心判断（一句话）。
- 紧跟 2-3 句话展开：今天最值得注意的结构性变化是什么？为什么重要？
- 如果趋势报告中有信号，用列表标注（📈 持续趋势、🆕 新兴方向、🔺 突然升温、🔑 新关键词、📉 消退关键词）。

#### 技术层信号
- **章节标题行**（二级）：`## 技术层信号`
- 第一行强度：`🔴 重大突破` / `🟡 渐进改善` / `⚪ 无显著变化`，然后空行。
- 用连贯段落回答：模型能力边界如何推进？有何新范式？句内 `[标题](URL)`。
- 趋势信号写入段落；无变化时 1-2 句。

#### 产业资本信号
- **章节标题行**（二级）：`## 产业资本信号`
- 第一行强度：`🔴 强信号` / `🟡 中等信号` / `⚪ 弱信号`，然后空行。
- 段落融合基础设施、应用层、资本流向；重点是钱、基础设施、应用落地；句内链接。

#### 潜在拐点判断
- **章节标题行**（二级）：`## 潜在拐点判断`
- 有则写拐点、依据、影响；无则写明未观察到及原因。

#### 明日观察点
- **章节标题行**（二级）：`## 明日观察点`
- 2-3 条编号列表。

#### 长期趋势坐标
- **章节标题行**（二级）：`## 长期趋势坐标`
- 1-2 段，月/季视角；可引用 overall_novelty、keyword_trends。

#### 数据洞察（必写）
- **章节标题行**（二级）：`## 数据洞察`
- 三级标题分子节，例如：
  - `### arXiv 摘要对比 — 模型能力提升曲线`（arxiv_capability）
  - `### GitHub Star 增速分析`（github_stars）
  - `### HuggingFace 模型下载量变化`（huggingface_downloads）
  - `### 趋势图`（chart_paths 与 `![arXiv能力曲线](charts/YYYY-MM-DD_arxiv_capability.png)` 等）
- 无数据则对应小节写「今日暂无数据」。

#### 参考来源
- **章节标题行**（二级）：`## 参考来源`
- 按论文、公司动态、开源生态、资本与行业分组；`- [标题](URL) — 来源`。

### 风格要求
- 总字数：1000-2000字（含参考来源列表）。
- 有变化的方向重点展开，无变化的方向快速带过。
- 正文优先内嵌 `[标题](链接)`。
"""


def _build_user_prompt(
    analysis: dict[str, Any],
    trend_report: dict[str, Any] | None = None,
    metrics_report: dict[str, Any] | None = None,
    chart_paths: list[str] | None = None,
) -> str:
    parts = [
        "## 今日结构分析\n",
        json.dumps(analysis, indent=2, ensure_ascii=False),
    ]

    if trend_report and trend_report.get("has_enough_history"):
        parts.append("\n\n## 趋势对比报告\n")
        parts.append(json.dumps(trend_report, indent=2, ensure_ascii=False))
    elif trend_report:
        parts.append(
            f"\n\n## 趋势对比\n{trend_report.get('message', '历史数据不足，暂无趋势分析。')}"
        )

    if metrics_report:
        parts.append("\n\n## 数据洞察报告 (metrics_report)\n")
        parts.append(json.dumps(metrics_report, indent=2, ensure_ascii=False))

    if chart_paths:
        parts.append("\n\n## 趋势图路径 (chart_paths)\n")
        parts.append(json.dumps(chart_paths, ensure_ascii=False))

    return "\n".join(parts)


def generate_post(
    analysis: dict[str, Any],
    trend_report: dict[str, Any] | None = None,
    metrics_report: dict[str, Any] | None = None,
    chart_paths: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    """
    Stage-2 generation: analysis + trends + metrics → Markdown insight blog post.

    Returns (markdown_string, token_usage_dict).
    """
    client = get_client()
    model = get_model()

    user_prompt = _build_user_prompt(
        analysis, trend_report,
        metrics_report=metrics_report,
        chart_paths=chart_paths,
    )
    logger.info("Sending analysis + trends to LLM (%s) for Stage-2 generation …", model)

    response = client.chat.completions.create(
        **extend_chat_completion_kwargs(
            {
                "model": model,
                "temperature": 0.5,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            }
        )
    )

    markdown: str = normalize_assistant_message_content(response.choices[0].message)

    usage = getattr(response, "usage", None)
    token_usage: dict[str, Any] = {
        "model": model,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
    }

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONTENT_DIR / f"{today}.md"
    out_path.write_text(markdown, encoding="utf-8")
    logger.info(
        "Blog post written to %s (%d chars)  (tokens: %d prompt + %d completion = %d total)",
        out_path, len(markdown),
        token_usage["prompt_tokens"],
        token_usage["completion_tokens"],
        token_usage["total_tokens"],
    )

    return markdown, token_usage


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample_analysis = {
        "date": "2026-02-26",
        "title": "Agent范式加速落地，资本密集涌入AI基础设施",
        "core_insight": "AI Agent从概念验证进入产业级部署，基础设施层融资密度创季度新高。",
        "keywords": ["agent", "infrastructure", "RAG", "inference"],
        "dimensions": {
            "technology": {
                "has_change": True,
                "intensity": "渐进改善",
                "model_capability": "多个开源模型在推理能力上逼近闭源前沿。",
                "new_paradigm": "Agent编排框架趋于标准化。",
                "evidence": [
                    {"title": "Sample Paper", "source": "arXiv cs.AI", "url": "https://arxiv.org/example"}
                ],
            },
            "infrastructure": {
                "has_change": True,
                "intensity": "重大突破",
                "inference_cost": "新一代推理芯片将单位成本降低40%。",
                "compute_trend": "云厂商开始提供Agent专用算力实例。",
                "evidence": [
                    {"title": "Sample Infra", "source": "TechCrunch AI", "url": "https://techcrunch.com/example"}
                ],
            },
            "application": {
                "has_change": False, "intensity": "无显著变化",
                "new_industries": "", "displacement": "", "evidence": [],
            },
            "capital": {
                "has_change": True, "intensity": "强信号",
                "funding_trend": "基础设施赛道本周第三笔大额融资。",
                "valuation": "头部AI公司估值继续上行。",
                "strategic_moves": "某大厂收购推理优化创业公司。",
                "evidence": [
                    {"title": "Sample Funding", "source": "Crunchbase News", "url": "https://crunchbase.com/example"}
                ],
            },
            "risk": {
                "has_change": False, "intensity": "弱信号",
                "regulation": "", "ethics_safety": "", "supply_chain": "", "evidence": [],
            },
        },
    }
    sample_trends = {
        "has_enough_history": True,
        "history_days": 15,
        "overall_novelty": 0.35,
        "signals": [
            {
                "type": "continuous_trend",
                "dimension": "技术层",
                "description": "技术层话题向量在最近7天内持续增强。",
                "trend_days": 7,
                "confidence": 0.80,
            },
            {
                "type": "emerging_topic",
                "dimension": "基础设施",
                "description": "基础设施出现全新话题（novelty 0.95）。",
                "novelty_score": 0.95,
                "confidence": 0.95,
            },
        ],
        "keyword_trends": {
            "new_keywords": ["agent-orchestration"],
            "rising_keywords": ["inference", "RAG"],
            "fading_keywords": ["diffusion"],
        },
    }
    md = generate_post(sample_analysis, sample_trends)
    print(md)
