"""
generate.py — Stage-2 prompt: turn structural analysis + trend signals
into a professional cloud computing industry insight blog post.

Takes the five-dimension structural change JSON from Stage-1 and the
trend comparison report, then asks the LLM to produce a Markdown blog
with a fixed structure focused on cloud & SaaS structural shifts.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from llm_client import chat_completion_with_retry, get_model, normalize_assistant_message_content

logger = logging.getLogger(__name__)

CONTENT_DIR = Path(__file__).resolve().parent.parent / "content"

SYSTEM_PROMPT = """\
你是一位顶级云计算与SaaS产业结构分析师，每天发布一篇深度洞察博客。

你将收到两份数据：
1. **今日结构分析JSON**：五个维度的结构性变化分析（聚焦 AWS、Google Cloud、Azure、阿里云、腾讯云、火山云 及 Salesforce、ServiceNow 等 SaaS 厂商）
2. **趋势对比报告JSON**：今日话题向量与过去30天的对比结果

### 核心原则
- 这不是新闻摘要，而是**结构性变化洞察**。
- 每一段文字都要回答："这意味着云计算/SaaS 产业格局发生了什么变化？"
- 不要罗列新闻，要提炼信号、判断拐点、给出观点。
- 语言犀利、专业、有判断力。不说废话，不堆砌形容词。
- 用中文撰写。

### Markdown 版式（与 AI Daily Insight 报告统一，便于同站阅读）
- 每个章节标题必须是**二级标题**，且**恰好两个井号**：`## 今日云计算结构性变化`、`## 技术层信号` 等。
- **禁止**输出 `## # 标题`、禁止用单个 `#` 作为章节标题。
- 强度标签（`🔴 重大突破` / `🟡 渐进改善` / `⚪ 无显著变化` 或资本三档）**单独一行**，下一空行后再写正文段落。
- 技术层、产业资本等以**连贯段落**为主，在句内用 `[文章标题](URL)` 引用证据；勿仅用 `> 证据文章` + 链接列表充当正文。若需集中罗列，可在段末加引用块列表作补充。
- 「数据洞察」下用 `### 小标题` 分条（如云厂商话题演化、SaaS 厂商话题演化、趋势图），与 AI 日报中数据小节层级一致。

### 博客结构（严格遵守以下章节顺序与标题文案）

#### Front-matter
以 YAML front-matter 开头：
```
---
title: "标题"
date: YYYY-MM-DD
tags: [关键标签]
---
```

#### 今日云计算结构性变化
- **章节标题行**（二级）：`## 今日云计算结构性变化`
- 用 `> ` 引用块写出今日核心判断（一句话）。
- 紧跟 2-3 句话展开：今天最值得注意的云计算/SaaS 结构性变化是什么？为什么重要？
- 如果趋势报告中有信号（突然增强/持续趋势/新兴话题），在这里用简洁的列表
  标注出来，例如：
  - 📈 **持续趋势**：技术层话题已连续N天增强
  - 🆕 **新兴方向**：某话题为30天内首次出现
  - 🔺 **突然升温**：某方向近3天突然集中出现
  - 🔑 **新关键词**：xxx, yyy
  - 📉 **消退关键词**：zzz

#### 技术层信号
- **章节标题行**（二级）：`## 技术层信号`
- 第一行写强度：`🔴 重大突破` / `🟡 渐进改善` / `⚪ 无显著变化`，然后空行。
- 用 2-4 段连贯正文回答：云原生/容器/K8s/Serverless、数据库/存储/网络、AI 与云融合有何进展？句内用 `[标题](URL)` 引用证据。
- 如果趋势报告显示该维度有趋势信号，在段落中体现历史对比。
- 无变化时 1-2 句话带过。

#### 产业资本信号
- **章节标题行**（二级）：`## 产业资本信号`
- 第一行写强度：`🔴 强信号` / `🟡 中等信号` / `⚪ 弱信号`，然后空行。
- 用连贯段落融合：**基础设施变化**（定价、区域、算力）、**应用层变化**（上云、数字化、垂直 SaaS）、**资本流向**（融资、估值、并购）；句内 `[标题](URL)`。
- 重点是：**钱在往哪里流？云基础设施在怎么变？SaaS 在哪里落地？**
- 无变化时简短说明。

#### 潜在拐点判断
- **章节标题行**（二级）：`## 潜在拐点判断`
- 基于今日信号 + 历史趋势，判断是否存在潜在拐点。
- 如果有拐点信号：是什么拐点、依据、可能影响；若无，写明「今日未观察到拐点信号」及原因。

#### 明日观察点
- **章节标题行**（二级）：`## 明日观察点`
- 2-3 条编号列表；每条说明关注什么、为什么、如何判断变化。

#### 长期趋势坐标
- **章节标题行**（二级）：`## 长期趋势坐标`
- 1-2 段话，将今日信号放到月/季尺度；可引用 overall_novelty、keyword_trends。

#### 数据洞察（必写）
- **章节标题行**（二级）：`## 数据洞察`
- 使用三级标题分子节，例如：
  - `### 云厂商话题演化`（cloud_topic_evolution：novelty、trend）
  - `### SaaS 厂商话题演化`（saas_topic_evolution）
  - `### 趋势图`（若有 chart_paths：`![云厂商话题曲线](charts/YYYY-MM-DD_cloud_novelty.png)` 等；无则说明省略）
- 无数据时对应小节写「今日暂无数据」。

#### 参考来源
- **章节标题行**（二级）：`## 参考来源`
- 列出今日引用过的**全部**文章；按云厂商、SaaS厂商、行业资讯分组。
- 每条：`- [文章标题](URL) — 来源名称`；无则跳过该组。

### 风格要求
- 总字数：1000-2000字（含参考来源列表）。
- 有变化的方向重点展开，无变化的方向快速带过。
- 正文优先内嵌 `[标题](链接)`，与 AI 产业日报版式一致。
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
    model = get_model()

    user_prompt = _build_user_prompt(
        analysis, trend_report,
        metrics_report=metrics_report,
        chart_paths=chart_paths,
    )
    logger.info("Sending analysis + trends to LLM (%s) for Stage-2 generation …", model)

    response = chat_completion_with_retry(
        model=model,
        temperature=0.5,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
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
        "date": "2026-03-02",
        "title": "云厂商加速 AI 基础设施布局，SaaS 并购升温",
        "core_insight": "AWS、Azure、GCP 密集发布 AI 算力新品，企业 SaaS 整合并购活跃。",
        "keywords": ["AWS", "Azure", "AI", "SaaS", "并购"],
        "dimensions": {
            "technology": {
                "has_change": True,
                "intensity": "渐进改善",
                "cloud_native": "容器与 Serverless 持续演进。",
                "infra_updates": "云厂商推出新一代 AI 推理实例。",
                "evidence": [{"title": "Sample", "source": "AWS Blog", "url": "https://example.com"}],
            },
            "infrastructure": {"has_change": False, "intensity": "无显著变化", "evidence": []},
            "application": {"has_change": False, "intensity": "无显著变化", "evidence": []},
            "capital": {
                "has_change": True,
                "intensity": "强信号",
                "funding_trend": "SaaS 并购活跃。",
                "evidence": [],
            },
            "risk": {"has_change": False, "intensity": "弱信号", "evidence": []},
        },
    }
    sample_trends = {
        "has_enough_history": True,
        "history_days": 15,
        "overall_novelty": 0.35,
        "signals": [],
        "keyword_trends": {"new_keywords": ["AI推理"], "rising_keywords": ["SaaS"], "fading_keywords": []},
    }
    md, _ = generate_post(sample_analysis, sample_trends)
    print(md)
