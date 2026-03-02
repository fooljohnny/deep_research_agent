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

from llm_client import get_client, get_model

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

### 博客结构（严格遵守以下六节）

#### Front-matter
以 YAML front-matter 开头：
```
---
title: "标题"
date: YYYY-MM-DD
tags: [关键标签]
---
```

#### # 今日云计算结构性变化
- 用 `> ` 引用块写出今日核心判断（一句话）。
- 紧跟 2-3 句话展开：今天最值得注意的云计算/SaaS 结构性变化是什么？为什么重要？
- 如果趋势报告中有信号（突然增强/持续趋势/新兴话题），在这里用简洁的列表
  标注出来，例如：
  - 📈 **持续趋势**：技术层话题已连续N天增强
  - 🆕 **新兴方向**：某话题为30天内首次出现
  - 🔺 **突然升温**：某方向近3天突然集中出现
  - 🔑 **新关键词**：xxx, yyy
  - 📉 **消退关键词**：zzz

#### # 技术层信号
- 变化强度标签开头：`🔴 重大突破` / `🟡 渐进改善` / `⚪ 无显著变化`
- 回答：云原生/容器/K8s/Serverless、数据库/存储/网络、AI 与云融合有何进展？
- 引用证据文章（含标题和链接）。
- 如果趋势报告显示该维度有趋势信号，体现历史对比视角。
- 无变化时 1-2 句话带过。

#### # 产业资本信号
- 变化强度标签开头：`🔴 强信号` / `🟡 中等信号` / `⚪ 弱信号`
- 融合三个子维度进行分析：
  - **基础设施变化**：云厂商定价、区域扩张、算力供应
  - **应用层变化**：企业上云、数字化转型、垂直 SaaS
  - **资本流向**：融资集中方向、估值变化、大厂战略投资/收购
- 重点是：**钱在往哪里流？云基础设施在怎么变？SaaS 在哪里落地？**
- 引用证据文章。
- 无变化时简短说明。

#### # 潜在拐点判断
- 基于今日信号 + 历史趋势，判断是否存在潜在拐点。
- 拐点 = 某个方向可能即将发生质变的转折点。
- 如果有拐点信号，说明：
  - 是什么拐点？
  - 依据是什么？（今日信号 + 历史趋势）
  - 如果发生，影响是什么？
- 如果今日没有拐点信号，也要明确说明"今日未观察到拐点信号"并简述原因。

#### # 明日观察点
- 列出 2-3 个明天值得重点关注的方向。
- 每个观察点说明：关注什么？为什么？怎么判断是否发生变化？
- 格式用编号列表。

#### # 长期趋势坐标
- 将今日观察放入更大的时间框架（月度/季度级别）。
- 回答：今天的信号在云计算/SaaS 产业演进的大图中处于什么位置？
- 用 1-2 段话收尾，给出结构性判断。
- 如果趋势报告有 overall_novelty 或 keyword_trends 数据，
  在这里引用作为"坐标感"的依据。

#### # 数据洞察（新增，必写）
根据提供的 metrics_report 数据，撰写以下小节（有数据则分析，无数据则简要说明"今日暂无数据"）：

1. **云厂商话题演化**
   - 若 cloud_topic_evolution 有 novelty_score 和 trend，说明今日云厂商内容与历史的延续/跃迁情况。
   - 趋势为「跃迁」表示话题显著更新；「延续」表示在既有方向深化。

2. **SaaS 厂商话题演化**
   - 若 saas_topic_evolution 有 novelty_score 和 trend，说明今日 SaaS 内容与历史的延续/跃迁情况。

3. **趋势图**
   - 若提供了 chart_paths，在文中引用：`![云厂商话题曲线](charts/YYYY-MM-DD_cloud_novelty.png)` 等。
   - 若无图表，可省略。

#### # 参考来源
- 列出今日分析中引用和参考的**所有**文章链接。
- 按来源类别分组（云厂商、SaaS厂商、行业资讯）。
- 每条格式：`- [文章标题](URL) — 来源名称`
- 包含分析JSON中所有维度 evidence 里的文章，不要遗漏。
- 如果某个来源类别没有引用文章，跳过该类别。

### 风格要求
- 总字数：1000-2000字（含参考来源列表）。
- 有变化的方向重点展开，无变化的方向快速带过。
- 正文中引用文章时仍使用 `[标题](链接)` 格式内嵌引用。
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
        model=model,
        temperature=0.5,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    markdown: str = response.choices[0].message.content or ""

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
