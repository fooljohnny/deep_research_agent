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

from llm_client import get_client, get_model

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

#### # 今日AI结构性变化
- 用 `> ` 引用块写出今日核心判断（一句话）。
- 紧跟 2-3 句话展开：今天最值得注意的结构性变化是什么？为什么重要？
- 如果趋势报告中有信号（突然增强/持续趋势/新兴话题），在这里用简洁的列表
  标注出来，例如：
  - 📈 **持续趋势**：技术层话题已连续N天增强
  - 🆕 **新兴方向**：某话题为30天内首次出现
  - 🔺 **突然升温**：某方向近3天突然集中出现
  - 🔑 **新关键词**：xxx, yyy
  - 📉 **消退关键词**：zzz

#### # 技术层信号
- 变化强度标签开头：`🔴 重大突破` / `🟡 渐进改善` / `⚪ 无显著变化`
- 回答两个问题：
  1. 模型能力边界是否被推进？怎么推进的？
  2. 是否出现新范式？为什么它重要？
- 引用证据文章（含标题和链接）。
- 如果趋势报告显示该维度有趋势信号，体现历史对比视角。
- 无变化时 1-2 句话带过。

#### # 产业资本信号
- 变化强度标签开头：`🔴 强信号` / `🟡 中等信号` / `⚪ 弱信号`
- 融合三个子维度进行分析：
  - **基础设施变化**：推理成本、芯片/算力趋势
  - **应用层变化**：新行业渗透、替代旧方案
  - **资本流向**：融资集中方向、估值变化、大厂战略投资/收购
- 重点是：**钱在往哪里流？基础设施在怎么变？应用在哪里落地？**
  这三者合在一起构成完整的产业资本图景。
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
- 回答：今天的信号在 AI 产业演进的大图中处于什么位置？
- 用 1-2 段话收尾，给出结构性判断。
- 如果趋势报告有 overall_novelty 或 keyword_trends 数据，
  在这里引用作为"坐标感"的依据。

### 风格要求
- 总字数：1000-1800字。
- 有变化的方向重点展开，无变化的方向快速带过。
- 每个引用的文章用 `[标题](链接)` 格式。
- 不要在文末单独列链接列表——链接在行文中自然引用即可。
"""


def _build_user_prompt(
    analysis: dict[str, Any],
    trend_report: dict[str, Any] | None = None,
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

    return "\n".join(parts)


def generate_post(
    analysis: dict[str, Any],
    trend_report: dict[str, Any] | None = None,
) -> str:
    """
    Stage-2 generation: analysis + trends → Markdown insight blog post.

    Returns the Markdown string and writes it to content/<date>.md.
    """
    client = get_client()
    model = get_model()

    user_prompt = _build_user_prompt(analysis, trend_report)
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

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    CONTENT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CONTENT_DIR / f"{today}.md"
    out_path.write_text(markdown, encoding="utf-8")
    logger.info("Blog post written to %s (%d chars)", out_path, len(markdown))

    return markdown


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
