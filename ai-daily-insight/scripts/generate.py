"""
generate.py — Stage-2 prompt: turn structural analysis + trend signals
into a Markdown insight post.

Takes the five-dimension structural change JSON from Stage-1 and the
trend comparison report, then asks the LLM to produce a Markdown blog
post focused on *structural shifts* and *historical context*.
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
你是一位AI产业结构洞察博主，每天发布一篇结构性变化分析。

你将收到两份数据：
1. **今日结构分析JSON**：五个维度（技术层、基础设施、应用层、资本信号、风险信号）
2. **趋势对比报告JSON**：今日话题向量与过去30天的对比结果

### 写作原则
- 这不是新闻摘要，而是结构性变化洞察。
- 核心问题是：今天的信息意味着AI产业格局发生了什么变化？
- 有变化的维度深入展开，无变化的维度一笔带过。
- **如果趋势报告中有信号，务必在相应维度中体现历史对比视角**：
  - 突然增强的话题：指出"这个方向在过去几天突然升温"
  - 持续增强的趋势：指出"这个趋势已经连续N天增强"
  - 新出现的话题：指出"这是30天内首次出现的全新方向"
- 如果趋势报告显示历史数据不足，则跳过趋势部分。
- 语言要犀利、有观点、不废话。
- 用中文撰写。

### 格式要求
- 以 YAML front-matter 开头（title, date, tags）。
- 第一段是"今日核心判断"（core_insight），用引用块格式（> ）。
- 如果有趋势信号，紧跟一个 ## 趋势雷达 section，用简洁的列表展示检测到的趋势信号。
- 然后按五个维度分节，使用 ## 标题：
  - 技术层：模型能力与新范式
  - 基础设施层：推理成本与算力
  - 应用层：场景扩展与替代效应
  - 资本信号：融资与战略
  - 风险信号：监管与伦理
- 每个维度开头标注变化强度标签：`🔴 重大突破` `🟡 渐进改善` `⚪ 无显著变化` `🔴 强信号` `🟡 中等信号` `⚪ 弱信号`
- 有变化的维度：分析变化本质 + 引用证据文章（含链接）+ 历史趋势对比（如有）。
- 无变化的维度：简短说明即可（1-2句话）。
- 最后一节 ## 值得关注的链接：列出分析中引用的关键文章链接。
- 总字数控制在 800-1500 字。
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
    Stage-2 generation: analysis + trends → Markdown blog post.

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
            }
        ],
        "keyword_trends": {
            "new_keywords": ["agent-orchestration"],
            "rising_keywords": ["inference", "RAG"],
            "fading_keywords": ["diffusion"],
        },
    }
    md = generate_post(sample_analysis, sample_trends)
    print(md)
