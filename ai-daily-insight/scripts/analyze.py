"""
analyze.py — Stage-1 prompt: AI industry structural change analysis.

This is NOT a summariser. It identifies *structural shifts* across five
dimensions: technology, infrastructure, applications, capital, and risk.
The output is a structured JSON that Stage-2 (generate.py) turns into
a Markdown insight post.
"""

import json
import logging
from typing import Any

from llm_client import get_client, get_model

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
你是一位AI产业结构分析师。

你的任务不是做新闻摘要，而是从今日新增的AI信息中识别"结构性变化信号"。

请从以下五个维度进行分析：

### 1. 技术层变化 (technology)
- 模型能力边界是否被推进？（如：新的SOTA、新模态、上下文长度突破）
- 是否出现新的范式？（如：新的训练方法、架构创新、推理策略）
- 证据：引用具体的文章标题和来源。
- 变化强度：重大突破 / 渐进改善 / 无显著变化

### 2. 基础设施变化 (infrastructure)
- 推理成本是否出现变化？（如：新的高效推理方案、模型蒸馏、量化）
- 芯片/算力趋势如何？（如：新硬件发布、供应链变化、云厂商定价）
- 证据：引用具体的文章标题和来源。
- 变化强度：重大突破 / 渐进改善 / 无显著变化

### 3. 应用层变化 (application)
- AI是否进入了新行业或新场景？（如：医疗、法律、教育、制造）
- 是否在替代旧的解决方案？（如：传统SaaS被AI-native产品替代）
- 证据：引用具体的文章标题和来源。
- 变化强度：重大突破 / 渐进改善 / 无显著变化

### 4. 资本信号 (capital)
- 融资方向是否出现集中趋势？（如：某细分赛道密集融资）
- 估值水平有何变化？
- 大厂战略投资/收购有何动向？
- 证据：引用具体的文章标题和来源。
- 信号强度：强信号 / 中等信号 / 弱信号

### 5. 风险信号 (risk)
- 监管/立法层面是否有新动向？
- 伦理/安全/对齐领域是否有新发现或新争议？
- 是否出现供应链/地缘风险？
- 证据：引用具体的文章标题和来源。
- 信号强度：强信号 / 中等信号 / 弱信号

### 输出规则
- 如果某个维度今日没有明显变化，写明"今日无显著变化"并简要说明原因。
- 不要编造不存在的变化——没有信号比虚假信号更有价值。
- 最后给出一个"今日核心判断"(core_insight)：用一句话概括今天最值得关注的结构性变化。
- 给出一个适合做标题的 title。
- 提取 8-15 个关键词(keywords)：反映今日核心话题的关键技术术语、公司名、产品名等。
  关键词应尽量规范化（如统一使用 "LLM" 而非 "大语言模型"，"RAG" 而非 "检索增强生成"）。

返回严格的JSON（不要 markdown 代码块），schema 如下：

{
  "date": "YYYY-MM-DD",
  "title": "string – 博客标题",
  "core_insight": "string – 一句话核心判断",
  "keywords": ["keyword1", "keyword2", "..."],
  "dimensions": {
    "technology": {
      "has_change": true/false,
      "intensity": "重大突破 | 渐进改善 | 无显著变化",
      "model_capability": "string – 模型能力边界变化分析",
      "new_paradigm": "string – 新范式分析",
      "evidence": [
        {"title": "article title", "source": "source name", "url": "url"}
      ]
    },
    "infrastructure": {
      "has_change": true/false,
      "intensity": "重大突破 | 渐进改善 | 无显著变化",
      "inference_cost": "string – 推理成本变化分析",
      "compute_trend": "string – 芯片/算力趋势分析",
      "evidence": [
        {"title": "article title", "source": "source name", "url": "url"}
      ]
    },
    "application": {
      "has_change": true/false,
      "intensity": "重大突破 | 渐进改善 | 无显著变化",
      "new_industries": "string – 新行业/新场景分析",
      "displacement": "string – 替代旧方案分析",
      "evidence": [
        {"title": "article title", "source": "source name", "url": "url"}
      ]
    },
    "capital": {
      "has_change": true/false,
      "intensity": "强信号 | 中等信号 | 弱信号",
      "funding_trend": "string – 融资方向分析",
      "valuation": "string – 估值变化分析",
      "strategic_moves": "string – 大厂战略动向分析",
      "evidence": [
        {"title": "article title", "source": "source name", "url": "url"}
      ]
    },
    "risk": {
      "has_change": true/false,
      "intensity": "强信号 | 中等信号 | 弱信号",
      "regulation": "string – 监管/立法分析",
      "ethics_safety": "string – 伦理/安全分析",
      "supply_chain": "string – 供应链/地缘风险分析",
      "evidence": [
        {"title": "article title", "source": "source name", "url": "url"}
      ]
    }
  }
}
"""


def _build_user_prompt(articles: list[dict[str, str]]) -> str:
    lines = [
        f"今日日期: {_today()}",
        f"信息条目数: {len(articles)}",
        "",
        "以下是今日新增的AI信息列表（按来源分类）：",
        "",
    ]

    by_category: dict[str, list[dict[str, str]]] = {}
    for a in articles:
        cat = a.get("category", "other")
        by_category.setdefault(cat, []).append(a)

    category_labels = {
        "papers": "论文",
        "company": "公司动态",
        "opensource": "开源生态",
        "industry": "资本与行业",
        "other": "其他",
    }

    idx = 1
    for cat in ["papers", "company", "opensource", "industry", "other"]:
        items = by_category.get(cat, [])
        if not items:
            continue
        lines.append(f"### {category_labels.get(cat, cat)}（{len(items)} 条）")
        lines.append("")
        for a in items:
            lines.append(f"{idx}. [{a['source']}] {a['title']}")
            lines.append(f"   URL: {a['url']}")
            if a.get("summary"):
                lines.append(f"   摘要: {a['summary'][:400]}")
            lines.append("")
            idx += 1

    return "\n".join(lines)


def _today() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _empty_analysis() -> dict[str, Any]:
    """Fallback when no articles are available."""
    empty_dim = {
        "has_change": False,
        "intensity": "无显著变化",
        "evidence": [],
    }
    return {
        "date": _today(),
        "title": f"AI产业结构日报 – {_today()}",
        "core_insight": "今日未采集到新信息，无法进行结构性分析。",
        "keywords": [],
        "dimensions": {
            "technology": {**empty_dim, "model_capability": "", "new_paradigm": ""},
            "infrastructure": {**empty_dim, "inference_cost": "", "compute_trend": ""},
            "application": {**empty_dim, "new_industries": "", "displacement": ""},
            "capital": {**empty_dim, "funding_trend": "", "valuation": "", "strategic_moves": ""},
            "risk": {**empty_dim, "regulation": "", "ethics_safety": "", "supply_chain": ""},
        },
    }


def analyze_articles(
    articles: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Stage-1: structural change analysis → JSON.

    Returns (analysis_dict, token_usage_dict).
    """
    empty_usage: dict[str, Any] = {
        "model": "", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    }

    if not articles:
        logger.warning("No articles to analyse — returning empty analysis.")
        return _empty_analysis(), empty_usage

    client = get_client()
    model = get_model()

    user_prompt = _build_user_prompt(articles)
    logger.info(
        "Sending %d articles to LLM (%s) for structural analysis …",
        len(articles), model,
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.3,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )

    raw = response.choices[0].message.content
    analysis: dict[str, Any] = json.loads(raw)  # type: ignore[arg-type]

    usage = getattr(response, "usage", None)
    token_usage: dict[str, Any] = {
        "model": model,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
    }
    logger.info(
        "Stage-1 complete: %s  (tokens: %d prompt + %d completion = %d total)",
        analysis.get("title", ""),
        token_usage["prompt_tokens"],
        token_usage["completion_tokens"],
        token_usage["total_tokens"],
    )
    logger.info("Core insight: %s", analysis.get("core_insight", ""))
    return analysis, token_usage


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sample = [
        {
            "source": "arXiv cs.AI",
            "category": "papers",
            "title": "Scaling Laws for Next-Generation Transformers",
            "url": "https://arxiv.org/abs/2026.00001",
            "summary": "We demonstrate new scaling laws that extend model capability boundaries.",
        },
        {
            "source": "TechCrunch AI",
            "category": "industry",
            "title": "AI Startup Raises $500M Series C for Enterprise Agents",
            "url": "https://techcrunch.com/example",
            "summary": "Major funding round signals concentrated capital in AI agent space.",
        },
    ]
    result = analyze_articles(sample)
    print(json.dumps(result, indent=2, ensure_ascii=False))
