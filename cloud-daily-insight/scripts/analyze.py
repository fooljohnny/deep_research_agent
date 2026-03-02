"""
analyze.py — Stage-1 prompt: Cloud computing industry structural change analysis.

Identifies *structural shifts* across five dimensions for cloud & SaaS:
technology, infrastructure, applications, capital, and risk.
The output is a structured JSON that Stage-2 (generate.py) turns into
a Markdown insight post.
"""

import json
import logging
import re
from typing import Any

from llm_client import get_client, get_model, chat_completion_with_retry

logger = logging.getLogger(__name__)

# Groq 免费版 TPM 限制约 12k，需控制 prompt 体积
MAX_ARTICLES_FOR_ANALYSIS = 45
MAX_SUMMARY_CHARS = 200

SYSTEM_PROMPT = """\
你是一位云计算与SaaS产业结构分析师。

你的任务不是做新闻摘要，而是从今日新增的云计算/SaaS信息中识别"结构性变化信号"。

请重点关注以下云厂商与SaaS厂商的进展：
- 云厂商：AWS、Google Cloud、Azure、阿里云、腾讯云、火山云(Volcengine)
- SaaS厂商：Salesforce、ServiceNow、Adobe、Workday、Zoom、Slack、HubSpot、Zendesk 等

请从以下五个维度进行分析：

### 1. 技术层变化 (technology)
- 云原生/容器/K8s/Serverless 是否有新突破？
- 数据库、存储、网络、安全等基础设施是否有重大更新？
- AI/ML 与云的融合进展如何？
- 证据：引用具体的文章标题和来源。
- 变化强度：重大突破 / 渐进改善 / 无显著变化

### 2. 基础设施变化 (infrastructure)
- 云厂商定价、区域扩张、算力供应有何变化？
- 芯片/GPU 供应、绿色数据中心趋势？
- 多云/混合云架构有何新动向？
- 证据：引用具体的文章标题和来源。
- 变化强度：重大突破 / 渐进改善 / 无显著变化

### 3. 应用层变化 (application)
- 云/SaaS 是否进入新行业或新场景？
- 企业上云、数字化转型有何新案例？
- 垂直行业 SaaS 有何进展？
- 证据：引用具体的文章标题和来源。
- 变化强度：重大突破 / 渐进改善 / 无显著变化

### 4. 资本信号 (capital)
- 云/SaaS 融资方向是否出现集中趋势？
- 估值水平、并购有何动向？
- 大厂战略投资/收购有何新动作？
- 证据：引用具体的文章标题和来源。
- 信号强度：强信号 / 中等信号 / 弱信号

### 5. 风险信号 (risk)
- 监管/合规（数据主权、隐私）有何新动向？
- 供应链/地缘风险？
- 安全事件、宕机事故？
- 证据：引用具体的文章标题和来源。
- 信号强度：强信号 / 中等信号 / 弱信号

### 输出规则
- 如果某个维度今日没有明显变化，写明"今日无显著变化"并简要说明原因。
- 不要编造不存在的变化——没有信号比虚假信号更有价值。
- 最后给出一个"今日核心判断"(core_insight)：用一句话概括今天最值得关注的云计算结构性变化。
- 给出一个适合做标题的 title。
- 提取 8-15 个关键词(keywords)：反映今日核心话题的关键技术术语、云厂商名、SaaS产品名等。

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
      "cloud_native": "string – 云原生/容器等技术变化分析",
      "infra_updates": "string – 数据库/存储/网络等基础设施分析",
      "evidence": [
        {"title": "article title", "source": "source name", "url": "url"}
      ]
    },
    "infrastructure": {
      "has_change": true/false,
      "intensity": "重大突破 | 渐进改善 | 无显著变化",
      "pricing_region": "string – 定价与区域扩张分析",
      "compute_supply": "string – 算力/芯片供应分析",
      "evidence": [
        {"title": "article title", "source": "source name", "url": "url"}
      ]
    },
    "application": {
      "has_change": true/false,
      "intensity": "重大突破 | 渐进改善 | 无显著变化",
      "new_industries": "string – 新行业/新场景分析",
      "digital_transformation": "string – 企业上云/数字化转型分析",
      "evidence": [
        {"title": "article title", "source": "source name", "url": "url"}
      ]
    },
    "capital": {
      "has_change": true/false,
      "intensity": "强信号 | 中等信号 | 弱信号",
      "funding_trend": "string – 融资方向分析",
      "valuation_mna": "string – 估值与并购分析",
      "strategic_moves": "string – 大厂战略动向分析",
      "evidence": [
        {"title": "article title", "source": "source name", "url": "url"}
      ]
    },
    "risk": {
      "has_change": true/false,
      "intensity": "强信号 | 中等信号 | 弱信号",
      "regulation": "string – 监管/合规分析",
      "supply_chain": "string – 供应链/地缘风险分析",
      "evidence": [
        {"title": "article title", "source": "source name", "url": "url"}
      ]
    }
  }
}
"""


def _sample_articles(articles: list[dict[str, str]], max_total: int) -> list[dict[str, str]]:
    """按类别均衡采样，保证云/SaaS/行业均有覆盖，且总条数不超过 max_total。"""
    by_category: dict[str, list[dict[str, str]]] = {}
    for a in articles:
        cat = a.get("category", "other")
        by_category.setdefault(cat, []).append(a)

    # 每类最多取若干条，保证多样性
    per_cat = max(8, max_total // 4)
    sampled: list[dict[str, str]] = []
    for cat in ["cloud", "saas", "industry", "other"]:
        items = by_category.get(cat, [])
        sampled.extend(items[:per_cat])
    return sampled[:max_total]


def _build_user_prompt(articles: list[dict[str, str]]) -> str:
    sampled = _sample_articles(articles, MAX_ARTICLES_FOR_ANALYSIS)
    lines = [
        f"今日日期: {_today()}",
        f"信息条目数: {len(sampled)}",
        "",
        "以下是今日新增的云计算/SaaS信息列表（按来源分类）：",
        "",
    ]

    by_category: dict[str, list[dict[str, str]]] = {}
    for a in sampled:
        cat = a.get("category", "other")
        by_category.setdefault(cat, []).append(a)

    category_labels = {
        "cloud": "云厂商",
        "saas": "SaaS厂商",
        "industry": "行业资讯",
        "other": "其他",
    }

    idx = 1
    for cat in ["cloud", "saas", "industry", "other"]:
        items = by_category.get(cat, [])
        if not items:
            continue
        lines.append(f"### {category_labels.get(cat, cat)}（{len(items)} 条）")
        lines.append("")
        for a in items:
            lines.append(f"{idx}. [{a['source']}] {a['title']}")
            lines.append(f"   URL: {a['url']}")
            if a.get("summary"):
                lines.append(f"   摘要: {a['summary'][:MAX_SUMMARY_CHARS]}")
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
        "title": f"云计算产业日报 – {_today()}",
        "core_insight": "今日未采集到新信息，无法进行结构性分析。",
        "keywords": [],
        "dimensions": {
            "technology": {**empty_dim, "cloud_native": "", "infra_updates": ""},
            "infrastructure": {**empty_dim, "pricing_region": "", "compute_supply": ""},
            "application": {**empty_dim, "new_industries": "", "digital_transformation": ""},
            "capital": {**empty_dim, "funding_trend": "", "valuation_mna": "", "strategic_moves": ""},
            "risk": {**empty_dim, "regulation": "", "supply_chain": ""},
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

    sampled_count = len(_sample_articles(articles, MAX_ARTICLES_FOR_ANALYSIS))
    user_prompt = _build_user_prompt(articles)
    logger.info(
        "Sending %d articles (sampled from %d) to LLM (%s) for structural analysis …",
        sampled_count, len(articles), model,
    )

    # groq/compound 对 response_format 支持不稳定，可能返回空；仅对非 compound 使用
    create_kwargs: dict[str, Any] = {
        "model": model,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    if "compound" not in model.lower():
        create_kwargs["response_format"] = {"type": "json_object"}
    response = chat_completion_with_retry(**create_kwargs)

    raw = response.choices[0].message.content or ""
    raw = raw.strip()
    # 移除可能的 markdown 代码块包裹
    if raw.startswith("```"):
        lines = raw.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines)
    # 若仍无内容，尝试从 executed_tools 等提取（compound 模型可能返回不同结构）
    if not raw and hasattr(response.choices[0].message, "executed_tools"):
        tools = getattr(response.choices[0].message, "executed_tools", [])
        for t in tools:
            if isinstance(t, dict) and "result" in t:
                raw = str(t.get("result", ""))
                break
    if not raw:
        logger.error("LLM returned empty content. Full response: %s", response)
        raise ValueError("LLM returned empty response")
    try:
        analysis: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError:
        # 尝试用正则提取 JSON 块
        m = re.search(r"\{[\s\S]*\}", raw)
        if m:
            analysis = json.loads(m.group(0))  # type: ignore[assignment]
        else:
            logger.error("LLM returned invalid JSON. Raw (first 500): %s", raw[:500])
            raise

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
            "source": "AWS Blog",
            "category": "cloud",
            "title": "Amazon EC2 introduces new Graviton4 instances",
            "url": "https://aws.amazon.com/blogs/aws/example",
            "summary": "New ARM-based instances with 40% better performance.",
        },
        {
            "source": "TechCrunch",
            "category": "industry",
            "title": "Cloud startup raises $200M for multi-cloud management",
            "url": "https://techcrunch.com/example",
            "summary": "Funding signals enterprise cloud adoption acceleration.",
        },
    ]
    result, usage = analyze_articles(sample)
    print(json.dumps(result, indent=2, ensure_ascii=False))
