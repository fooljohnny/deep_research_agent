"""
analyze.py — Stage-1 prompt: Cloud computing industry structural change analysis.

Identifies *structural shifts* across five dimensions for cloud & SaaS:
technology, infrastructure, applications, capital, and risk.
The output is a structured JSON that Stage-2 (generate.py) turns into
a Markdown insight post.

支持分批分析：将文章拆成多批（每批≤30篇），每批调用一次 LLM，最后合并结果。
"""

import json
import logging
import os
import re
import time
from typing import Any

from llm_client import get_client, get_model, chat_completion_with_retry

logger = logging.getLogger(__name__)


def _json_response_format_enabled() -> bool:
    """OpenAI-style json_object mode; disable for gateways that return empty content."""
    v = os.environ.get("LLM_JSON_RESPONSE_FORMAT")
    if v is None or not str(v).strip():
        return True
    return str(v).strip().lower() not in ("0", "false", "no", "off")


def _normalize_assistant_content(msg: Any) -> str:
    """Collect assistant text from content, multimodal parts, or reasoning-style fields."""
    c = getattr(msg, "content", None)
    if isinstance(c, list):
        texts: list[str] = []
        for p in c:
            if isinstance(p, dict):
                if p.get("type") == "text" and p.get("text"):
                    texts.append(str(p["text"]))
                elif p.get("text"):
                    texts.append(str(p["text"]))
            elif isinstance(p, str):
                texts.append(p)
        joined = "\n".join(texts).strip()
        if joined:
            return joined
    elif isinstance(c, str) and c.strip():
        return c.strip()
    elif c not in (None, "") and str(c).strip():
        return str(c).strip()

    for attr in ("reasoning_content", "reasoning"):
        v = getattr(msg, attr, None)
        if isinstance(v, str) and v.strip():
            return v.strip()

    if hasattr(msg, "model_dump"):
        d = msg.model_dump(mode="python")
        for key in ("content", "reasoning_content", "reasoning", "text"):
            val = d.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
    return ""


# 每批最多文章数，控制单次 prompt 体积
BATCH_SIZE = 30
MAX_SUMMARY_CHARS = 150
# 批次间等待（秒），缓解 TPM 限流
BATCH_DELAY_SEC = int(os.environ.get("BATCH_DELAY_SEC", "65"))

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

MERGE_SYSTEM_PROMPT = """\
你是云计算产业结构分析师。你收到多份「分批分析」的结构化 JSON，每份来自不同文章子集。

请将它们合并为**一份**统一的结构分析 JSON，要求：
1. 合并各维度的 evidence，去重（同 URL 只保留一条），按重要性排序，每维度最多保留 10 条
2. 各维度的分析文本：综合多份的结论，取信号更强者；若多份都无变化则保持「今日无显著变化」
3. intensity：取多份中最强的（重大突破>渐进改善>无显著变化；强信号>中等>弱）
4. core_insight：用一句话概括所有批次中最值得关注的结构性变化
5. title：综合性的博客标题
6. keywords：合并去重，保留 8-15 个核心关键词

返回严格的 JSON，schema 与输入一致，不要 markdown 代码块。
"""


def _split_into_batches(articles: list[dict[str, str]]) -> list[list[dict[str, str]]]:
    """按类别均衡拆分为多批，每批最多 BATCH_SIZE 篇，轮询保证每批都有云/SaaS/行业覆盖。"""
    by_category: dict[str, list[dict[str, str]]] = {}
    for a in articles:
        cat = a.get("category", "other")
        by_category.setdefault(cat, []).append(a)

    cats = ["cloud", "saas", "industry", "other"]
    # 轮询交错，保证每批多样性
    ordered: list[dict[str, str]] = []
    max_len = max(len(by_category.get(c, [])) for c in cats)
    for i in range(max_len):
        for c in cats:
            items = by_category.get(c, [])
            if i < len(items):
                ordered.append(items[i])

    return [
        ordered[i : i + BATCH_SIZE]
        for i in range(0, len(ordered), BATCH_SIZE)
    ]


def _build_user_prompt(articles: list[dict[str, str]], batch_label: str = "") -> str:
    lines = [
        f"今日日期: {_today()}",
        f"信息条目数: {len(articles)}",
        "",
    ]
    if batch_label:
        lines.append(f"{batch_label}\n")
    lines.append("以下是云计算/SaaS信息列表（按来源分类）：\n")

    by_category: dict[str, list[dict[str, str]]] = {}
    for a in articles:
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


def _parse_llm_json(raw: str, response: Any) -> dict[str, Any]:
    """解析 LLM 返回的 JSON，处理 markdown 包裹与空内容。"""
    text = (raw or "").strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    if not text and hasattr(response.choices[0].message, "executed_tools"):
        tools = getattr(response.choices[0].message, "executed_tools", [])
        for t in tools:
            if isinstance(t, dict) and "result" in t:
                text = str(t.get("result", "")).strip()
                break

    if not text:
        choice = response.choices[0]
        fr = getattr(choice, "finish_reason", None)
        dump = (
            choice.message.model_dump(mode="python")
            if hasattr(choice.message, "model_dump")
            else repr(choice.message)
        )
        logger.error("Empty assistant body: finish_reason=%s message=%s", fr, dump)
        raise ValueError(
            "LLM returned an empty message. Some gateways omit content when json_object mode "
            "is unsupported — set LLM_JSON_RESPONSE_FORMAT=0 to disable JSON mode."
        )

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE | re.DOTALL)
    if m:
        inner = m.group(1).strip()
        try:
            return json.loads(inner)
        except json.JSONDecodeError:
            pass

    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    logger.error("Invalid JSON. Raw (first 800): %r", text[:800])
    raise json.JSONDecodeError("Response is not valid JSON", text, 0)


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


def _analyze_batch(
    articles: list[dict[str, str]],
    batch_idx: int,
    total_batches: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """分析单批文章，返回 (analysis, token_usage)。"""
    model = get_model()
    label = f"第 {batch_idx + 1}/{total_batches} 批" if total_batches > 1 else ""
    user_prompt = _build_user_prompt(articles, batch_label=label)

    create_kwargs: dict[str, Any] = {
        "model": model,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    if "compound" not in model.lower() and _json_response_format_enabled():
        create_kwargs["response_format"] = {"type": "json_object"}
    response = chat_completion_with_retry(**create_kwargs)
    raw = _normalize_assistant_content(response.choices[0].message)
    analysis = _parse_llm_json(raw, response)

    usage = getattr(response, "usage", None)
    token_usage: dict[str, Any] = {
        "model": model,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
    }
    return analysis, token_usage


def _merge_analyses(analyses: list[dict[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    """将多份分批分析合并为一份，调用 LLM。"""
    model = get_model()
    user_prompt = (
        f"今日日期: {_today()}\n\n"
        f"以下是 {len(analyses)} 份分批分析结果，请合并为一份：\n\n"
        + json.dumps(analyses, indent=2, ensure_ascii=False)
    )
    create_kwargs: dict[str, Any] = {
        "model": model,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": MERGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    }
    if "compound" not in model.lower() and _json_response_format_enabled():
        create_kwargs["response_format"] = {"type": "json_object"}
    response = chat_completion_with_retry(**create_kwargs)
    raw = _normalize_assistant_content(response.choices[0].message)
    merged = _parse_llm_json(raw, response)
    if "date" not in merged or not merged["date"]:
        merged["date"] = _today()

    usage = getattr(response, "usage", None)
    token_usage: dict[str, Any] = {
        "model": model,
        "prompt_tokens": getattr(usage, "prompt_tokens", 0) if usage else 0,
        "completion_tokens": getattr(usage, "completion_tokens", 0) if usage else 0,
        "total_tokens": getattr(usage, "total_tokens", 0) if usage else 0,
    }
    return merged, token_usage


def analyze_articles(
    articles: list[dict[str, str]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Stage-1: 分批结构分析 → 合并 → JSON。

    将文章拆成多批（每批≤BATCH_SIZE），每批调用一次 LLM，最后合并。
    Returns (analysis_dict, token_usage_dict)。
    """
    empty_usage: dict[str, Any] = {
        "model": "", "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
    }

    if not articles:
        logger.warning("No articles to analyse — returning empty analysis.")
        return _empty_analysis(), empty_usage

    model = get_model()
    batches = _split_into_batches(articles)
    logger.info(
        "Stage-1: %d articles → %d batch(es), %d per batch, model=%s",
        len(articles), len(batches), BATCH_SIZE, model,
    )

    batch_analyses: list[dict[str, Any]] = []
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    for i, batch in enumerate(batches):
        if i > 0:
            logger.info("Waiting %ds for TPM buffer before next batch …", BATCH_DELAY_SEC)
            time.sleep(BATCH_DELAY_SEC)
        logger.info("Analyzing batch %d/%d (%d articles) …", i + 1, len(batches), len(batch))
        analysis, usage = _analyze_batch(batch, i, len(batches))
        batch_analyses.append(analysis)
        for k in total_usage:
            total_usage[k] += usage.get(k, 0)

    if len(batch_analyses) == 1:
        analysis = batch_analyses[0]
        token_usage = {
            "model": model,
            "prompt_tokens": total_usage["prompt_tokens"],
            "completion_tokens": total_usage["completion_tokens"],
            "total_tokens": total_usage["total_tokens"],
        }
    else:
        logger.info("Waiting %ds before merge …", BATCH_DELAY_SEC)
        time.sleep(BATCH_DELAY_SEC)
        logger.info("Merging %d batch analyses …", len(batch_analyses))
        analysis, merge_usage = _merge_analyses(batch_analyses)
        token_usage = {
            "model": model,
            "prompt_tokens": total_usage["prompt_tokens"] + merge_usage.get("prompt_tokens", 0),
            "completion_tokens": total_usage["completion_tokens"] + merge_usage.get("completion_tokens", 0),
            "total_tokens": total_usage["total_tokens"] + merge_usage.get("total_tokens", 0),
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
