"""
Stage-2 resilience for gateway content-policy blocks (e.g. MaaS 81011).
"""

from __future__ import annotations

import json
import re
from typing import Any

from openai import PermissionDeniedError

_URL_RE = re.compile(r"https?://\S+")


def is_gateway_sensitive_input_error(exc: BaseException) -> bool:
    if isinstance(exc, PermissionDeniedError):
        blob = str(exc).lower()
        if "81011" in blob or "sensitive" in blob:
            return True
    text = str(exc).lower()
    if "81011" in text or ("modelarts" in text and "forbidden" in text):
        return True
    if "sensitive information" in text:
        return True
    resp = getattr(exc, "response", None)
    if resp is not None:
        try:
            body = getattr(resp, "text", None) or ""
            if body and ("81011" in body or "sensitive" in body.lower()):
                return True
        except Exception:
            pass
    return False


def strip_urls_from_value(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: strip_urls_from_value(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [strip_urls_from_value(x) for x in obj]
    if isinstance(obj, str):
        return _URL_RE.sub("[链接已省略]", obj)
    return obj


def _fmt_evidence(items: list[Any]) -> str:
    lines: list[str] = []
    for ev in items or []:
        if not isinstance(ev, dict):
            continue
        title = str(ev.get("title") or "条目").strip()
        src = str(ev.get("source") or "").strip()
        url = str(ev.get("url") or "").strip()
        if url and url.startswith("http"):
            lines.append(f"- [{title}]({url})" + (f" — {src}" if src else ""))
        elif src:
            lines.append(f"- {title} — {src}")
        else:
            lines.append(f"- {title}")
    return "\n".join(lines) if lines else "- （无）"


def build_fallback_markdown_cloud(
    analysis: dict[str, Any],
    trend_report: dict[str, Any] | None,
    metrics_report: dict[str, Any] | None,
    chart_paths: list[str] | None,
    reason: str,
) -> str:
    today = str(analysis.get("date") or "")[:10]
    title = str(analysis.get("title") or f"云计算产业日报 – {today}")
    tags = analysis.get("keywords") or []
    if not isinstance(tags, list):
        tags = []
    tags_yaml = json.dumps(tags[:16], ensure_ascii=False)
    lines: list[str] = [
        "---",
        f'title: "{title.replace(chr(34), chr(39))}"',
        f"date: {today}",
        f"tags: {tags_yaml}",
        "---",
        "",
        "## 今日云计算结构性变化",
        "",
        f"> {analysis.get('core_insight', '（无）')}",
        "",
    ]
    kt = (trend_report or {}).get("keyword_trends") or {}
    if isinstance(kt, dict):
        parts = []
        if kt.get("new_keywords"):
            parts.append("**新关键词**：" + "、".join(str(x) for x in kt["new_keywords"][:15]))
        if kt.get("rising_keywords"):
            parts.append("**升温**：" + "、".join(str(x) for x in kt["rising_keywords"][:15]))
        if kt.get("fading_keywords"):
            parts.append("**消退**：" + "、".join(str(x) for x in kt["fading_keywords"][:15]))
        if parts:
            lines.append("\n".join(parts))
            lines.append("")

    dims = analysis.get("dimensions") or {}
    labels = [
        ("technology", "## 技术层信号", ["cloud_native", "infra_updates"]),
        ("infrastructure", "## 基础设施信号", ["pricing_region", "compute_supply"]),
        ("application", "## 应用层信号", ["new_industries", "digital_transformation"]),
        ("capital", "## 产业资本信号", ["funding_trend", "valuation_mna", "strategic_moves"]),
        ("risk", "## 风险信号", ["regulation", "supply_chain"]),
    ]
    for key, heading, fields in labels:
        block = dims.get(key) if isinstance(dims, dict) else None
        if not isinstance(block, dict):
            continue
        lines.append(heading)
        lines.append("")
        lines.append(str(block.get("intensity") or "⚪ 见结构数据"))
        lines.append("")
        paras = []
        for f in fields:
            t = block.get(f)
            if isinstance(t, str) and t.strip():
                paras.append(t.strip())
        if paras:
            lines.append("\n\n".join(paras))
            lines.append("")
        ev = _fmt_evidence(block.get("evidence") if isinstance(block.get("evidence"), list) else [])
        lines.append("### 证据摘录")
        lines.append("")
        lines.append(ev)
        lines.append("")

    lines.append("## 潜在拐点判断")
    lines.append("")
    lines.append("（Stage-2 因网关内容策略未生成评述；请以结构分析 JSON 为准。）")
    lines.append("")
    lines.append("## 明日观察点")
    lines.append("")
    lines.append("1. 关注今日 evidence 中云/SaaS 赛道后续进展。")
    lines.append("2. 结合趋势报告验证信号是否持续。")
    lines.append("")
    lines.append("## 长期趋势坐标")
    lines.append("")
    nov = (trend_report or {}).get("overall_novelty")
    if nov is not None:
        lines.append(f"趋势报告 overall_novelty ≈ {nov}。")
    lines.append("")
    lines.append("## 数据洞察")
    lines.append("")
    if metrics_report:
        lines.append("```json")
        lines.append(json.dumps(metrics_report, indent=2, ensure_ascii=False)[:12000])
        lines.append("```")
    else:
        lines.append("今日暂无 metrics 摘要。")
    lines.append("")
    if chart_paths:
        lines.append("### 趋势图")
        lines.append("")
        for rel in chart_paths:
            name = rel.split("/")[-1].replace("_", " ").replace(".png", "")
            lines.append(f"![{name}]({rel})")
        lines.append("")
    lines.append("## 参考来源")
    lines.append("")
    lines.append("见各维度 **证据摘录**。")
    lines.append("")
    lines.append(f"> **说明**：{reason}")
    return "\n".join(lines)
