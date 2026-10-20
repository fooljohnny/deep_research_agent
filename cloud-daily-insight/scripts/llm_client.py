"""
llm_client.py — Shared LLM client factory.

Supports multiple providers (Groq, OpenAI, or any OpenAI-compatible API)
via environment variables. Both analyze.py and generate.py import from here.

Environment variables
---------------------
LLM_PROVIDER   : "groq" (default) | "openai" | "custom"
LLM_API_KEY    : API key for the chosen provider  (required)
LLM_MODEL      : Model name (default depends on provider)
LLM_BASE_URL   : Override the API base URL (optional; auto-set per provider)
LLM_EXTRA_BODY : Optional JSON merged into chat.completions.create as extra_body.
LLM_THINKING_ENABLED : "1"/"true"/"enabled" → {"thinking":{"type":"enabled"}} (Huawei MaaS).
"""

import json
import logging
import os
import time
from typing import Any

import openai

logger = logging.getLogger(__name__)

from openai import RateLimitError


def _first_balanced_brace_object(s: str, start: int) -> str | None:
    """Return substring from first `{` through matching `}` (strings/escapes aware), or None."""
    if start < 0 or start >= len(s) or s[start] != "{":
        return None
    depth = 0
    i = start
    in_str = False
    esc = False
    while i < len(s):
        c = s[i]
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
            i += 1
            continue
        if c == '"':
            in_str = True
            i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
        i += 1
    return None


def parse_llm_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object from LLM output (tolerates trailing junk / nested `}` in strings)."""
    text = (text or "").strip()
    if not text:
        return None

    dec = json.JSONDecoder()
    for idx in range(len(text)):
        if text[idx] != "{":
            continue
        try:
            obj, _end = dec.raw_decode(text, idx)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass
        chunk = _first_balanced_brace_object(text, idx)
        if chunk:
            try:
                obj = json.loads(chunk)
                if isinstance(obj, dict):
                    return obj
            except json.JSONDecodeError:
                pass
    return None

# 429 TPM 限流时重试间隔（秒），需 ≥60 以跨过 compound 子模型（gpt-oss 8K TPM）的分钟窗口
RETRY_DELAY_SEC = int(os.environ.get("RETRY_DELAY_SEC", "65"))
MAX_RETRIES = 3

PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "groq/compound",  # 无每日 token 上限，避免 TPD 429
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
    },
}


def get_client() -> openai.OpenAI:
    """Return a configured OpenAI-compatible client."""
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    api_key = os.environ.get("LLM_API_KEY", "")

    if not api_key:
        raise EnvironmentError(
            "LLM_API_KEY is not set. "
            "Add it as a repository secret and pass it via the workflow."
        )

    defaults = PROVIDER_DEFAULTS.get(provider, {})
    base_url = os.environ.get("LLM_BASE_URL", defaults.get("base_url", ""))

    kwargs: dict[str, str] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    logger.info("LLM provider: %s  base_url: %s", provider, base_url or "(default)")
    return openai.OpenAI(**kwargs)


def get_model() -> str:
    """Return the model name to use."""
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    defaults = PROVIDER_DEFAULTS.get(provider, {})
    model = os.environ.get("LLM_MODEL", defaults.get("model", "groq/compound"))
    return model


def normalize_assistant_message_content(msg: Any) -> str:
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


def get_chat_completion_extra_body() -> dict[str, Any]:
    raw = (os.environ.get("LLM_EXTRA_BODY") or "").strip()
    if raw:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise EnvironmentError(f"LLM_EXTRA_BODY must be valid JSON: {e}") from e
        if not isinstance(obj, dict):
            raise EnvironmentError("LLM_EXTRA_BODY must be a JSON object")
        return obj
    if os.environ.get("LLM_THINKING_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on", "enabled",
    ):
        return {"thinking": {"type": "enabled"}}
    return {}


def extend_chat_completion_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    extra = get_chat_completion_extra_body()
    if not extra:
        return kwargs
    merged = dict(extra)
    merged.update(kwargs.get("extra_body") or {})
    return {**kwargs, "extra_body": merged}


def chat_completion_with_retry(**kwargs) -> openai.types.chat.ChatCompletion:
    """调用 chat.completions.create，遇 429 时等待后重试。"""
    client = get_client()
    kwargs = extend_chat_completion_kwargs(dict(kwargs))
    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            last_err = e
            if attempt < MAX_RETRIES - 1:
                logger.warning(
                    "Rate limit 429, waiting %ds before retry (%d/%d) …",
                    RETRY_DELAY_SEC, attempt + 1, MAX_RETRIES,
                )
                time.sleep(RETRY_DELAY_SEC)
            else:
                raise
    raise last_err  # type: ignore[misc]
