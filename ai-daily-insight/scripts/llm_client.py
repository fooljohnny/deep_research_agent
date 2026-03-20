"""
llm_client.py — Shared LLM client factory.

Supports multiple providers (Groq, OpenAI, or any OpenAI-compatible API)
via environment variables. Both analyze.py and generate.py import from here.

Environment variables
---------------------
LLM_PROVIDER   : "groq" (default) | "openai" | "deepseek" | "custom"
LLM_API_KEY    : API key for the chosen provider  (required)
LLM_MODEL      : Model name (default depends on provider)
LLM_BASE_URL   : If set, overrides the default base URL for *any* provider.
                 Required for `custom` (e.g. third-party DeepSeek-compatible gateways).
LLM_EXTRA_BODY : Optional JSON object merged into chat.completions.create as extra_body
                 (e.g. Huawei ModelArts MaaS: {"thinking":{"type":"enabled"}}).
LLM_THINKING_ENABLED : If "1"/"true"/"enabled", sends thinking mode (shortcut for MaaS example).
"""

import json
import logging
import os
from typing import Any

import openai

logger = logging.getLogger(__name__)

PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "llama-3.3-70b-versatile",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
    },
    # Official DeepSeek SaaS only. Third-party / private deployments: use LLM_PROVIDER=custom
    # + LLM_BASE_URL from the vendor (OpenAI-compatible, usually …/v1).
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "model": "deepseek-chat",
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

    model_env = (os.environ.get("LLM_MODEL") or "").lower()
    if provider == "openai" and "deepseek" in model_env:
        raise EnvironmentError(
            "LLM_PROVIDER is 'openai' but LLM_MODEL looks like a DeepSeek model. "
            "That sends requests to api.openai.com and causes 401 with a DeepSeek API key. "
            "Use LLM_PROVIDER='deepseek' for official api.deepseek.com, or 'custom' with "
            "LLM_BASE_URL set to your vendor's OpenAI-compatible base URL (third-party / private)."
        )

    if provider == "custom" and not (base_url or "").strip():
        raise EnvironmentError(
            "LLM_PROVIDER is 'custom' but LLM_BASE_URL is empty. "
            "Set LLM_BASE_URL to your OpenAI-compatible base URL "
            "(e.g. https://your-vendor.example.com/v1). "
            "In GitHub Actions, add a repository variable LLM_BASE_URL and pass it in the workflow env."
        )

    kwargs: dict[str, str] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url

    logger.info("LLM provider: %s  base_url: %s", provider, base_url or "(default)")
    return openai.OpenAI(**kwargs)


def get_model() -> str:
    """Return the model name to use."""
    provider = os.environ.get("LLM_PROVIDER", "groq").lower()
    defaults = PROVIDER_DEFAULTS.get(provider, {})
    model = os.environ.get("LLM_MODEL", defaults.get("model", "llama-3.3-70b-versatile"))
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
    """Vendor extensions (e.g. Huawei MaaS ``thinking``) passed via OpenAI SDK ``extra_body``."""
    raw = (os.environ.get("LLM_EXTRA_BODY") or "").strip()
    if raw:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            raise EnvironmentError(f"LLM_EXTRA_BODY must be valid JSON: {e}") from e
        if not isinstance(obj, dict):
            raise EnvironmentError("LLM_EXTRA_BODY must be a JSON object, e.g. {\"thinking\":{\"type\":\"enabled\"}}")
        return obj
    if os.environ.get("LLM_THINKING_ENABLED", "").strip().lower() in (
        "1", "true", "yes", "on", "enabled",
    ):
        return {"thinking": {"type": "enabled"}}
    return {}


def extend_chat_completion_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Merge ``get_chat_completion_extra_body()`` into ``chat.completions.create`` kwargs."""
    extra = get_chat_completion_extra_body()
    if not extra:
        return kwargs
    merged = dict(extra)
    merged.update(kwargs.get("extra_body") or {})
    return {**kwargs, "extra_body": merged}
