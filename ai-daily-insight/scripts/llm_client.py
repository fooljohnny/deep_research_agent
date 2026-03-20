"""
llm_client.py — Shared LLM client factory.

Supports multiple providers (Groq, OpenAI, or any OpenAI-compatible API)
via environment variables. Both analyze.py and generate.py import from here.

Environment variables
---------------------
LLM_PROVIDER   : "groq" (default) | "openai" | "deepseek" | "custom"
LLM_API_KEY    : API key for the chosen provider  (required)
LLM_MODEL      : Model name (default depends on provider)
LLM_BASE_URL   : Override the API base URL (optional; auto-set per provider)
"""

import os
import logging

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
            "Set LLM_PROVIDER to 'deepseek' (recommended) or 'custom' with LLM_BASE_URL="
            "'https://api.deepseek.com/v1'."
        )

    if provider == "custom" and not (base_url or "").strip():
        raise EnvironmentError(
            "LLM_PROVIDER is 'custom' but LLM_BASE_URL is empty. "
            "Set LLM_BASE_URL to your OpenAI-compatible base URL "
            "(e.g. https://api.deepseek.com/v1). "
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
