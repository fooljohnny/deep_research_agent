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
"""

import os
import logging
import time

import openai

logger = logging.getLogger(__name__)

from openai import RateLimitError

# 429 TPM 限流时重试间隔（秒）
RETRY_DELAY_SEC = 20
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


def chat_completion_with_retry(**kwargs) -> openai.types.chat.ChatCompletion:
    """调用 chat.completions.create，遇 429 时等待后重试。"""
    client = get_client()
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
