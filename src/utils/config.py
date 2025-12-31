from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI


DEFAULT_SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1/"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-0528-Qwen3-8B"


@dataclass(frozen=True)
class LLMSettings:
    provider: str
    api_key: str
    model: str
    base_url: Optional[str]


def _resolve_settings() -> LLMSettings:
    siliconflow_key = os.getenv("SILICONFLOW_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if siliconflow_key:
        model = (
            os.getenv("LLM_MODEL")
            or os.getenv("SILICONFLOW_MODEL")
            or DEFAULT_MODEL
        )
        base_url = os.getenv("SILICONFLOW_BASE_URL") or DEFAULT_SILICONFLOW_BASE_URL
        return LLMSettings(
            provider="siliconflow",
            api_key=siliconflow_key,
            model=model,
            base_url=base_url,
        )

    if openai_key:
        model = os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL
        base_url = os.getenv("OPENAI_BASE_URL")
        return LLMSettings(
            provider="openai",
            api_key=openai_key,
            model=model,
            base_url=base_url,
        )

    raise RuntimeError(
        "Missing API key. Set SILICONFLOW_API_KEY or OPENAI_API_KEY in your environment."
    )


def get_llm(
    model: Optional[str] = None,
    temperature: float = 0.1,
    max_tokens: Optional[int] = None,
    base_url: Optional[str] = None,
) -> ChatOpenAI:
    settings = _resolve_settings()
    model_name = model or settings.model
    resolved_base_url = base_url if base_url is not None else settings.base_url

    kwargs = {
        "model": model_name,
        "api_key": settings.api_key,
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if resolved_base_url:
        kwargs["base_url"] = resolved_base_url

    return ChatOpenAI(**kwargs)


def get_llm_settings() -> LLMSettings:
    return _resolve_settings()
