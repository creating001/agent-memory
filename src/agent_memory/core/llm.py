from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx
from openai import OpenAI


@dataclass(frozen=True)
class ChatResult:
    content: str
    tokens: int


class ChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
        default_seed: int | None = None,
        default_top_p: float | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if timeout_seconds is not None:
            kwargs["timeout"] = timeout_seconds
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        kwargs["http_client"] = make_http_client(base_url or "")
        self.client = OpenAI(**kwargs)
        self.default_seed = default_seed
        self.default_top_p = default_top_p

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 8192,
        thinking: str = "default",
        response_format: dict[str, Any] | None = None,
        seed: int | None = None,
        top_p: float | None = None,
    ) -> ChatResult:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        effective_seed = self.default_seed if seed is None else seed
        effective_top_p = self.default_top_p if top_p is None else top_p
        if effective_seed is not None:
            kwargs["seed"] = effective_seed
        if effective_top_p is not None:
            kwargs["top_p"] = effective_top_p
        if response_format:
            kwargs["response_format"] = response_format
        if thinking != "default":
            kwargs["extra_body"] = {"thinking": {"type": thinking}}

        response = self.client.chat.completions.create(**kwargs)
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "total_tokens", 0) or 0)
        return ChatResult(response.choices[0].message.content or "", tokens)


def extract_json_object(text: str) -> dict[str, Any] | None:
    cleaned = text.strip()
    if not cleaned:
        return None
    try:
        value = json.loads(cleaned)
        return value if isinstance(value, dict) else None
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        value = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def is_local_url(url: str) -> bool:
    host = urlparse(url).hostname or ""
    return host in {"127.0.0.1", "localhost", "::1"}


def make_http_client(base_url: str) -> httpx.Client:
    if is_local_url(base_url):
        return httpx.Client(trust_env=False)

    proxy = first_http_proxy()
    if proxy:
        return httpx.Client(trust_env=False, proxy=proxy)
    return httpx.Client(trust_env=False)


def first_http_proxy() -> str:
    for name in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        value = os.environ.get(name, "")
        if value.startswith(("http://", "https://")):
            return value
    return ""
