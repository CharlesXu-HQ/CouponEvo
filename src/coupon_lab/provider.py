"""Small OpenAI-compatible Chat Completions client for Agent calls."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlsplit


@dataclass(frozen=True)
class ApiProvider:
    url: str
    model: str
    api_key: str = field(repr=False)
    thinking: str = "enabled"
    iteration_effort: str = "high"
    review_effort: str = "max"

    def __post_init__(self) -> None:
        parts = urlsplit(self.url)
        if parts.scheme not in {"http", "https"} or not parts.netloc or parts.username or parts.password or parts.query or parts.fragment:
            raise ValueError("provider_url must be an HTTP(S) base URL or chat/completions endpoint without credentials or query")
        if not self.model or not self.api_key:
            raise ValueError("provider model and API key are required")
        if self.thinking not in {"enabled", "omit"}:
            raise ValueError("thinking must be enabled or omitted when using high/max reasoning")
        if self.iteration_effort not in {"high", "max"} or self.review_effort not in {"high", "max"}:
            raise ValueError("iteration_effort and review_effort must be high or max")

    @property
    def endpoint(self) -> str:
        url = self.url.rstrip("/")
        return url if url.endswith("/chat/completions") else url + "/chat/completions"


def request_json(provider: ApiProvider, effort: str, messages: list[dict], *,
                 max_tokens: int, timeout: int = 180) -> dict:
    payload = {"model": provider.model, "reasoning_effort": effort,
               "response_format": {"type": "json_object"},
               "max_tokens": max_tokens, "messages": messages}
    if provider.thinking == "enabled":
        payload["thinking"] = {"type": "enabled"}
    request = urllib.request.Request(
        provider.endpoint, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {provider.api_key}",
                 "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"Agent provider returned HTTP {error.code}; check model and reasoning effort support") from None
    choice = result["choices"][0]
    if choice["finish_reason"] != "stop":
        raise RuntimeError(f"Agent provider response incomplete: {choice['finish_reason']}")
    answer = json.loads(choice["message"]["content"])
    if not isinstance(answer, dict):
        raise ValueError("Agent provider must return a JSON object")
    return answer
