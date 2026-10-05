"""Small OpenAI-compatible Chat Completions client for Agent calls."""

from __future__ import annotations

import json
import math
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from urllib.parse import urlsplit


class IncompleteResponseError(RuntimeError):
    def __init__(self, reason: str, content: str | None = None):
        self.reason = reason
        self.content = content
        super().__init__(f"Agent provider response incomplete: {reason}")


class InvalidJSONResponseError(ValueError):
    def __init__(self, content: str, parse_error: json.JSONDecodeError):
        self.content = content
        self.parse_error = parse_error
        super().__init__(f"Agent provider returned invalid JSON: {parse_error.msg} "
                         f"(line {parse_error.lineno}, column {parse_error.colno})")


@dataclass(frozen=True)
class ApiProvider:
    url: str
    model: str
    api_key: str = field(repr=False)
    thinking: str = "enabled"
    iteration_effort: str = "high"
    review_effort: str = "max"
    request_timeout_seconds: float | None = None
    token_budgets: dict[str, int] = field(default_factory=dict)

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
        timeout = self.request_timeout_seconds
        if timeout is not None and (isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or
                                    not math.isfinite(timeout) or timeout <= 0):
            raise ValueError("request_timeout_seconds must be a positive finite number")
        stages = {"proposal", "analysis", "review", "reflection", "diagnosis"}
        if (not isinstance(self.token_budgets, dict) or set(self.token_budgets) - stages or
                any(type(value) is not int or value <= 0 for value in self.token_budgets.values())):
            raise ValueError("token_budgets must map proposal, analysis, review, reflection or diagnosis to positive integers")

    def token_budget(self, stage: str, default: int) -> int:
        return self.token_budgets.get(stage, default)

    @property
    def endpoint(self) -> str:
        url = self.url.rstrip("/")
        return url if url.endswith("/chat/completions") else url + "/chat/completions"


def request_json(provider: ApiProvider, effort: str, messages: list[dict], *,
                 max_tokens: int, timeout: float | None = None) -> dict:
    if timeout is None:
        timeout = provider.request_timeout_seconds if provider.request_timeout_seconds is not None else 180
    deadline = time.monotonic() + timeout
    payload = {"model": provider.model, "reasoning_effort": effort,
               "response_format": {"type": "json_object"},
               "max_tokens": max_tokens, "messages": messages}
    if provider.thinking == "enabled":
        payload["thinking"] = {"type": "enabled"}
    request = urllib.request.Request(
        provider.endpoint, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {provider.api_key}",
                 "Content-Type": "application/json"})

    completed = queue.Queue(maxsize=1)
    cancelled = threading.Event()

    def read_response() -> None:
        if cancelled.is_set():
            return
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                if cancelled.is_set():
                    return
                read1 = getattr(response, "read1", None)
                if callable(read1):
                    chunks = []
                    while not cancelled.is_set():
                        chunk = read1(65536)
                        if not chunk:
                            break
                        chunks.append(chunk)
                    if cancelled.is_set():
                        return
                    raw = b"".join(chunks)
                else:
                    raw = response.read()
            if not cancelled.is_set():
                completed.put((True, json.loads(raw)))
        except Exception as error:
            if not cancelled.is_set():
                completed.put((False, error))

    if deadline <= time.monotonic():
        raise TimeoutError(f"Agent provider timed out after {timeout}s")
    threading.Thread(target=read_response, daemon=True).start()
    try:
        succeeded, result = completed.get(timeout=max(0, deadline - time.monotonic()))
    except queue.Empty:
        cancelled.set()
        raise TimeoutError(f"Agent provider timed out after {timeout}s") from None
    except BaseException:
        cancelled.set()
        raise
    if not succeeded:
        if isinstance(result, urllib.error.HTTPError):
            raise RuntimeError(f"Agent provider returned HTTP {result.code}; check model and reasoning effort support") from None
        raise result
    choice = result["choices"][0]
    content = choice.get("message", {}).get("content")
    if choice["finish_reason"] != "stop":
        raise IncompleteResponseError(choice["finish_reason"], content if isinstance(content, str) else None)
    try:
        answer = json.loads(content)
    except json.JSONDecodeError as error:
        raise InvalidJSONResponseError(content, error) from None
    if not isinstance(answer, dict):
        raise ValueError("Agent provider must return a JSON object")
    return answer
