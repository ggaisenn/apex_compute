"""Minimal Gemini REST client using urllib and an environment-sourced key."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from .redact import redact


@dataclass
class GeminiError(RuntimeError):
    code: str
    message: str
    next_step: str

    def __str__(self) -> str:
        return f"{self.code}: {self.message}. {self.next_step}"


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        endpoint: str | None = None,
        timeout: float = 20.0,
        retries: int = 2,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY", "")
        self.model = model or os.environ.get("APEX_GEMINI_MODEL", "gemma-4")
        self.endpoint = endpoint or os.environ.get(
            "APEX_GEMINI_ENDPOINT", "https://generativelanguage.googleapis.com"
        )
        self.timeout = max(1.0, min(float(timeout), 120.0))
        self.retries = max(0, min(int(retries), 3))

    def generate(self, prompt: str) -> str:
        if not self.api_key:
            raise GeminiError(
                "no_key", "GEMINI_API_KEY is not set", "set the key or use --offline"
            )
        endpoint = self.endpoint.rstrip("/")
        if "{model}" in endpoint:
            endpoint = endpoint.format(model=self.model)
        elif not endpoint.endswith(":generateContent"):
            endpoint = f"{endpoint}/v1beta/models/{self.model}:generateContent"
        body = json.dumps({
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 1024},
        }).encode("utf-8")
        request = urllib.request.Request(
            endpoint,
            data=body,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": self.api_key,
            },
            method="POST",
        )

        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                return self._response_text(payload)
            except urllib.error.HTTPError as exc:
                body_text = redact(exc.read().decode("utf-8", errors="replace"), self.api_key)
                code, message, retry_delay = self._http_error(
                    exc.code, body_text, exc.headers.get("Retry-After")
                )
                if retry_delay is not None and attempt < self.retries:
                    time.sleep(retry_delay)
                    continue
                raise GeminiError(code, message, self._next_step(code)) from None
            except (TimeoutError, OSError, urllib.error.URLError) as exc:
                if attempt < self.retries:
                    time.sleep(min(0.25 * (2**attempt), 1.0))
                    continue
                code = "timeout" if isinstance(exc, TimeoutError) else "network_unavailable"
                raise GeminiError(code, redact(str(exc), self.api_key),
                                  self._next_step(code)) from None
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise GeminiError("malformed_response", str(exc),
                                  self._next_step("malformed_response")) from None
        raise GeminiError("server_error", "model request failed", self._next_step("server_error"))

    @staticmethod
    def _response_text(payload: object) -> str:
        if not isinstance(payload, dict):
            raise GeminiError("malformed_response", "response root is not an object",
                              GeminiClient._next_step("malformed_response"))
        candidates = payload.get("candidates")
        if not isinstance(candidates, list) or not candidates:
            prompt_feedback = payload.get("promptFeedback", {})
            if isinstance(prompt_feedback, dict) and prompt_feedback.get("blockReason"):
                raise GeminiError("blocked_by_safety", "request was blocked",
                                  GeminiClient._next_step("blocked_by_safety"))
            raise GeminiError("malformed_response", "response contains no candidate",
                              GeminiClient._next_step("malformed_response"))
        candidate = candidates[0]
        finish_reason = candidate.get("finishReason") if isinstance(candidate, dict) else None
        if finish_reason == "SAFETY":
            raise GeminiError("blocked_by_safety", "response was blocked by safety filters",
                              GeminiClient._next_step("blocked_by_safety"))
        content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
        parts = content.get("parts", []) if isinstance(content, dict) else []
        if not isinstance(parts, list):
            raise GeminiError("malformed_response", "candidate parts are malformed",
                              GeminiClient._next_step("malformed_response"))
        text_parts = [part["text"] for part in parts
                      if isinstance(part, dict) and isinstance(part.get("text"), str)]
        if not text_parts:
            raise GeminiError("malformed_response", "candidate contains no text",
                              GeminiClient._next_step("malformed_response"))
        return "".join(text_parts)

    @staticmethod
    def _http_error(
        code: int, body: str, retry_after: str | None = None
    ) -> tuple[str, str, float | None]:
        if code in (401, 403):
            return "auth_failed", "Gemini rejected the API key", None
        if code == 404:
            return "model_unavailable", "configured model or endpoint was not found", None
        if code == 429:
            return "rate_limited", f"Gemini rate limit ({body[:300]})", _retry_delay(retry_after)
        if code >= 500:
            return "server_error", f"Gemini service error {code}", 0.5
        return "bad_request", f"Gemini rejected the request ({code})", None

    @staticmethod
    def _next_step(code: str) -> str:
        guidance = {
            "auth_failed": "check GEMINI_API_KEY without sharing it",
            "network_unavailable": "check the network or use --offline",
            "timeout": "retry with a longer --timeout or use --offline",
            "rate_limited": "wait and retry, or use --offline",
            "server_error": "retry later or use --offline",
            "blocked_by_safety": "rephrase the request",
            "malformed_response": "retry or use --offline",
            "model_unavailable": "set APEX_GEMINI_MODEL to an available model",
            "bad_request": "check the configured model and request settings",
        }
        return guidance.get(code, "use --offline")


def _retry_delay(value: str | None) -> float | None:
    """Honor a bounded Retry-After response without retrying an unspecified 429."""
    if not value:
        return None
    try:
        delay = float(value)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
        except (TypeError, ValueError, OverflowError):
            return None
    return max(0.0, min(delay, 5.0))
