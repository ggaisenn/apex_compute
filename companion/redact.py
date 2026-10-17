"""Credential scrubbing for child-process and model-facing text."""

from __future__ import annotations

import re

_KEY_SHAPES = (
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}\b"),
    re.compile(r"\b(?:sk|key)-[A-Za-z0-9_-]{20,}\b"),
    re.compile(r"(?i)(authorization\s*:\s*bearer\s+)[^\s,;]+"),
    re.compile(r"(?i)(x-goog-api-key\s*:\s*)[^\s,;]+"),
)


def redact(text: object, secret: str | None = None) -> str:
    value = str(text)
    if secret:
        value = value.replace(secret, "[REDACTED]")
    value = re.sub(r"(?i)(api[_-]?key\s*[=:]\s*)[^\s,;]+", r"\1[REDACTED]", value)
    for pattern in _KEY_SHAPES:
        if pattern.groups:
            value = pattern.sub(lambda match: (match.group(1) or "") + "[REDACTED]", value)
        else:
            value = pattern.sub("[REDACTED]", value)
    return value
