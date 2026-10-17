"""Mark unsupported numerical statements in generated explanations."""

from __future__ import annotations

import re
from typing import Any

_NUMBER = re.compile(
    r"(?<![A-Za-z])(-?\d+(?:,\d{3})*(?:\.\d+)?(?:[eE][+-]?\d+)?)"
    r"\s*(gflops|gf/s|ms|milliseconds|us|µs|microseconds|ns|nanoseconds|%|percent|x|times)?",
    re.IGNORECASE,
)


def check_explanation(
    text: str,
    facts: dict[str, Any],
    maximum_unverified_ratio: float = 0.25,
) -> dict[str, Any]:
    known = [float(value) for value in facts.get("numbers", [])]
    citations = list(_NUMBER.finditer(text))
    unverified: list[str] = []

    def replace(match: re.Match[str]) -> str:
        original = match.group(0)
        token = match.group(1).replace(",", "")
        try:
            value = float(token)
        except ValueError:
            return original
        unit = (match.group(2) or "").casefold()
        if unit in {"ns", "nanoseconds"}:
            value /= 1_000_000.0
        elif unit in {"us", "µs", "microseconds"}:
            value /= 1_000.0
        verified = any(
            abs(value - candidate) <= max(0.005, abs(candidate) * 0.02)
            for candidate in known
        )
        if verified:
            return original
        unverified.append(original.strip())
        return f"[unverified: {original.strip()}]"

    annotated = _NUMBER.sub(replace, text)
    ratio = len(unverified) / max(1, len(citations))
    withheld = bool(citations) and ratio > maximum_unverified_ratio
    if withheld:
        annotated = "The generated explanation was withheld because too many figures could not be checked against the results."
    return {"text": annotated, "unverified": unverified,
            "unverified_ratio": ratio, "withheld": withheld}
