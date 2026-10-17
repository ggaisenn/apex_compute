"""Extract exactly one JSON plan object from plain or fenced model text."""

from __future__ import annotations

import json
from typing import Any

from .ops import PlanError, validate_plan


class PlanParseError(ValueError):
    """Model output did not contain exactly one valid object."""


def parse_plan(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise PlanParseError("model response was empty")
    decoder = json.JSONDecoder()
    index = 0
    while index < len(text):
        start = text.find("{", index)
        if start < 0:
            break
        try:
            value, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            index = start + 1
            continue

        if isinstance(value, dict) and "operations" in value:
            try:
                validate_plan(value)
            except PlanError:
                # Ignore schema descriptions and other JSON that cannot be
                # safely translated into an allow-listed operation plan.
                index = start + 1
                continue
            # Gemma can emit a draft and then repeat or revise it. Use the
            # first complete, valid plan; the CLI displays it for approval.
            return value

        # Keep looking inside unrelated JSON wrappers for the plan object.
        index = start + 1

    raise PlanParseError("response must contain a valid JSON operation plan")
