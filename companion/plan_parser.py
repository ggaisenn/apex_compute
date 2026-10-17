"""Extract exactly one JSON plan object from plain or fenced model text."""

from __future__ import annotations

import json
from typing import Any


class PlanParseError(ValueError):
    """Model output did not contain exactly one valid object."""


def parse_plan(text: str) -> dict[str, Any]:
    if not isinstance(text, str) or not text.strip():
        raise PlanParseError("model response was empty")
    decoder = json.JSONDecoder()
    objects: list[Any] = []
    index = 0
    in_string = False
    escaped = False
    depth = 0
    start = -1
    while index < len(text):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"' and depth > 0:
            in_string = True
        elif char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                candidate = text[start:index + 1]
                try:
                    value, consumed = decoder.raw_decode(candidate)
                except json.JSONDecodeError as exc:
                    raise PlanParseError(f"invalid plan JSON: {exc.msg}") from exc
                if consumed != len(candidate):
                    raise PlanParseError("plan JSON has trailing content")
                objects.append(value)
                if len(objects) > 1:
                    raise PlanParseError("response contains more than one JSON object")
        index += 1
    if depth != 0:
        raise PlanParseError("plan JSON object is truncated")
    if len(objects) != 1:
        raise PlanParseError("response must contain exactly one JSON object")
    if not isinstance(objects[0], dict):
        raise PlanParseError("plan root must be a JSON object")
    return objects[0]
