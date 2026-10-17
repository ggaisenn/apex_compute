"""Offline fixtures and conservative local request interpretation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "demos.json"
_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}


def normalize_request(text: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", text.casefold()))


def load_fixtures(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fixture_for_request(text: str, fixtures: dict[str, Any] | None = None) -> dict[str, Any] | None:
    data = fixtures if fixtures is not None else load_fixtures()
    normalized = normalize_request(text)
    for fixture in data.get("demos", []):
        phrase = normalize_request(str(fixture.get("match", "")))
        if phrase and phrase in normalized:
            return fixture
    return None


def interpret_offline(text: str) -> dict[str, Any]:
    fixture = fixture_for_request(text)
    if fixture is not None:
        return fixture["plan"]
    normalized = text.casefold()
    if any(word in normalized for word in ("benchmark", "measure", "performance", "gflops")):
        op = "benchmark"
    elif any(word in normalized for word in ("validate", "correctness", "check the result")):
        op = "validate"
    elif any(word in normalized for word in ("environment", "cpu features", "machine info")):
        return {"operations": [{"op": "environment"}]}
    elif any(word in normalized for word in ("explain", "interpret")):
        return {"operations": [{"op": "explain_last", "focus": "overall"}]}
    else:
        return {"operations": [{"op": "clarify",
                                  "question": "Would you like a benchmark, validation, or environment summary?"}]}

    size = 128
    shape = re.search(r"\b(\d{1,4})\s*(?:x|by)\s*(\d{1,4})\b", normalized)
    if shape:
        size = max(int(shape.group(1)), int(shape.group(2)))
    else:
        numbers = re.findall(r"\b(\d{1,4})\b", normalized)
        if numbers:
            size = int(numbers[0])
    size = min(max(size, 1), 2048)

    threads = 1
    thread_match = re.search(r"\b(\d{1,3}|one|two|three|four|five|six|seven|eight|nine|ten)\s+threads?\b", normalized)
    if thread_match:
        token = thread_match.group(1)
        threads = int(token) if token.isdigit() else _NUMBER_WORDS[token]
    fields: dict[str, Any] = {
        "op": op, "sizes": [size], "threads": [threads],
        "kernels": ["auto"], "profile": "smoke", "seed": 1,
    }
    if op == "benchmark":
        fields.update({"variants": ["naive", "engine"], "repetitions": 5, "warmup": 1})
        operations = [fields]
        if "explain" in normalized or "explain the result" in normalized:
            operations.append({"op": "explain_last", "focus": "overall"})
    else:
        fields["repetitions"] = 1
        operations = [fields]
    return {"operations": operations}
