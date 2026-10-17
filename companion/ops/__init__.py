"""Strict operation allow-list and normalization for model-produced plans."""

from __future__ import annotations

import os
from typing import Any

MAX_STEPS = 3
MAX_SIZE = 2048
MAX_SIZES = 8
MAX_REPETITIONS = 10
MAX_WARMUP = 5
MAX_TIME_BUDGET_SECONDS = 120
MAX_TOTAL_CUBIC_WORK = 50_000_000_000
MAX_THREADS = max(1, min(os.cpu_count() or 1, 256))

KERNELS = {"auto", "scalar", "neon", "avx512"}
VARIANTS = {"naive", "engine", "reference"}
PROFILES = {"smoke", "ci", "full"}
FOCI = {"speedup", "scaling", "cache", "accuracy", "overall"}


class PlanError(ValueError):
    """The proposed operation plan is outside the local allow-list."""


def _fields(value: dict[str, Any], required: set[str], allowed: set[str], label: str) -> None:
    missing = required - value.keys()
    extra = value.keys() - allowed
    if missing:
        raise PlanError(f"{label} is missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise PlanError(f"{label} has unsupported fields: {', '.join(sorted(extra))}")


def _integer(value: Any, label: str, low: int, high: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlanError(f"{label} must be an integer")
    if not low <= value <= high:
        raise PlanError(f"{label} must be between {low} and {high}")
    return value


def _enum(value: Any, label: str, allowed: set[str]) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise PlanError(f"{label} must be one of: {', '.join(sorted(allowed))}")
    return value


def _int_list(
    value: Any, label: str, low: int, high: int, max_items: int
) -> list[int]:
    if not isinstance(value, list) or not value or len(value) > max_items:
        raise PlanError(f"{label} must contain 1 to {max_items} values")
    normalized = [_integer(item, label, low, high) for item in value]
    if len(set(normalized)) != len(normalized):
        raise PlanError(f"{label} values must be unique")
    return normalized


def _string_list(value: Any, label: str, allowed: set[str], max_items: int) -> list[str]:
    if not isinstance(value, list) or not value or len(value) > max_items:
        raise PlanError(f"{label} must contain 1 to {max_items} values")
    normalized = [_enum(item, label, allowed) for item in value]
    if len(set(normalized)) != len(normalized):
        raise PlanError(f"{label} values must be unique")
    return normalized


def _common_run_fields(raw: dict[str, Any], op: str) -> dict[str, Any]:
    allowed = {
        "op", "sizes", "threads", "kernels", "profile", "seed",
        "repetitions", "warmup", "time_budget", "variants",
    }
    _fields(raw, {"op"}, allowed, op)
    sizes = _int_list(raw.get("sizes", [128]), "sizes", 1, MAX_SIZE, MAX_SIZES)
    threads = _int_list(raw.get("threads", [1]), "threads", 1, MAX_THREADS, 8)
    kernels = _string_list(raw.get("kernels", ["auto"]), "kernels", KERNELS, 4)
    profile = _enum(raw.get("profile", "smoke"), "profile", PROFILES)
    seed = _integer(raw.get("seed", 1), "seed", 0, 2**63 - 1)
    repetitions = _integer(raw.get("repetitions", 5), "repetitions", 1, MAX_REPETITIONS)
    warmup = _integer(raw.get("warmup", 1), "warmup", 0, MAX_WARMUP)
    time_budget = _integer(raw.get("time_budget", 0), "time_budget", 0, MAX_TIME_BUDGET_SECONDS)
    total_work = sum(size**3 for size in sizes) * max(1, repetitions)
    total_work *= len(threads) * len(kernels)
    if op == "benchmark":
        variants = _string_list(
            raw.get("variants", ["naive", "engine"]), "variants", VARIANTS, 3
        )
        total_work *= len(variants)
    else:
        variants = []
    if total_work > MAX_TOTAL_CUBIC_WORK:
        raise PlanError("requested sizes and repetitions exceed the local compute budget")
    normalized: dict[str, Any] = {
        "op": op, "sizes": sizes, "threads": threads, "kernels": kernels,
        "profile": profile, "seed": seed, "repetitions": repetitions,
        "warmup": warmup, "time_budget": time_budget,
    }
    if op == "benchmark":
        normalized["variants"] = variants
    elif "variants" in raw:
        raise PlanError("validate does not accept variants")
    return normalized


def validate_plan(value: Any) -> dict[str, list[dict[str, Any]]]:
    if not isinstance(value, dict):
        raise PlanError("plan root must be an object")
    _fields(value, {"operations"}, {"operations"}, "plan")
    raw_operations = value["operations"]
    if not isinstance(raw_operations, list) or not 1 <= len(raw_operations) <= MAX_STEPS:
        raise PlanError(f"operations must contain 1 to {MAX_STEPS} steps")

    operations: list[dict[str, Any]] = []
    for position, raw in enumerate(raw_operations, 1):
        label = f"operation {position}"
        if not isinstance(raw, dict):
            raise PlanError(f"{label} must be an object")
        op = _enum(raw.get("op"), f"{label}.op", {
            "benchmark", "validate", "environment", "explain_last",
            "clarify", "decline",
        })
        if op in {"benchmark", "validate"}:
            operations.append(_common_run_fields(raw, op))
        elif op == "environment":
            _fields(raw, {"op"}, {"op"}, label)
            operations.append({"op": op})
        elif op == "explain_last":
            _fields(raw, {"op"}, {"op", "focus"}, label)
            operations.append({"op": op, "focus": _enum(raw.get("focus", "overall"),
                                                            f"{label}.focus", FOCI)})
        else:
            field = "question" if op == "clarify" else "reason"
            _fields(raw, {"op", field}, {"op", field}, label)
            text = raw[field]
            if not isinstance(text, str) or not text.strip() or len(text) > 300:
                raise PlanError(f"{label}.{field} must be 1 to 300 characters")
            operations.append({"op": op, field: text.strip()})
    return {"operations": operations}
