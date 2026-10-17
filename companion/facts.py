"""Deterministic facts extraction and display formatting for shell JSON logs."""

from __future__ import annotations

import json
import html
from typing import Any


def _fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def _cell(value: Any) -> str:
    return html.escape(str(value).replace("|", "&#124;").replace("\n", " ").replace("\r", " "))


def _numeric(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _numbers(value: Any, output: set[float]) -> None:
    if isinstance(value, bool) or value is None:
        return
    if isinstance(value, (int, float)):
        output.add(float(value))
    elif isinstance(value, dict):
        for item in value.values():
            _numbers(item, output)
    elif isinstance(value, list):
        for item in value:
            _numbers(item, output)


def extract_facts(document: Any) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ValueError("result log must contain a JSON object")
    schema = document.get("schema", {})
    name = schema.get("name") if isinstance(schema, dict) else None
    lines: list[str] = []
    numbers: set[float] = set()
    if name in {"apex.bench.log", "apex.runner.result"}:
        env = document.get("environment", {})
        if isinstance(env, dict):
            lines.extend([
                "Environment: " + ", ".join(
                    f"{label}={_cell(env.get(key, 'unknown'))}"
                    for label, key in (("architecture", "architecture"), ("OS", "os"),
                                       ("runner", "runner"))
                ),
                "",
            ])
        lines.extend([
            "| Shape | Variant | Threads | Latency (ms) | GFLOPS | Speedup |",
            "|---|---|---:|---:|---:|---:|",
        ])
        results = document.get("results")
        if results is None:
            results = [document]
        if not isinstance(results, list):
            raise ValueError("benchmark results must be an array")
        for record in results:
            if not isinstance(record, dict):
                continue
            shape = record.get("shape", {})
            if not isinstance(shape, dict):
                shape = {}
            shape_text = "×".join(_cell(shape.get(axis, "?")) for axis in ("m", "n", "k"))
            latency_ns = record.get("median_ns", 0.0)
            try:
                latency_ms = _numeric(latency_ns) / 1_000_000.0
                gflops = _numeric(record.get("gflops", 0.0))
                speedup = _numeric(record.get("speedup", 1.0), 1.0)
            except (TypeError, ValueError):
                latency_ms, gflops, speedup = 0.0, 0.0, 1.0
            lines.append(
                f"| {shape_text} | {_cell(record.get('variant', 'unknown'))} "
                f"| {_cell(record.get('threads', 1))} | {latency_ms:.4f} "
                f"| {gflops:.4f} | {speedup:.3f}× |"
            )
            numbers.update(_numeric(shape.get(axis, 0)) for axis in ("m", "n", "k"))
            numbers.update((_numeric(record.get("threads", 1)), latency_ms, gflops, speedup))
    elif name == "apex.validate.result":
        summary = document.get("summary", {})
        checks = document.get("checks", [])
        failures = [item for item in checks if isinstance(item, dict) and not item.get("passed", False)]
        lines.extend([
            f"Validation: {summary.get('checks', len(checks))} checks, "
            f"{summary.get('failures', len(failures))} failures.",
            "",
            "| Shape | Kernel | Status | Max error | Tolerance | Result |",
            "|---|---|---|---:|---:|---|",
        ])
        for record in checks:
            if not isinstance(record, dict):
                continue
            shape = record.get("shape", {})
            shape_text = "×".join(_cell(shape.get(axis, "?")) for axis in ("m", "n", "k"))
            lines.append(
                f"| {shape_text} | {_cell(record.get('kernel', '?'))} | {_cell(record.get('status', '?'))} "
                f"| {_fmt(_numeric(record.get('checksum_error', 0.0)))} "
                f"| {_fmt(_numeric(record.get('tolerance', 0.0)))} "
                f"| {'PASS' if record.get('passed') else 'FAIL'} |"
            )
            shape = record.get("shape", {})
            numbers.update(_numeric(shape.get(axis, 0)) for axis in ("m", "n", "k"))
            numbers.update((_numeric(record.get("checksum_error", 0.0)),
                            _numeric(record.get("tolerance", 0.0))))
        numbers.update((_numeric(summary.get("checks", len(checks))),
                        _numeric(summary.get("failures", len(failures)))))
    elif name == "apex.runner.info":
        lines.append("Environment information:")
        for key in ("runner", "architecture", "os", "kernel", "features", "cache_bytes"):
            if key in document:
                lines.append(f"- {key}: {_fmt(json.dumps(document[key], sort_keys=True))}")
        _numbers(document, numbers)
    else:
        raise ValueError(f"unsupported result schema: {name!r}")
    return {"schema": name, "markdown": "\n".join(lines),
            "numbers": sorted(numbers), "document": document}
