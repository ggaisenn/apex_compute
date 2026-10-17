"""Dependency-free Markdown and SVG reporting for APEX benchmark logs."""

from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any


def _number(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def _bar_chart(title: str, axis_label: str, points: list[tuple[str, float]]) -> str:
    width, height = 900, 420
    left, top, chart_w, chart_h = 70, 30, 790, 300
    maximum = max((value for _, value in points), default=1.0) or 1.0
    bars = []
    slot = chart_w / max(1, len(points))
    for index, (label, value) in enumerate(points):
        bar_height = max(0.0, value) / maximum * chart_h
        x = left + index * slot + slot * 0.12
        y = top + chart_h - bar_height
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{slot * 0.76:.1f}" '
            f'height="{bar_height:.1f}" fill="#3568a8"><title>{html.escape(label)}: '
            f'{value:.3f} {html.escape(axis_label)}</title></rect>'
        )
    labels = []
    for index, (label, _) in enumerate(points):
        x = left + index * slot + slot / 2
        labels.append(
            f'<text x="{x:.1f}" y="{top + chart_h + 22}" text-anchor="middle" '
            f'font-size="10">{html.escape(label[:20])}</text>'
        )
    return "\n".join([
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="450" y="20" text-anchor="middle" font-family="sans-serif" '
        f'font-size="16">{html.escape(title)}</text>',
        f'<line x1="{left}" y1="{top + chart_h}" x2="{left + chart_w}" '
        f'y2="{top + chart_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + chart_h}" stroke="#333"/>',
        *bars, *labels,
        f'<text x="18" y="180" transform="rotate(-90 18 180)" '
        f'font-family="sans-serif" font-size="12">{html.escape(axis_label)}</text>',
        '</svg>',
    ])


def generate_report(
    source: str | Path, output_dir: str | Path | None = None
) -> tuple[Path, Path, Path]:
    source_path = Path(source)
    data = json.loads(source_path.read_text(encoding="utf-8"))
    results = data.get("results")
    if not isinstance(results, list):
        raise ValueError("benchmark log must contain a results array")
    destination = Path(output_dir) if output_dir else source_path.parent
    destination.mkdir(parents=True, exist_ok=True)
    markdown_path = destination / f"{source_path.stem}-report.md"
    svg_path = destination / f"{source_path.stem}-throughput.svg"
    speedup_path = destination / f"{source_path.stem}-speedup.svg"

    env = data.get("environment", {})
    lines = [
        f"# APEX Compute benchmark report: {source_path.name}",
        "",
        "## Environment", "",
        f"- Runner: `{env.get('runner', 'unknown')}`",
        f"- Architecture: `{env.get('architecture', 'unknown')}`",
        f"- Operating system: `{env.get('os', 'unknown')}`",
        "",
        "## Results", "",
        "| M × N × K | Variant | Threads | Latency (ms) | GFLOPS | Speedup |",
        "|---:|:---|---:|---:|---:|---:|",
    ]
    points: list[tuple[str, float]] = []
    speedups: list[tuple[str, float]] = []
    for result in results:
        shape = result.get("shape", {})
        variant = str(result.get("variant", "unknown"))
        m, n, k = shape.get("m", 0), shape.get("n", 0), shape.get("k", 0)
        latency = _number(result.get("median_ns"))
        gflops = _number(result.get("gflops"))
        lines.append(
            f"| {m} × {n} × {k} | {variant} | {result.get('threads', 1)} "
            f"| {latency / 1_000_000:.4f} | {gflops:.3f} | {_number(result.get('speedup'), 1.0):.3f}× |"
        )
        points.append((f"{m}x{n}x{k} {variant}", gflops))
        speedups.append((f"{m}x{n}x{k} {variant}", _number(result.get("speedup"), 1.0)))
    lines.extend(["", "Generated from the recorded runner JSON output.", ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    svg_path.write_text(
        _bar_chart("APEX Compute throughput (GFLOPS)", "GFLOPS", points),
        encoding="utf-8",
    )
    speedup_path.write_text(
        _bar_chart("APEX Compute speedup", "Speedup", speedups), encoding="utf-8"
    )
    return markdown_path, svg_path, speedup_path
