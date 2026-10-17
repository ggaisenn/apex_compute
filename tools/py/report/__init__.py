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
    width = max(900, 120 + len(points) * 48)
    height = 520
    left, top, chart_w, chart_h = 70, 30, width - 120, 300
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
        y = top + chart_h + 92
        labels.append(
            f'<text x="{x:.1f}" y="{y}" text-anchor="end" '
            f'transform="rotate(-55 {x:.1f} {y})" font-size="9">'
            f'{html.escape(label[:24])}</text>'
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
    info = env.get("info", {})
    features = info.get("features", {})
    cache = info.get("cache_bytes", {})
    methodology = data.get("methodology", {})
    # Older logs did not record the requested kernel. Their result order still
    # places each naive timing immediately before its matching engine run.
    naive_times: dict[tuple[Any, ...], float] = {}
    speedup_by_index: dict[int, float] = {}
    for index, result in enumerate(results):
        shape = result.get("shape", {})
        thread_count = result.get(
            "threads_requested", result.get("threads", 1)
        )
        shape_key = (shape.get("m"), shape.get("n"), shape.get("k"), thread_count)
        requested_kernel = result.get("kernel_requested")
        if result.get("variant") == "naive":
            naive_times[shape_key + (requested_kernel,)] = _number(
                result.get("median_ns")
            )
            speedup_by_index[index] = 1.0
        else:
            baseline = naive_times.get(shape_key + (requested_kernel,))
            if baseline is None:
                baseline = naive_times.get(shape_key + (None,))
            # Legacy logs with repeated naive rows use the latest baseline,
            # which corresponds to the current kernel's measurement batch.
            if baseline is None:
                baseline = naive_times.get(shape_key + ("__legacy__",))
            if baseline and _number(result.get("median_ns")) > 0:
                speedup_by_index[index] = baseline / _number(result["median_ns"])
            else:
                speedup_by_index[index] = _number(result.get("speedup"), 1.0)

    # Give untagged legacy measurements a batch-local baseline by walking in
    # recorded order. The previous map above is keyed by request for new logs;
    # this pass handles the old format without changing measured durations.
    latest_legacy_naive: dict[tuple[Any, ...], float] = {}
    for index, result in enumerate(results):
        if result.get("kernel_requested") is not None:
            continue
        shape = result.get("shape", {})
        thread_count = result.get(
            "threads_requested", result.get("threads", 1)
        )
        key = (shape.get("m"), shape.get("n"), shape.get("k"), thread_count)
        if result.get("variant") == "naive":
            latest_legacy_naive[key] = _number(result.get("median_ns"))
            speedup_by_index[index] = 1.0
        else:
            baseline = latest_legacy_naive.get(key)
            if baseline and _number(result.get("median_ns")) > 0:
                speedup_by_index[index] = baseline / _number(result["median_ns"])

    lines = [
        f"# APEX Compute benchmark report: {source_path.name}",
        "",
        "## Environment", "",
        f"- Runner: `{env.get('runner', 'unknown')}`",
        f"- Architecture: `{env.get('architecture', 'unknown')}`",
        f"- Operating system: `{env.get('os', 'unknown')}`",
        f"- SIMD features: NEON `{features.get('neon', 'unknown')}`, AVX-512 `{features.get('avx512f', 'unknown')}`",
        f"- Cache sizes (L1d / L2 / L3): `{cache.get('l1d', 'unknown')}` / `{cache.get('l2', 'unknown')}` / `{cache.get('l3', 'unknown')}` bytes",
        f"- Runs: `{methodology.get('launches', 'unknown')}` launch(es), `{methodology.get('warmup', 'unknown')}` warm-up(s), `{methodology.get('repetitions', 'unknown')}` repetitions per launch; cache state `{methodology.get('cache_state', 'unknown')}`",
        "",
        "## Results", "",
        "| M × N × K | Variant | Kernel | Threads requested / used | Latency (ms) | GFLOPS | Speedup |",
        "|---:|:---|:---|---:|---:|---:|---:|",
    ]
    points: list[tuple[str, float]] = []
    speedups: list[tuple[str, float]] = []
    for index, result in enumerate(results):
        shape = result.get("shape", {})
        variant = str(result.get("variant", "unknown"))
        kernel = str(result.get("kernel", variant))
        m, n, k = shape.get("m", 0), shape.get("n", 0), shape.get("k", 0)
        latency = _number(result.get("median_ns"))
        gflops = _number(result.get("gflops"))
        threads_requested = int(_number(
            result.get("threads_requested", result.get("threads", 1)), 1.0
        ))
        threads_used = int(_number(result.get("threads", 1), 1.0))
        lines.append(
            f"| {m} × {n} × {k} | {variant} | {kernel} "
            f"| {threads_requested} / {threads_used} "
            f"| {latency / 1_000_000:.4f} | {gflops:.3f} | {speedup_by_index[index]:.3f}× |"
        )
        chart_label = f"N{m} {variant}/{kernel} t{threads_requested}/{threads_used}"
        points.append((chart_label, gflops))
        speedups.append((chart_label, speedup_by_index[index]))
    lines.extend([
        "",
        "Speedup compares each engine run with its corresponding naive run for the same shape and requested kernel. Legacy logs without requested-kernel metadata use the most recent preceding naive run for that shape.",
        "",
        "Generated from the recorded runner JSON output; timings are unchanged.",
        "",
    ])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    svg_path.write_text(
        _bar_chart("APEX Compute throughput (GFLOPS)", "GFLOPS", points),
        encoding="utf-8",
    )
    speedup_path.write_text(
        _bar_chart("APEX Compute speedup", "Speedup", speedups), encoding="utf-8"
    )
    return markdown_path, svg_path, speedup_path
