#!/usr/bin/env python3
"""Benchmark APEX runner variants and export dependency-free reports."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent / "py"))
from report import generate_report  # noqa: E402
from runner_client import RunnerClient, RunnerError  # noqa: E402


def positive_list(value: str) -> list[int]:
    try:
        parsed = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("values must be positive integers")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure in-process GEMM variants with apex-runner.",
        epilog="Example: apex-bench.py --sizes 128,256,512 --variants naive,engine --json-out run.json",
    )
    parser.add_argument("--runner", help="Path to apex-runner; defaults to discovery")
    parser.add_argument("--timeout", type=float, default=120.0, help="Per-run timeout in seconds (default: 120)")
    parser.add_argument("--kernels", default="auto", help="Comma-separated kernel selections")
    parser.add_argument("--sizes", type=positive_list, default=[64, 128], help="Square dimensions (default: 64,128)")
    parser.add_argument("--threads", type=positive_list, default=[1], help="Worker counts (default: 1)")
    parser.add_argument("--profile", choices=("smoke", "ci", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--format", choices=("table", "json", "both"), default="table")
    parser.add_argument("--json-out", type=Path, help="Write structured results to this file")
    parser.add_argument("--variants", default="naive,engine", help="Comma-separated naive,engine,reference")
    parser.add_argument("--reps", type=int, default=5, help="Timed runs per case (default: 5)")
    parser.add_argument("--warmup", type=int, default=2, help="Warm-up runs per case (default: 2)")
    parser.add_argument("--launches", type=int, default=1, help="Independent process launches (default: 1)")
    parser.add_argument("--time-budget", type=float, default=0.0, help="Optional minimum measurement seconds")
    parser.add_argument("--cache-state", choices=("warm", "cold"), default="warm")
    parser.add_argument("--peak-gflops", type=float, help="Optional hardware peak for efficiency reporting")
    parser.add_argument("--baseline", type=Path, help="Prior benchmark JSON to compare")
    parser.add_argument("--threshold", type=float, default=0.0, help="Regression threshold percentage")
    parser.add_argument("--report", type=Path, help="Generate Markdown and SVG from an existing JSON log")
    parser.add_argument("--report-out", type=Path, help="Directory for generated report files")
    parser.add_argument("--environment", action="store_true", help="Print the runner environment record and exit")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.environment:
        try:
            client = RunnerClient(args.runner, args.timeout)
            payload = json.dumps(client.info, indent=2) + "\n"
            if args.json_out:
                args.json_out.parent.mkdir(parents=True, exist_ok=True)
                args.json_out.write_text(payload, encoding="utf-8")
            if not args.quiet:
                print(payload, end="")
            return 0
        except (RunnerError, OSError) as exc:
            print(f"apex-bench: {exc}", file=sys.stderr)
            return 3
    if args.report:
        try:
            markdown, throughput_svg, speedup_svg = generate_report(args.report, args.report_out)
            print(f"Wrote {markdown}\nWrote {throughput_svg}\nWrote {speedup_svg}")
            return 0
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"apex-bench: cannot generate report: {exc}", file=sys.stderr)
            return 2
    if args.reps <= 0 or args.warmup < 0 or args.launches <= 0 or args.threshold < 0 or args.time_budget < 0:
        parser.error("--reps and --launches must be positive; warm-up, time budget, and threshold nonnegative")
    if args.peak_gflops is not None and args.peak_gflops <= 0:
        parser.error("--peak-gflops must be positive")
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    kernels = [item.strip() for item in args.kernels.split(",") if item.strip()]
    if not variants or any(item not in {"naive", "engine", "reference"} for item in variants):
        parser.error("--variants accepts naive, engine, and reference")
    if not kernels or any(item not in {"auto", "scalar", "neon", "avx512"} for item in kernels):
        parser.error("--kernels accepts auto, scalar, neon, and avx512")
    if args.profile == "ci" and args.sizes == [64, 128]:
        args.sizes = [16, 64, 128]
    elif args.profile == "full" and args.sizes == [64, 128]:
        args.sizes = [16, 32, 64, 128, 256, 512]
    try:
        client = RunnerClient(args.runner, args.timeout)
        results = []
        if not args.quiet and args.format in ("table", "both"):
            print(f"{'Size':>5} {'Variant':<10} {'Kernel':<14} {'Threads':>8}  "
                  f"{'Latency':>14}  {'Throughput':>14}")
        for size in args.sizes:
            for threads in args.threads:
                for kernel in kernels:
                    for variant in variants:
                        durations: list[int] = []
                        last = {}
                        launches = 0
                        target_ns = int(args.time_budget * 1_000_000_000)
                        while launches < args.launches or (target_ns > 0 and sum(durations) < target_ns):
                            last = client.call("run", {
                                "m": size, "n": size, "k": size,
                                "threads": threads, "seed": args.seed,
                                "variant": variant, "kernel": kernel,
                                "warmup": args.warmup, "reps": args.reps,
                            })
                            if last.get("status") != "Ok":
                                raise RunnerError(f"run returned {last.get('status')} for {size}/{variant}/{kernel}")
                            durations.extend(int(value) for value in last.get("durations_ns", []))
                            launches += 1
                            if not last.get("durations_ns"):
                                raise RunnerError("runner returned no timed samples")
                        median_ns = float(statistics.median(durations)) if durations else 0.0
                        operations = 2.0 * size * size * size
                        gflops = operations / median_ns if median_ns > 0 else 0.0
                        record = {
                            "shape": {"m": size, "n": size, "k": size},
                            "variant": variant, "kernel_requested": kernel,
                            "kernel": last.get("kernel_used", kernel),
                            "threads": last.get("threads_used", 1),
                            "threads_requested": threads, "durations_ns": durations,
                            "median_ns": median_ns, "gflops": gflops,
                            "checksum": last.get("c_hash"),
                            "speedup": 1.0,
                        }
                        if args.peak_gflops:
                            record["peak_efficiency_percent"] = gflops / args.peak_gflops * 100.0
                        results.append(record)
                        if not args.quiet and args.format in ("table", "both"):
                            print(f"{size:>5} {variant:<10} {record['kernel']:<14} "
                                  f"{threads:>5}  {median_ns / 1e6:>10.4f} ms  "
                                  f"{gflops:>9.3f} GFLOPS")
        naive_times = {
            (item["shape"]["m"], item.get("threads_requested", item["threads"]),
             item.get("kernel_requested", "auto")): item["median_ns"]
            for item in results if item["variant"] == "naive"
        }
        for item in results:
            requested_threads = item.get("threads_requested", item["threads"])
            requested_kernel = item.get("kernel_requested", "auto")
            baseline_ns = naive_times.get(
                (item["shape"]["m"], requested_threads, requested_kernel)
            )
            if baseline_ns and item["median_ns"] > 0:
                item["speedup"] = baseline_ns / item["median_ns"]

        regression = False
        if args.baseline:
            try:
                baseline_doc = json.loads(args.baseline.read_text(encoding="utf-8"))
                baseline_results = baseline_doc.get("results", [])
            except (OSError, json.JSONDecodeError) as exc:
                raise RunnerError(f"Could not read baseline {args.baseline}: {exc}") from exc
            baseline_map = {
                (item.get("shape", {}).get("m"), item.get("variant"),
                 item.get("threads_requested", item.get("threads", 1))): item
                for item in baseline_results
            }
            for item in results:
                key = (item["shape"]["m"], item["variant"],
                       item.get("threads_requested", item["threads"]))
                previous = baseline_map.get(key)
                if previous and float(previous.get("median_ns", 0.0)) > 0.0:
                    previous_ns = float(previous["median_ns"])
                    item["baseline_median_ns"] = previous_ns
                    item["regression_percent"] = (item["median_ns"] / previous_ns - 1.0) * 100.0
                    regression |= item["regression_percent"] > args.threshold

        document = {
            "schema": {"name": "apex.bench.log", "version": 1},
            "environment": {"runner": client.info.get("runner"),
                            "architecture": client.info.get("architecture"),
                            "os": client.info.get("os"), "info": client.info},
            "methodology": {"warmup": args.warmup, "repetitions": args.reps,
                            "launches": args.launches, "cache_state": args.cache_state,
                            "seed": args.seed, "kernels": kernels},
            "results": results,
        }
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        if args.format in ("json", "both"):
            print(json.dumps(document, indent=2))
        if args.report_out:
            report_source = args.json_out
            if report_source is None:
                report_source = Path("apex-bench-results.json")
                report_source.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
            markdown, throughput_svg, speedup_svg = generate_report(report_source, args.report_out)
            print(f"Wrote {markdown}\nWrote {throughput_svg}\nWrote {speedup_svg}")
        return 5 if regression else 0
    except (RunnerError, OSError, ValueError) as exc:
        print(f"apex-bench: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
