#!/usr/bin/env python3
"""Run deterministic correctness smoke checks through apex-runner."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent / "py"))
from runner_client import RunnerClient, RunnerError  # noqa: E402


def integer_list(value: str) -> list[int]:
    try:
        parsed = [int(item) for item in value.split(",")]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected comma-separated positive integers") from exc
    if not parsed or any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all sizes and thread counts must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Differentially check runner kernels against the scalar reference.",
        epilog="Example: apex-validate.py --sizes 31,32,33 --kernels scalar,auto",
    )
    parser.add_argument("--runner", help="Path to apex-runner; defaults to discovery")
    parser.add_argument("--timeout", type=float, default=60.0, help="Runner timeout in seconds (default: 60)")
    parser.add_argument("--kernels", default="auto,scalar", help="Comma-separated kernels (default: auto,scalar)")
    parser.add_argument("--sizes", type=integer_list, default=[16, 32], help="Square dimensions (default: 16,32)")
    parser.add_argument("--threads", type=integer_list, default=[1], help="Thread counts (default: 1)")
    parser.add_argument("--profile", choices=("smoke", "ci", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--format", choices=("table", "json", "both"), default="table")
    parser.add_argument("--json-out", type=Path, help="Write structured results to this file")
    parser.add_argument("--fail-fast", action="store_true", help="Stop after the first mismatch")
    parser.add_argument("--list-cases", action="store_true", help="List built-in validation scenarios")
    parser.add_argument("--case", help="Run a named scenario (currently: gemm-smoke)")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_cases:
        print("gemm-smoke\tseeded engine/reference comparison over requested sizes")
        return 0
    if args.case and args.case != "gemm-smoke":
        parser.error(f"unknown --case {args.case!r}; supported case: gemm-smoke")

    if args.profile == "ci" and args.sizes == [16, 32]:
        args.sizes = [1, 15, 16, 17, 31, 32, 33]
    elif args.profile == "full" and args.sizes == [16, 32]:
        args.sizes = [1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129]
    kernels = [name.strip() for name in args.kernels.split(",") if name.strip()]
    if not kernels or any(name not in {"auto", "scalar", "neon", "avx512"} for name in kernels):
        parser.error("--kernels accepts auto, scalar, neon, and avx512")

    try:
        client = RunnerClient(args.runner, args.timeout)
        selftest = client.call("selftest")
        if selftest.get("status") != "Ok":
            raise RunnerError(f"runner selftest failed: {selftest!r}")
        results = []
        failures = 0
        for size in args.sizes:
            for threads in args.threads:
                for kernel in kernels:
                    common = {"m": size, "n": size, "k": size,
                              "threads": threads, "seed": args.seed}
                    reference = client.call("run", {**common, "variant": "reference", "kernel": "scalar"})
                    candidate = client.call("run", {**common, "variant": "engine", "kernel": kernel})
                    status = candidate.get("status", "UnknownStatus")
                    ref_status = reference.get("status", "UnknownStatus")
                    error = float(candidate.get("max_abs_error", math.nan))
                    scale = float(reference.get("max_element_scale", 0.0))
                    tolerance = max(1e-45, (2 * size + 3) * 2 ** -24 * scale * 2.0)
                    skipped = status in {"UnsupportedKernel", "KernelUnavailable"}
                    passed = skipped or (status == "Ok" and ref_status == "Ok" and
                                           math.isfinite(error) and error <= tolerance)
                    record = {"shape": {"m": size, "n": size, "k": size},
                              "kernel": kernel, "threads": candidate.get("threads_used", 1),
                              "threads_requested": threads, "status": status,
                              "reference_status": ref_status, "checksum_error": error,
                              "tolerance": tolerance, "passed": passed, "skipped": skipped}
                    results.append(record)
                    failures += int(not passed)
                    if not args.quiet and args.format in ("table", "both"):
                        label = "SKIP" if skipped else "PASS" if passed else "FAIL"
                        print(f"{label} {size}x{size}x{size} "
                              f"kernel={kernel} threads={threads} error={error:.3e} tol={tolerance:.3e}")
                    if args.fail_fast and not passed:
                        break
                if args.fail_fast and failures:
                    break
            if args.fail_fast and failures:
                break
        document = {"schema": {"name": "apex.validate.result", "version": 1},
                    "environment": client.info, "checks": results,
                    "summary": {"checks": len(results), "failures": failures}}
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        if args.format in ("json", "both"):
            print(json.dumps(document, indent=2))
        return 0 if failures == 0 else 1
    except RunnerError as exc:
        print(f"apex-validate: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
