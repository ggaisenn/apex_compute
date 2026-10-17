"""Build shell argument lists from normalized allow-listed operations only."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def build_command(
    operation: dict[str, Any], repo_root: Path, json_path: Path
) -> list[str] | None:
    op = operation.get("op")
    if op == "benchmark":
        script = repo_root / "tools" / "apex-bench.py"
        arguments = [
            "--sizes", ",".join(str(value) for value in operation["sizes"]),
            "--threads", ",".join(str(value) for value in operation["threads"]),
            "--kernels", ",".join(operation["kernels"]),
            "--variants", ",".join(operation["variants"]),
            "--profile", operation["profile"], "--seed", str(operation["seed"]),
            "--reps", str(operation["repetitions"]), "--warmup", str(operation["warmup"]),
            "--time-budget", str(operation["time_budget"]),
        ]
    elif op == "validate":
        script = repo_root / "tools" / "apex-validate.py"
        arguments = [
            "--sizes", ",".join(str(value) for value in operation["sizes"]),
            "--threads", ",".join(str(value) for value in operation["threads"]),
            "--kernels", ",".join(operation["kernels"]),
            "--profile", operation["profile"], "--seed", str(operation["seed"]),
        ]
    elif op == "environment":
        script = repo_root / "tools" / "apex-bench.py"
        arguments = ["--environment"]
    else:
        return None
    return [sys.executable, str(script), *arguments,
            "--format", "json", "--quiet", "--json-out", str(json_path)]
