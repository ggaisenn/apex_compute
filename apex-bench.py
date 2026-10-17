#!/usr/bin/env python3
"""Convenience entry point for the Layer 6 benchmark shell."""

from pathlib import Path
from runpy import run_path

main = run_path(str(Path(__file__).resolve().parent / "tools" / "apex-bench.py"))["main"]

if __name__ == "__main__":
    raise SystemExit(main())
