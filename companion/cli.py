"""Command-line entry point for the optional APEX companion."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any, Sequence

from .build import build_command
from .check import check_explanation
from .executor import ExecutionError, run_shell
from .facts import extract_facts
from .gemini import GeminiClient, GeminiError
from .intent import fixture_for_request, interpret_offline, load_fixtures
from .ops import PlanError, validate_plan
from .plan_parser import PlanParseError, parse_plan
from .prompts import explanation_prompt, interpretation_prompt
from .redact import redact

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="apex-companion",
        description="Turn a bounded natural-language request into approved APEX shell operations.",
        epilog="Examples: --offline 'benchmark a small square multiply' --yes; "
        "'validate edge cases' (requires GEMINI_API_KEY)",
    )
    parser.add_argument("request", nargs="*", help="Benchmark, validation, or environment request")
    parser.add_argument("--offline", action="store_true", help="Use recorded interpretations; no API key or network")
    parser.add_argument("--replay-results", action="store_true", help="Use fixture result logs without local binaries")
    parser.add_argument("--yes", action="store_true", help="Run the displayed approved plan without prompting")
    parser.add_argument("--timeout", type=float, default=180.0, help="Maximum child/model request time in seconds")
    parser.add_argument("--model", help="Gemini model name (defaults to APEX_GEMINI_MODEL)")
    parser.add_argument("--endpoint", help="Gemini API base URL (defaults to APEX_GEMINI_ENDPOINT)")
    parser.add_argument("--fixture-file", type=Path, help="Override the offline fixture JSON file")
    return parser


def _display_command(command: list[str]) -> str:
    return subprocess_quote(command)


def subprocess_quote(command: list[str]) -> str:
    if os.name == "nt":
        import subprocess
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def _confirm() -> bool:
    try:
        return input("Run the approved local operations? [y/N] ").strip().casefold() in {"y", "yes"}
    except EOFError:
        return False


def _fixture_for_plan(
    request: str, fixture_file: Path | None
) -> dict[str, Any] | None:
    try:
        data = load_fixtures(fixture_file) if fixture_file else load_fixtures()
        return fixture_for_request(request, data)
    except (OSError, ValueError, TypeError):
        return None


def _read_result(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionError(f"could not read shell JSON output: {redact(exc)}") from None
    if not isinstance(value, dict):
        raise ExecutionError("shell JSON output must be an object")
    return value


def _failure_exit(returncode: int) -> int:
    return 1 if returncode == 1 else 3 if returncode == 3 else 4


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    request = " ".join(args.request).strip()
    if not request:
        if sys.stdin.isatty():
            request = input("What would you like to run? ").strip()
        if not request:
            parser.error("provide a request, for example: 'benchmark a 128 by 128 multiply'")
    if args.timeout <= 0 or args.timeout > 600:
        parser.error("--timeout must be between 0 and 600 seconds")
    if args.replay_results and not args.offline:
        parser.error("--replay-results requires --offline")

    fixture = _fixture_for_plan(request, args.fixture_file)
    try:
        if args.offline:
            print("Mode: offline recorded interpretation")
            raw_plan = fixture["plan"] if fixture else interpret_offline(request)
        else:
            client = GeminiClient(model=args.model, endpoint=args.endpoint, timeout=args.timeout)
            prompt = interpretation_prompt(request)
            raw_plan = parse_plan(client.generate(prompt))
        plan = validate_plan(raw_plan)
    except GeminiError as exc:
        print(redact(str(exc), os.environ.get("GEMINI_API_KEY")), file=sys.stderr)
        return 3
    except (PlanParseError, PlanError, OSError, ValueError) as exc:
        print(f"apex-companion: {redact(exc)}", file=sys.stderr)
        return 2

    executable_items: list[tuple[int, dict[str, Any], list[str], Path]] = []
    temporary = tempfile.TemporaryDirectory(prefix="apex-companion-")
    output_root = Path(temporary.name)
    try:
        for index, operation in enumerate(plan["operations"]):
            command = build_command(operation, REPO_ROOT, output_root / f"result-{index}.json")
            if command:
                executable_items.append((index, operation, command, output_root / f"result-{index}.json"))

        print("Validated plan:")
        for index, operation in enumerate(plan["operations"], 1):
            if operation["op"] in {"clarify", "decline"}:
                detail = operation.get("question", operation.get("reason", ""))
                print(f"  {index}. {operation['op']}: {redact(detail)}")
            elif operation["op"] == "explain_last":
                print(f"  {index}. explain last result ({operation['focus']})")
            else:
                command_item = next(item for item in executable_items if item[0] == index - 1)
                print(f"  {index}. {operation['op']}: {_display_command(command_item[2])}")

        if executable_items and not args.yes and not _confirm():
            print("Cancelled; no operation was started.")
            return 0

        last_document: dict[str, Any] | None = None
        last_facts: dict[str, Any] | None = None
        for index, operation in enumerate(plan["operations"]):
            op = operation["op"]
            if op == "clarify":
                print(operation["question"])
                continue
            if op == "decline":
                print(operation["reason"])
                continue
            if op == "explain_last":
                if last_facts is None:
                    print("No result is available to explain yet.")
                    continue
                if args.offline:
                    narrative = (fixture or {}).get("narrative", "The facts table is the verified result summary.")
                else:
                    try:
                        narrative = GeminiClient(
                            model=args.model, endpoint=args.endpoint, timeout=args.timeout
                        ).generate(explanation_prompt(last_facts, operation["focus"]))
                    except GeminiError as exc:
                        print(f"Explanation unavailable: {redact(exc, os.environ.get('GEMINI_API_KEY'))}")
                        continue
                checked = check_explanation(narrative, last_facts)
                print("Interpretation:")
                print(checked["text"])
                if checked["unverified"]:
                    print("Unverified figures: " + ", ".join(checked["unverified"]))
                continue

            item = next(item for item in executable_items if item[0] == index)
            _, _, command, json_path = item
            if args.replay_results:
                replay = (fixture or {}).get("replay_results")
                if not isinstance(replay, dict):
                    print("No recorded result matches this offline request; omit --replay-results to run locally.",
                          file=sys.stderr)
                    return 3
                last_document = replay
            else:
                result = run_shell(command, REPO_ROOT, args.timeout)
                if result.returncode != 0:
                    detail = result.stderr.strip() or result.stdout.strip()
                    print(redact(detail) or f"shell exited with code {result.returncode}", file=sys.stderr)
                    return _failure_exit(result.returncode)
                last_document = _read_result(json_path)
            try:
                last_facts = extract_facts(last_document)
            except ValueError as exc:
                print(f"apex-companion: {redact(exc)}", file=sys.stderr)
                return 4
            print("Verified facts:")
            print(last_facts["markdown"])
            if last_document.get("summary", {}).get("failures", 0):
                return 1
        return 0
    except (ExecutionError, OSError, StopIteration) as exc:
        print(f"apex-companion: {redact(exc, os.environ.get('GEMINI_API_KEY'))}", file=sys.stderr)
        return 3
    finally:
        temporary.cleanup()
