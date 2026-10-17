"""Standard-library client for the versioned apex-runner process contract."""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

CONTRACT_MAJOR = 1
CONTRACT_MINOR = 0


class RunnerError(RuntimeError):
    """A discovery, contract, process, or JSON-output failure."""


def _executable_name() -> str:
    return "apex-runner.exe" if os.name == "nt" else "apex-runner"


def discover_runner(explicit: str | None = None) -> Path:
    name = _executable_name()
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("APEX_RUNNER"):
        candidates.append(Path(os.environ["APEX_RUNNER"]).expanduser())
    root = Path(__file__).resolve().parents[3]
    locations = [root / "runner-location.json", root / "build" / "runner-location.json"]
    locations.extend(root / "build" / preset / "runner-location.json"
                     for preset in ("debug", "release", "portable", "asan"))
    for location in locations:
        try:
            metadata = json.loads(location.read_text(encoding="utf-8"))
            executable = metadata.get("runner") or metadata.get("path")
            if executable:
                candidates.append(Path(executable))
        except (OSError, ValueError, TypeError):
            pass
    for preset in ("debug", "release", "portable", "asan"):
        base = root / "build" / preset
        candidates.extend((base / name, base / "Release" / name, base / "Debug" / name))
    for candidate in candidates:
        if candidate.is_file() and (os.name == "nt" or os.access(candidate, os.X_OK)):
            return candidate.resolve()
    searched = ", ".join(str(path) for path in candidates) or "no candidate paths"
    raise RunnerError(
        f"Could not find {name}. Searched: {searched}. Build with CMake or pass --runner PATH."
    )


class RunnerClient:
    def __init__(self, runner: str | None = None, timeout: float = 60.0) -> None:
        self.path = discover_runner(runner)
        self.timeout = timeout
        self.info = self.invoke(["--mode", "info"])
        contract = self.info.get("contract", {})
        if contract.get("major") != CONTRACT_MAJOR or contract.get("minor", -1) < CONTRACT_MINOR:
            raise RunnerError(f"Incompatible runner contract: {contract!r}")
        runner_arch = str(self.info.get("architecture", "unknown")).lower()
        host_arch = platform.machine().lower()
        aliases = {"amd64": "x86_64", "x64": "x86_64", "aarch64": "arm64"}
        if aliases.get(runner_arch, runner_arch) != aliases.get(host_arch, host_arch):
            raise RunnerError(
                f"Runner architecture {runner_arch!r} does not match host {host_arch!r}."
            )

    def invoke(self, arguments: Sequence[str]) -> dict[str, Any]:
        command = [str(self.path), *arguments]
        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.timeout,
                shell=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RunnerError(f"Runner timed out after {self.timeout:g}s: {command!r}") from exc
        except OSError as exc:
            raise RunnerError(f"Could not start runner {self.path}: {exc}") from exc
        if completed.returncode != 0:
            detail = completed.stderr.strip() or completed.stdout.strip()
            if completed.returncode < 0:
                raise RunnerError(f"Runner crashed (signal {-completed.returncode}): {command!r}")
            raise RunnerError(
                f"Runner exited with code {completed.returncode}: {detail or command!r}"
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RunnerError(
                f"Runner violated its JSON contract; output was {completed.stdout!r}"
            ) from exc
        if not isinstance(result, dict):
            raise RunnerError("Runner output must be one JSON object.")
        return result

    def call(self, mode: str, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
        args = ["--mode", mode]
        for key, value in (options or {}).items():
            option = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(option)
            elif value is not None:
                args.extend((option, str(value)))
        return self.invoke(args)
