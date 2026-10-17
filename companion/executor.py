"""Bounded child-process execution for the two approved Python shells."""

from __future__ import annotations

import os
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .redact import redact


@dataclass
class ExecutionResult:
    returncode: int
    stdout: str
    stderr: str
    command: list[str]


class ExecutionError(RuntimeError):
    pass


def run_shell(
    command: Sequence[str], cwd: Path, timeout: float, environment: dict[str, str] | None = None
) -> ExecutionResult:
    if not command or timeout <= 0:
        raise ExecutionError("invalid child-process request")
    child_env = dict(os.environ if environment is None else environment)
    for secret_name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        child_env.pop(secret_name, None)
    start_new_session = os.name != "nt"
    creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    process = subprocess.Popen(
        list(command), cwd=str(cwd), env=child_env, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        encoding="utf-8", errors="replace", shell=False,
        start_new_session=start_new_session, creationflags=creation_flags,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        stdout, stderr = process.communicate()
        raise ExecutionError(
            f"child process timed out after {timeout:.1f}s: "
            f"{redact(' '.join(command))}; {redact(stderr)}"
        ) from None
    return ExecutionResult(
        process.returncode, redact(stdout), redact(stderr), list(command)
    )
