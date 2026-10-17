"""Load the companion's small, non-executable project configuration file."""

from __future__ import annotations

import os
from pathlib import Path

_ALLOWED_KEYS = frozenset({
    "GEMINI_API_KEY",
    "APEX_GEMINI_MODEL",
    "APEX_GEMINI_ENDPOINT",
})


def load_dotenv(path: Path) -> None:
    """Load allow-listed KEY=value entries without overriding the process env.

    The file is parsed as data, never evaluated as shell code. Supports blank
    lines, comments, an optional ``export`` prefix, and matching quotes.
    """
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return

    for raw_line in lines:
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or name not in _ALLOWED_KEYS:
            continue

        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        if value:
            os.environ.setdefault(name, value)
