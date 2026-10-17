"""Small local HTTP server that emulates the Gemini generateContent response."""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlparse


DEFAULT_PLAN = {
    "operations": [
        {
            "op": "benchmark",
            "sizes": [8],
            "threads": [1],
            "kernels": ["scalar"],
            "variants": ["naive", "engine"],
            "profile": "smoke",
            "seed": 1,
            "repetitions": 1,
            "warmup": 0,
        }
    ]
}


class MockGeminiHandler(BaseHTTPRequestHandler):
    """Return deterministic model text without logging request headers or prompts."""

    server_version = "APEXMockGemini/1"
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API name
        if urlparse(self.path).path == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": {"message": "unknown endpoint"}})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API name
        if not urlparse(self.path).path.endswith(":generateContent"):
            self._send(404, {"error": {"message": "unknown endpoint"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 1_000_000:
                self._send(413, {"error": {"message": "request body size rejected"}})
                return
            request = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
            self._send(400, {"error": {"message": "invalid JSON request"}})
            return

        prompt = _prompt_text(request)
        response_text = self.server.response_text  # type: ignore[attr-defined]
        if response_text is None:
            response_text = (
                "The verified output shows the requested operation completed. "
                "Use the facts table above for measured values."
                if "BEGIN_FACTS_JSON" in prompt
                else json.dumps(DEFAULT_PLAN, separators=(",", ":"))
            )
        self._send(200, {
            "candidates": [{
                "content": {"role": "model", "parts": [{"text": response_text}]},
                "finishReason": "STOP",
            }]
        })

    def log_message(self, _format: str, *_args: Any) -> None:
        # API keys and user prompts must not appear in local server logs.
        return

    def _send(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(encoded)


def _prompt_text(request: object) -> str:
    if not isinstance(request, dict):
        return ""
    contents = request.get("contents")
    if not isinstance(contents, list):
        return ""
    text: list[str] = []
    for content in contents:
        if not isinstance(content, dict):
            continue
        parts = content.get("parts")
        if not isinstance(parts, list):
            continue
        text.extend(
            part["text"] for part in parts
            if isinstance(part, dict) and isinstance(part.get("text"), str)
        )
    return "\n".join(text)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1", help="bind address (default: loopback only)")
    parser.add_argument("--port", type=int, default=8765, help="HTTP port; use 0 for an ephemeral port")
    parser.add_argument("--response", help="fixed response text for development scenarios")
    args = parser.parse_args()
    if args.port < 0 or args.port > 65535:
        parser.error("--port must be between 0 and 65535")
    server = ThreadingHTTPServer((args.host, args.port), MockGeminiHandler)
    server.daemon_threads = True
    server.response_text = args.response
    print(f"Mock Gemini ready at http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
