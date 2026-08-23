#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proxy OpenAI-compatible HTTP traffic and log token usage.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--upstream-base-url", required=True)
    parser.add_argument("--log-path", required=True)
    return parser.parse_args()


def _normalize_prefix(path: str) -> str:
    stripped = (path or "").strip()
    if not stripped or stripped == "/":
        return ""
    return "/" + stripped.strip("/")


def _extract_usage(payload: Any) -> dict[str, int]:
    if not isinstance(payload, dict):
        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    usage = payload.get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}

    input_tokens = usage.get("prompt_tokens")
    if input_tokens is None:
        input_tokens = usage.get("input_tokens")
    if input_tokens is None:
        input_tokens = usage.get("total_tokens")

    output_tokens = usage.get("completion_tokens")
    if output_tokens is None:
        output_tokens = usage.get("output_tokens")

    try:
        input_value = int(input_tokens or 0)
    except Exception:
        input_value = 0
    try:
        output_value = int(output_tokens or 0)
    except Exception:
        output_value = 0

    total_tokens = usage.get("total_tokens")
    try:
        total_value = int(total_tokens or (input_value + output_value))
    except Exception:
        total_value = input_value + output_value

    if total_value < input_value + output_value:
        total_value = input_value + output_value

    return {
        "input_tokens": input_value,
        "output_tokens": output_value,
        "total_tokens": total_value,
    }


def _classify_path(path: str) -> str:
    normalized = path.lower()
    if normalized.endswith("/chat/completions") or "/chat/completions" in normalized:
        return "chat_completions"
    if normalized.endswith("/responses") or "/responses" in normalized:
        return "responses"
    if normalized.endswith("/embeddings") or "/embeddings" in normalized:
        return "embeddings"
    return "other"


class ProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address: tuple[str, int], handler: type[BaseHTTPRequestHandler], *, upstream_base_url: str, log_path: Path) -> None:
        super().__init__(server_address, handler)
        self.session = requests.Session()
        self.upstream_base_url = upstream_base_url.rstrip("/")
        self.upstream_prefix = _normalize_prefix(urlparse(self.upstream_base_url).path)
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def build_upstream_url(self, incoming_path: str) -> str:
        path_only = incoming_path.split("?", 1)[0]
        query = ""
        if "?" in incoming_path:
            query = "?" + incoming_path.split("?", 1)[1]

        normalized_path = path_only
        if self.upstream_prefix and normalized_path.startswith(self.upstream_prefix):
            normalized_path = normalized_path[len(self.upstream_prefix) :]
        if not normalized_path.startswith("/"):
            normalized_path = "/" + normalized_path

        parsed = urlparse(self.upstream_base_url)
        base_root = f"{parsed.scheme}://{parsed.netloc}{self.upstream_prefix}"
        return urljoin(base_root.rstrip("/") + "/", normalized_path.lstrip("/")) + query

    def append_log(self, payload: dict[str, Any]) -> None:
        with self.log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(payload, ensure_ascii=False) + "\n")


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def server_state(self) -> ProxyServer:
        return self.server  # type: ignore[return-value]

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/health"):
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self._forward()

    def do_POST(self) -> None:  # noqa: N802
        self._forward()

    def do_DELETE(self) -> None:  # noqa: N802
        self._forward()

    def do_PUT(self) -> None:  # noqa: N802
        self._forward()

    def do_PATCH(self) -> None:  # noqa: N802
        self._forward()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def _forward(self) -> None:
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        body = self.rfile.read(content_length) if content_length else b""
        upstream_url = self.server_state.build_upstream_url(self.path)

        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in {"host", "content-length", "accept-encoding", "connection"}
        }

        status_code = 599
        response_headers: dict[str, str] = {}
        response_body = b""
        usage_payload = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        model_name = ""
        error = ""
        try:
            response = self.server_state.session.request(
                method=self.command,
                url=upstream_url,
                headers=headers,
                data=body,
                timeout=180,
            )
            status_code = response.status_code
            response_body = response.content
            response_headers = dict(response.headers)

            try:
                payload = response.json()
            except Exception:
                payload = None

            usage_payload = _extract_usage(payload)
            if isinstance(payload, dict):
                model_name = str(payload.get("model") or "")
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            response_body = json.dumps({"error": error}).encode("utf-8")
            response_headers = {"Content-Type": "application/json; charset=utf-8"}
        finally:
            log_record = {
                "method": self.command,
                "path": self.path,
                "kind": _classify_path(self.path),
                "upstream_url": upstream_url,
                "status_code": status_code,
                "model": model_name,
                "input_tokens": usage_payload["input_tokens"],
                "output_tokens": usage_payload["output_tokens"],
                "total_tokens": usage_payload["total_tokens"],
                "request_bytes": len(body),
                "response_bytes": len(response_body),
                "error": error,
            }
            self.server_state.append_log(log_record)

        self.send_response(status_code if status_code != 599 else 502)
        passthrough_headers = {
            key: value
            for key, value in response_headers.items()
            if key.lower() not in {"content-length", "transfer-encoding", "content-encoding", "connection"}
        }
        for key, value in passthrough_headers.items():
            self.send_header(key, value)
        self.send_header("Content-Length", str(len(response_body)))
        self.end_headers()
        if response_body:
            self.wfile.write(response_body)


def main() -> int:
    args = parse_args()
    server = ProxyServer(
        (args.host, args.port),
        ProxyHandler,
        upstream_base_url=args.upstream_base_url,
        log_path=Path(args.log_path).resolve(),
    )
    print(json.dumps({"status": "ready", "host": args.host, "port": args.port}), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
