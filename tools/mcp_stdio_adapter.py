#!/usr/bin/env python3
"""StdIO MCP adapter that proxies the existing REST/SSE server.

Speaks Model Context Protocol JSON-RPC 2.0 over stdio and forwards
requests to the FastAPI server already running at the configured base URL.
"""
import json
import os
import select
import sys
from typing import Any, Dict, Optional

import requests

BASE_URL = os.getenv("MCP_SERVER_BASE_URL", "http://localhost:8010")
TIMEOUT = float(os.getenv("MCP_ADAPTER_TIMEOUT", "30"))


# Some MCP clients use LSP-style Content-Length framing; others send one JSON
# object per line; a few send raw JSON without delimiters. Detect the mode from
# the first incoming bytes and respond using the same framing.
FRAME_MODE: Optional[str] = None  # "content-length" | "newline" | "json"

# stdin byte buffer for incremental parsing
_INBUF = b""


LOG_PATH = os.getenv("MCP_ADAPTER_LOG", "/tmp/mcp_adapter.log")


def log(message: str) -> None:
    """Write diagnostics to stderr and a temp log without breaking framing."""
    line = message + "\n"
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as fp:
            fp.write(line)
    except Exception:
        pass
    sys.stderr.write(line)
    sys.stderr.flush()


def read_message() -> Optional[Dict[str, Any]]:
    """Read a single JSON-RPC message.

    Supports both:
    - LSP-style Content-Length framing
    - Newline-delimited JSON (one object per line)
    """
    global FRAME_MODE, _INBUF  # noqa: PLW0603

    def _read_more() -> bool:
        """Read more bytes into the buffer. Returns False on EOF."""
        global _INBUF  # noqa: PLW0603
        rlist, _, _ = select.select([sys.stdin], [], [], 0.5)
        if not rlist:
            return True
        chunk = os.read(sys.stdin.fileno(), 65536)
        if chunk == b"":
            return False
        _INBUF += chunk
        return True

    # Ensure we have some bytes
    while _INBUF == b"":
        if not _read_more():
            return None

    # Detect framing mode if unknown
    if FRAME_MODE is None:
        sample = _INBUF.lstrip()
        if sample.lower().startswith(b"content-length:"):
            FRAME_MODE = "content-length"
            log("frame mode: content-length")
        elif b"\n" in _INBUF:
            # likely newline-delimited JSON
            FRAME_MODE = "newline"
            log("frame mode: newline")
        elif sample.startswith((b"{", b"[")):
            FRAME_MODE = "json"
            log("frame mode: json")
        else:
            # keep reading until we can decide
            if not _read_more():
                return None

    # Parse according to detected mode
    if FRAME_MODE == "content-length":
        while True:
            # Support both CRLF and LF header termination
            header_end = _INBUF.find(b"\r\n\r\n")
            delim_len = 4
            if header_end < 0:
                header_end = _INBUF.find(b"\n\n")
                delim_len = 2
            if header_end < 0:
                if not _read_more():
                    return None
                continue

            header_blob = _INBUF[:header_end].decode("utf-8", errors="ignore")
            headers: Dict[str, str] = {}
            content_length = None
            for line in header_blob.splitlines():
                if ":" not in line:
                    continue
                k, v = line.split(":", 1)
                headers[k.strip().lower()] = v.strip()
            if "content-length" in headers:
                try:
                    content_length = int(headers["content-length"])
                except ValueError:
                    content_length = None
            if content_length is None:
                log(f"missing content-length; headers={headers}")
                # drop headers and try again
                _INBUF = _INBUF[header_end + delim_len :]
                continue

            needed = header_end + delim_len + content_length
            if len(_INBUF) < needed:
                if not _read_more():
                    return None
                continue

            body = _INBUF[header_end + delim_len : needed]
            _INBUF = _INBUF[needed:]
            try:
                return json.loads(body.decode("utf-8"))
            except json.JSONDecodeError as exc:
                log(f"Failed to decode JSON-RPC payload: {exc}")
                return None

    if FRAME_MODE == "newline":
        while b"\n" not in _INBUF:
            if not _read_more():
                return None
        line, _INBUF = _INBUF.split(b"\n", 1)
        if line.strip() == b"":
            return None
        try:
            return json.loads(line.decode("utf-8"))
        except json.JSONDecodeError as exc:
            log(f"Failed to decode newline JSON payload: {exc}")
            return None

    # FRAME_MODE == "json": parse a single JSON value from the buffer
    decoder = json.JSONDecoder()
    while True:
        try:
            text = _INBUF.decode("utf-8")
        except UnicodeDecodeError:
            if not _read_more():
                return None
            continue
        try:
            obj, idx = decoder.raw_decode(text.lstrip())
            # compute bytes consumed by the decoded prefix
            prefix = text[: len(text) - len(text.lstrip()) + idx]
            consumed = len(prefix.encode("utf-8"))
            _INBUF = _INBUF[consumed:]
            return obj
        except json.JSONDecodeError:
            if not _read_more():
                return None


def send_message(payload: Dict[str, Any]) -> None:
    """Send a JSON-RPC message using the detected framing mode."""
    data = json.dumps(payload)
    raw = data.encode("utf-8")
    log(f"send frame={FRAME_MODE} id={payload.get('id')}")
    if FRAME_MODE in {"newline", "json"}:
        sys.stdout.buffer.write(raw + b"\n")
    else:
        sys.stdout.buffer.write(f"Content-Length: {len(raw)}\r\n\r\n".encode("utf-8"))
        sys.stdout.buffer.write(raw)
    sys.stdout.buffer.flush()


def json_rpc_error(_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": _id, "error": {"code": code, "message": message}}


def handle_initialize(message: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": message.get("id"),
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {},
                "resources": {},
                "logging": {},
            },
            "serverInfo": {"name": "protein-binder-mcp-adapter", "version": "1.0.0"},
        },
    }


def fetch_tools(session: requests.Session) -> Dict[str, Any]:
    resp = session.get(f"{BASE_URL}/mcp/v1/tools", timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return {"jsonrpc": "2.0", "result": {"tools": data.get("tools", [])}}


def call_tool(session: requests.Session, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name == "design_protein_binder":
        resp = session.post(f"{BASE_URL}/api/jobs", json=arguments, timeout=TIMEOUT)
    elif name == "get_job_status":
        job_id = arguments.get("job_id")
        resp = session.get(f"{BASE_URL}/api/jobs/{job_id}", timeout=TIMEOUT)
    elif name == "list_jobs":
        resp = session.get(f"{BASE_URL}/api/jobs", timeout=TIMEOUT)
    else:
        raise ValueError(f"Unknown tool: {name}")

    resp.raise_for_status()
    text = json.dumps(resp.json(), indent=2)
    return {
        "content": [{"type": "text", "text": text}],
        "isError": False,
    }


def list_resources(session: requests.Session) -> Dict[str, Any]:
    resp = session.get(f"{BASE_URL}/mcp/v1/resources", timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return {"resources": data.get("resources", [])}


def read_resource(session: requests.Session, uri: str) -> Dict[str, Any]:
    job_id = uri.replace("job://", "")
    resp = session.get(f"{BASE_URL}/mcp/v1/resources/{job_id}", timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    return {"contents": data.get("contents", [])}


def main() -> None:
    session = requests.Session()
    log(f"adapter started cwd={os.getcwd()} argv={sys.argv}")
    while True:
        message = read_message()
        if message is None:
            log("no message received; exiting")
            break

        method = message.get("method")
        msg_id = message.get("id")

        # Trace incoming methods to help diagnose initialization issues
        log(f"recv method={method} id={msg_id}")

        try:
            if method == "initialize":
                response = handle_initialize(message)
                send_message(response)
            elif method == "tools/list":
                result = fetch_tools(session)
                result.update({"id": msg_id, "jsonrpc": "2.0"})
                send_message(result)
            elif method == "tools/call":
                params = message.get("params", {})
                name = params.get("name")
                arguments = params.get("arguments", {}) or {}
                result = call_tool(session, name, arguments)
                send_message({"jsonrpc": "2.0", "id": msg_id, "result": result})
            elif method == "resources/list":
                result = list_resources(session)
                send_message({"jsonrpc": "2.0", "id": msg_id, "result": result})
            elif method == "resources/read":
                params = message.get("params", {})
                uri = params.get("uri") or params.get("path")
                if not uri:
                    raise ValueError("Missing resource uri")
                result = read_resource(session, uri)
                send_message({"jsonrpc": "2.0", "id": msg_id, "result": result})
            elif method in {"shutdown", "exit"}:
                send_message({"jsonrpc": "2.0", "id": msg_id, "result": None})
                break
            else:
                send_message(json_rpc_error(msg_id, -32601, f"Method not found: {method}"))
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response else 500
            send_message(json_rpc_error(msg_id, -32000, f"HTTP {status}: {exc}"))
        except Exception as exc:  # pylint: disable=broad-except
            send_message(json_rpc_error(msg_id, -32603, str(exc)))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
