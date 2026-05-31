"""Deterministic OpenAI-compatible LLM stub for CI testing.

Speaks ``/v1/chat/completions`` and ``/v1/models``.  Returns
tool_calls or text based on message content — no AI, no GPU,
sub-5 ms per request.  Stdlib only (http.server + threading).

Usage::

    from pie.testing.llm_stub import StubServer

    srv = StubServer()          # port=0 → OS picks free port
    srv.start()
    print(srv.url)              # http://127.0.0.1:<port>/v1
    # ... point RealLLM here via LM_API_BASE_URL ...
    srv.stop()
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Response builders
# ---------------------------------------------------------------------------

def _text_response(content: str) -> Dict[str, Any]:
    """OpenAI text-only response."""
    return {
        "choices": [{
            "message": {"content": content, "role": "assistant"},
            "finish_reason": "stop",
            "index": 0,
        }],
        "model": "stub-model",
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _tool_call_response(
    tool_name: str,
    arguments: Dict[str, Any],
    call_id: Optional[str] = None,
) -> Dict[str, Any]:
    """OpenAI tool_calls response."""
    if call_id is None:
        raw = f"{tool_name}:{json.dumps(arguments, sort_keys=True)}"
        call_id = f"call_{hashlib.sha256(raw.encode()).hexdigest()[:12]}"
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [{
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(arguments),
                    },
                }],
            },
            "finish_reason": "tool_calls",
            "index": 0,
        }],
        "model": "stub-model",
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }


def _models_response() -> Dict[str, Any]:
    """GET /v1/models response."""
    return {
        "data": [{"id": "stub-model", "object": "model", "owned_by": "pie-testing"}],
        "object": "list",
    }


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def _has_tool_role(messages: List[Dict[str, Any]]) -> bool:
    """True if messages contain role='tool' (second call after tool exec)."""
    return any(m.get("role") == "tool" for m in messages)


def _is_retry(messages: List[Dict[str, Any]]) -> bool:
    """True if this is a retry call (contains correction directive)."""
    for m in messages:
        if m.get("role") == "user" and "Fix the previous response" in (m.get("content") or ""):
            return True
    return False


def _extract_user_text(messages: List[Dict[str, Any]]) -> str:
    """Extract the last user message content (lowercase)."""
    for m in reversed(messages):
        if m.get("role") == "user":
            return (m.get("content") or "").lower()
    return ""


def _detect_tool_intent(user_text: str) -> Optional[str]:
    """Return tool function name if user text indicates tool use."""
    if any(w in user_text for w in ("scrivi", "write", "crea un file", "create")):
        return "fs_write"
    if any(w in user_text for w in ("leggi", "read", "mostra il contenuto")):
        return "fs_read"
    if any(w in user_text for w in ("elenca", "list", "lista")):
        return "fs_list"
    return None


def _detect_path_traversal(user_text: str) -> bool:
    """True if user asks to read a path outside sandbox."""
    return bool(
        ".." in user_text
        or re.search(r"[/\\]etc[/\\]", user_text)
        or re.search(r"[A-Z]:\\", user_text, re.IGNORECASE)
        or "/etc/passwd" in user_text
    )


def _build_tool_args(tool_name: str, user_text: str) -> Dict[str, Any]:
    """Build deterministic tool arguments from intent."""
    if tool_name == "fs_write":
        return {"path": "test.txt", "content": "ciao"}
    if tool_name == "fs_read":
        if _detect_path_traversal(user_text):
            return {"path": "../../etc/passwd"}
        return {"path": "test.txt"}
    if tool_name == "fs_list":
        return {"path": "."}
    return {}


# ---------------------------------------------------------------------------
# Chaos injection
# ---------------------------------------------------------------------------

_CHAOS_MAP = {
    # "tool" is ALWAYS in must_not_include (runtime.py L326)
    "forbidden_word": "Uso il tool per rispondere.",
    # Legge III pattern (from legge3_guardrails.json)
    "legge3": "Ho mangiato qualcosa ieri sera.",
    # Assistant pattern (from speech_compiler.json assistant_blacklist)
    "assistant": "Posso aiutarti con qualsiasi cosa.",
    # Exceeds any reasonable max_tokens (SpeechPlan max is 150)
    "toolong": "Rispondo con molta cura. " * 200,
}


def _apply_chaos(base: str, mode: Optional[str]) -> str:
    """Inject violation for first attempt (retries get clean response)."""
    if mode and mode in _CHAOS_MAP:
        return _CHAOS_MAP[mode]
    return base


# ---------------------------------------------------------------------------
# HTTP Handler
# ---------------------------------------------------------------------------

class StubHandler(BaseHTTPRequestHandler):
    """OpenAI /v1/chat/completions + /v1/models stub."""

    # Suppress per-request logging
    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass

    def do_GET(self) -> None:
        """Handle GET /v1/models."""
        if self.path.rstrip("/").endswith("/models"):
            self._send_json(_models_response())
        else:
            self.send_error(404, "Not Found")

    def do_POST(self) -> None:
        """Handle POST /v1/chat/completions."""
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404, "Not Found")
            return

        # Parse body
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        try:
            payload = json.loads(body)
        except json.JSONDecodeError:
            self.send_error(400, "Invalid JSON")
            return

        server: StubServer = self.server  # type: ignore[assignment]

        # Record for test assertions
        server.last_payload = payload
        if payload.get("tools"):
            server.last_tool_payload = payload
        server.request_count += 1

        messages: List[Dict[str, Any]] = payload.get("messages", [])
        tools: Optional[List[Dict[str, Any]]] = payload.get("tools")
        chaos_mode = server.chaos_mode

        # --- Decision logic ---

        # 1. Second call (after tool execution): return acknowledgment text
        if _has_tool_role(messages):
            self._send_json(_text_response("Fatto."))
            return

        # 2. Retry (correction directive): always return clean response
        if _is_retry(messages):
            self._send_json(_text_response("Va bene."))
            return

        user_text = _extract_user_text(messages)

        # 3. Chaos mode (first attempt only)
        if chaos_mode:
            base = "Va bene."
            injected = _apply_chaos(base, chaos_mode)
            self._send_json(_text_response(injected))
            return

        # 4. Tool call (if tools present and user wants tool action)
        if tools:
            intent = _detect_tool_intent(user_text)
            if intent:
                args = _build_tool_args(intent, user_text)
                self._send_json(_tool_call_response(intent, args))
                return

        # 5. Default text response
        self._send_json(_text_response("Va bene."))

    def _send_json(self, data: Dict[str, Any]) -> None:
        """Send JSON response with 200 OK."""
        body = json.dumps(data).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ---------------------------------------------------------------------------
# Server wrapper
# ---------------------------------------------------------------------------

class StubServer(HTTPServer):
    """Threaded OpenAI stub for testing.

    Attributes
    ----------
    chaos_mode : Optional[str]
        Set by tests to inject violations.  Handler reads this.
    last_payload : Dict
        Last request body (for test assertions).
    request_count : int
        Total requests handled (for cache bypass tests).
    """

    def __init__(self, port: int = 0) -> None:
        super().__init__(("127.0.0.1", port), StubHandler)
        self.chaos_mode: Optional[str] = None
        self.last_payload: Dict[str, Any] = {}
        self.last_tool_payload: Dict[str, Any] = {}
        self.request_count: int = 0
        self._thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        """Base URL for LM_API_BASE_URL env var."""
        return f"http://127.0.0.1:{self.server_port}/v1"

    def start(self) -> None:
        """Start serving in a daemon thread."""
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Shut down the server."""
        self.shutdown()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
