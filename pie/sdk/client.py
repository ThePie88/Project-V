"""PieClient — typed Python SDK for the Pie kernel API (V6b-E2).

Wraps HTTP calls to the FastAPI server with type-safe return types.
Uses ``httpx`` if available, falls back to ``urllib`` for zero-dep usage.
"""

from __future__ import annotations

import json
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional, TypedDict


# ---------------------------------------------------------------------------
# Return types (TypedDict for type safety)
# ---------------------------------------------------------------------------


class SessionInfo(TypedDict):
    session_id: str
    created_at: float
    seed_id: str
    kernel_version: str
    api_version: str
    schema_version: str


class TelemetryDelta(TypedDict):
    cv_channels: Dict[str, float]
    spikes: List[Dict[str, Any]]
    gating_decisions: List[Dict[str, Any]]
    budget: Dict[str, Any]
    new_event_count: int


class TurnResult(TypedDict):
    session_id: str
    response: str
    turn_count: int
    state: Dict[str, Any]
    telemetry_delta: TelemetryDelta
    telemetry_version: str
    kernel_version: str
    api_version: str
    schema_version: str


class StateSnapshot(TypedDict):
    session_id: str
    state: Dict[str, Any]
    kernel_version: str
    api_version: str
    schema_version: str


class JournalPage(TypedDict):
    session_id: str
    events: List[Dict[str, Any]]
    total: int
    offset: int
    limit: int
    kernel_version: str
    api_version: str
    schema_version: str


class SnapshotInfo(TypedDict):
    session_id: str
    hash: str
    kernel_version: str
    api_version: str
    schema_version: str


class TelemetryBulk(TypedDict):
    session_id: str
    telemetry_version: str
    cv_channels: Dict[str, List[float]]
    spike_rate: Dict[str, List[int]]
    gating_decisions: List[Dict[str, Any]]
    budget_summary: Dict[str, Any]
    kernel_version: str
    api_version: str
    schema_version: str


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PieAPIError(Exception):
    """Raised when the API returns a non-2xx response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class PieClient:
    """Typed Python SDK for the Pie kernel API.

    Parameters
    ----------
    base_url:
        API base URL (default: ``http://127.0.0.1:8000``).
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8000") -> None:
        self._base = base_url.rstrip("/")

    # -- HTTP helpers -------------------------------------------------------

    def _request(
        self,
        method: str,
        path: str,
        body: Optional[Dict[str, Any]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Make an HTTP request and return parsed JSON."""
        url = f"{self._base}{path}"

        if params:
            query_parts = []
            for k, v in params.items():
                if v is not None:
                    query_parts.append(f"{k}={v}")
            if query_parts:
                url += "?" + "&".join(query_parts)

        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=data,
            method=method,
            headers={"Content-Type": "application/json"} if data else {},
        )

        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", errors="replace")
            try:
                err = json.loads(body_text)
                detail = err.get("detail", body_text)
            except json.JSONDecodeError:
                detail = body_text
            raise PieAPIError(e.code, detail) from e

    # -- Public API ---------------------------------------------------------

    def create_session(
        self,
        seed_id: str = "SEED_V0",
        session_id: Optional[str] = None,
    ) -> SessionInfo:
        """POST /api/v1/session/create"""
        body: Dict[str, Any] = {"seed_id": seed_id}
        if session_id is not None:
            body["session_id"] = session_id
        return self._request("POST", "/api/v1/session/create", body=body)  # type: ignore[return-value]

    def turn(self, session_id: str, user_input: str) -> TurnResult:
        """POST /api/v1/session/{id}/turn"""
        return self._request(  # type: ignore[return-value]
            "POST",
            f"/api/v1/session/{session_id}/turn",
            body={"user_input": user_input},
        )

    def get_state(self, session_id: str) -> StateSnapshot:
        """GET /api/v1/session/{id}/state"""
        return self._request("GET", f"/api/v1/session/{session_id}/state")  # type: ignore[return-value]

    def get_journal(
        self,
        session_id: str,
        limit: int = 100,
        offset: int = 0,
        event_type: Optional[str] = None,
    ) -> JournalPage:
        """GET /api/v1/session/{id}/journal"""
        params: Dict[str, Any] = {"limit": limit, "offset": offset}
        if event_type:
            params["event_type"] = event_type
        return self._request(  # type: ignore[return-value]
            "GET",
            f"/api/v1/session/{session_id}/journal",
            params=params,
        )

    def get_telemetry(
        self,
        session_id: str,
        turn_start: int = 0,
        turn_end: Optional[int] = None,
    ) -> TelemetryBulk:
        """GET /api/v1/session/{id}/telemetry"""
        params: Dict[str, Any] = {"turn_start": turn_start}
        if turn_end is not None:
            params["turn_end"] = turn_end
        return self._request(  # type: ignore[return-value]
            "GET",
            f"/api/v1/session/{session_id}/telemetry",
            params=params,
        )

    def save_snapshot(self, session_id: str) -> SnapshotInfo:
        """POST /api/v1/session/{id}/snapshot"""
        return self._request(  # type: ignore[return-value]
            "POST", f"/api/v1/session/{session_id}/snapshot"
        )

    def restore_snapshot(self, session_id: str) -> SnapshotInfo:
        """POST /api/v1/session/{id}/restore"""
        return self._request(  # type: ignore[return-value]
            "POST", f"/api/v1/session/{session_id}/restore"
        )

    def reward(
        self,
        session_id: str,
        target_turn: int,
        value: int,
        source: str = "thumb",
        reason: Optional[str] = None,
        actor_id: str = "local_ui",
    ) -> Dict[str, Any]:
        """POST /api/v1/session/{id}/reward"""
        body: Dict[str, Any] = {
            "target_turn": target_turn,
            "value": value,
            "source": source,
            "actor_id": actor_id,
        }
        if reason is not None:
            body["reason"] = reason
        return self._request(
            "POST", f"/api/v1/session/{session_id}/reward", body=body,
        )
