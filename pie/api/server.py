"""FastAPI versioned API server for the Pie kernel (V6b-E2).

All endpoints under ``/api/v1/``. Namespaced to avoid conflicts with
the E3 dashboard routes (``/api/session/``).

Bind to 127.0.0.1 by default (security: no external exposure).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .engine import (
    API_VERSION,
    KERNEL_VERSION,
    SCHEMA_VERSION,
    TELEMETRY_VERSION,
    SessionEngine,
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class CreateSessionRequest(BaseModel):
    seed_id: str = "SEED_V0"
    session_id: Optional[str] = None


class TurnRequest(BaseModel):
    user_input: str


class RewardRequest(BaseModel):
    target_turn: int
    value: int
    source: str = "thumb"
    reason: Optional[str] = None
    actor_id: str = "local_ui"


class ErrorResponse(BaseModel):
    error: str
    detail: str
    status_code: int


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Pie Kernel API",
    version=API_VERSION,
    description="Versioned API for the Pie kernel (V6b-E2). GET-only reads, POST for mutations.",
)

# Engine singleton — initialized on first request
_engine: Optional[SessionEngine] = None


def _get_engine() -> SessionEngine:
    global _engine
    if _engine is None:
        sessions_root = Path(os.environ.get("PIE_SESSIONS_ROOT", "sessions"))
        seeds_root = Path(os.environ.get("PIE_SEEDS_ROOT", "."))
        llm = os.environ.get("PIE_LLM_PROVIDER", "fake")
        engine_type = os.environ.get("PIE_ENGINE", "ode")
        no_cache = os.environ.get("PIE_NO_CACHE", "0") == "1"
        _engine = SessionEngine(
            sessions_root=sessions_root,
            seeds_root=seeds_root,
            llm=llm,
            engine=engine_type,
            no_cache=no_cache,
        )
    return _engine


def set_engine(engine: Optional[SessionEngine]) -> None:
    """Override the engine singleton (for testing)."""
    global _engine
    _engine = engine


# ---------------------------------------------------------------------------
# Middleware — response headers
# ---------------------------------------------------------------------------


@app.middleware("http")
async def add_version_headers(request: Request, call_next):  # type: ignore[no-untyped-def]
    response = await call_next(request)
    response.headers["X-Schema-Version"] = SCHEMA_VERSION
    response.headers["X-API-Version"] = API_VERSION
    response.headers["X-Kernel-Version"] = KERNEL_VERSION
    response.headers["X-Telemetry-Version"] = TELEMETRY_VERSION
    return response


# ---------------------------------------------------------------------------
# Error handler
# ---------------------------------------------------------------------------


@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "detail": str(exc.detail),
            "status_code": exc.status_code,
        },
        headers={
            "X-Schema-Version": SCHEMA_VERSION,
            "X-API-Version": API_VERSION,
            "X-Kernel-Version": KERNEL_VERSION,
        },
    )


# ---------------------------------------------------------------------------
# Routes — /api/v1/
# ---------------------------------------------------------------------------


@app.post("/api/v1/session/create")
def create_session(req: CreateSessionRequest) -> Dict[str, Any]:
    """Create a new session from an allowlisted seed."""
    engine = _get_engine()
    try:
        return engine.create_session(
            seed_id=req.seed_id,
            session_id=req.session_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/v1/session/{session_id}/turn")
def process_turn(session_id: str, req: TurnRequest) -> Dict[str, Any]:
    """Process one turn, returns response + state + telemetry_delta."""
    engine = _get_engine()
    try:
        return engine.process_turn(session_id, req.user_input)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/v1/session/{session_id}/reward")
def submit_reward(session_id: str, req: RewardRequest) -> Dict[str, Any]:
    """Submit a reward signal for a completed turn."""
    engine = _get_engine()
    try:
        return engine.submit_reward(
            session_id, req.target_turn, req.value,
            req.source, req.reason, req.actor_id,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/api/v1/session/{session_id}/state")
def get_state(session_id: str) -> Dict[str, Any]:
    """Current state snapshot (read-only)."""
    engine = _get_engine()
    try:
        return engine.get_state(session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/session/{session_id}/journal")
def get_journal(
    session_id: str,
    limit: int = Query(default=100, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    event_type: Optional[str] = Query(default=None),
) -> Dict[str, Any]:
    """Journal events with pagination (read-only)."""
    engine = _get_engine()
    try:
        return engine.get_journal(
            session_id, limit=limit, offset=offset, event_type=event_type
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/api/v1/session/{session_id}/telemetry")
def get_telemetry(
    session_id: str,
    turn_start: int = Query(default=0, ge=0),
    turn_end: Optional[int] = Query(default=None),
) -> Dict[str, Any]:
    """Bulk telemetry for loading existing sessions."""
    engine = _get_engine()
    try:
        return engine.get_telemetry(
            session_id, turn_start=turn_start, turn_end=turn_end
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/v1/session/{session_id}/snapshot")
def save_snapshot(session_id: str) -> Dict[str, Any]:
    """Save engine snapshot (E1 integration)."""
    engine = _get_engine()
    try:
        return engine.save_snapshot(session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/api/v1/session/{session_id}/restore")
def restore_snapshot(session_id: str) -> Dict[str, Any]:
    """Restore engine snapshot (E1 integration)."""
    engine = _get_engine()
    try:
        return engine.restore_snapshot(session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
