"""FastAPI read-only dashboard for Ivy sessions (V6b-E3).

GET-only routes. No POST/PUT/DELETE/PATCH. No write imports.
No state mutations. Reads session files via DashboardReader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .reader import DashboardReader

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Ivy Dashboard",
    version="0.1.0",
    description="Read-only session inspector for the Pie kernel.",
)

# Sessions root — configurable via environment or default
import os

_SESSIONS_ROOT = Path(os.environ.get("PIE_SESSIONS_ROOT", "sessions"))

# Static files
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_reader(session_id: str) -> DashboardReader:
    """Build DashboardReader for a session, or raise 404."""
    session_dir = _SESSIONS_ROOT / session_id
    if not session_dir.is_dir():
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    return DashboardReader(session_dir)


# ---------------------------------------------------------------------------
# Routes — GET only
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def dashboard_home() -> HTMLResponse:
    """Serve the static HTML dashboard."""
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Ivy Dashboard</h1><p>index.html not found.</p>")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/api/sessions")
def list_sessions() -> List[Dict[str, Any]]:
    """List all sessions with metadata."""
    sessions: List[Dict[str, Any]] = []
    if not _SESSIONS_ROOT.exists():
        return sessions
    for child in sorted(_SESSIONS_ROOT.iterdir()):
        if not child.is_dir():
            continue
        meta_path = child / "session_meta.json"
        if meta_path.exists():
            import json

            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                sessions.append(meta)
            except (json.JSONDecodeError, OSError):
                continue
    return sessions


@app.get("/api/session/{session_id}/meta")
def session_meta(session_id: str) -> Dict[str, Any]:
    """Session metadata."""
    return _get_reader(session_id).session_meta()


@app.get("/api/session/{session_id}/state")
def session_state(session_id: str) -> Dict[str, Any]:
    """Current drives + affect snapshot."""
    return _get_reader(session_id).state_latest()


@app.get("/api/session/{session_id}/gating")
def session_gating(session_id: str) -> List[Dict[str, Any]]:
    """CV_GATING decisions table."""
    return _get_reader(session_id).cv_gating_table()


@app.get("/api/session/{session_id}/cv")
def session_cv(session_id: str) -> Dict[str, List[float]]:
    """CV channel time series."""
    return _get_reader(session_id).cv_channels_timeseries()


@app.get("/api/session/{session_id}/spikes")
def session_spikes(session_id: str) -> Dict[str, List[int]]:
    """Spike rate per neuron per turn."""
    return _get_reader(session_id).spike_rate()


@app.get("/api/session/{session_id}/budget")
def session_budget(session_id: str) -> Dict[str, Any]:
    """Budget summary + time series."""
    reader = _get_reader(session_id)
    return {
        "summary": reader.budget_summary(),
        "timeseries": reader.budget_timeseries(),
    }


@app.get("/api/session/{session_id}/tools")
def session_tools(session_id: str) -> Dict[str, Any]:
    """Tool stats (calls, denies, rate)."""
    return _get_reader(session_id).tool_stats()


@app.get("/api/session/{session_id}/memory")
def session_memory(session_id: str) -> Dict[str, int]:
    """Memory counts by type."""
    return _get_reader(session_id).memory_counts()
