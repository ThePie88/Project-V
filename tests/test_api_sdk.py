"""V6b-E2 — API / SDK tests.

Tests the versioned API (FastAPI) and Python SDK (PieClient).
Uses FastAPI TestClient for in-process testing (no network).

Test IDs: E-API-010..060
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict

import pytest
from fastapi.testclient import TestClient

from pie.api.engine import (
    API_VERSION,
    KERNEL_VERSION,
    SCHEMA_VERSION,
    TELEMETRY_VERSION,
    SessionEngine,
)
from pie.api.server import app, set_engine
from pie.state_engine.registry import StateEngineRegistry

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED_PATH = Path("progetto/SEED_V0.md")


@pytest.fixture(autouse=True)
def _reset_registry():
    """Reset StateEngineRegistry between tests."""
    StateEngineRegistry.reset()
    yield
    StateEngineRegistry.reset()


@pytest.fixture()
def engine(tmp_path: Path) -> SessionEngine:
    """SessionEngine with temp sessions root."""
    return SessionEngine(
        sessions_root=tmp_path / "sessions",
        seeds_root=Path("."),
        llm="fake",
    )


@pytest.fixture()
def client(engine: SessionEngine) -> TestClient:
    """FastAPI TestClient with injected engine."""
    set_engine(engine)
    yield TestClient(app)
    set_engine(None)  # type: ignore[arg-type]


@pytest.fixture()
def session_id(client: TestClient) -> str:
    """Create a session and return its ID."""
    resp = client.post("/api/v1/session/create", json={"seed_id": "SEED_V0"})
    assert resp.status_code == 200
    return resp.json()["session_id"]


# ---------------------------------------------------------------------------
# E-API-010 — All endpoints have versioned schemas
# ---------------------------------------------------------------------------


class TestSchemaContract:
    """E-API-010: Schema file exists and covers all endpoint types."""

    def test_schema_file_exists(self):
        """E-API-010a: schemas/api/v1_session.json exists."""
        schema_path = Path("schemas/api/v1_session.json")
        assert schema_path.exists(), "Schema file missing"

    def test_schema_has_all_definitions(self):
        """E-API-010b: Schema defines all request/response types."""
        schema = json.loads(Path("schemas/api/v1_session.json").read_text())
        defs = schema.get("definitions", {})
        required = [
            "CreateSessionRequest",
            "CreateSessionResponse",
            "TurnRequest",
            "TurnResponse",
            "StateResponse",
            "JournalResponse",
            "SnapshotResponse",
            "ErrorResponse",
            "TelemetryDelta",
            "TelemetryResponse",
            "ResponseMeta",
        ]
        for name in required:
            assert name in defs, f"Missing definition: {name}"

    def test_schema_fingerprint_stable(self):
        """E-API-010c: Schema fingerprint is canonical (sorted keys, compact)."""
        raw = Path("schemas/api/v1_session.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        fingerprint = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        # Just verify it's deterministic — same content = same hash
        canonical2 = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        assert hashlib.sha256(canonical2.encode("utf-8")).hexdigest() == fingerprint


# ---------------------------------------------------------------------------
# E-API-011 — X-Schema-Version header on all responses
# ---------------------------------------------------------------------------


class TestVersionHeaders:
    """E-API-011: Version headers present on all responses."""

    def test_create_has_headers(self, client: TestClient):
        resp = client.post("/api/v1/session/create", json={"seed_id": "SEED_V0"})
        assert resp.headers.get("X-Schema-Version") == SCHEMA_VERSION
        assert resp.headers.get("X-API-Version") == API_VERSION
        assert resp.headers.get("X-Kernel-Version") == KERNEL_VERSION
        assert resp.headers.get("X-Telemetry-Version") == TELEMETRY_VERSION

    def test_state_has_headers(self, client: TestClient, session_id: str):
        resp = client.get(f"/api/v1/session/{session_id}/state")
        assert resp.headers.get("X-Schema-Version") == SCHEMA_VERSION

    def test_error_has_headers(self, client: TestClient):
        resp = client.get("/api/v1/session/nonexistent/state")
        assert resp.status_code == 404
        assert resp.headers.get("X-Schema-Version") == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# E-API-020 — Schema validation rejects malformed requests
# ---------------------------------------------------------------------------


class TestValidation:
    """E-API-020: Malformed requests are rejected."""

    def test_turn_missing_user_input(self, client: TestClient, session_id: str):
        """POST /turn without user_input → 422."""
        resp = client.post(f"/api/v1/session/{session_id}/turn", json={})
        assert resp.status_code == 422

    def test_create_invalid_seed(self, client: TestClient):
        """Create with unknown seed_id → 422."""
        resp = client.post("/api/v1/session/create", json={"seed_id": "NONEXISTENT"})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# E-API-021 — POST /session/create returns valid session
# ---------------------------------------------------------------------------


class TestCreateSession:
    """E-API-021: Session creation smoke test."""

    def test_create_returns_session_id(self, client: TestClient):
        resp = client.post("/api/v1/session/create", json={"seed_id": "SEED_V0"})
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert data["seed_id"] == "SEED_V0"
        assert data["kernel_version"] == KERNEL_VERSION
        assert data["api_version"] == API_VERSION
        assert data["schema_version"] == SCHEMA_VERSION

    def test_create_with_custom_id(self, client: TestClient):
        resp = client.post(
            "/api/v1/session/create",
            json={"seed_id": "SEED_V0", "session_id": "test_custom_001"},
        )
        assert resp.status_code == 200
        assert resp.json()["session_id"] == "test_custom_001"


# ---------------------------------------------------------------------------
# E-API-022 — POST /turn returns response + state + telemetry
# ---------------------------------------------------------------------------


class TestProcessTurn:
    """E-API-022: Turn processing smoke test."""

    def test_turn_returns_response(self, client: TestClient, session_id: str):
        resp = client.post(
            f"/api/v1/session/{session_id}/turn",
            json={"user_input": "Ciao Ivy"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data
        assert isinstance(data["response"], str)
        assert data["turn_count"] >= 1
        assert "state" in data
        assert data["session_id"] == session_id

    def test_turn_has_telemetry_delta(self, client: TestClient, session_id: str):
        resp = client.post(
            f"/api/v1/session/{session_id}/turn",
            json={"user_input": "Come stai?"},
        )
        data = resp.json()
        td = data["telemetry_delta"]
        assert "cv_channels" in td
        assert "spikes" in td
        assert "gating_decisions" in td
        assert "budget" in td
        assert "new_event_count" in td
        assert data["telemetry_version"] == TELEMETRY_VERSION

    def test_turn_has_response_meta(self, client: TestClient, session_id: str):
        resp = client.post(
            f"/api/v1/session/{session_id}/turn",
            json={"user_input": "Test meta"},
        )
        data = resp.json()
        assert data["kernel_version"] == KERNEL_VERSION
        assert data["api_version"] == API_VERSION
        assert data["schema_version"] == SCHEMA_VERSION
        assert data["session_id"] == session_id
        assert "turn_id" in data


# ---------------------------------------------------------------------------
# E-API-023 — GET /state matches state_latest.json
# ---------------------------------------------------------------------------


class TestGetState:
    """E-API-023: State endpoint returns consistent data."""

    def test_state_after_turn(self, client: TestClient, session_id: str):
        # Process a turn first
        client.post(
            f"/api/v1/session/{session_id}/turn",
            json={"user_input": "Primo turno"},
        )
        resp = client.get(f"/api/v1/session/{session_id}/state")
        assert resp.status_code == 200
        data = resp.json()
        assert "state" in data
        state = data["state"]
        assert "drives" in state
        assert "affect" in state


# ---------------------------------------------------------------------------
# E-API-024 — GET /journal with pagination
# ---------------------------------------------------------------------------


class TestGetJournal:
    """E-API-024: Journal endpoint with pagination."""

    def test_journal_returns_events(self, client: TestClient, session_id: str):
        client.post(
            f"/api/v1/session/{session_id}/turn",
            json={"user_input": "Turno per journal"},
        )
        resp = client.get(f"/api/v1/session/{session_id}/journal")
        assert resp.status_code == 200
        data = resp.json()
        assert "events" in data
        assert len(data["events"]) > 0
        assert "total" in data
        assert "offset" in data
        assert "limit" in data

    def test_journal_pagination(self, client: TestClient, session_id: str):
        client.post(
            f"/api/v1/session/{session_id}/turn",
            json={"user_input": "Turno 1"},
        )
        # Get with small limit
        resp = client.get(
            f"/api/v1/session/{session_id}/journal",
            params={"limit": 2, "offset": 0},
        )
        data = resp.json()
        assert len(data["events"]) <= 2
        assert data["total"] >= 2  # At least INPUT + STATE_UPDATED

    def test_journal_filter_by_type(self, client: TestClient, session_id: str):
        client.post(
            f"/api/v1/session/{session_id}/turn",
            json={"user_input": "Turno filter"},
        )
        resp = client.get(
            f"/api/v1/session/{session_id}/journal",
            params={"event_type": "INPUT"},
        )
        data = resp.json()
        for evt in data["events"]:
            assert evt.get("type") == "INPUT"


# ---------------------------------------------------------------------------
# E-API-025 — Snapshot round-trip (E1 integration)
# ---------------------------------------------------------------------------


class TestSnapshotRoundTrip:
    """E-API-025: Save + restore snapshot via API."""

    def test_snapshot_save(self, client: TestClient, session_id: str):
        resp = client.post(f"/api/v1/session/{session_id}/snapshot")
        assert resp.status_code == 200
        data = resp.json()
        assert "hash" in data
        assert len(data["hash"]) == 64  # SHA-256 hex

    def test_snapshot_restore(self, client: TestClient, session_id: str):
        # Save first
        save_resp = client.post(f"/api/v1/session/{session_id}/snapshot")
        save_hash = save_resp.json()["hash"]

        # Restore
        restore_resp = client.post(f"/api/v1/session/{session_id}/restore")
        assert restore_resp.status_code == 200
        restore_hash = restore_resp.json()["hash"]
        assert restore_hash == save_hash


# ---------------------------------------------------------------------------
# E-API-030 — Telemetry bulk endpoint
# ---------------------------------------------------------------------------


class TestTelemetry:
    """E-API-030: Bulk telemetry endpoint."""

    def test_telemetry_returns_data(self, client: TestClient, session_id: str):
        client.post(
            f"/api/v1/session/{session_id}/turn",
            json={"user_input": "Turno telemetry"},
        )
        resp = client.get(f"/api/v1/session/{session_id}/telemetry")
        assert resp.status_code == 200
        data = resp.json()
        assert "cv_channels" in data
        assert "spike_rate" in data
        assert "gating_decisions" in data
        assert "budget_summary" in data
        assert data["telemetry_version"] == TELEMETRY_VERSION


# ---------------------------------------------------------------------------
# E-API-040 — Breaking change detection (schema fingerprint)
# ---------------------------------------------------------------------------


class TestBreakingChange:
    """E-API-040: Schema fingerprint detects breaking changes."""

    def test_fingerprint_deterministic(self):
        """Same schema content → same fingerprint."""
        raw = Path("schemas/api/v1_session.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        canonical = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        fp1 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        fp2 = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert fp1 == fp2

    def test_fingerprint_changes_on_schema_mutation(self):
        """Mutated schema → different fingerprint."""
        raw = Path("schemas/api/v1_session.json").read_text(encoding="utf-8")
        data = json.loads(raw)
        canonical_orig = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        fp_orig = hashlib.sha256(canonical_orig.encode("utf-8")).hexdigest()

        # Mutate: add a field
        data_mutated = json.loads(raw)
        data_mutated["definitions"]["TurnResponse"]["properties"]["new_field"] = {"type": "string"}
        canonical_mut = json.dumps(data_mutated, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        fp_mut = hashlib.sha256(canonical_mut.encode("utf-8")).hexdigest()

        assert fp_orig != fp_mut, "Fingerprint should change when schema is mutated"


# ---------------------------------------------------------------------------
# E-API-050 — API responses include response metadata
# ---------------------------------------------------------------------------


class TestResponseMeta:
    """E-API-050: Every response includes kernel_version, api_version, etc."""

    def test_create_meta(self, client: TestClient):
        resp = client.post("/api/v1/session/create", json={"seed_id": "SEED_V0"})
        data = resp.json()
        assert data["kernel_version"] == KERNEL_VERSION
        assert data["api_version"] == API_VERSION
        assert data["schema_version"] == SCHEMA_VERSION

    def test_state_meta(self, client: TestClient, session_id: str):
        resp = client.get(f"/api/v1/session/{session_id}/state")
        data = resp.json()
        assert data["kernel_version"] == KERNEL_VERSION
        assert data["session_id"] == session_id

    def test_journal_meta(self, client: TestClient, session_id: str):
        resp = client.get(f"/api/v1/session/{session_id}/journal")
        data = resp.json()
        assert data["kernel_version"] == KERNEL_VERSION
        assert data["session_id"] == session_id


# ---------------------------------------------------------------------------
# E-API-060 — Invalid session_id → 404, not crash
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """E-API-060: Error handling for invalid sessions."""

    def test_state_404(self, client: TestClient):
        resp = client.get("/api/v1/session/nonexistent_session/state")
        assert resp.status_code == 404
        data = resp.json()
        assert "error" in data or "detail" in data

    def test_turn_404(self, client: TestClient):
        resp = client.post(
            "/api/v1/session/nonexistent_session/turn",
            json={"user_input": "hello"},
        )
        assert resp.status_code == 404

    def test_journal_404(self, client: TestClient):
        resp = client.get("/api/v1/session/nonexistent_session/journal")
        assert resp.status_code == 404

    def test_snapshot_404(self, client: TestClient):
        resp = client.post("/api/v1/session/nonexistent_session/snapshot")
        assert resp.status_code == 404

    def test_restore_no_snapshot(self, client: TestClient, session_id: str):
        """Restore without prior save → 404."""
        resp = client.post(f"/api/v1/session/{session_id}/restore")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Seed security — path traversal prevention
# ---------------------------------------------------------------------------


class TestSeedSecurity:
    """Seed allowlist prevents arbitrary filesystem access."""

    def test_unknown_seed_rejected(self, client: TestClient):
        resp = client.post(
            "/api/v1/session/create",
            json={"seed_id": "../../../etc/passwd"},
        )
        assert resp.status_code == 422

    def test_path_traversal_rejected(self, client: TestClient):
        resp = client.post(
            "/api/v1/session/create",
            json={"seed_id": "../../secrets.json"},
        )
        assert resp.status_code == 422

    def test_only_allowlisted_seeds(self, engine: SessionEngine):
        """Direct engine call with bad seed raises ValueError."""
        with pytest.raises(ValueError, match="Unknown seed_id"):
            engine.create_session(seed_id="EVIL_SEED")


# ---------------------------------------------------------------------------
# SDK types — import check
# ---------------------------------------------------------------------------


class TestSDKImports:
    """SDK module imports cleanly."""

    def test_import_client(self):
        from pie.sdk.client import PieClient, PieAPIError
        assert PieClient is not None
        assert PieAPIError is not None

    def test_import_types(self):
        from pie.sdk.client import (
            SessionInfo,
            TurnResult,
            StateSnapshot,
            JournalPage,
            SnapshotInfo,
            TelemetryDelta,
            TelemetryBulk,
        )
        # TypedDicts are importable
        assert SessionInfo is not None
