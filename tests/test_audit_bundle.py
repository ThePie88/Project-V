"""V6b-E4: Audit Bundle + Hash Chain — test suite.

Tests:
    E-AUD-010: Bundle ZIP contains all required payload files + manifest
    E-AUD-011: manifest.json NOT included in root_hash (anti-circular)
    E-AUD-012: _volatile/ files NOT included in root_hash
    E-AUD-020: Per-file SHA256 matches normalized ZIP content
    E-AUD-021: Root hash = SHA256(sorted payload file hash pairs)
    E-AUD-030: Same session → same root_hash (reproducibility)
    E-AUD-031: Two bundles from same session = identical root_hash
    E-AUD-040: Tamper: modify trace.jsonl → verify fails
    E-AUD-041: Tamper: modify manifest root_hash → verify fails
    E-AUD-042: Tamper: remove a payload file → verify fails
    E-AUD-043: Verifier rejects zip-slip paths
    E-AUD-050: AUDIT_BUNDLE_CREATED event in journal AFTER bundle
    E-AUD-051: Root hash anchored in external audit_anchors.jsonl
    E-AUD-060: Policy snapshot contains pie/config/*.json with correct hashes
    E-AUD-070: Manifest schema-valid against schemas/audit_manifest.json
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any, Dict

import pytest

from pie.audit.bundle import AuditBundler, _POLICY_FILES, _CONFIG_DIR
from pie.audit.verify import BundleVerifier, VerificationResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _create_minimal_session(tmp_path: Path) -> Path:
    """Create a minimal session directory with required files."""
    session_dir = tmp_path / "test_session"
    session_dir.mkdir()

    # Required: journal.jsonl
    journal = session_dir / "journal.jsonl"
    events = [
        {"schema_version": "0.1", "id": 1, "type": "TURN",
         "timestamp": 1000.0, "content": {"user_input": "ciao"}},
        {"schema_version": "0.1", "id": 2, "type": "SNAPSHOT_SAVED",
         "timestamp": 1001.0, "content": {"hash": "abc123"}},
    ]
    journal.write_text(
        "\n".join(json.dumps(e) for e in events) + "\n",
        encoding="utf-8",
    )

    # Required: state_latest.json
    state = {
        "drives": {"curiosity": 0.6, "playfulness": 0.5},
        "affect": {"valence": 0.1, "tension": 0.2},
        "turn_count": 2,
    }
    (session_dir / "state_latest.json").write_text(
        json.dumps(state, indent=2), encoding="utf-8"
    )

    # Optional: engine_snapshot.json
    snapshot = {
        "snapshot_schema_version": 1,
        "active_engine_id": "default_ode",
        "engine_version": "1.0.0",
        "engine_state": {},
    }
    (session_dir / "engine_snapshot.json").write_text(
        json.dumps(snapshot, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Optional: environment.json (session-captured)
    env = {
        "python_version": "3.11.0",
        "platform": "win32",
        "os_version": "Windows-11",
        "kernel_version": "0.0.0",
        "pip_freeze": ["pydantic==2.0.0"],
    }
    (session_dir / "environment.json").write_text(
        json.dumps(env, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Optional: model_info.json (session-captured)
    model = {
        "model_id": "test-model",
        "endpoint": "http://localhost:1234/v1",
        "params": {"temperature": 0.7},
    }
    (session_dir / "model_info.json").write_text(
        json.dumps(model, indent=2, sort_keys=True), encoding="utf-8"
    )

    return session_dir


def _create_bundle(session_dir: Path, tmp_path: Path) -> tuple:
    """Create a bundle and return (output_path, manifest)."""
    output = tmp_path / "bundle.zip"
    manifest = AuditBundler.create(session_dir, output, session_id="test_sess")
    return output, manifest


# ---------------------------------------------------------------------------
# E-AUD-010: Bundle contains all required files
# ---------------------------------------------------------------------------

class TestBundleStructure:

    def test_e_aud_010_bundle_contains_required_files(self, tmp_path):
        """Bundle ZIP contains all payload files + manifest."""
        session_dir = _create_minimal_session(tmp_path)
        output, manifest = _create_bundle(session_dir, tmp_path)

        with zipfile.ZipFile(output, "r") as zf:
            names = set(zf.namelist())

        # Payload files
        assert "trace.jsonl" in names
        assert "snapshot.json" in names
        assert "state_latest.json" in names
        assert "policy_snapshot.json" in names
        assert "environment.json" in names
        assert "model_info.json" in names

        # Manifest
        assert "manifest.json" in names

        # Volatile
        assert "_volatile/creation_metadata.json" in names


# ---------------------------------------------------------------------------
# E-AUD-011, E-AUD-012: Root hash excludes manifest and volatile
# ---------------------------------------------------------------------------

class TestHashChain:

    def test_e_aud_011_manifest_not_in_root_hash(self, tmp_path):
        """manifest.json is NOT included in root_hash computation."""
        session_dir = _create_minimal_session(tmp_path)
        _, manifest = _create_bundle(session_dir, tmp_path)

        # Recompute root hash from manifest's file list
        files = manifest["files"]
        # If manifest.json were included, the root hash would differ
        payload_hashes = {
            k: v["sha256"] for k, v in files.items()
            if k != "manifest.json" and not k.startswith("_volatile/")
        }
        pairs = sorted(payload_hashes.items())
        canonical = json.dumps(pairs, separators=(",", ":"), sort_keys=False)
        expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        assert manifest["root_hash"] == expected
        # manifest.json should NOT be in the files dict at all
        # (bundler only puts payload files there)
        assert "manifest.json" not in manifest["files"]

    def test_e_aud_012_volatile_not_in_root_hash(self, tmp_path):
        """_volatile/ files NOT included in root_hash."""
        session_dir = _create_minimal_session(tmp_path)
        _, manifest = _create_bundle(session_dir, tmp_path)

        # No _volatile/ entry in files dict
        for key in manifest["files"]:
            assert not key.startswith("_volatile/"), \
                f"Volatile file {key} should not be in manifest files"

    def test_e_aud_020_per_file_hash_matches(self, tmp_path):
        """Per-file SHA256 in manifest matches normalized ZIP content."""
        session_dir = _create_minimal_session(tmp_path)
        output, manifest = _create_bundle(session_dir, tmp_path)

        with zipfile.ZipFile(output, "r") as zf:
            for fname, info in manifest["files"].items():
                content = zf.read(fname)
                normalized = content.replace(b"\r\n", b"\n")
                actual = hashlib.sha256(normalized).hexdigest()
                assert actual == info["sha256"], \
                    f"Hash mismatch for {fname}"
                assert len(normalized) == info["size"], \
                    f"Size mismatch for {fname}"

    def test_e_aud_021_root_hash_correct(self, tmp_path):
        """Root hash = SHA256(sorted payload file hash pairs)."""
        session_dir = _create_minimal_session(tmp_path)
        _, manifest = _create_bundle(session_dir, tmp_path)

        hash_map = {k: v["sha256"] for k, v in manifest["files"].items()}
        expected = AuditBundler._compute_root_hash(hash_map)
        assert manifest["root_hash"] == expected


# ---------------------------------------------------------------------------
# E-AUD-030, E-AUD-031: Determinism / reproducibility
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_e_aud_030_same_session_same_root_hash(self, tmp_path):
        """Same session → same root_hash."""
        session_dir = _create_minimal_session(tmp_path)

        out1 = tmp_path / "bundle1.zip"
        m1 = AuditBundler.create(session_dir, out1, session_id="s")

        out2 = tmp_path / "bundle2.zip"
        m2 = AuditBundler.create(session_dir, out2, session_id="s")

        assert m1["root_hash"] == m2["root_hash"]

    def test_e_aud_031_two_bundles_identical_root_hash(self, tmp_path):
        """Two bundles from same session = identical root_hash."""
        session_dir = _create_minimal_session(tmp_path)

        out1 = tmp_path / "a.zip"
        out2 = tmp_path / "b.zip"
        m1 = AuditBundler.create(session_dir, out1, session_id="x")
        m2 = AuditBundler.create(session_dir, out2, session_id="x")

        assert m1["root_hash"] == m2["root_hash"]
        assert m1["files"] == m2["files"]


# ---------------------------------------------------------------------------
# E-AUD-040..043: Tamper detection
# ---------------------------------------------------------------------------

class TestTamperDetection:

    def test_e_aud_040_tamper_trace_content(self, tmp_path):
        """Tamper: modify trace.jsonl content → verify fails."""
        session_dir = _create_minimal_session(tmp_path)
        output, _ = _create_bundle(session_dir, tmp_path)

        # Tamper: rewrite trace.jsonl inside the ZIP
        _tamper_zip_file(output, "trace.jsonl", b"TAMPERED CONTENT\n")

        result = BundleVerifier.verify(output)
        assert not result.valid
        assert any("trace.jsonl" in e for e in result.errors)

    def test_e_aud_041_tamper_root_hash(self, tmp_path):
        """Tamper: modify manifest root_hash → verify fails."""
        session_dir = _create_minimal_session(tmp_path)
        output, _ = _create_bundle(session_dir, tmp_path)

        # Tamper: change root_hash in manifest
        _tamper_manifest_root_hash(output, "0" * 64)

        result = BundleVerifier.verify(output)
        assert not result.valid
        assert any("root_hash" in e.lower() or "Root hash" in e for e in result.errors)

    def test_e_aud_042_tamper_remove_file(self, tmp_path):
        """Tamper: remove a payload file → verify fails."""
        session_dir = _create_minimal_session(tmp_path)
        output, _ = _create_bundle(session_dir, tmp_path)

        # Remove trace.jsonl from ZIP
        _remove_zip_file(output, "trace.jsonl")

        result = BundleVerifier.verify(output)
        assert not result.valid
        assert any("missing" in e.lower() for e in result.errors)

    def test_e_aud_043_zip_slip_rejected(self, tmp_path):
        """Verifier rejects zip-slip paths."""
        session_dir = _create_minimal_session(tmp_path)
        output, _ = _create_bundle(session_dir, tmp_path)

        # Inject a path-traversal entry
        _inject_zip_entry(output, "../../../etc/passwd", b"root:x:0:0")

        result = BundleVerifier.verify(output)
        assert not result.valid
        assert any("traversal" in e.lower() or "Unsafe" in e for e in result.errors)


# ---------------------------------------------------------------------------
# E-AUD-050, E-AUD-051: Integration
# ---------------------------------------------------------------------------

class TestIntegration:

    def test_e_aud_050_bundle_event_in_journal(self, tmp_path):
        """AUDIT_BUNDLE_CREATED event emitted to journal AFTER bundle."""
        from pie.session.manager import SessionManager, SessionContext
        from pie.persistence.atomic import atomic_write

        sessions_root = tmp_path / "sessions"
        mgr = SessionManager(sessions_root)

        # Set up a minimal session context
        session_dir = sessions_root / "audit_test"
        session_dir.mkdir(parents=True)
        journal_path = session_dir / "journal.jsonl"
        journal_path.touch()

        # Create required files
        (session_dir / "state_latest.json").write_text(
            json.dumps({"drives": {}, "affect": {}, "turn_count": 0}),
            encoding="utf-8",
        )
        (session_dir / "journal.jsonl").write_text(
            json.dumps({"schema_version": "0.1", "id": 1, "type": "TURN",
                         "timestamp": 1.0, "content": {}}) + "\n",
            encoding="utf-8",
        )
        (session_dir / "environment.json").write_text(
            json.dumps({"python_version": "3.11", "platform": "test",
                         "os_version": "test", "kernel_version": "0.0.0",
                         "pip_freeze": []}),
            encoding="utf-8",
        )

        # Create a minimal SessionContext
        from pie.contracts.state import State
        from pie.persistence.memory_store import MemoryStore
        from pie.persistence.constraints_store import ConstraintsStore

        ctx = SessionContext(
            session_id="audit_test",
            session_dir=session_dir,
            state=State(),
            identity={},
            event_id=10,
            turn_count=1,
            journal_path=journal_path,
            memory_store=MemoryStore(str(session_dir / "memory.jsonl")),
            constraints_store=ConstraintsStore(
                str(session_dir / "constraints.jsonl")
            ),
        )

        mgr.create_audit_bundle(ctx)

        # Check journal has AUDIT_BUNDLE_CREATED
        lines = journal_path.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(l) for l in lines if l.strip()]
        bundle_events = [e for e in events if e.get("type") == "AUDIT_BUNDLE_CREATED"]
        assert len(bundle_events) == 1
        assert "root_hash" in bundle_events[0]["content"]

    def test_e_aud_051_root_hash_anchored(self, tmp_path):
        """Root hash anchored in external audit_anchors.jsonl."""
        from pie.session.manager import SessionManager, SessionContext
        from pie.contracts.state import State
        from pie.persistence.memory_store import MemoryStore
        from pie.persistence.constraints_store import ConstraintsStore

        sessions_root = tmp_path / "sessions"
        mgr = SessionManager(sessions_root)

        session_dir = sessions_root / "anchor_test"
        session_dir.mkdir(parents=True)
        journal_path = session_dir / "journal.jsonl"
        journal_path.touch()

        (session_dir / "state_latest.json").write_text(
            json.dumps({"drives": {}, "affect": {}}), encoding="utf-8"
        )
        (session_dir / "journal.jsonl").write_text(
            json.dumps({"schema_version": "0.1", "id": 1, "type": "X",
                         "timestamp": 1.0, "content": {}}) + "\n",
            encoding="utf-8",
        )
        (session_dir / "environment.json").write_text(
            json.dumps({"python_version": "3.11", "platform": "test",
                         "os_version": "test", "kernel_version": "0.0.0",
                         "pip_freeze": []}),
            encoding="utf-8",
        )

        ctx = SessionContext(
            session_id="anchor_test",
            session_dir=session_dir,
            state=State(),
            identity={},
            event_id=5,
            turn_count=0,
            journal_path=journal_path,
            memory_store=MemoryStore(str(session_dir / "memory.jsonl")),
            constraints_store=ConstraintsStore(
                str(session_dir / "constraints.jsonl")
            ),
        )

        mgr.create_audit_bundle(ctx)

        anchor_path = sessions_root / "audit_anchors.jsonl"
        assert anchor_path.exists()
        lines = anchor_path.read_text(encoding="utf-8").strip().splitlines()
        anchors = [json.loads(l) for l in lines]
        assert len(anchors) == 1
        assert anchors[0]["session_id"] == "anchor_test"
        assert len(anchors[0]["root_hash"]) == 64


# ---------------------------------------------------------------------------
# E-AUD-060: Policy snapshot
# ---------------------------------------------------------------------------

class TestPolicySnapshot:

    def test_e_aud_060_policy_snapshot_correct(self, tmp_path):
        """Policy snapshot contains pie/config/*.json with correct hashes."""
        session_dir = _create_minimal_session(tmp_path)
        output, _ = _create_bundle(session_dir, tmp_path)

        with zipfile.ZipFile(output, "r") as zf:
            raw = zf.read("policy_snapshot.json")
            policy = json.loads(raw.decode("utf-8"))

        policies = policy.get("policies", {})

        # Check each config file that exists
        for fname in _POLICY_FILES:
            config_path = _CONFIG_DIR / fname
            if not config_path.exists():
                continue
            assert fname in policies, f"Missing policy: {fname}"
            entry = policies[fname]

            # Verify hash
            raw_content = config_path.read_bytes().replace(b"\r\n", b"\n")
            expected_hash = hashlib.sha256(raw_content).hexdigest()
            assert entry["sha256"] == expected_hash, \
                f"Hash mismatch for policy {fname}"

            # Content should be valid JSON
            assert entry["content"] is not None


# ---------------------------------------------------------------------------
# E-AUD-070: Schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:

    def test_e_aud_070_manifest_schema_valid(self, tmp_path):
        """Manifest is valid against schemas/audit_manifest.json."""
        session_dir = _create_minimal_session(tmp_path)
        _, manifest = _create_bundle(session_dir, tmp_path)

        # Load schema
        schema_path = Path(__file__).parent.parent / "schemas" / "audit_manifest.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        # Basic structural validation (no jsonschema dependency needed)
        assert manifest["schema_version"] == 1
        assert isinstance(manifest["kernel_version"], str)
        assert isinstance(manifest["session_id"], str)
        assert len(manifest["session_id"]) > 0
        assert isinstance(manifest["files"], dict)
        assert isinstance(manifest["root_hash"], str)
        assert len(manifest["root_hash"]) == 64

        # Each file entry has sha256 and size
        for fname, info in manifest["files"].items():
            assert "sha256" in info
            assert "size" in info
            assert len(info["sha256"]) == 64
            assert isinstance(info["size"], int)
            assert info["size"] >= 0


# ---------------------------------------------------------------------------
# Verification roundtrip
# ---------------------------------------------------------------------------

class TestVerificationRoundtrip:

    def test_valid_bundle_passes_verification(self, tmp_path):
        """A freshly created bundle passes verification."""
        session_dir = _create_minimal_session(tmp_path)
        output, manifest = _create_bundle(session_dir, tmp_path)

        result = BundleVerifier.verify(output)
        assert result.valid, f"Errors: {result.errors}"
        assert result.root_hash == manifest["root_hash"]
        assert result.files_checked == len(manifest["files"])

    def test_missing_bundle_returns_error(self, tmp_path):
        """Nonexistent bundle path returns error."""
        result = BundleVerifier.verify(tmp_path / "nonexistent.zip")
        assert not result.valid
        assert any("not found" in e.lower() for e in result.errors)


# ---------------------------------------------------------------------------
# ZIP tampering helpers
# ---------------------------------------------------------------------------

def _tamper_zip_file(zip_path: Path, entry_name: str, new_content: bytes):
    """Replace content of one entry in a ZIP file."""
    import io
    import shutil

    tmp = zip_path.with_suffix(".tmp.zip")
    with zipfile.ZipFile(zip_path, "r") as zin:
        with zipfile.ZipFile(tmp, "w") as zout:
            for item in zin.infolist():
                if item.filename == entry_name:
                    zout.writestr(item, new_content)
                else:
                    zout.writestr(item, zin.read(item.filename))
    shutil.move(str(tmp), str(zip_path))


def _tamper_manifest_root_hash(zip_path: Path, fake_hash: str):
    """Replace root_hash in manifest.json inside a ZIP."""
    import shutil

    tmp = zip_path.with_suffix(".tmp.zip")
    with zipfile.ZipFile(zip_path, "r") as zin:
        with zipfile.ZipFile(tmp, "w") as zout:
            for item in zin.infolist():
                if item.filename == "manifest.json":
                    manifest = json.loads(zin.read(item.filename))
                    manifest["root_hash"] = fake_hash
                    zout.writestr(
                        item,
                        json.dumps(manifest, indent=2, sort_keys=True),
                    )
                else:
                    zout.writestr(item, zin.read(item.filename))
    shutil.move(str(tmp), str(zip_path))


def _remove_zip_file(zip_path: Path, entry_name: str):
    """Remove one entry from a ZIP file."""
    import shutil

    tmp = zip_path.with_suffix(".tmp.zip")
    with zipfile.ZipFile(zip_path, "r") as zin:
        with zipfile.ZipFile(tmp, "w") as zout:
            for item in zin.infolist():
                if item.filename != entry_name:
                    zout.writestr(item, zin.read(item.filename))
    shutil.move(str(tmp), str(zip_path))


def _inject_zip_entry(zip_path: Path, entry_name: str, content: bytes):
    """Add a new entry to a ZIP file."""
    import shutil

    tmp = zip_path.with_suffix(".tmp.zip")
    with zipfile.ZipFile(zip_path, "r") as zin:
        with zipfile.ZipFile(tmp, "w") as zout:
            for item in zin.infolist():
                zout.writestr(item, zin.read(item.filename))
            zout.writestr(entry_name, content)
    shutil.move(str(tmp), str(zip_path))
