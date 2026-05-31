"""Audit bundle creation (V6b-E4).

Creates deterministic ZIP bundles with SHA-256 hash chain for session
forensics.  Every payload file is hashed, and the root hash is computed
from the sorted (filename, hash) pairs of payload files only.

Security properties:
- manifest.json is NEVER included in root_hash (anti-circular)
- _volatile/ files are excluded from root_hash
- All text content is LF-normalized before hashing AND before ZIP storage
- Only allowlisted files are read from session_dir (no globs, no user paths)
- Root hash must be anchored externally for tamper-evidence
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Allowlists
# ---------------------------------------------------------------------------

# Session files: disk_name → bundle_name
_SESSION_FILE_MAP: Dict[str, str] = {
    "journal.jsonl": "trace.jsonl",
    "engine_snapshot.json": "snapshot.json",
    "state_latest.json": "state_latest.json",
    "environment.json": "environment.json",
    "model_info.json": "model_info.json",
}

# Required session files (must exist for a valid bundle)
_REQUIRED_SESSION_FILES = {"journal.jsonl", "state_latest.json"}

# Optional session files (omitted silently if missing)
_OPTIONAL_SESSION_FILES = {
    "engine_snapshot.json",
    "environment.json",
    "model_info.json",
}

# Policy config files to snapshot
_CONFIG_DIR = Path(__file__).parent.parent / "config"
_POLICY_FILES: List[str] = [
    "romance_guardrails.json",
    "tools_config.json",
    "legge3_guardrails.json",
    "speech_compiler.json",
    "crystallization_rules_v0.json",
]

# Fixed ZIP timestamp (earliest allowed: 1980-01-01 00:00:00)
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


# ---------------------------------------------------------------------------
# AuditBundler
# ---------------------------------------------------------------------------


class AuditBundler:
    """Create deterministic audit bundles for session forensics."""

    @classmethod
    def create(
        cls,
        session_dir: Path,
        output_path: Path,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create audit bundle ZIP from session-captured files.

        Parameters
        ----------
        session_dir:
            Path to the session directory containing captured files.
        output_path:
            Where to write the ZIP bundle.
        session_id:
            Optional session ID for manifest metadata.

        Returns
        -------
        dict
            The manifest dict including ``root_hash``.
        """
        # 1. Collect payload files (normalized)
        payload_files = cls._collect_payload_files(session_dir)

        # 2. Build policy snapshot
        policy_content = cls._build_policy_snapshot()
        payload_files["policy_snapshot.json"] = policy_content

        # 3. Compute per-file hashes
        file_hashes: Dict[str, Dict[str, Any]] = {}
        for name, content in sorted(payload_files.items()):
            file_hashes[name] = {
                "sha256": cls._compute_file_hash(content),
                "size": len(content),
            }

        # 4. Compute root hash (payload only, no manifest, no volatile)
        hash_map = {name: info["sha256"] for name, info in file_hashes.items()}
        root_hash = cls._compute_root_hash(hash_map)

        # 5. Build manifest
        manifest = {
            "schema_version": 1,
            "kernel_version": "0.0.0",
            "session_id": session_id or session_dir.name,
            "files": file_hashes,
            "root_hash": root_hash,
            "volatile_files": ["_volatile/creation_metadata.json"],
        }
        manifest_bytes = json.dumps(
            manifest, indent=2, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")

        # 6. Build volatile metadata (bundle-time, NOT in root_hash)
        volatile = cls._build_volatile_metadata()

        # 7. Write ZIP (deterministic order, fixed timestamps)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            output_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6
        ) as zf:
            # Manifest first
            cls._write_zip_entry(zf, "manifest.json", manifest_bytes)

            # Payload files in alphabetical order
            for name in sorted(payload_files.keys()):
                cls._write_zip_entry(zf, name, payload_files[name])

            # Volatile files
            cls._write_zip_entry(
                zf,
                "_volatile/creation_metadata.json",
                volatile,
            )

        return manifest

    # -- File collection ----------------------------------------------------

    @classmethod
    def _collect_payload_files(cls, session_dir: Path) -> Dict[str, bytes]:
        """Collect and LF-normalize session files from allowlist."""
        result: Dict[str, bytes] = {}

        for disk_name, bundle_name in _SESSION_FILE_MAP.items():
            path = session_dir / disk_name
            if disk_name in _REQUIRED_SESSION_FILES and not path.exists():
                raise FileNotFoundError(
                    f"Required session file missing: {disk_name}"
                )
            if not path.exists():
                continue
            raw = path.read_bytes()
            result[bundle_name] = cls._normalize(raw)

        return result

    @staticmethod
    def _build_policy_snapshot() -> bytes:
        """Snapshot all pie/config/*.json from hardcoded allowlist."""
        policies: Dict[str, Any] = {}

        for fname in sorted(_POLICY_FILES):
            path = _CONFIG_DIR / fname
            if not path.exists():
                continue
            raw = path.read_bytes()
            normalized = raw.replace(b"\r\n", b"\n")
            sha = hashlib.sha256(normalized).hexdigest()
            try:
                content = json.loads(normalized.decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                content = None
            policies[fname] = {"sha256": sha, "content": content}

        result = json.dumps(
            {"policies": policies},
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
        )
        return result.encode("utf-8")

    @staticmethod
    def _build_volatile_metadata() -> bytes:
        """Build bundle-time volatile metadata (excluded from root_hash)."""
        import time

        meta = {
            "created_at": time.time(),
            "pid": os.getpid(),
            "hostname": platform.node(),
        }
        return json.dumps(
            meta, indent=2, sort_keys=True, ensure_ascii=False
        ).encode("utf-8")

    # -- Hashing ------------------------------------------------------------

    @staticmethod
    def _compute_file_hash(content: bytes) -> str:
        """SHA256 hex digest of content (already normalized)."""
        return hashlib.sha256(content).hexdigest()

    @staticmethod
    def _compute_root_hash(file_hashes: Dict[str, str]) -> str:
        """Root hash from sorted PAYLOAD file hash pairs.

        EXCLUDES manifest.json and _volatile/* — these are never in
        file_hashes to begin with (caller passes only payload).
        """
        pairs = sorted(file_hashes.items())
        canonical = json.dumps(pairs, separators=(",", ":"), sort_keys=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # -- Normalization ------------------------------------------------------

    @staticmethod
    def _normalize(content: bytes) -> bytes:
        """Replace CRLF with LF for cross-platform determinism."""
        return content.replace(b"\r\n", b"\n")

    # -- ZIP helpers --------------------------------------------------------

    @staticmethod
    def _write_zip_entry(
        zf: zipfile.ZipFile, name: str, data: bytes
    ) -> None:
        """Write a single entry with fixed timestamp."""
        info = zipfile.ZipInfo(filename=name, date_time=_ZIP_EPOCH)
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, data)

    # -- Safety -------------------------------------------------------------

    @staticmethod
    def validate_zip_entry_name(name: str) -> bool:
        """Reject path traversal patterns.

        Returns True if name is safe, False if it contains:
        - ``..`` components
        - absolute paths (starts with / or drive letter)
        - backslash characters
        """
        if "\\" in name:
            return False
        if name.startswith("/") or (len(name) >= 2 and name[1] == ":"):
            return False
        parts = name.replace("/", os.sep).split(os.sep)
        if ".." in parts:
            return False
        return True
