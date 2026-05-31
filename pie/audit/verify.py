"""Audit bundle verification (V6b-E4).

Recomputes SHA-256 hashes for every payload file in a bundle ZIP
and compares them against the manifest.  Detects:
- Modified file content (hash mismatch)
- Missing files (declared in manifest but absent from ZIP)
- Tampered root_hash (recomputed vs stored)
- Path traversal attempts (zip-slip)
"""

from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class VerificationResult:
    """Result of bundle verification."""

    valid: bool
    root_hash: str
    errors: List[str] = field(default_factory=list)
    files_checked: int = 0


class BundleVerifier:
    """Verify audit bundle integrity."""

    @classmethod
    def verify(cls, bundle_path: Path) -> VerificationResult:
        """Verify a bundle ZIP.

        Returns a VerificationResult with ``valid=True`` if all checks pass.
        """
        errors: List[str] = []
        files_checked = 0

        if not bundle_path.exists():
            return VerificationResult(
                valid=False,
                root_hash="",
                errors=[f"Bundle not found: {bundle_path}"],
            )

        try:
            with zipfile.ZipFile(bundle_path, "r") as zf:
                # 1. Safety: check all entry names
                for name in zf.namelist():
                    if not cls._is_safe_entry_name(name):
                        errors.append(
                            f"Unsafe ZIP entry name (path traversal): {name}"
                        )
                if errors:
                    return VerificationResult(
                        valid=False, root_hash="", errors=errors
                    )

                # 2. Read manifest
                if "manifest.json" not in zf.namelist():
                    return VerificationResult(
                        valid=False,
                        root_hash="",
                        errors=["manifest.json missing from bundle"],
                    )

                manifest_raw = zf.read("manifest.json")
                try:
                    manifest = json.loads(manifest_raw.decode("utf-8"))
                except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                    return VerificationResult(
                        valid=False,
                        root_hash="",
                        errors=[f"Invalid manifest.json: {exc}"],
                    )

                # 3. Verify per-file hashes
                file_errors, checked = cls._verify_file_hashes(manifest, zf)
                errors.extend(file_errors)
                files_checked = checked

                # 4. Verify root hash
                root_error = cls._verify_root_hash(manifest)
                if root_error:
                    errors.append(root_error)

                stored_root = manifest.get("root_hash", "")

        except zipfile.BadZipFile as exc:
            return VerificationResult(
                valid=False,
                root_hash="",
                errors=[f"Invalid ZIP file: {exc}"],
            )

        return VerificationResult(
            valid=len(errors) == 0,
            root_hash=stored_root,
            errors=errors,
            files_checked=files_checked,
        )

    @classmethod
    def _verify_file_hashes(
        cls,
        manifest: Dict[str, Any],
        zf: zipfile.ZipFile,
    ) -> tuple:
        """Check each file hash in manifest against ZIP content.

        Returns (errors_list, files_checked_count).
        """
        errors: List[str] = []
        checked = 0
        files_info = manifest.get("files", {})
        zip_names = set(zf.namelist())

        for fname, info in sorted(files_info.items()):
            expected_hash = info.get("sha256", "")
            expected_size = info.get("size", -1)

            if fname not in zip_names:
                errors.append(
                    f"File declared in manifest but missing from ZIP: {fname}"
                )
                continue

            content = zf.read(fname)
            # Normalize for hash comparison (same as bundler)
            normalized = cls._normalize(content)
            actual_hash = hashlib.sha256(normalized).hexdigest()

            if actual_hash != expected_hash:
                errors.append(
                    f"Hash mismatch for {fname}: "
                    f"expected {expected_hash[:16]}..., "
                    f"got {actual_hash[:16]}..."
                )

            if expected_size >= 0 and len(normalized) != expected_size:
                errors.append(
                    f"Size mismatch for {fname}: "
                    f"expected {expected_size}, got {len(normalized)}"
                )

            checked += 1

        return errors, checked

    @classmethod
    def _verify_root_hash(cls, manifest: Dict[str, Any]) -> Optional[str]:
        """Recompute root hash and compare with stored value.

        Returns error string or None if OK.
        """
        stored = manifest.get("root_hash", "")
        files_info = manifest.get("files", {})

        # Rebuild hash map (payload only — exclude manifest and volatile)
        volatile = set(manifest.get("volatile_files", []))
        hash_map = {}
        for fname, info in files_info.items():
            if fname == "manifest.json":
                continue
            if fname.startswith("_volatile/") or fname in volatile:
                continue
            hash_map[fname] = info.get("sha256", "")

        # Recompute root hash
        pairs = sorted(hash_map.items())
        canonical = json.dumps(pairs, separators=(",", ":"), sort_keys=False)
        recomputed = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

        if recomputed != stored:
            return (
                f"Root hash mismatch: stored {stored[:16]}..., "
                f"recomputed {recomputed[:16]}..."
            )
        return None

    # -- Helpers ------------------------------------------------------------

    @staticmethod
    def _normalize(content: bytes) -> bytes:
        """Replace CRLF with LF (same normalization as bundler)."""
        return content.replace(b"\r\n", b"\n")

    @staticmethod
    def _is_safe_entry_name(name: str) -> bool:
        """Check for path traversal patterns."""
        if "\\" in name:
            return False
        if name.startswith("/"):
            return False
        if len(name) >= 2 and name[1] == ":":
            return False
        # Check for .. in path components
        parts = name.split("/")
        if ".." in parts:
            return False
        return True
