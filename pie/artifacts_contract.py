"""Artifacts contract metadata for V1 hardening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

ARTIFACTS_SCHEMA_VERSION = "1.0.0"


def load_artifacts_contract(repo_root: Path | None = None) -> Dict[str, Any]:
    root = repo_root or Path(__file__).resolve().parents[1]
    path = root / "schemas" / "artifacts_contract.json"
    if not path.exists():
        raise RuntimeError(
            f"schemas/ not found at {path.parent}. "
            "This command requires a repo checkout — run from the repository "
            "root or pass --repo-root."
        )
    return json.loads(path.read_text(encoding="utf-8"))
