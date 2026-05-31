"""Identity bootstrap from SEED files (V3.0).

Loads the SEED_V0 identity file and produces the initial
``identity_snapshot.json`` used to anchor Ivy's identity in a session.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


def load_seed(seed_path: Path) -> Dict[str, Any]:
    """Load and validate a SEED identity file.

    The SEED file (e.g. ``progetto/SEED_V0.md``) contains JSON despite
    its ``.md`` extension.  This function reads it and performs minimal
    validation.

    Raises
    ------
    FileNotFoundError
        If *seed_path* does not exist.
    ValueError
        If the file is not valid JSON or is missing required sections.
    """
    if not seed_path.exists():
        raise FileNotFoundError(f"SEED file not found: {seed_path}")

    text = seed_path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"SEED file is not valid JSON: {exc}") from exc

    # Minimal validation: identity section must exist
    if "identity" not in data:
        raise ValueError("SEED file missing required 'identity' section")
    identity = data["identity"]
    if "name" not in identity:
        raise ValueError("SEED identity missing required 'name' field")

    return data


def build_identity_snapshot(
    seed_data: Dict[str, Any],
    seed_path: Path,
) -> Dict[str, Any]:
    """Build the ``identity_snapshot.json`` content from SEED data.

    The snapshot includes the full SEED content plus metadata about
    when and from where the identity was bootstrapped.  Timestamps
    are for logging only and do not influence decisions.
    """
    return {
        "schema_version": "0.1",
        "bootstrapped_from": str(seed_path),
        "bootstrapped_at": time.time(),
        **seed_data,
    }
