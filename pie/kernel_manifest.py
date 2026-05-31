"""Golden manifest builder for the public kernel surface."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Type

from .contracts import Event, State, SpeechPlan, MemoryRecord, ConstraintRecord
from .kernel_release import KERNEL_RELEASE, KERNEL_FROZEN, PUBLIC_SCHEMA_VERSION

MODEL_CLASSES: List[Type[Any]] = [
    Event,
    State,
    SpeechPlan,
    MemoryRecord,
    ConstraintRecord,
]


def _model_schema(model: Type[Any]) -> Dict[str, Any]:
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()  # type: ignore[attr-defined]


def _model_fields(model: Type[Any]) -> List[str]:
    if hasattr(model, "model_fields"):
        return list(model.model_fields.keys())  # type: ignore[attr-defined]
    return []


def _fingerprint(payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def build_manifest() -> Dict[str, Any]:
    models: List[Dict[str, Any]] = []
    for model in MODEL_CLASSES:
        schema = _model_schema(model)
        models.append(
            {
                "name": model.__name__,
                "fields": _model_fields(model),
                "schema": schema,
                "schema_fingerprint": _fingerprint(schema),
            }
        )
    exam_artifacts = [
        {
            "name": "trace_exam.jsonl",
            "pattern": "trace_exam*.jsonl",
            "type": "jsonl",
            "required_keys": ["schema_version", "id", "type", "timestamp", "content"],
        },
        {
            "name": "snapshot_exam.json",
            "pattern": "snapshot_exam.json",
            "type": "json",
            "required_keys": [
                "schema_version",
                "turn_count",
                "creator_anchor",
                "drives",
                "affect",
            ],
        },
        {
            "name": "exam_report.md",
            "pattern": "exam_report.md",
            "type": "text",
            "required_keys": [],
        },
        {
            "name": "memory.jsonl",
            "pattern": "memory.jsonl",
            "type": "jsonl",
            "required_keys": [
                "schema_version",
                "memory_id",
                "logical_time",
                "type",
                "content",
                "source_refs",
            ],
        },
        {
            "name": "memory_snapshot.json",
            "pattern": "memory_snapshot.json",
            "type": "json",
            "required_keys": [
                "identity_summary",
                "preferences_active",
                "beliefs_top",
                "trust_scores",
            ],
        },
        {
            "name": "constraints.jsonl",
            "pattern": "constraints.jsonl",
            "type": "jsonl",
            "required_keys": [
                "schema_version",
                "constraint_id",
                "logical_time",
                "family",
                "rule_id",
                "severity",
                "status",
                "commit_policy",
                "effects",
                "trigger_events",
                "explanation",
            ],
        },
        {
            "name": "constraints_snapshot.json",
            "pattern": "constraints_snapshot.json",
            "type": "json",
            "required_keys": [
                "constraint_id",
                "logical_time",
                "family",
                "rule_id",
                "severity",
                "status",
                "commit_policy",
                "effects",
                "trigger_events",
                "explanation",
            ],
        },
    ]
    return {
        "kernel_release": KERNEL_RELEASE,
        "kernel_frozen": KERNEL_FROZEN,
        "public_schema_version": PUBLIC_SCHEMA_VERSION,
        "models": models,
        "exam_artifacts": exam_artifacts,
    }


def canonical_manifest_json(manifest: Dict[str, Any]) -> str:
    return json.dumps(manifest, sort_keys=True, ensure_ascii=True)


def load_manifest(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, sort_keys=True, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )


def resolve_artifacts(output_dir: Path, manifest: Dict[str, Any]) -> Dict[str, Path]:
    resolved: Dict[str, Path] = {}
    for artifact in manifest.get("exam_artifacts", []):
        name = artifact.get("name")
        pattern = artifact.get("pattern", name)
        if not pattern:
            continue
        matches = sorted(output_dir.glob(pattern))
        if matches:
            resolved[name] = matches[0]
        else:
            resolved[name] = output_dir / str(name)
    return resolved


def validate_exam_artifacts(
    output_dir: Path,
    manifest: Dict[str, Any],
    *,
    validate_file: Any,
) -> None:
    resolved = resolve_artifacts(output_dir, manifest)
    for artifact in manifest.get("exam_artifacts", []):
        name = artifact.get("name")
        if not name:
            continue
        path = resolved.get(name, output_dir / str(name))
        if not path.exists():
            raise ValueError(f"Missing artifact: {name}")
        artifact_type = artifact.get("type")
        required_keys = artifact.get("required_keys", [])
        if artifact_type == "jsonl":
            validate_file(str(path))
            lines = [ln for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            for line in lines:
                obj = json.loads(line)
                _require_keys(obj, required_keys, name)
        elif artifact_type == "json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    _require_keys(item, required_keys, name)
                if name == "constraints_snapshot.json":
                    for item in data:
                        ConstraintRecord.model_validate(item)
            else:
                _require_keys(data, required_keys, name)
                if name == "snapshot_exam.json":
                    State.model_validate(data)
        else:
            # text artifacts only need to exist
            continue


def _require_keys(obj: Dict[str, Any], required: Iterable[str], name: str) -> None:
    for key in required:
        if key not in obj:
            raise ValueError(f"Artifact {name} missing key: {key}")
