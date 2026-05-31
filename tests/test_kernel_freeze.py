"""Kernel freeze tests for manifest and exam artifacts."""

from __future__ import annotations

from pathlib import Path

from pie.kernel_manifest import (
    build_manifest,
    canonical_manifest_json,
    load_manifest,
    validate_exam_artifacts,
)
from pie.runtime import run, validate_file


def test_manifest_matches_golden() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "pie" / "kernel_manifest.json"
    golden = load_manifest(manifest_path)
    current = build_manifest()
    assert canonical_manifest_json(golden) == canonical_manifest_json(current)


def test_exam_artifacts_match_manifest(tmp_path) -> None:
    out_dir = tmp_path / "artifacts"
    run(exam=True, llm="fake", output_dir=str(out_dir))
    manifest_path = Path(__file__).resolve().parents[1] / "pie" / "kernel_manifest.json"
    manifest = load_manifest(manifest_path)
    validate_exam_artifacts(out_dir, manifest, validate_file=validate_file)
