from __future__ import annotations

import os
from pathlib import Path

import pytest

from pie.artifacts_check import validate_golden_artifacts, validate_run_artifacts
from pie.artifacts_contract import ARTIFACTS_SCHEMA_VERSION, load_artifacts_contract
from pie.matrix_runner import MatrixConfig, run_matrix


def _run_matrix(tmp_path: Path, config: MatrixConfig, *, allow_online: bool = False, golden_dir: Path | None = None, golden_dir_auto: bool = False) -> Path:
    exit_code, summary = run_matrix(
        [config],
        output_root=tmp_path,
        golden_dir=golden_dir,
        golden_dir_auto=golden_dir_auto,
        allow_online=allow_online,
    )
    assert exit_code == 0
    run_id = summary["runs"][0]["run_id"]
    return tmp_path / run_id


def test_artifacts_contract_version_matches_manifest() -> None:
    contract = load_artifacts_contract()
    assert contract["schema_version"] == ARTIFACTS_SCHEMA_VERSION


def test_artifacts_check_offline_fast(tmp_path: Path) -> None:
    config = MatrixConfig(
        preset="offline_fast",
        voice="none",
        cache="no-cache",
        replay="on",
        golden="off",
        crash_test="off",
        policy="valid",
        tools="sandbox",
        rate_limit="off",
        seed=None,
        only=["exam", "replay"],
    )
    run_dir = _run_matrix(tmp_path, config)
    errors = validate_run_artifacts(run_dir)
    assert errors == []


def test_artifacts_check_policy_tamper(tmp_path: Path) -> None:
    config = MatrixConfig(
        preset="policy_tamper",
        voice="none",
        cache="no-cache",
        replay="off",
        golden="off",
        crash_test="off",
        policy="tampered",
        tools="none",
        rate_limit="off",
        seed=None,
        only=["policy"],
    )
    run_dir = _run_matrix(tmp_path, config)
    errors = validate_run_artifacts(run_dir)
    assert errors == []


def test_artifacts_check_golden_bundle(tmp_path: Path) -> None:
    golden_root = tmp_path / "golden"
    config = MatrixConfig(
        preset="offline_fast",
        voice="none",
        cache="no-cache",
        replay="on",
        golden="on",
        crash_test="off",
        policy="valid",
        tools="sandbox",
        rate_limit="off",
        seed=None,
        only=["exam", "replay", "golden"],
    )
    run_dir = _run_matrix(tmp_path, config, golden_dir=golden_root, golden_dir_auto=True)
    assert run_dir.exists()
    golden_dir = golden_root / "offline_fast"
    errors = validate_golden_artifacts(golden_dir)
    assert errors == []


def _online_enabled() -> bool:
    if os.environ.get("PIE_ENABLE_ONLINE_TESTS") != "1":
        return False
    return all(
        os.environ.get(name)
        for name in ["PIE_REAL_BASE_URL", "PIE_REAL_API_KEY", "PIE_REAL_MODEL"]
    )


@pytest.mark.skipif(not _online_enabled(), reason="online tests disabled")
def test_artifacts_check_online_smoke(tmp_path: Path) -> None:
    config = MatrixConfig(
        preset="online_smoke",
        voice="real",
        cache="no-cache",
        replay="on",
        golden="off",
        crash_test="off",
        policy="valid",
        tools="none",
        rate_limit="off",
        seed=None,
        only=["exam", "conformance"],
    )
    run_dir = _run_matrix(tmp_path, config, allow_online=True)
    errors = validate_run_artifacts(run_dir)
    assert errors == []
