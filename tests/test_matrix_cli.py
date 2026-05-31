"""Tests for test-matrix CLI parsing."""

from __future__ import annotations

import pytest

import pie.cli as cli


def test_test_matrix_default_preset(monkeypatch) -> None:
    captured = {}

    def fake_run_matrix(configs, **kwargs):
        captured["configs"] = configs
        return 0, {"runs": []}

    monkeypatch.setattr(cli, "run_matrix", fake_run_matrix)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["test-matrix"])
    assert excinfo.value.code == 0
    configs = captured["configs"]
    assert len(configs) == 1
    assert configs[0].preset == "offline_fast"
    assert configs[0].voice == "none"


def test_test_matrix_only_parsing(monkeypatch) -> None:
    captured = {}

    def fake_run_matrix(configs, **kwargs):
        captured["configs"] = configs
        return 0, {"runs": []}

    monkeypatch.setattr(cli, "run_matrix", fake_run_matrix)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["test-matrix", "--only", "exam,replay"])
    assert excinfo.value.code == 0
    configs = captured["configs"]
    assert configs[0].only == ["exam", "replay"]


def test_test_matrix_unknown_preset(monkeypatch) -> None:
    def fake_run_matrix(configs, **kwargs):
        return 0, {"runs": []}

    monkeypatch.setattr(cli, "run_matrix", fake_run_matrix)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["test-matrix", "--preset", "does_not_exist"])
    assert excinfo.value.code == 1
