"""Tests for kernel-check CLI branching."""

from __future__ import annotations

import pytest

import pie.cli as cli


def _patch_kernel_check(monkeypatch, calls) -> None:
    def fake_run(*, exam, llm, output_dir, no_cache=False):
        calls.append((llm, no_cache))

    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "validate_exam_artifacts", lambda *args, **kwargs: None)
    monkeypatch.setattr(cli, "replay", lambda *args, **kwargs: True)
    dummy_manifest = {"models": [], "exam_artifacts": []}
    monkeypatch.setattr(cli, "load_manifest", lambda path: dummy_manifest)
    monkeypatch.setattr(cli, "build_manifest", lambda: dummy_manifest)
    monkeypatch.setattr(cli, "canonical_manifest_json", lambda m: "ok")


def test_kernel_check_fake_only(monkeypatch, capsys) -> None:
    calls = []
    _patch_kernel_check(monkeypatch, calls)
    monkeypatch.delenv("LM_API_BASE_URL", raising=False)
    monkeypatch.delenv("LM_API_KEY", raising=False)
    monkeypatch.delenv("PIE_REAL_BASE_URL", raising=False)
    monkeypatch.delenv("PIE_REAL_API_KEY", raising=False)
    monkeypatch.delenv("PIE_REAL_MODEL", raising=False)
    cli.main(["kernel-check"])
    assert calls == [("fake", False)]
    captured = capsys.readouterr()
    assert "REAL_LLM_CHECK_SKIPPED" in captured.out


def test_kernel_check_with_real_missing_config(monkeypatch, capsys) -> None:
    calls = []
    _patch_kernel_check(monkeypatch, calls)
    monkeypatch.delenv("LM_API_BASE_URL", raising=False)
    monkeypatch.delenv("LM_API_KEY", raising=False)
    monkeypatch.delenv("PIE_REAL_BASE_URL", raising=False)
    monkeypatch.delenv("PIE_REAL_API_KEY", raising=False)
    monkeypatch.delenv("PIE_REAL_MODEL", raising=False)
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["kernel-check", "--with-real"])
    assert excinfo.value.code == 1
    assert calls == [("fake", False)]
    captured = capsys.readouterr()
    assert "REAL_LLM_NOT_CONFIGURED" in captured.out


def test_kernel_check_with_real_unreachable(monkeypatch, capsys) -> None:
    calls = []
    _patch_kernel_check(monkeypatch, calls)
    monkeypatch.setattr(
        cli,
        "_get_real_llm_config",
        lambda: (
            {"base_url": "http://127.0.0.1:1234/v1", "api_key": "x", "model": "m"},
            "",
        ),
    )
    monkeypatch.setattr(cli, "_check_real_llm_reachable", lambda *args, **kwargs: (False, "boom"))
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["kernel-check", "--with-real"])
    assert excinfo.value.code == 1
    assert calls == [("fake", False)]
    captured = capsys.readouterr()
    assert "REAL_LLM_UNREACHABLE" in captured.out
