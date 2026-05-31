"""Tests for V3.5 — Quality Gate + Packaging.

Test IDs:
- E-QA-010: Fresh venv pipeline (pyproject.toml installable)
- E-QA-020: Structured logging present
- E-QA-030: README + ARCH.md presenti
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------
# E-QA-010 — Packaging installable
# -----------------------------------------------------------------------


class TestPackaging:
    """Verify the project is installable and has correct structure."""

    def test_pyproject_toml_exists(self):
        """pyproject.toml must exist at project root."""
        assert (PROJECT_ROOT / "pyproject.toml").exists()

    def test_pyproject_has_required_fields(self):
        """pyproject.toml must declare name, version, dependencies."""
        content = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert "name" in content
        assert "version" in content
        assert "pydantic" in content

    def test_pie_package_importable(self):
        """The pie package must be importable."""
        import pie
        assert hasattr(pie, "__file__")

    def test_all_core_modules_importable(self):
        """All core kernel modules must import without error."""
        modules = [
            "pie.contracts",
            "pie.contracts.event",
            "pie.contracts.state",
            "pie.contracts.speech_plan",
            "pie.contracts.memory",
            "pie.contracts.constraint",
            "pie.runtime",
            "pie.cli",
            "pie.decisioning",
            "pie.determinism",
            "pie.persistence.atomic",
            "pie.persistence.memory_store",
            "pie.persistence.constraints_store",
            "pie.session",
            "pie.session.manager",
            "pie.session.bootstrap",
            "pie.counterfactuals",
            "pie.killswitch",
            "pie.metabolism",
            "pie.routines",
            "pie.state_engine",
            "pie.crystallization",
            "pie.memory.policy",
            "pie.memory.view",
        ]
        for mod in modules:
            __import__(mod)

    def test_cli_help_works(self):
        """CLI --help must exit 0."""
        result = subprocess.run(
            [sys.executable, "-m", "pie.cli", "--help"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0
        assert "run" in result.stdout
        assert "validate" in result.stdout


# -----------------------------------------------------------------------
# E-QA-020 — Structured logging / error handling
# -----------------------------------------------------------------------


class TestStructuredLogging:
    """Verify that errors produce structured, actionable output."""

    def test_validate_invalid_file_structured_error(self):
        """validate on bad file must produce structured error message."""
        result = subprocess.run(
            [sys.executable, "-m", "pie.cli", "validate", "nonexistent_file.json"],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 1
        assert "failed" in result.stdout.lower()

    def test_resume_nonexistent_session_error(self):
        """Resuming a bad session must produce a clear error."""
        from pie.session.manager import SessionManager
        mgr = SessionManager(Path("nonexistent_sessions_dir"))
        with pytest.raises(FileNotFoundError):
            mgr.resume("does_not_exist")

    def test_load_seed_bad_json_structured_error(self):
        """Bad SEED JSON must produce ValueError with context."""
        from pie.session.bootstrap import load_seed
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write("not json at all")
            f.flush()
            with pytest.raises(ValueError, match="not valid JSON"):
                load_seed(Path(f.name))


# -----------------------------------------------------------------------
# E-QA-030 — README + ARCH.md presenti
# -----------------------------------------------------------------------


class TestDocumentation:
    """Verify that README.md and ARCH.md are present and non-empty."""

    def test_readme_exists(self):
        """README.md must exist at project root."""
        readme = PROJECT_ROOT / "README.md"
        assert readme.exists()
        content = readme.read_text(encoding="utf-8")
        assert len(content) > 100

    def test_readme_has_required_sections(self):
        """README.md must have install, run, test, troubleshooting."""
        content = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert "## Install" in content
        assert "## Run" in content
        assert "## Test" in content
        assert "## Troubleshooting" in content

    def test_arch_md_exists(self):
        """ARCH.md must exist."""
        arch = PROJECT_ROOT / "progetto" / "ARCH.md"
        assert arch.exists()
        content = arch.read_text(encoding="utf-8")
        assert len(content) > 100

    def test_spec_files_present(self):
        """Core spec files must all be present."""
        for name in ["V1.md", "V2.md", "V3.md", "SPEC.md", "DONE.md", "TESTS.md"]:
            path = PROJECT_ROOT / "progetto" / name
            assert path.exists(), f"Missing spec file: {name}"

    def test_seed_file_present(self):
        """SEED_V0.md must exist and be valid JSON."""
        seed = PROJECT_ROOT / "progetto" / "SEED_V0.md"
        assert seed.exists()
        data = json.loads(seed.read_text(encoding="utf-8"))
        assert "identity" in data

    def test_kernel_manifest_present(self):
        """kernel_manifest.json must exist."""
        manifest = PROJECT_ROOT / "pie" / "kernel_manifest.json"
        assert manifest.exists()
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert "contracts" in data or "schema_fingerprints" in data or len(data) > 0
