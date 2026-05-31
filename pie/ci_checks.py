"""Local CI quality checks for lint/format and type checking."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List
import py_compile


def _iter_files(root: Path, suffixes: Iterable[str]) -> Iterable[Path]:
    for suffix in suffixes:
        yield from root.rglob(f"*{suffix}")


def _lint_json(paths: Iterable[Path]) -> List[str]:
    errors: List[str] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            text = path.read_text(encoding="utf-8")
            json.loads(text)
        except Exception as exc:
            errors.append(f"{path}: {exc}")
    return errors


def _lint_jsonl(paths: Iterable[Path]) -> List[str]:
    errors: List[str] = []
    for path in paths:
        if not path.exists():
            continue
        try:
            for idx, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                json.loads(line)
        except Exception as exc:
            errors.append(f"{path}:{idx}: {exc}")
    return errors


def lint_format_check(repo_root: Path) -> int:
    json_targets = [
        repo_root / "pie" / "config",
        repo_root / "examples",
        repo_root / "artifacts" / "golden",
        repo_root / "schemas",
        repo_root / "progetto",
        repo_root / "pie",
    ]
    json_files: List[Path] = []
    jsonl_files: List[Path] = []
    for root in json_targets:
        if not root.exists():
            continue
        json_files.extend(_iter_files(root, [".json"]))
        jsonl_files.extend(_iter_files(root, [".jsonl"]))
    errors = _lint_json(json_files) + _lint_jsonl(jsonl_files)
    if errors:
        print("Lint/format check failed:")
        for error in errors:
            print(error)
        return 1
    print("Lint/format check passed.")
    return 0


def type_check(repo_root: Path) -> int:
    targets = [repo_root / "pie", repo_root / "tests", repo_root / "scripts"]
    errors: List[str] = []
    for root in targets:
        if not root.exists():
            continue
        for path in _iter_files(root, [".py"]):
            try:
                py_compile.compile(str(path), doraise=True)
            except Exception as exc:
                errors.append(f"{path}: {exc}")
    if errors:
        print("Type check failed:")
        for error in errors:
            print(error)
        return 1
    print("Type check passed.")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local CI checks")
    parser.add_argument(
        "check",
        choices=["lint", "typecheck"],
        help="Which check to run",
    )
    args = parser.parse_args(argv)
    repo_root = Path(__file__).resolve().parents[1]
    if args.check == "lint":
        return lint_format_check(repo_root)
    return type_check(repo_root)


if __name__ == "__main__":
    sys.exit(main())
