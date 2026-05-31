"""Artifacts contract validation for V1 hardening."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .artifacts_contract import ARTIFACTS_SCHEMA_VERSION, load_artifacts_contract


def validate_run_artifacts(run_dir: Path, *, accept_schema_bump: bool = False) -> List[str]:
    errors: List[str] = []
    contract = _load_contract(errors, accept_schema_bump=accept_schema_bump)
    if not contract:
        return errors
    if not run_dir.exists():
        return [f"run_dir_missing: {run_dir}"]
    summary_path = run_dir / "matrix_summary.json"
    summary = _load_json(summary_path, errors)
    if summary is None:
        return errors
    matrix_schema = _load_schema(contract, "matrix_summary.json", errors)
    if matrix_schema:
        _validate_schema(summary, matrix_schema, "matrix_summary.json", errors)
    _check_artifacts_version(summary, "matrix_summary.json", contract, accept_schema_bump, errors)

    suites = summary.get("suites", {})
    allowed = set(contract.get("allowed_suites", []))
    if not isinstance(suites, dict):
        errors.append("matrix_summary.json: suites must be object")
        suites = {}
    for suite in suites.keys():
        if allowed and suite not in allowed:
            errors.append(f"matrix_summary.json: unknown suite '{suite}'")

    required_files = []
    for item in contract.get("run_required_files", {}).get("base", []):
        required_files.append(item)
    for suite in suites.keys():
        required_files.extend(contract.get("run_required_files", {}).get(suite, []))

    _validate_required_files(run_dir, required_files, contract, errors, accept_schema_bump)
    return errors


def validate_golden_artifacts(golden_dir: Path, *, accept_schema_bump: bool = False) -> List[str]:
    errors: List[str] = []
    contract = _load_contract(errors, accept_schema_bump=accept_schema_bump)
    if not contract:
        return errors
    if not golden_dir.exists():
        return [f"golden_dir_missing: {golden_dir}"]
    required_files = contract.get("golden_required_files", [])
    _validate_required_files(golden_dir, required_files, contract, errors, accept_schema_bump)
    meta_path = golden_dir / "golden_meta.json"
    meta = _load_json(meta_path, errors)
    if meta is not None:
        meta_schema = _load_schema(contract, "golden_meta.json", errors)
        if meta_schema:
            _validate_schema(meta, meta_schema, "golden_meta.json", errors)
        _check_artifacts_version(meta, "golden_meta.json", contract, accept_schema_bump, errors)
    return errors


def _load_contract(errors: List[str], *, accept_schema_bump: bool) -> Dict[str, Any]:
    contract = load_artifacts_contract()
    if not isinstance(contract, dict):
        errors.append("artifacts_contract_invalid")
        return {}
    version = contract.get("schema_version")
    if version is None:
        errors.append("artifacts_contract_missing_version")
        return contract
    if version != ARTIFACTS_SCHEMA_VERSION and not accept_schema_bump:
        errors.append(
            f"artifacts_schema_version_mismatch: contract={version} code={ARTIFACTS_SCHEMA_VERSION}"
        )
    return contract


def _check_artifacts_version(
    data: Dict[str, Any],
    label: str,
    contract: Dict[str, Any],
    accept_schema_bump: bool,
    errors: List[str],
) -> None:
    version = data.get("artifacts_schema_version")
    expected = contract.get("schema_version")
    if version is None:
        errors.append(f"{label}: missing artifacts_schema_version")
        return
    if expected and version != expected and not accept_schema_bump:
        errors.append(
            f"{label}: artifacts_schema_version {version} != contract {expected}"
        )


def _validate_required_files(
    root: Path,
    required: Iterable[str],
    contract: Dict[str, Any],
    errors: List[str],
    accept_schema_bump: bool,
) -> None:
    for pattern in required:
        matches = list(root.glob(pattern))
        if not matches:
            errors.append(f"missing_required_file: {pattern}")
            continue
        for path in matches:
            _validate_file(path, contract, errors, accept_schema_bump)


def _validate_file(
    path: Path,
    contract: Dict[str, Any],
    errors: List[str],
    accept_schema_bump: bool,
) -> None:
    if path.suffix == ".md":
        _validate_markdown(path, errors)
        return
    if path.suffix == ".jsonl":
        schema = _schema_for_file(path.name, contract, errors)
        if schema:
            _validate_jsonl(path, schema, errors)
        else:
            _validate_jsonl(path, None, errors)
        return
    if path.suffix == ".json":
        schema = _schema_for_file(path.name, contract, errors)
        data = _load_json(path, errors)
        if data is None:
            return
        if schema:
            _validate_schema(data, schema, path.name, errors)
        return


def _validate_markdown(path: Path, errors: List[str]) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        errors.append(f"{path.name}: {exc}")
        return
    if not lines:
        errors.append(f"{path.name}: empty")
        return
    if not lines[0].lstrip().startswith("#"):
        errors.append(f"{path.name}: missing header")


def _validate_jsonl(path: Path, schema: Dict[str, Any] | None, errors: List[str]) -> None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception as exc:
        errors.append(f"{path.name}: {exc}")
        return
    for idx, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except Exception as exc:
            errors.append(f"{path.name}:{idx}: {exc}")
            continue
        if schema:
            _validate_schema(obj, schema, f"{path.name}:{idx}", errors)


def _load_json(path: Path, errors: List[str]) -> Dict[str, Any] | None:
    if not path.exists():
        errors.append(f"missing_required_file: {path.name}")
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{path.name}: {exc}")
        return None


def _load_schema(
    contract: Dict[str, Any], key: str, errors: List[str]
) -> Dict[str, Any] | None:
    schema_path = contract.get("schemas", {}).get(key)
    if not schema_path:
        return None
    path = Path(__file__).resolve().parents[1] / schema_path
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        errors.append(f"{schema_path}: {exc}")
        return None


def _schema_for_file(
    name: str, contract: Dict[str, Any], errors: List[str]
) -> Dict[str, Any] | None:
    if name.startswith("conformance_") and name.endswith(".json"):
        return _load_schema(contract, "conformance", errors)
    return _load_schema(contract, name, errors)


def _validate_schema(
    data: Any, schema: Dict[str, Any], path: str, errors: List[str]
) -> None:
    expected_type = schema.get("type")
    if expected_type and not _type_matches(data, expected_type):
        errors.append(f"{path}: expected {expected_type}")
        return

    if _type_includes(expected_type, "object"):
        if not isinstance(data, dict):
            errors.append(f"{path}: expected object")
            return
        required = schema.get("required", [])
        for key in required:
            if key not in data:
                errors.append(f"{path}: missing '{key}'")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, value in data.items():
            if key in properties:
                _validate_schema(value, properties[key], f"{path}.{key}", errors)
            else:
                if additional is False:
                    errors.append(f"{path}: unexpected '{key}'")
                elif isinstance(additional, dict):
                    _validate_schema(value, additional, f"{path}.{key}", errors)
        return

    if _type_includes(expected_type, "array"):
        if not isinstance(data, list):
            errors.append(f"{path}: expected array")
            return
        items = schema.get("items")
        if items:
            for idx, item in enumerate(data):
                _validate_schema(item, items, f"{path}[{idx}]", errors)
        return

    if "enum" in schema:
        if data not in schema["enum"]:
            errors.append(f"{path}: value '{data}' not in enum")


def _type_matches(value: Any, expected: Any) -> bool:
    if isinstance(expected, list):
        return any(_type_matches(value, item) for item in expected)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _type_includes(expected: Any, target: str) -> bool:
    if expected is None:
        return False
    if isinstance(expected, list):
        return target in expected
    return expected == target


def main(argv: List[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate artifacts against the frozen contract")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", help="Path to a run directory")
    group.add_argument("--golden", help="Path to a golden baseline directory")
    parser.add_argument(
        "--accept-schema-bump",
        action="store_true",
        help="Allow schema version mismatch (developer only)",
    )
    args = parser.parse_args(argv)
    if args.run:
        errors = validate_run_artifacts(
            Path(args.run), accept_schema_bump=args.accept_schema_bump
        )
    else:
        errors = validate_golden_artifacts(
            Path(args.golden), accept_schema_bump=args.accept_schema_bump
        )
    if errors:
        print("Artifacts check failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("Artifacts check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
