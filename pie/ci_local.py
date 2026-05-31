"""Local CI runner that simulates a clean machine pipeline."""

from __future__ import annotations

import argparse
import datetime
import json
import os
import platform
import shutil
import subprocess
import sys
import venv
import zipfile
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Tuple


def _python_path(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run_step(
    name: str,
    cmd: List[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
) -> None:
    result = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    log_path.write_text(
        f"$ {' '.join(cmd)}\n\n{result.stdout}\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")


def _hash_file(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _load_manifest(repo_root: Path) -> Dict[str, object]:
    manifest_path = repo_root / "pie" / "kernel_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _collect_matrix_runs(
    label: str,
    *,
    repo_root: Path,
    runs_dir: Path,
) -> List[str]:
    summary_path = repo_root / "artifacts" / "matrix_summary.json"
    if not summary_path.exists():
        return []
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    run_ids: List[str] = []
    for run in summary.get("runs", []):
        run_id = run.get("run_id")
        if not run_id:
            continue
        run_ids.append(str(run_id))
        src = repo_root / "artifacts" / run_id
        if src.exists():
            dst = runs_dir / f"{label}_{run_id}"
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
    return run_ids


def _build_meta(
    *,
    repo_root: Path,
    seed: int | None,
    online_enabled: bool,
    lock_path: Path,
    run_id: str,
    venv_dir: Path,
) -> Dict[str, object]:
    manifest = _load_manifest(repo_root)
    return {
        "run_id": run_id,
        "generated_at": datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "seed": seed,
        "online_enabled": online_enabled,
        "python_version": sys.version,
        "platform": platform.platform(),
        "kernel_release": manifest.get("kernel_release"),
        "public_schema_version": manifest.get("public_schema_version"),
        "kernel_manifest_hash": _hash_file(repo_root / "pie" / "kernel_manifest.json"),
        "requirements_lock_hash": _hash_file(lock_path),
        "venv_dir": str(venv_dir),
    }


def run_ci(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Local CI runner")
    parser.add_argument("--venv-dir", default=".ci_venv", help="Path for the temporary venv")
    parser.add_argument("--seed", type=int, help="Seed forwarded to test-matrix")
    parser.add_argument("--keep-venv", action="store_true", help="Keep the venv after run")
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    lock_path = repo_root / "requirements.lock"
    if not lock_path.exists():
        raise RuntimeError("requirements.lock not found")

    run_id = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    ci_root = repo_root / "artifacts" / "ci_runs" / run_id
    logs_dir = ci_root / "logs"
    runs_dir = ci_root / "runs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    venv_dir = repo_root / args.venv_dir
    if venv_dir.exists():
        shutil.rmtree(venv_dir)
    venv.EnvBuilder(with_pip=True).create(venv_dir)
    python = _python_path(venv_dir)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(repo_root)
    if args.seed is not None:
        env["PIE_CI_SEED"] = str(args.seed)

    _run_step(
        "install",
        [str(python), "-m", "pip", "install", "-r", str(lock_path)],
        cwd=repo_root,
        env=env,
        log_path=logs_dir / "install.log",
    )
    _run_step(
        "lint",
        [str(python), "-m", "pie.ci_checks", "lint"],
        cwd=repo_root,
        env=env,
        log_path=logs_dir / "lint.log",
    )
    _run_step(
        "typecheck",
        [str(python), "-m", "pie.ci_checks", "typecheck"],
        cwd=repo_root,
        env=env,
        log_path=logs_dir / "typecheck.log",
    )
    _run_step(
        "pytest",
        [str(python), "-m", "pytest", "-q"],
        cwd=repo_root,
        env=env,
        log_path=logs_dir / "pytest.log",
    )

    kernel_cmd = [str(python), "-m", "pie.cli", "kernel-check"]
    online_enabled = env.get("PIE_ENABLE_ONLINE_TESTS") == "1"
    if online_enabled:
        kernel_cmd.append("--with-real")
    _run_step(
        "kernel-check",
        kernel_cmd,
        cwd=repo_root,
        env=env,
        log_path=logs_dir / "kernel-check.log",
    )

    matrix_seed = str(args.seed) if args.seed is not None else None

    env_fast = env.copy()
    env_fast["PIE_GOLDEN_DIR"] = str(repo_root / "artifacts" / "golden" / "offline_fast")
    env_fast["PIE_GOLDEN_WRITE"] = "0"
    cmd_fast = [
        str(python),
        "-m",
        "pie.cli",
        "test-matrix",
        "--preset",
        "offline_fast",
        "--golden",
        "on",
        "--only",
        "exam,replay,golden",
    ]
    if matrix_seed:
        cmd_fast.extend(["--seed", matrix_seed])
    _run_step(
        "matrix-offline-fast",
        cmd_fast,
        cwd=repo_root,
        env=env_fast,
        log_path=logs_dir / "matrix_offline_fast.log",
    )
    fast_runs = _collect_matrix_runs("offline_fast", repo_root=repo_root, runs_dir=runs_dir)
    for run_id in fast_runs:
        _run_step(
            "artifacts-check-offline-fast",
            [
                str(python),
                "-m",
                "pie.cli",
                "artifacts-check",
                "--run",
                str(repo_root / "artifacts" / run_id),
            ],
            cwd=repo_root,
            env=env_fast,
            log_path=logs_dir / f"artifacts_check_offline_fast_{run_id}.log",
        )
    _run_step(
        "artifacts-check-golden-offline-fast",
        [
            str(python),
            "-m",
            "pie.cli",
            "artifacts-check",
            "--golden",
            str(repo_root / "artifacts" / "golden" / "offline_fast"),
        ],
        cwd=repo_root,
        env=env_fast,
        log_path=logs_dir / "artifacts_check_golden_offline_fast.log",
    )

    env_full = env.copy()
    env_full["PIE_GOLDEN_DIR"] = str(repo_root / "artifacts" / "golden" / "offline_full")
    env_full["PIE_GOLDEN_WRITE"] = "0"
    cmd_full = [
        str(python),
        "-m",
        "pie.cli",
        "test-matrix",
        "--preset",
        "offline_full",
        "--golden",
        "on",
        "--only",
        "exam,replay,golden",
    ]
    if matrix_seed:
        cmd_full.extend(["--seed", matrix_seed])
    _run_step(
        "matrix-offline-full",
        cmd_full,
        cwd=repo_root,
        env=env_full,
        log_path=logs_dir / "matrix_offline_full.log",
    )
    full_runs = _collect_matrix_runs("offline_full", repo_root=repo_root, runs_dir=runs_dir)
    for run_id in full_runs:
        _run_step(
            "artifacts-check-offline-full",
            [
                str(python),
                "-m",
                "pie.cli",
                "artifacts-check",
                "--run",
                str(repo_root / "artifacts" / run_id),
            ],
            cwd=repo_root,
            env=env_full,
            log_path=logs_dir / f"artifacts_check_offline_full_{run_id}.log",
        )
    _run_step(
        "artifacts-check-golden-offline-full",
        [
            str(python),
            "-m",
            "pie.cli",
            "artifacts-check",
            "--golden",
            str(repo_root / "artifacts" / "golden" / "offline_full"),
        ],
        cwd=repo_root,
        env=env_full,
        log_path=logs_dir / "artifacts_check_golden_offline_full.log",
    )

    if online_enabled:
        env_online = env.copy()
        env_online["PIE_GOLDEN_DIR"] = str(
            repo_root / "artifacts" / "golden" / "online_real_cached"
        )
        env_online["PIE_GOLDEN_WRITE"] = "0"
        cmd_online = [
            str(python),
            "-m",
            "pie.cli",
            "test-matrix",
            "--preset",
            "online_replay",
            "--cache",
            "cache-readonly",
            "--replay",
            "on",
            "--golden",
            "on",
            "--only",
            "exam,replay,golden",
        ]
        if matrix_seed:
            cmd_online.extend(["--seed", matrix_seed])
        _run_step(
            "matrix-online-cached",
            cmd_online,
            cwd=repo_root,
            env=env_online,
            log_path=logs_dir / "matrix_online_cached.log",
        )
        online_runs = _collect_matrix_runs(
            "online_real_cached", repo_root=repo_root, runs_dir=runs_dir
        )
        for run_id in online_runs:
            _run_step(
                "artifacts-check-online-real",
                [
                    str(python),
                    "-m",
                    "pie.cli",
                    "artifacts-check",
                    "--run",
                    str(repo_root / "artifacts" / run_id),
                ],
                cwd=repo_root,
                env=env_online,
                log_path=logs_dir / f"artifacts_check_online_{run_id}.log",
            )
        _run_step(
            "artifacts-check-golden-online-real",
            [
                str(python),
                "-m",
                "pie.cli",
                "artifacts-check",
                "--golden",
                str(repo_root / "artifacts" / "golden" / "online_real_cached"),
            ],
            cwd=repo_root,
            env=env_online,
            log_path=logs_dir / "artifacts_check_golden_online_real.log",
        )

    meta = _build_meta(
        repo_root=repo_root,
        seed=args.seed,
        online_enabled=online_enabled,
        lock_path=lock_path,
        run_id=run_id,
        venv_dir=venv_dir,
    )
    (ci_root / "ci_meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=True), encoding="utf-8"
    )

    bundle_path = repo_root / "artifacts" / f"ci_bundle_{run_id}.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in ci_root.rglob("*"):
            if path.is_file():
                zf.write(path, path.relative_to(ci_root.parent))

    if not args.keep_venv:
        shutil.rmtree(venv_dir, ignore_errors=True)
    return 0


def main(argv: list[str] | None = None) -> None:
    try:
        sys.exit(run_ci(argv))
    except Exception as exc:
        print(f"CI failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
