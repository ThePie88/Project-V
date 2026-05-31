"""Command line interface for the Pie Kernel.

This script uses the built‑in ``argparse`` module instead of Typer
because the Typer dependency is not available in the execution
environment.  Three subcommands are exposed:

* ``run`` – run the kernel either interactively or in exam mode.
* ``validate`` – validate a JSON/JSONL file against the contracts.
* ``replay`` – replay a trace file and verify deterministic decisions.

Run ``python -m pie.cli --help`` for usage information.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path
from typing import List

from .runtime import run, run_session, validate_file, replay, CONFORMANCE_THRESHOLD
from .persistence.constraints_store import ConstraintsStore
from .kernel_manifest import (
    build_manifest,
    canonical_manifest_json,
    load_manifest,
    resolve_artifacts,
    validate_exam_artifacts,
)
from .matrix_runner import MatrixConfig, PRESETS, run_matrix


def _normalize_base_url(base_url: str) -> str:
    cleaned = (base_url or "").strip().rstrip("/")
    if not cleaned:
        return ""
    if cleaned.endswith("/v1"):
        return cleaned
    if "/v1/" in cleaned:
        return cleaned.split("/v1/", 1)[0] + "/v1"
    return f"{cleaned}/v1"


def _get_real_llm_config() -> tuple[dict[str, str] | None, str]:
    base_url = os.environ.get("LM_API_BASE_URL", "").strip()
    if not base_url:
        base_url = os.environ.get("PIE_REAL_BASE_URL", "").strip()
    api_key = os.environ.get("LM_API_KEY", "").strip()
    if not api_key:
        api_key = os.environ.get("PIE_REAL_API_KEY", "").strip()
    model = os.environ.get("LM_API_MODEL", "").strip()
    if not model:
        model = os.environ.get("PIE_REAL_MODEL", "qwen3-vl-30b-a3b-instruct").strip()
    if not base_url:
        return None, "missing_base_url"
    if not api_key:
        return None, "missing_api_key"
    normalized = _normalize_base_url(base_url)
    return {"base_url": normalized, "api_key": api_key, "model": model}, ""


def _check_real_llm_reachable(base_url: str, api_key: str) -> tuple[bool, str]:
    url = _normalize_base_url(base_url).rstrip("/") + "/models"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            status = getattr(resp, "status", 200)
        if status >= 400:
            return False, f"status_{status}"
        return True, "ok"
    except Exception as exc:
        return False, str(exc)


def _load_conformance_summary(output_dir: Path) -> dict[str, object]:
    files = sorted(output_dir.glob("conformance_*.json"))
    if not files:
        raise ValueError("missing conformance report")
    return json.loads(files[0].read_text(encoding="utf-8"))


def _check_conformance(summary: dict[str, object]) -> tuple[bool, str]:
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        return False, "missing metrics"
    total = int(metrics.get("total", 0) or 0)
    ok = int(metrics.get("ok", 0) or 0)
    retry = int(metrics.get("retry", 0) or 0)
    fallback = int(metrics.get("fallback", 0) or 0)
    if total <= 0:
        return False, "empty metrics"
    ok_rate = (ok + retry) / total
    fallback_rate = fallback / total
    fallback_threshold = max(0.0, 1.0 - CONFORMANCE_THRESHOLD)
    if ok_rate < CONFORMANCE_THRESHOLD:
        return False, f"ok_rate {round(ok_rate, 4)} < {CONFORMANCE_THRESHOLD}"
    if fallback_rate > fallback_threshold:
        return False, f"fallback_rate {round(fallback_rate, 4)} > {fallback_threshold}"
    return True, "ok"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pie Kernel CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    # run command
    run_parser = subparsers.add_parser("run", help="Run the kernel")
    run_parser.add_argument(
        "--exam",
        action="store_true",
        help="Execute the fixed exam scenario instead of interactive mode",
    )
    run_parser.add_argument(
        "--turns",
        type=int,
        default=1,
        help="Number of turns to run in interactive mode (ignored in exam mode)",
    )
    run_parser.add_argument(
        "--output-dir",
        default="artifacts",
        help="Directory to write artefacts in exam mode",
    )
    run_parser.add_argument(
        "--llm",
        choices=["fake", "real"],
        default="fake",
        help="Which language model provider to use: 'fake' or 'real'",
    )
    run_parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable LLM cache reads (useful for audit runs)",
    )
    run_parser.add_argument(
        "--engine",
        choices=["ode", "neural"],
        default="ode",
        help="StateEngine backend: 'ode' (default) or 'neural' (V4 Izhikevich+ESN)",
    )
    run_parser.add_argument(
        "--new-session",
        action="store_true",
        help="Create a new persistent Ivy session (V3.0)",
    )
    run_parser.add_argument(
        "--session",
        type=str,
        default=None,
        help="Resume an existing session by ID (V3.0)",
    )
    run_parser.add_argument(
        "--seed-path",
        type=str,
        default="progetto/SEED_V0.md",
        help="Path to SEED identity file for session bootstrap",
    )
    # kill-switch command (V3.3)
    ks_parser = subparsers.add_parser(
        "kill-switch", help="Activate or deactivate the kill-switch (Legge IV)"
    )
    ks_parser.add_argument(
        "action", choices=["on", "off"], help="Activate (on) or deactivate (off)"
    )
    ks_parser.add_argument(
        "--session", type=str, required=True, help="Session ID to apply kill-switch to"
    )
    # validate command
    val_parser = subparsers.add_parser(
        "validate", help="Validate a JSON or JSONL file against contracts"
    )
    val_parser.add_argument("file", help="Path to JSON/JSONL file to validate")
    # replay command
    replay_parser = subparsers.add_parser(
        "replay", help="Replay a trace file and verify deterministic decisions"
    )
    replay_parser.add_argument("trace", help="Path to a trace.jsonl file to replay")
    replay_parser.add_argument(
        "--llm",
        choices=["fake", "real"],
        default="fake",
        help="Provider used when reconstructing LLM output events during replay"
    )
    kernel_parser = subparsers.add_parser(
        "kernel-check", help="Run kernel freeze checks (manifest + exam + replay)"
    )
    kernel_parser.add_argument(
        "--with-real",
        action="store_true",
        help="Include optional real LLM exam check (LM_API_* or PIE_REAL_* required)",
    )
    artifacts_parser = subparsers.add_parser(
        "artifacts-check", help="Validate artifacts against the frozen contract"
    )
    artifacts_group = artifacts_parser.add_mutually_exclusive_group(required=True)
    artifacts_group.add_argument("--run", help="Path to a run directory")
    artifacts_group.add_argument("--golden", help="Path to a golden baseline directory")
    artifacts_parser.add_argument(
        "--accept-schema-bump",
        action="store_true",
        help="Allow schema version mismatch (developer only)",
    )
    ci_parser = subparsers.add_parser(
        "ci-local", help="Run the local CI pipeline in a clean venv"
    )
    ci_parser.add_argument(
        "--venv-dir", default=".ci_venv", help="Path for the temporary venv"
    )
    ci_parser.add_argument(
        "--seed", type=int, help="Seed forwarded to test-matrix runs"
    )
    ci_parser.add_argument(
        "--keep-venv", action="store_true", help="Keep the venv after run"
    )
    matrix_parser = subparsers.add_parser(
        "test-matrix", help="Run the V1 matrix test runner"
    )
    matrix_parser.add_argument(
        "--preset",
        default="offline_fast",
        help=f"Preset name(s), comma-separated. Available: {', '.join(sorted(PRESETS.keys()))}",
    )
    matrix_parser.add_argument(
        "--voice",
        choices=["none", "fake", "real"],
        help="Override voice mode for the run",
    )
    matrix_parser.add_argument(
        "--cache",
        choices=["no-cache", "cache-write", "cache-readonly"],
        help="Cache mode for LLM calls",
    )
    matrix_parser.add_argument(
        "--replay",
        choices=["on", "off"],
        help="Replay mode",
    )
    matrix_parser.add_argument(
        "--golden",
        choices=["on", "off"],
        help="Golden diff mode",
    )
    matrix_parser.add_argument(
        "--crash-test",
        choices=["on", "off"],
        help="Crash simulation mode",
    )
    matrix_parser.add_argument(
        "--policy",
        choices=["valid", "tampered"],
        help="Policy verification mode",
    )
    matrix_parser.add_argument(
        "--tools",
        choices=["none", "sandbox"],
        help="Tools harness mode",
    )
    matrix_parser.add_argument(
        "--rate-limit",
        choices=["on", "off"],
        help="Tool rate limiting mode",
    )
    matrix_parser.add_argument(
        "--seed",
        type=int,
        help="Deterministic seed for test metadata",
    )
    matrix_parser.add_argument(
        "--only",
        help="Comma-separated suite list (exam, replay, conformance, golden, crash, policy, tools, kernel)",
    )
    approve_parser = subparsers.add_parser(
        "approve_constraint", help="Approve a pending constraint"
    )
    approve_parser.add_argument("constraint_id", help="Constraint id to approve")
    approve_parser.add_argument(
        "--path",
        default="artifacts/constraints.jsonl",
        help="Path to constraints.jsonl",
    )
    approve_parser.add_argument(
        "--explanation",
        default="Constraint approved by creator.",
        help="Explanation to store in the decision record",
    )
    reject_parser = subparsers.add_parser(
        "reject_constraint", help="Reject a pending constraint"
    )
    reject_parser.add_argument("constraint_id", help="Constraint id to reject")
    reject_parser.add_argument(
        "--path",
        default="artifacts/constraints.jsonl",
        help="Path to constraints.jsonl",
    )
    reject_parser.add_argument(
        "--explanation",
        default="Constraint rejected by creator.",
        help="Explanation to store in the decision record",
    )
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.new_session or args.session:
            from .session.manager import SessionManager
            mgr = SessionManager()
            if args.new_session:
                ctx = mgr.create(
                    seed_path=Path(args.seed_path),
                    session_id=args.session,
                )
                print(f"Created session: {ctx.session_id}")
            else:
                ctx = mgr.resume(args.session)
                print(f"Resumed session: {ctx.session_id} (turn {ctx.turn_count})")
            run_session(
                session_ctx=ctx,
                turns=args.turns,
                llm=args.llm,
                no_cache=args.no_cache,
                interactive=not args.exam,
            )
        else:
            run(
                exam=args.exam,
                turns=args.turns,
                llm=args.llm,
                output_dir=args.output_dir,
                no_cache=args.no_cache,
                engine=args.engine,
            )
    elif args.command == "kill-switch":
        from .killswitch import KillSwitchState
        session_dir = Path("sessions") / args.session
        if not session_dir.is_dir():
            print(f"Session not found: {args.session}")
            sys.exit(1)
        ks_path = session_dir / "killswitch_report.json"
        ks = KillSwitchState.load(ks_path)
        journal_path = session_dir / "journal.jsonl"
        if args.action == "on":
            ks_content = ks.activate(phase="cli_command")
            from .contracts import Event
            from .persistence.atomic import atomic_append
            # Determine next event_id from journal
            last_eid = 0
            if journal_path.exists():
                for line in journal_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            evt = json.loads(line)
                            eid = evt.get("id", 0)
                            if eid > last_eid:
                                last_eid = eid
                        except json.JSONDecodeError:
                            pass
            event = Event.new(last_eid + 1, "KILL_SWITCH_ON", ks_content)
            atomic_append(journal_path, json.dumps(event.to_json()) + "\n")
            ks.save(ks_path)
            print(f"Kill-switch ATTIVATO per sessione {args.session}.")
        else:
            ks_content = ks.deactivate()
            from .contracts import Event
            from .persistence.atomic import atomic_append
            last_eid = 0
            if journal_path.exists():
                for line in journal_path.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        try:
                            evt = json.loads(line)
                            eid = evt.get("id", 0)
                            if eid > last_eid:
                                last_eid = eid
                        except json.JSONDecodeError:
                            pass
            event = Event.new(last_eid + 1, "KILL_SWITCH_OFF", ks_content)
            atomic_append(journal_path, json.dumps(event.to_json()) + "\n")
            ks.save(ks_path)
            print(f"Kill-switch DISATTIVATO per sessione {args.session}.")
    elif args.command == "validate":
        try:
            validate_file(args.file)
            print("Validation succeeded.")
            sys.exit(0)
        except ValueError as exc:
            print(f"Validation failed: {exc}")
            sys.exit(1)
    elif args.command == "replay":
        replay(args.trace, provider=args.llm)
    elif args.command == "approve_constraint":
        store = ConstraintsStore(args.path)
        record = store.append_decision(
            args.constraint_id,
            "active",
            explanation=args.explanation,
        )
        print(f"Approved constraint {record.constraint_id}.")
    elif args.command == "reject_constraint":
        store = ConstraintsStore(args.path)
        record = store.append_decision(
            args.constraint_id,
            "rejected",
            explanation=args.explanation,
        )
        print(f"Rejected constraint {record.constraint_id}.")
    elif args.command == "kernel-check":
        manifest_path = Path(__file__).with_name("kernel_manifest.json")
        if not manifest_path.exists():
            print(f"Kernel manifest not found: {manifest_path}")
            sys.exit(1)
        golden = load_manifest(manifest_path)
        current = build_manifest()
        if canonical_manifest_json(golden) != canonical_manifest_json(current):
            print("Kernel manifest mismatch: public surface changed.")
            sys.exit(1)
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                run(exam=True, llm="fake", output_dir=tmpdir)
                out_dir = Path(tmpdir)
                validate_exam_artifacts(out_dir, golden, validate_file=validate_file)
                resolved = resolve_artifacts(out_dir, golden)
                trace_path = resolved.get("trace_exam.jsonl", out_dir / "trace_exam.jsonl")
                if not replay(str(trace_path), provider="fake"):
                    print("Kernel replay failed.")
                    sys.exit(1)
            if not args.with_real:
                config, reason = _get_real_llm_config()
                if config is None:
                    print(f"REAL_LLM_CHECK_SKIPPED: {reason}")
                else:
                    print("REAL_LLM_CHECK_SKIPPED: disabled (use --with-real)")
                print("Kernel check passed.")
                return
            config, reason = _get_real_llm_config()
            if config is None:
                print("REAL_LLM_NOT_CONFIGURED")
                sys.exit(1)
            reachable, detail = _check_real_llm_reachable(
                config["base_url"], config["api_key"]
            )
            if not reachable:
                print("REAL_LLM_UNREACHABLE")
                sys.exit(1)
            with tempfile.TemporaryDirectory() as real_tmpdir:
                run(
                    exam=True,
                    llm="real",
                    output_dir=real_tmpdir,
                    no_cache=True,
                )
                real_dir = Path(real_tmpdir)
                validate_exam_artifacts(real_dir, golden, validate_file=validate_file)
                resolved_real = resolve_artifacts(real_dir, golden)
                real_trace = resolved_real.get(
                    "trace_exam.jsonl", real_dir / "trace_exam.jsonl"
                )
                if not replay(str(real_trace), provider="real"):
                    print("REAL_LLM_CHECK_FAILED: replay mismatch")
                    sys.exit(1)
                summary = _load_conformance_summary(real_dir)
                ok, detail = _check_conformance(summary)
                if not ok:
                    print(f"REAL_LLM_CHECK_FAILED: {detail}")
                    sys.exit(1)
        except Exception as exc:
            print(f"Kernel check failed: {exc}")
            sys.exit(1)
        print("Kernel check passed (include real).")
    elif args.command == "artifacts-check":
        from .artifacts_check import validate_run_artifacts, validate_golden_artifacts

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
            sys.exit(1)
        print("Artifacts check passed.")
        sys.exit(0)
    elif args.command == "ci-local":
        from .ci_local import run_ci

        ci_args: List[str] = []
        if args.venv_dir:
            ci_args.extend(["--venv-dir", args.venv_dir])
        if args.seed is not None:
            ci_args.extend(["--seed", str(args.seed)])
        if args.keep_venv:
            ci_args.append("--keep-venv")
        try:
            sys.exit(run_ci(ci_args))
        except Exception as exc:
            print(f"CI failed: {exc}")
            sys.exit(1)
    elif args.command == "test-matrix":
        presets = [item.strip() for item in args.preset.split(",") if item.strip()]
        configs: List[MatrixConfig] = []
        only = [item.strip() for item in (args.only or "").split(",") if item.strip()]
        for preset in presets:
            base = PRESETS.get(preset, {}).copy()
            if not base:
                print(f"Unknown preset: {preset}")
                sys.exit(1)
            config = MatrixConfig(
                preset=preset,
                voice=args.voice or base.get("voice", "fake"),
                cache=args.cache or base.get("cache", "no-cache"),
                replay=args.replay or base.get("replay", "on"),
                golden=args.golden or base.get("golden", "off"),
                crash_test=args.crash_test or base.get("crash_test", "off"),
                policy=args.policy or base.get("policy", "valid"),
                tools=args.tools or base.get("tools", "none"),
                rate_limit=args.rate_limit or base.get("rate_limit", "off"),
                seed=args.seed,
                only=only,
                replay_cache=base.get("replay_cache"),
            )
            configs.append(config)
        golden_dir_env = os.environ.get("PIE_GOLDEN_DIR")
        if golden_dir_env:
            golden_dir = Path(golden_dir_env)
            golden_dir_auto = False
        else:
            golden_dir = Path("artifacts/golden")
            golden_dir_auto = True
        allow_online = os.environ.get("PIE_ENABLE_ONLINE_TESTS", "0") == "1"
        exit_code, summary = run_matrix(
            configs,
            output_root=Path("artifacts"),
            golden_dir=golden_dir if golden_dir else None,
            golden_dir_auto=golden_dir_auto,
            allow_online=allow_online,
        )
        run_states = [run.get("overall") for run in summary.get("runs", [])]
        if run_states and all(state == "SKIP" for state in run_states):
            overall = "SKIP"
        else:
            overall = "PASS" if exit_code == 0 else "FAIL"
        print(f"Matrix result: {overall}")
        print(f"Matrix summary: {summary}")
        sys.exit(exit_code)
    else:
        parser.print_help()


if __name__ == "__main__":
    main(sys.argv[1:])
