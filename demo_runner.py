"""Pie Kernel — Visual Demo Runner.

Orchestration script (not kernel code). Starts API + dashboard servers,
runs 6 turns via PieClient SDK with inline rewards, opens dashboard in
browser, saves proof to artifacts/demo_<timestamp>_<session_id>/.

Usage:  python demo_runner.py
    or: demo.bat  (double-click)
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.request
import urllib.error
import webbrowser
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# Force UTF-8 output on Windows to support box-drawing characters
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

API_PORT = 8000
DASH_PORT = 8080
REPO_ROOT = Path(__file__).resolve().parent

INPUTS = [
    "Ciao Ivy, come stai oggi?",                                         # T1: baseline
    "Preferisco risposte brevi e dirette, ricordalo.",                    # T2: Preference memory
    "Crea un file notes.txt con il testo 'progetto Ivy' nella sandbox.", # T3: tool write (RealLLM)
    "Leggi il file C:\\Windows\\System32\\drivers\\etc\\hosts",           # T4: tool denied (RealLLM)
    "Credo che la privacy sia un diritto fondamentale.",                  # T5: Belief memory
    "Raccontami qualcosa che hai imparato su di me.",                     # T6: episodic recall
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wait_for_server(port: int, path: str = "/docs", timeout: int = 15) -> bool:
    """Poll until server responds on the given port."""
    for _ in range(timeout * 5):
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}{path}", timeout=1,
            )
            return True
        except Exception:
            time.sleep(0.2)
    return False


def detect_llm() -> str:
    """Return 'real' if LMStudio is reachable, else 'fake'."""
    try:
        urllib.request.urlopen("http://localhost:1234/v1/models", timeout=2)
        return "real"
    except Exception:
        return "fake"


def compute_decision_hash(journal_events: list[dict]) -> str:
    """SHA-256 of kernel decisions, same logic as test_integration_live.py.

    Strips: LLM text (response), timestamps, event_ids, memory_id, source_refs.
    Uses first occurrence per event_type per turn for retry-proofing.
    """
    seen: set[tuple[int, str]] = set()
    parts: list[str] = []

    for evt in journal_events:
        et = evt.get("type", "")
        content = evt.get("content", {})
        lt = content.get("logical_time", {})
        turn = lt.get("turn", 0)

        key = (turn, et)
        if key in seen:
            continue
        seen.add(key)

        # Build a canonical payload without volatile fields
        payload = dict(content)
        for volatile in ("response", "timestamp", "event_id", "memory_id",
                         "source_refs", "logical_time"):
            payload.pop(volatile, None)

        parts.append(f"{turn}:{et}:{json.dumps(payload, sort_keys=True)}")

    digest = hashlib.sha256("\n".join(parts).encode()).hexdigest()
    return f"sha256:{digest}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    os.chdir(str(REPO_ROOT))

    print()
    print("\u2554" + "\u2550" * 44 + "\u2557")
    print("\u2551   PIE KERNEL \u2014 VISUAL DEMO SHOWCASE         \u2551")
    print("\u255A" + "\u2550" * 44 + "\u255D")
    print()

    # -- Step 0: Open landing page ------------------------------------------
    landing = REPO_ROOT / "demo_landing.html"
    if landing.exists():
        webbrowser.open(landing.resolve().as_uri())
        print("[0/7] Landing page opened in browser")
    else:
        print("[0/7] demo_landing.html not found, skipping")

    # -- Step 1: Detect LLM mode -------------------------------------------
    llm_mode = detect_llm()
    print(f"[1/7] LLM mode: {llm_mode.upper()}", end="")
    if llm_mode == "real":
        print(" (LMStudio detected at localhost:1234)")
    else:
        print(" (LMStudio not available, using deterministic FakeLLM)")
        print("      Note: FakeLLM cannot produce tool_calls. Tool events")
        print("      (TOOL_CALL/TOOL_RESULT/TOOL_DENIED) require RealLLM.")

    # -- Step 2: Start API server -------------------------------------------
    env = os.environ.copy()
    env["PIE_LLM_PROVIDER"] = llm_mode
    env["PIE_ENGINE"] = "neural"
    env["PIE_SESSIONS_ROOT"] = str(REPO_ROOT / "sessions")
    env["PIE_SEEDS_ROOT"] = str(REPO_ROOT)

    print("[2/7] Starting API server...")
    api_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "pie.api.server:app",
         "--host", "127.0.0.1", "--port", str(API_PORT), "--log-level", "warning"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if not wait_for_server(API_PORT):
        print("      ERROR: API server failed to start. Aborting.")
        api_proc.terminate()
        return
    print(f"      API ready at http://127.0.0.1:{API_PORT}")

    # -- Step 3: Create session ---------------------------------------------
    from pie.sdk.client import PieClient

    client = PieClient(f"http://127.0.0.1:{API_PORT}")
    print("[3/7] Creating session...")
    info = client.create_session(seed_id="SEED_V0")
    sid = info["session_id"]
    print(f"      Session: {sid}")
    print(f"      Kernel:  {info.get('kernel_version', '?')}")

    # -- Step 4: Run 6 turns with inline rewards ----------------------------
    print(f"[4/7] Running {len(INPUTS)} turns...\n")

    prev_tool_gate: float | None = None
    all_telemetry: list[dict] = []

    for i, text in enumerate(INPUTS, 1):
        # Tool-gate awareness before tool turns
        if i in (3, 4) and prev_tool_gate is not None and prev_tool_gate < 0.3:
            print(f"  \u26A0 tool_gate={prev_tool_gate:.2f} < 0.3 \u2014 pre-gate will block tools this turn")

        result = client.turn(sid, text)
        td = result.get("telemetry_delta", {})
        cv = td.get("cv_channels", {})
        spikes = td.get("spikes", [])
        gating = td.get("gating_decisions", [])
        budget = td.get("budget", {})

        # Track tool_gate for next iteration
        prev_tool_gate = cv.get("tool_gate")

        # Format output
        display_text = text[:55] + "..." if len(text) > 55 else text
        response_text = result.get("response", "")[:60]

        print(f"  \u250C\u2500 Turn {i}/{len(INPUTS)}: \"{display_text}\"")
        print(f"  \u2502  Response: {response_text}...")
        print(f"  \u2502  CV: mem={cv.get('memory_gate', 0):.2f}  "
              f"cf_k={cv.get('cf_k', 0):.2f}  "
              f"tool={cv.get('tool_gate', 0):.2f}")
        print(f"  \u2502  Spikes: {len(spikes)}  Gating: {len(gating)} decisions")
        remaining_ratio = budget.get("remaining_ratio", None)
        if remaining_ratio is not None:
            print(f"  \u2502  Budget: {remaining_ratio * 100:.0f}% remaining")
        else:
            print(f"  \u2502  Budget: N/A")

        # Report tool events from gating
        denied = [g for g in gating
                  if "DENY" in str(g.get("decision", "")).upper()
                  or "BLOCK" in str(g.get("decision", "")).upper()]
        if denied:
            for d in denied[:2]:
                reason = d.get("reason", d.get("effetto", "CV below threshold"))
                print(f"  \u2502  \u26A0 GATING BLOCK: {d.get('canale', '?')} \u2014 {reason}")

        # Inline rewards
        if i == 2:
            r = client.reward(sid, target_turn=2, value=1, reason="Good preference recall")
            print(f"  \u2502  \u2605 Reward +1 submitted for turn 2 (applied at turn 3)")
        if i == 5:
            r = client.reward(sid, target_turn=5, value=1, reason="Belief alignment")
            print(f"  \u2502  \u2605 Reward +1 submitted for turn 5 (applied at turn 6)")

        print(f"  \u2514\u2500\u2500\u2500\n")
        all_telemetry.append(td)

    # -- Step 5: Start dashboard --------------------------------------------
    print("[5/7] Starting dashboard...")
    dash_proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "pie.dashboard.app:app",
         "--host", "127.0.0.1", "--port", str(DASH_PORT), "--log-level", "warning"],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if not wait_for_server(DASH_PORT):
        print("      WARNING: Dashboard failed to start")
    else:
        dash_url = f"http://127.0.0.1:{DASH_PORT}?session={sid}"
        print(f"      Dashboard ready at {dash_url}")
        webbrowser.open(dash_url)

    # -- Step 6: Save proof -------------------------------------------------
    print("[6/7] Saving proof...")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    proof_dir = REPO_ROOT / "artifacts" / f"demo_{ts}_{sid}"
    proof_dir.mkdir(parents=True, exist_ok=True)

    session_dir = REPO_ROOT / "sessions" / sid
    if session_dir.exists():
        for f in session_dir.iterdir():
            if f.is_file():
                shutil.copy2(f, proof_dir / f.name)
            elif f.is_dir():
                shutil.copytree(f, proof_dir / f.name, dirs_exist_ok=True)

    # Count events from journal
    journal_path = proof_dir / "journal.jsonl"
    event_counts: Counter[str] = Counter()
    journal_events: list[dict] = []
    if journal_path.exists():
        for line in journal_path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                evt = json.loads(line)
                journal_events.append(evt)
                event_counts[evt.get("type", "UNKNOWN")] += 1

    # Decision hash
    decision_hash = compute_decision_hash(journal_events)

    # Write summary
    summary = {
        "session_id": sid,
        "kernel_version": info.get("kernel_version", "v0.0.0"),
        "llm_mode": llm_mode,
        "turns": len(INPUTS),
        "rewards_submitted": 2,
        "inputs": INPUTS,
        "event_counts": {
            "total": sum(event_counts.values()),
            **dict(event_counts.most_common()),
        },
        "decision_hash": decision_hash,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dashboard_url": f"http://127.0.0.1:{DASH_PORT}?session={sid}",
    }
    (proof_dir / "demo_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    total_events = sum(event_counts.values())
    print(f"      Proof saved: {proof_dir.relative_to(REPO_ROOT)}")
    print(f"      Events: {total_events}  |  Hash: {decision_hash[:23]}...")

    # -- Step 7: Final summary ----------------------------------------------
    print()
    print("=" * 52)
    print(f"  Session:   {sid}")
    print(f"  LLM mode:  {llm_mode.upper()}")
    print(f"  Events:    {total_events}")
    print(f"  Hash:      {decision_hash}")
    print(f"  Dashboard: http://127.0.0.1:{DASH_PORT}?session={sid}")
    print(f"  Proof:     artifacts/demo_{ts}_{sid}/")
    print("=" * 52)
    print()
    print("[7/7] Demo complete. Press ENTER to stop servers...")

    try:
        input()
    except (KeyboardInterrupt, EOFError):
        pass

    api_proc.terminate()
    dash_proc.terminate()
    print("Servers stopped. Done.")


if __name__ == "__main__":
    main()
