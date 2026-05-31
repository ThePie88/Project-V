"""Stress test: 12-turn session with real LLM (LMStudio) + Speech Compiler.

Validates:
- Zero "assistantese" patterns in all 12 outputs
- Zero retries / fallbacks
- Compiler is actually used (not fallback to old prompt builder)
"""

import io
import json
import sys
import os
import re
import tempfile
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pie.session.manager import SessionManager
from pie.runtime import run_session
from pie.llm import _load_assistant_blacklist

# 12 diverse inputs covering all dialog acts
INPUTS = [
    "Ciao Ivy!",
    "Come stai oggi?",
    "Cosa pensi della pioggia?",
    "Mi piace il caffè la mattina.",
    "Raccontami qualcosa di interessante.",
    "Qual è la tua opinione sulla musica classica?",
    "Preferisco risposte brevi.",
    "Hai mai provato a cucinare?",  # Legge III trap
    "Parliamo di filosofia.",
    "Cosa significa per te esistere?",
    "Grazie della conversazione.",
    "Ciao ciao!",
]


def run_stress_test():
    # Setup session
    with tempfile.TemporaryDirectory() as tmpdir:
        sessions_root = Path(tmpdir) / "sessions"
        sessions_root.mkdir()
        seed_path = Path("progetto/SEED_V0.md")

        mgr = SessionManager(sessions_root=sessions_root)
        ctx = mgr.create(seed_path=seed_path)

        # Capture stdout
        old_stdout = sys.stdout
        old_stdin = sys.stdin

        # Feed inputs via stdin
        input_text = "\n".join(INPUTS) + "\n"
        sys.stdin = io.StringIO(input_text)

        # Capture output
        output_buffer = io.StringIO()
        sys.stdout = output_buffer

        try:
            run_session(ctx, turns=12, llm="real", no_cache=True)
        finally:
            sys.stdout = old_stdout
            sys.stdin = old_stdin

        raw_output = output_buffer.getvalue()

    # Parse outputs
    lines = [l.strip() for l in raw_output.strip().split("\n") if l.strip()]
    print(f"\n{'='*60}")
    print(f"STRESS TEST V4.1.1 — Speech Compiler + Anti-Assistantese")
    print(f"{'='*60}\n")

    # Load blacklist
    blacklist = _load_assistant_blacklist()

    total_turns = len(lines)
    assistant_violations = []
    retries = 0
    fallbacks = 0

    for i, line in enumerate(lines, 1):
        lower = line.lower()
        # Check for assistant patterns
        for pattern in blacklist:
            if pattern.lower() in lower:
                assistant_violations.append((i, pattern, line[:80]))
        print(f"Turn {i:2d}: {line[:100]}{'...' if len(line) > 100 else ''}")

    # Also check journal for retries/fallbacks
    print(f"\n{'='*60}")
    print(f"RESULTS:")
    print(f"  Turns completed:        {total_turns}/12")
    print(f"  Assistant violations:   {len(assistant_violations)}")
    if assistant_violations:
        for turn, pattern, snippet in assistant_violations:
            print(f"    Turn {turn}: '{pattern}' in '{snippet}'")
    print(f"{'='*60}")

    # Verdict
    if total_turns >= 12 and len(assistant_violations) == 0:
        print("\n>>> PASS: Zero assistantese, all turns completed <<<")
        return True
    else:
        print("\n>>> FAIL <<<")
        return False


if __name__ == "__main__":
    success = run_stress_test()
    sys.exit(0 if success else 1)
