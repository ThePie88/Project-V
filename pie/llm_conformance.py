"""LLM conformance testing for the Pie Kernel.

This module defines utilities to run a suite of conformance tests
against a language model provider.  A conformance test verifies that
the model's output respects must‑include and must‑not‑include
constraints, stays within a maximum token limit and does not mention
internal concepts like tools, memory or goals.  The suite is defined
in JSON format under ``examples/llm_conformance/cases.json``.  The
results are written to ``artifacts/conformance_<model>.json`` and
include metrics such as the percentage of tests that passed on the
first try, after retries, or only via fallback.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict, Any

from .contracts.speech_plan import SpeechPlan
from .llm import generate_with_policy, DEFAULT_MAX_RETRIES


@dataclass
class TestCase:
    plan: Dict[str, Any]
    must_include: List[str]
    must_not_include: List[str]
    max_tokens: int
    verbosity: int


@dataclass
class TestResult:
    case_index: int
    outcome: str  # "ok", "retry", "fallback"
    attempts: int
    generated: str


def load_cases(path: str) -> List[TestCase]:
    """Load conformance test cases from a JSON file.

    Parameters
    ----------
    path:
        Path to the JSON file containing an array of test case objects.

    Returns
    -------
    list of TestCase
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    cases: List[TestCase] = []
    for item in data:
        cases.append(
            TestCase(
                plan=item["plan"],
                must_include=item.get("must_include", []),
                must_not_include=item.get("must_not_include", []),
                max_tokens=item.get("max_tokens", 50),
                verbosity=item.get("verbosity", 50),
            )
        )
    return cases


def _token_count(text: str) -> int:
    """Approximate token count by counting words.

    This approximation is sufficient for conformance testing.  A more
    accurate tokeniser can be substituted in future versions.
    """
    return len(text.strip().split())


def _check_conformance(text: str, case: TestCase) -> Optional[str]:
    """Check whether the generated text conforms to the test case.

    Parameters
    ----------
    text:
        The generated response.
    case:
        The test case containing constraints.

    Returns
    -------
    None if the text conforms; otherwise a string describing the reason for non‑conformance.
    """
    if not isinstance(text, str) or not text.strip():
        return "invalid format"
    lower = text.lower()
    # Check must_not_include first (case insensitive)
    for forb in case.must_not_include:
        if forb.lower() in lower:
            return f"must_not_include: {forb}"
    # Check must_include: at least one of the listed substrings must appear
    if case.must_include:
        missing = [req for req in case.must_include if req.lower() not in lower]
        if missing:
            return f"missing required: {', '.join(missing)}"
    # Check max tokens
    tokens = _token_count(text)
    if tokens > case.max_tokens:
        return f"too many tokens: {tokens} > {case.max_tokens}"
    return None


def run_conformance(
    provider: str,
    model_name: str,
    cases_path: str = "examples/llm_conformance/cases.json",
    output_dir: str = "artifacts",
    max_retries: int = DEFAULT_MAX_RETRIES,
    use_cache: bool = True,
    record_cache: bool = True,
) -> Dict[str, Any]:
    """Run the LLM conformance suite and write results to disk.

    Parameters
    ----------
    provider:
        The LLM provider to use ("real" or "fake").
    model_name:
        The identifier for the model used; used in naming the output file.
    cases_path:
        Path to the JSON file containing test cases.
    output_dir:
        Directory in which to write the conformance results.
    max_retries:
        Maximum number of attempts to obtain a conformant response before
        falling back.

    Returns
    -------
    dict
        A dictionary summarising the conformance metrics and details.
    """
    cases = load_cases(cases_path)
    results: List[TestResult] = []
    ok_count = 0
    retry_count = 0
    fallback_count = 0
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    for idx, case in enumerate(cases, start=1):
        # Build speech plan for this case
        plan = SpeechPlan(
            plan_id=idx,
            intent=case.plan["intent"],
            arguments=case.plan.get("arguments", {}),
            must_include=case.must_include,
            must_not_include=case.must_not_include,
            max_tokens=case.max_tokens,
            verbosity=case.verbosity,
            facts_allowed=case.plan.get("facts_allowed", []),
            output_format=case.plan.get("output_format", "TEXT"),
        )
        result = generate_with_policy(
            plan,
            provider=provider,
            max_retries=max_retries,
            use_cache=use_cache,
            record_cache=record_cache,
        )
        outcome = result.outcome
        if outcome == "ok":
            ok_count += 1
        elif outcome == "retry":
            retry_count += 1
        else:
            fallback_count += 1
        results.append(
            TestResult(
                case_index=idx,
                outcome=outcome,
                attempts=result.attempts,
                generated=result.text,
            )
        )
    total = len(cases)
    metrics = {
        "total": total,
        "ok": ok_count,
        "retry": retry_count,
        "fallback": fallback_count,
        "ok_percent": round(ok_count / total * 100, 2),
        "retry_percent": round(retry_count / total * 100, 2),
        "fallback_percent": round(fallback_count / total * 100, 2),
    }
    # Serialize results
    summary = {
        "model": model_name,
        "provider": provider,
        "metrics": metrics,
        "results": [asdict(r) for r in results],
    }
    # Write to file
    out_path = os.path.join(output_dir, f"conformance_{model_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary
