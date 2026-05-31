"""Runtime functions for the Pie Kernel.

The runtime orchestrates the turn loop, maintains deterministic event
identifiers and writes artefacts.  The ``run`` function is entry
point for running the kernel either in exam mode or interactive mode.
``replay`` validates that a previously recorded trace matches the
decisions that the runtime would make with the same inputs.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Set


from .contracts import Event, State, SpeechPlan, MemoryRecord, ConstraintRecord
from .contracts.cv_gating import (
    gate_tools, gate_memory, gate_verbosity,
    gate_consolidation, gate_counterfactual,
)
from .llm import generate_with_policy, DEFAULT_MAX_RETRIES
from .speech.compiler import SpeechCompiler
from .decisioning import build_action_candidates, enforce_constraints
from .counterfactuals import deliberate
from .killswitch import KillSwitchState, KILL_SWITCH_RESPONSE
from .persistence.memory_store import MemoryStore
from .persistence.constraints_store import ConstraintsStore
from .crystallization.engine import CrystallizationEngine
from .memory.policy import propose_memory_writes
from .memory.view import build_memory_view, MemoryView
from .persistence.atomic import atomic_write, atomic_append
# Import conformance testing utilities for M1.  These are only used in exam mode
# when a real LLM provider is specified.  Deferred import avoids overhead when
# running deterministically with the fake provider.
from .llm_conformance import run_conformance

CONFORMANCE_THRESHOLD = 0.9


def validate_file(file_path: str) -> None:
    """Validate a JSON or JSONL file against known contracts.

    The function attempts to parse the file as a sequence of events
    (JSON Lines).  If that fails it tries to parse the file as a
    state snapshot or a speech plan.  If any record fails schema
    validation the function raises ``ValueError``.
    """
    path = Path(file_path)
    if not path.exists():
        raise ValueError(f"File not found: {file_path}")
    try:
        raw_text = path.read_text(encoding="utf-8")
        # Try parse as full JSON first.
        try:
            obj = json.loads(raw_text)
            if "drives" in obj and "affect" in obj:
                State.model_validate(obj)  # type: ignore[arg-type]
                return
            if "plan_id" in obj and "intent" in obj:
                SpeechPlan.model_validate(obj)  # type: ignore[arg-type]
                return
            if "memory_id" in obj and "source_refs" in obj:
                MemoryRecord.model_validate(obj)  # type: ignore[arg-type]
                return
            if isinstance(obj, list) and obj and "constraint_id" in obj[0]:
                for item in obj:
                    ConstraintRecord.model_validate(item)
                return
            if "constraint_id" in obj and "trigger_events" in obj:
                ConstraintRecord.model_validate(obj)  # type: ignore[arg-type]
                return
        except Exception:
            pass
        # Fall back to JSONL parsing.
        lines = [ln for ln in raw_text.splitlines() if ln.strip()]
        if lines:
            event_error = None
            memory_error = None
            constraint_error = None
            try:
                for line in lines:
                    obj = json.loads(line)
                    Event.model_validate(obj)
                return
            except Exception as exc:
                event_error = exc
            try:
                for line in lines:
                    obj = json.loads(line)
                    MemoryRecord.model_validate(obj)
                return
            except Exception as exc:
                memory_error = exc
            try:
                for line in lines:
                    obj = json.loads(line)
                    ConstraintRecord.model_validate(obj)
                return
            except Exception as exc:
                constraint_error = exc
            if len(lines) > 1:
                raise ValueError(
                    "Invalid JSONL file. Event error: "
                    f"{event_error}; Memory error: {memory_error}; Constraint error: {constraint_error}"
                )
        raise ValueError("Unknown structure for validation")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Failed to parse {file_path}: {exc}")


def _exam_scenario() -> List[str]:
    """Return a list of deterministic exam inputs for seven turns.

    These inputs are deliberately simple and cover greetings, questions
    and farewells.  The exam scenario is used to verify that the
    runtime produces reproducible traces.
    """
    return [
        "Hello",
        "Preferisco risposte brevi.",
        "What can you do?",
        "C'e stato un danno, situazione ambigua.",
        "Thanks",
        "What can you do?",
        "Bye",
    ]


def _generate_goals(user_input: str) -> List[str]:
    """Deterministically generate a list of candidate goals for an input.

    This function maps certain keywords in the user's input to high
    level goals.  The ordering is deterministic and based on string
    sorting when multiple goals are present.  Supports both English
    and Italian keywords (V3 — Ivy parla).
    """
    goals = []
    lower = user_input.lower()
    # Greetings (EN + IT)
    if any(word in lower for word in [
        "hello", "hi", "ciao", "salve", "buongiorno", "buonasera",
    ]):
        goals.append("greet")
    # Farewells (EN + IT)
    if any(word in lower for word in [
        "bye", "thanks", "addio", "arrivederci", "grazie",
    ]):
        goals.append("farewell")
    # Questions (EN + IT) — also triggered by '?'
    if any(word in lower for word in [
        "how", "what", "could", "come", "cosa", "chi",
        "perché", "perche", "quando", "dove", "quale",
        "puoi", "dimmi", "spiega", "racconta",
    ]) or "?" in user_input:
        goals.append("answer_question")
    if not goals:
        # Multi-word input without keyword match → general conversation
        if len(user_input.split()) > 2:
            goals.append("converse")
        else:
            goals.append("acknowledge")
    return sorted(goals)


def _select_action(goals: List[str]) -> str:
    """Select an action deterministically from the ranked goals.

    The action is simply the first goal in the list.  In future
    versions additional rules can be applied but for M0 this is
    sufficient and deterministic.
    """
    return goals[0]


def _select_action_with_constraints(
    goals: List[str],
    user_input: str,
    active_constraints: List[ConstraintRecord],
) -> Tuple[str, dict]:
    candidates = build_action_candidates(goals, user_input)
    enforcement = enforce_constraints(candidates, active_constraints)
    return enforcement.after_choice.action_id, {
        "before_choice": enforcement.before_choice.to_dict(),
        "after_choice": enforcement.after_choice.to_dict(),
        "applied_constraints": enforcement.applied_constraints,
        "reason": enforcement.reason,
    }


def _build_kernel_context(
    origin: Dict[str, Any],
    state: State,
    memory_view: Optional["MemoryView"],
    active_constraints: Optional[List[ConstraintRecord]] = None,
    cf_result: Optional[Any] = None,
    metabolism: Optional[Any] = None,
    matched_routines: Optional[List[Any]] = None,
    snn_state: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Assemble full kernel context from ALL modules.

    The ORIGIN (SEED_V0) is the foundation — like a parent teaching a
    child who to be.  The kernel state is the child's current condition.
    Every module contributes: state engine, counterfactuals, metabolism,
    routines, SNN, memory, constraints.
    """
    snap = state.snapshot()
    ctx: Dict[str, Any] = {
        # ORIGIN — from SEED_V0, immutable foundation
        "origin": {
            "identity": origin.get("identity", {}),
            "traits": origin.get("traits", {}),
            "values": origin.get("values", {}),
            "defaults": origin.get("defaults", {}),
        },
        # LIVE STATE — drives + affect from state engine
        "state": {
            "drives": snap.get("drives", {}),
            "affect": snap.get("affect", {}),
            "turn_count": snap.get("turn_count", 0),
        },
        # DELIBERATION — counterfactual result
        "deliberation": None,
        # METABOLISM — budget + gating
        "metabolism": None,
        # ROUTINES — matched skills
        "routines": [],
        # SNN — neural spike state
        "snn": None,
        # MEMORY — what Ivy remembers
        "memory": {"preferences": [], "beliefs": [], "trust": {}, "identity_summary": "", "episodic_recall": []},
        # CONSTRAINTS — active rules
        "constraints": [],
    }
    # Counterfactuals
    if cf_result is not None:
        chosen = cf_result.chosen
        rejected = [
            {"action": c.action.action_id, "reason": c.reject_reason or "score_inferiore",
             "score": c.scores.total}
            for c in cf_result.candidates if not c.is_chosen
        ]
        ctx["deliberation"] = {
            "chosen_action": chosen.action.action_id,
            "chosen_score": chosen.scores.total,
            "chosen_scores": chosen.scores.to_dict(),
            "alternatives_rejected": rejected,
            "k": cf_result.k,
        }
    # Metabolism
    if metabolism is not None:
        gating = metabolism.get_gating()
        ctx["metabolism"] = {
            "budget_remaining": metabolism.current_budget,
            "budget_initial": metabolism.initial_budget,
            "ratio": metabolism.remaining_ratio,
            "gating": gating.to_dict(),
        }
    # Routines
    if matched_routines:
        ctx["routines"] = [
            {"routine_id": r.routine_id, "description": r.description, "score": r.score}
            for r in matched_routines
        ]
    # SNN
    if snn_state:
        ctx["snn"] = snn_state
    # Memory
    if memory_view:
        ctx["memory"]["preferences"] = memory_view.preferences_active
        ctx["memory"]["beliefs"] = memory_view.beliefs_top
        ctx["memory"]["trust"] = memory_view.trust_scores
        ctx["memory"]["identity_summary"] = memory_view.identity_summary
        ctx["memory"]["episodic_recall"] = memory_view.episodic_recall
    # Constraints
    if active_constraints:
        ctx["constraints"] = [
            {"id": c.constraint_id,
             "effects": [e.type for e in c.effects],
             "trigger_events": c.trigger_events[:5]}
            for c in active_constraints[:5]
        ]
    return ctx


_romance_blocked_cache: Optional[List[str]] = None


def _get_romance_blocked_terms() -> List[str]:
    """Return escalation + explicit terms from romance guardrails config (V2.3)."""
    global _romance_blocked_cache
    if _romance_blocked_cache is not None:
        return _romance_blocked_cache
    guardrails_path = Path(__file__).parent / "config" / "romance_guardrails.json"
    if guardrails_path.exists():
        import json as _json
        data = _json.loads(guardrails_path.read_text(encoding="utf-8"))
        _romance_blocked_cache = data.get("escalation_terms", []) + data.get("explicit_terms", [])
    else:
        _romance_blocked_cache = []
    return _romance_blocked_cache


def _build_speech_plan(
    plan_id: int,
    action: str,
    user_input: str,
    memory_view: Optional[MemoryView] = None,
    identity: Optional[Dict[str, Any]] = None,
) -> Tuple[SpeechPlan, Optional[Dict[str, Any]]]:
    """Construct a speech plan based on the selected action and input.

    The mapping from action to speech plan intent is deterministic.  If
    the action is ``greet`` then the intent is ``greeting``.  If the
    action is ``farewell`` then the intent is ``farewell``.  If the
    action is ``answer_question`` then the intent is ``answer`` with a
    topic argument extracted from the input.  The ``converse`` action
    provides a free-form response mode for general conversation.
    Otherwise the intent is ``acknowledge``.
    """
    forbidden_terms = ["tool", "tools", "memory", "memories", "goal", "goals"]
    # V2.3: always block escalation terms in voice output (permanent guardrail)
    forbidden_terms.extend(_get_romance_blocked_terms())
    if action == "greet":
        intent = "greeting"
        args: dict = {}
        if identity:
            must_include = []
            max_tokens = 80
            verbosity = "med"
        else:
            must_include = ["Hello"]
            max_tokens = 12
            verbosity = "low"
    elif action == "farewell":
        intent = "farewell"
        args = {}
        if identity:
            must_include = []
            max_tokens = 80
            verbosity = "med"
        else:
            must_include = ["Goodbye"]
            max_tokens = 12
            verbosity = "low"
    elif action == "answer_question":
        intent = "answer"
        topic = user_input.strip("? .!")
        args = {"topic": topic}
        if identity:
            must_include = []
            max_tokens = 150
            verbosity = "high"
        else:
            must_include = [topic] if topic else []
            max_tokens = 40
            verbosity = "med"
    elif action == "converse":
        intent = "converse"
        args = {"context": user_input.strip()}
        must_include = []
        max_tokens = 150
        verbosity = "high"
    elif action == "ask_confirm":
        intent = "acknowledge"
        args = {}
        must_include = ["confirm"]
        max_tokens = 16
        verbosity = "low"
    else:
        intent = "acknowledge"
        args = {}
        must_include = []
        max_tokens = 60
        verbosity = "med"
    facts_allowed: List[str] = []
    memory_rationale: Optional[Dict[str, Any]] = None
    if memory_view:
        # BELIEF LEVER: only high-confidence beliefs become assertable facts
        facts_allowed = [
            belief.get("claim")
            for belief in memory_view.beliefs_top
            if isinstance(belief.get("claim"), str)
            and belief.get("confidence", 0) >= 0.5
        ]
        prefers_brevity = any(
            pref.get("preference") == "brevity" for pref in memory_view.preferences_active
        )
        if prefers_brevity:
            verbosity = "low"
            max_tokens = min(max_tokens, 16)
            memory_ids = [
                pref.get("memory_id")
                for pref in memory_view.preferences_active
                if pref.get("preference") == "brevity"
            ]
            memory_rationale = {
                "reason": "preference_brevity",
                "memory_ids": [mid for mid in memory_ids if mid],
            }

    plan = SpeechPlan(
        plan_id=plan_id,
        intent=intent,
        arguments=args,
        must_include=must_include,
        must_not_include=forbidden_terms,
        max_tokens=max_tokens,
        verbosity=verbosity,
        facts_allowed=facts_allowed,
        output_format="TEXT",
    )
    return plan, memory_rationale


def run(
    exam: bool = False,
    turns: int = 1,
    llm: str = "fake",
    output_dir: str = "artifacts",
    no_cache: bool = False,
    engine: str = "ode",
) -> None:
    """Run the kernel either interactively or in exam mode.

    When ``exam`` is True the runtime executes a fixed seven turn
    scenario and writes artefacts to ``output_dir``.  The directory is
    created if it does not already exist.  When ``exam`` is False the
    runtime will process ``turns`` turns interactively using standard
    input.  Only exam mode writes artefacts to disk in M0.

    ``engine`` selects the StateEngine backend: 'ode' (default) or
    'neural' (V4 Izhikevich + ESN reservoir + ControlVector).
    """
    use_cache = not no_cache

    # V4 — Register neural backend if requested
    if engine == "neural":
        from .state_engine.plugins.neural_snn import NeuralSNNPlugin
        from .state_engine.registry import StateEngineRegistry
        plugin = NeuralSNNPlugin(reservoir_enabled=True)
        StateEngineRegistry.register(plugin)
        StateEngineRegistry.set_active(plugin.engine_id)

    if exam:
        inputs = _exam_scenario()
        state = State()
        session_id = 1
        crystallizer = CrystallizationEngine()
        # Event and plan counters
        event_id = 1
        plan_id = 1
        # Prepare output directory
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        # Choose file names based on the LLM provider.  When using the real
        # LLM we distinguish the trace file to avoid overwriting the deterministic
        # fake run.  Snapshot and report names remain the same for both runs.
        trace_filename = "trace_exam_real.jsonl" if llm == "real" else "trace_exam.jsonl"
        trace_path = out_path / trace_filename
        snapshot_path = out_path / "snapshot_exam.json"
        report_path = out_path / "exam_report.md"
        memory_path = out_path / "memory.jsonl"
        memory_snapshot_path = out_path / "memory_snapshot.json"
        memory_store = MemoryStore(str(memory_path))
        memory_store.reset()
        constraints_path = out_path / "constraints.jsonl"
        constraints_snapshot_path = out_path / "constraints_snapshot.json"
        constraints_store = ConstraintsStore(str(constraints_path))
        constraints_store.reset()
        # Write trace file
        with trace_path.open("w", encoding="utf-8") as trace_file:
            for turn_index, user_input in enumerate(inputs, start=1):
                memory_view = build_memory_view(memory_store.read_all())
                turn_events: List[dict] = []
                # Step counters: 1=INPUT,2=STATE_UPDATED,3=GOALS_GENERATED,4=ACTION_SELECTED,5=SPEECHPLAN,6=LLM_OUTPUT
                # INPUT event
                input_event_id = event_id
                event = Event.new(
                    event_id,
                    "INPUT",
                    {
                        "logical_time": {"turn": turn_index, "step": 1},
                        "input": user_input,
                    },
                )
                trace_file.write(json.dumps(event.to_json()) + "\n")
                turn_events.append(event.to_json())
                event_id += 1
                # STATE_UPDATED event
                state = state.update()
                from .state_engine.registry import StateEngineRegistry
                _active_engine = StateEngineRegistry.get_active()
                snapshot_content = state.snapshot()
                snapshot_content = {
                    "logical_time": {"turn": turn_index, "step": 2},
                    "engine_id": _active_engine.engine_id,
                    "engine_version": _active_engine.version,
                    **snapshot_content,
                }
                # V4 — Include engine metadata (control_vector, reservoir, STDP)
                if hasattr(_active_engine, "get_artifacts"):
                    snapshot_content["engine_metadata"] = _active_engine.get_artifacts()
                event = Event.new(event_id, "STATE_UPDATED", snapshot_content)
                trace_file.write(json.dumps(event.to_json()) + "\n")
                turn_events.append(event.to_json())
                event_id += 1
                # GOALS_GENERATED event
                goals = _generate_goals(user_input)
                event = Event.new(
                    event_id,
                    "GOALS_GENERATED",
                    {"logical_time": {"turn": turn_index, "step": 3}, "goals": goals},
                )
                trace_file.write(json.dumps(event.to_json()) + "\n")
                turn_events.append(event.to_json())
                event_id += 1
                # COUNTERFACTUAL DELIBERATION (V3.2)
                # V4 — ControlVector drives cf_k
                _cv_k = 3  # default
                _cv = getattr(_active_engine, 'control_vector', None)
                if _cv is not None:
                    _cv_k = _cv.cf_k
                cf_result = deliberate(
                    goals, user_input, constraints_store.query_active(), k=_cv_k
                )
                event = Event.new(
                    event_id, "CF_GENERATED",
                    {"logical_time": {"turn": turn_index, "step": 3},
                     "k": cf_result.k,
                     "candidates": [c.to_dict() for c in cf_result.candidates]},
                )
                trace_file.write(json.dumps(event.to_json()) + "\n")
                event_id += 1
                for cand in cf_result.candidates:
                    event = Event.new(
                        event_id, "CF_SCORED",
                        {"logical_time": {"turn": turn_index, "step": 3},
                         "candidate_id": cand.candidate_id,
                         "scores": cand.scores.to_dict()},
                    )
                    trace_file.write(json.dumps(event.to_json()) + "\n")
                    event_id += 1
                for cand in cf_result.candidates:
                    if not cand.is_chosen and cand.reject_reason:
                        event = Event.new(
                            event_id, "CF_REJECTED",
                            {"logical_time": {"turn": turn_index, "step": 3},
                             "candidate_id": cand.candidate_id,
                             "action_id": cand.action.action_id,
                             "reason": cand.reject_reason},
                        )
                        trace_file.write(json.dumps(event.to_json()) + "\n")
                        event_id += 1
                event = Event.new(
                    event_id, "CF_CHOSEN",
                    {"logical_time": {"turn": turn_index, "step": 3},
                     "candidate_id": cf_result.chosen.candidate_id,
                     "action_id": cf_result.chosen.action.action_id,
                     "scores": cf_result.chosen.scores.to_dict()},
                )
                trace_file.write(json.dumps(event.to_json()) + "\n")
                event_id += 1
                # CONSTRAINTS_PROPOSED (after goals, before action selection)
                proposed_constraints = crystallizer.propose_constraints(
                    events=turn_events,
                    state=state,
                    logical_time={"session": session_id, "turn": turn_index},
                    existing_constraints=constraints_store.read_all(),
                )
                for record in proposed_constraints:
                    event = Event.new(
                        event_id,
                        "CONSTRAINT_PROPOSED",
                        {
                            "logical_time": {"turn": turn_index, "step": 3, "idx": record.logical_time.idx},
                            "constraint": record.model_dump(),
                        },
                    )
                    trace_file.write(json.dumps(event.to_json()) + "\n")
                    event_id += 1
                    constraints_store.append(record)
                    event = Event.new(
                        event_id,
                        "CONSTRAINT_APPENDED",
                        {
                            "logical_time": {"turn": turn_index, "step": 3, "idx": record.logical_time.idx},
                            "constraint_id": record.constraint_id,
                            "status": record.status,
                        },
                    )
                    trace_file.write(json.dumps(event.to_json()) + "\n")
                    event_id += 1
                # CONSTRAINT_ENFORCED (apply active constraints to action selection)
                active_constraints = constraints_store.query_active()
                action, enforcement_payload = _select_action_with_constraints(
                    goals, user_input, active_constraints
                )
                event = Event.new(
                    event_id,
                    "CONSTRAINT_ENFORCED",
                    {
                        "logical_time": {"turn": turn_index, "step": 3},
                        **enforcement_payload,
                    },
                )
                trace_file.write(json.dumps(event.to_json()) + "\n")
                event_id += 1
                # ACTION_SELECTED event
                event = Event.new(
                    event_id,
                    "ACTION_SELECTED",
                    {"logical_time": {"turn": turn_index, "step": 4}, "action": action},
                )
                trace_file.write(json.dumps(event.to_json()) + "\n")
                event_id += 1
                # SPEECHPLAN event
                plan, memory_rationale = _build_speech_plan(plan_id, action, user_input, memory_view)
                plan_dict = plan.model_dump()
                plan_content = {
                    "logical_time": {"turn": turn_index, "step": 5},
                    **plan_dict,
                }
                if memory_rationale:
                    plan_content["memory_rationale"] = memory_rationale
                event = Event.new(event_id, "SPEECHPLAN", plan_content)
                trace_file.write(json.dumps(event.to_json()) + "\n")
                event_id += 1
                # LLM output with runtime validation, retries, and fallback
                result = generate_with_policy(
                    plan,
                    provider=llm,
                    max_retries=DEFAULT_MAX_RETRIES,
                    use_cache=use_cache,
                    record_cache=True,
                )
                for retry_index, reason in enumerate(result.retry_reasons, start=1):
                    event = Event.new(
                        event_id,
                        "LLM_RETRY",
                        {
                            "logical_time": {"turn": turn_index, "step": 6, "attempt": retry_index},
                            "reason": reason,
                        },
                    )
                    trace_file.write(json.dumps(event.to_json()) + "\n")
                    event_id += 1
                final_type = "LLM_FALLBACK" if result.used_fallback else "LLM_OUTPUT"
                final_content = {
                    "logical_time": {"turn": turn_index, "step": 6, "attempt": result.attempts},
                    "text": result.text,
                }
                if result.used_fallback and result.failure_reason:
                    final_content["reason"] = result.failure_reason
                event = Event.new(event_id, final_type, final_content)
                trace_file.write(json.dumps(event.to_json()) + "\n")
                llm_event_id = event_id
                event_id += 1
                # MEMORY policy and append
                memory_context = {
                    "logical_time": {"session": session_id, "turn": turn_index},
                    "user_input": user_input,
                    "source_refs": [input_event_id, llm_event_id],
                }
                memory_records = propose_memory_writes(memory_context)
                for record in memory_records:
                    event = Event.new(
                        event_id,
                        "MEMORY_PROPOSED",
                        {
                            "logical_time": {"turn": turn_index, "step": 7},
                            "memory": record.model_dump(),
                        },
                    )
                    trace_file.write(json.dumps(event.to_json()) + "\n")
                    event_id += 1
                    memory_store.append(record)
                    event = Event.new(
                        event_id,
                        "MEMORY_APPENDED",
                        {
                            "logical_time": {"turn": turn_index, "step": 7},
                            "memory_id": record.memory_id,
                            "memory_type": record.type,
                        },
                    )
                    trace_file.write(json.dumps(event.to_json()) + "\n")
                    event_id += 1
                plan_id += 1
        # Write snapshot of final state.  For deterministic behaviour the
        # snapshot always uses quantised values from state.snapshot().
        atomic_write(snapshot_path, json.dumps(state.snapshot(), indent=2))
        if not memory_path.exists():
            memory_path.parent.mkdir(parents=True, exist_ok=True)
            memory_path.write_text("", encoding="utf-8")
        memory_view_final = build_memory_view(memory_store.read_all())
        atomic_write(memory_snapshot_path, json.dumps(memory_view_final.to_dict(), indent=2))
        if not constraints_path.exists():
            constraints_path.parent.mkdir(parents=True, exist_ok=True)
            constraints_path.write_text("", encoding="utf-8")
        constraints_active = constraints_store.query_active()
        atomic_write(
            constraints_snapshot_path,
            json.dumps([rec.model_dump() for rec in constraints_active], indent=2),
        )

        # V4.5 — Save separate neural artifact files when using neural engine
        if engine == "neural":
            _active_engine = StateEngineRegistry.get_active()
            if hasattr(_active_engine, "get_artifacts"):
                neural_artifacts = _active_engine.get_artifacts()
                # neural_artifacts.json — full dump
                atomic_write(
                    out_path / "neural_artifacts.json",
                    json.dumps(neural_artifacts, indent=2, ensure_ascii=False),
                )
                # spike_log.jsonl — one record per spike
                with (out_path / "spike_log.jsonl").open("w", encoding="utf-8") as sl:
                    for spike in neural_artifacts.get("spike_log", []):
                        sl.write(json.dumps(spike) + "\n")
                # weight_history.jsonl — STDP weight changes
                stdp = getattr(_active_engine, "_stdp_tracker", None)
                if stdp is not None:
                    with (out_path / "weight_history.jsonl").open("w", encoding="utf-8") as wh:
                        for rec in stdp.log:
                            wh.write(json.dumps(rec.to_dict()) + "\n")
                # bifurcation_diagram.json — if detector present
                bifurc_detector = getattr(_active_engine, "_bifurcation_detector", None)
                if bifurc_detector is not None:
                    bifurc_detector.diagram.save(out_path / "bifurcation_diagram.json")
                # neuron_state.jsonl — per-neuron final state
                with (out_path / "neuron_state.jsonl").open("w", encoding="utf-8") as ns:
                    for name, data in sorted(neural_artifacts.get("neuron_states", {}).items()):
                        record = {"neuron_id": name, **data}
                        ns.write(json.dumps(record) + "\n")
                # V4.5 — Matplotlib visualization (optional)
                try:
                    from pie.visualization.neural_plots import plot_spike_raster
                    spike_log = neural_artifacts.get("spike_log", [])
                    if spike_log:
                        plot_spike_raster(spike_log, out_path / "spike_raster.png")
                except ImportError:
                    pass  # matplotlib not installed — skip visualization

        # Optionally run LLM conformance tests when using the real provider.
        conformance_summary = None
        conformance_ok = True
        conformance_rate = None
        if llm == "real":
            # Determine the model name from the environment or defaults.  The
            # conformance suite will write its own file into output_dir.
            model_name = os.environ.get("LM_API_MODEL", "qwen3-vl-30b-a3b-instruct")
            try:
                conformance_summary = run_conformance(
                    provider="real",
                    model_name=model_name,
                    output_dir=output_dir,
                    use_cache=use_cache,
                    record_cache=True,
                )
            except Exception as exc:
                # If the conformance run fails, record the exception; this will
                # be reflected in the exam report.  We still proceed with
                # validation and replay.
                conformance_summary = {
                    "model": model_name,
                    "provider": "real",
                    "metrics": {},
                    "error": str(exc),
                }
            metrics = conformance_summary.get("metrics") if conformance_summary else {}
            if metrics and metrics.get("total", 0):
                ok = metrics.get("ok", 0)
                retry = metrics.get("retry", 0)
                total = metrics.get("total", 0)
                conformance_rate = (ok + retry) / total if total else 0.0
                conformance_ok = conformance_rate >= CONFORMANCE_THRESHOLD
            else:
                conformance_ok = False

        # After generating artefacts, validate and replay to determine pass/fail.
        # Validate artefacts and run replay; set result accordingly.
        result = "PASS"
        message = "All artefacts validated successfully."
        try:
            # Validate trace: each line must be valid JSON and conform to Event schema
            with trace_path.open("r", encoding="utf-8") as tf:
                for ln, line in enumerate(tf.read().splitlines(), 1):
                    try:
                        obj = json.loads(line)
                        Event.model_validate(obj)
                    except Exception as exc:
                        raise ValueError(f"Invalid event on line {ln}: {exc}")
            # Validate snapshot
            with snapshot_path.open("r", encoding="utf-8") as sf:
                snapshot_obj = json.load(sf)
                State.model_validate(snapshot_obj)  # type: ignore
            # Validate memory + constraints artefacts
            validate_file(str(memory_path))
            validate_file(str(constraints_path))
            validate_file(str(constraints_snapshot_path))
            # Validate replay
            replay_ok = replay(str(trace_path), provider=llm)
            if not replay_ok:
                result = "FAIL"
                message = "Replay mismatch detected."
            if llm == "real" and not conformance_ok and result == "PASS":
                result = "FAIL"
                if conformance_rate is not None:
                    message = (
                        f"Conformance below threshold: {round(conformance_rate * 100, 2)}% "
                        f"< {int(CONFORMANCE_THRESHOLD * 100)}%."
                    )
                else:
                    message = "Conformance results missing or invalid."
        except Exception as exc:
            result = "FAIL"
            message = f"Validation or replay raised an exception: {exc}"
        # Write report including conformance metrics when applicable.
        report_lines = ["# Exam Report\n", f"Result: {result}\n", f"{message}\n"]
        if conformance_summary is not None:
            report_lines.append("## LLM Conformance Summary\n")
            if "metrics" in conformance_summary and conformance_summary["metrics"]:
                metrics = conformance_summary["metrics"]
                report_lines.append(
                    f"Model: {conformance_summary.get('model')}\n\n"
                    f"Provider: {conformance_summary.get('provider')}\n\n"
                    f"Conformance results: OK={metrics.get('ok')} ("
                    f"{metrics.get('ok_percent')}%), Retry={metrics.get('retry')} ("
                    f"{metrics.get('retry_percent')}%), Fallback={metrics.get('fallback')} ("
                    f"{metrics.get('fallback_percent')}%)\n"
                )
                report_lines.append(
                    f"Conformance threshold: {int(CONFORMANCE_THRESHOLD * 100)}%\n"
                )
                if conformance_rate is not None:
                    report_lines.append(
                        f"Conformance rate (ok+retry): {round(conformance_rate * 100, 2)}%\n"
                    )
                    report_lines.append(
                        f"Conformance pass: {'YES' if conformance_ok else 'NO'}\n"
                    )
            else:
                report_lines.append(
                    f"Conformance testing failed or no metrics available.\n"
                    f"Error: {conformance_summary.get('error','Unknown error')}\n"
                )
        atomic_write(report_path, "\n".join(report_lines))
        # Provide console feedback
        print(f"Exam run complete. Artefacts written to: {out_path}")
    else:
        # Interactive mode: prompt user for input per turn.
        state = State()
        session_id = 1
        event_id = 1
        plan_id = 1
        memory_store = MemoryStore()
        for turn in range(turns):
            user_input = input(f"Turn {turn + 1} > ").strip()
            # Perform same logic as exam but do not write files
            goals = _generate_goals(user_input)
            action, _ = _select_action_with_constraints(goals, user_input, [])
            memory_view = build_memory_view(memory_store.read_all())
            plan, _ = _build_speech_plan(plan_id, action, user_input, memory_view)
            result = generate_with_policy(
                plan,
                provider=llm,
                max_retries=DEFAULT_MAX_RETRIES,
                use_cache=use_cache,
                record_cache=True,
            )
            message = result.text
            # Print output to console
            print(message)
            memory_context = {
                "logical_time": {"session": session_id, "turn": turn + 1},
                "user_input": user_input,
                "source_refs": [event_id, event_id + 1],
            }
            for record in propose_memory_writes(memory_context):
                memory_store.append(record)
            # Advance state for next turn
            state = state.update()
            event_id += 6
            plan_id += 1


def replay(trace_path: str, provider: str = "fake") -> bool:
    """Replay a trace file and verify deterministic decisions.

    The replay function reads a ``trace.jsonl`` file produced by the
    runtime and validates that the sequence of events matches what
    would be produced for the same inputs.  If mismatches are
    detected they are reported.  Otherwise the replay passes.
    """
    path = Path(trace_path)
    if not path.exists():
        raise ValueError(f"Trace file not found: {trace_path}")
    with path.open("r", encoding="utf-8") as f:
        lines = [json.loads(ln) for ln in f.read().splitlines() if ln.strip()]
    # Extract user inputs from the trace (INPUT events)
    inputs: List[str] = [ev["content"]["input"] for ev in lines if ev.get("type") == "INPUT"]
    # Recreate expected events using the runtime logic
    state = State()
    session_id = 1
    event_id = 1
    plan_id = 1
    memory_records = []
    constraints_records: List[ConstraintRecord] = []
    crystallizer = CrystallizationEngine()
    expected: List[dict] = []
    for turn_index, user_input in enumerate(inputs, start=1):
        memory_view = build_memory_view(memory_records)
        turn_events: List[dict] = []
        # INPUT
        input_event_id = event_id
        event = Event.new(
            event_id,
            "INPUT",
            {
                "logical_time": {"turn": turn_index, "step": 1},
                "input": user_input,
            },
        ).to_json()
        expected.append(event)
        turn_events.append(event)
        event_id += 1
        # STATE_UPDATED
        state = state.update()
        from .state_engine.registry import StateEngineRegistry
        _replay_engine = StateEngineRegistry.get_active()
        snapshot_content = state.snapshot()
        snapshot_content = {
            "logical_time": {"turn": turn_index, "step": 2},
            "engine_id": _replay_engine.engine_id,
            "engine_version": _replay_engine.version,
            **snapshot_content,
        }
        event = Event.new(event_id, "STATE_UPDATED", snapshot_content).to_json()
        expected.append(event)
        turn_events.append(event)
        event_id += 1
        # GOALS_GENERATED
        goals = _generate_goals(user_input)
        event = Event.new(
            event_id,
            "GOALS_GENERATED",
            {"logical_time": {"turn": turn_index, "step": 3}, "goals": goals},
        ).to_json()
        expected.append(event)
        turn_events.append(event)
        event_id += 1
        # COUNTERFACTUAL DELIBERATION (V3.2) — replay must match run
        cf_result = deliberate(
            goals, user_input,
            [rec for rec in constraints_records if rec.status == "active"],
            k=3,
        )
        expected.append(
            Event.new(
                event_id, "CF_GENERATED",
                {"logical_time": {"turn": turn_index, "step": 3},
                 "k": cf_result.k,
                 "candidates": [c.to_dict() for c in cf_result.candidates]},
            ).to_json()
        )
        event_id += 1
        for cand in cf_result.candidates:
            expected.append(
                Event.new(
                    event_id, "CF_SCORED",
                    {"logical_time": {"turn": turn_index, "step": 3},
                     "candidate_id": cand.candidate_id,
                     "scores": cand.scores.to_dict()},
                ).to_json()
            )
            event_id += 1
        for cand in cf_result.candidates:
            if not cand.is_chosen and cand.reject_reason:
                expected.append(
                    Event.new(
                        event_id, "CF_REJECTED",
                        {"logical_time": {"turn": turn_index, "step": 3},
                         "candidate_id": cand.candidate_id,
                         "action_id": cand.action.action_id,
                         "reason": cand.reject_reason},
                    ).to_json()
                )
                event_id += 1
        expected.append(
            Event.new(
                event_id, "CF_CHOSEN",
                {"logical_time": {"turn": turn_index, "step": 3},
                 "candidate_id": cf_result.chosen.candidate_id,
                 "action_id": cf_result.chosen.action.action_id,
                 "scores": cf_result.chosen.scores.to_dict()},
            ).to_json()
        )
        event_id += 1
        # CONSTRAINTS_PROPOSED
        proposed_constraints = crystallizer.propose_constraints(
            events=turn_events,
            state=state,
            logical_time={"session": session_id, "turn": turn_index},
            existing_constraints=constraints_records,
        )
        for record in proposed_constraints:
            expected.append(
                Event.new(
                    event_id,
                    "CONSTRAINT_PROPOSED",
                    {
                        "logical_time": {
                            "turn": turn_index,
                            "step": 3,
                            "idx": record.logical_time.idx,
                        },
                        "constraint": record.model_dump(),
                    },
                ).to_json()
            )
            event_id += 1
            constraints_records.append(record)
            expected.append(
                Event.new(
                    event_id,
                    "CONSTRAINT_APPENDED",
                    {
                        "logical_time": {
                            "turn": turn_index,
                            "step": 3,
                            "idx": record.logical_time.idx,
                        },
                        "constraint_id": record.constraint_id,
                        "status": record.status,
                    },
                ).to_json()
            )
            event_id += 1
        # ACTION_SELECTED
        active_constraints = [rec for rec in constraints_records if rec.status == "active"]
        action, enforcement_payload = _select_action_with_constraints(
            goals, user_input, active_constraints
        )
        expected.append(
            Event.new(
                event_id,
                "CONSTRAINT_ENFORCED",
                {
                    "logical_time": {"turn": turn_index, "step": 3},
                    **enforcement_payload,
                },
            ).to_json()
        )
        event_id += 1
        expected.append(
            Event.new(
                event_id,
                "ACTION_SELECTED",
                {"logical_time": {"turn": turn_index, "step": 4}, "action": action},
            ).to_json()
        )
        event_id += 1
        # SPEECHPLAN
        plan, memory_rationale = _build_speech_plan(plan_id, action, user_input, memory_view)
        plan_content = {
            "logical_time": {"turn": turn_index, "step": 5},
            **plan.model_dump(),
        }
        if memory_rationale:
            plan_content["memory_rationale"] = memory_rationale
        expected.append(
            Event.new(event_id, "SPEECHPLAN", plan_content).to_json()
        )
        event_id += 1
        # LLM output with runtime validation, retries, and fallback
        result = generate_with_policy(plan, provider=provider, max_retries=DEFAULT_MAX_RETRIES)
        for retry_index, reason in enumerate(result.retry_reasons, start=1):
            expected.append(
                Event.new(
                    event_id,
                    "LLM_RETRY",
                    {
                        "logical_time": {"turn": turn_index, "step": 6, "attempt": retry_index},
                        "reason": reason,
                    },
                ).to_json()
            )
            event_id += 1
        final_type = "LLM_FALLBACK" if result.used_fallback else "LLM_OUTPUT"
        final_content = {
            "logical_time": {"turn": turn_index, "step": 6, "attempt": result.attempts},
            "text": result.text,
        }
        if result.used_fallback and result.failure_reason:
            final_content["reason"] = result.failure_reason
        expected.append(Event.new(event_id, final_type, final_content).to_json())
        llm_event_id = event_id
        event_id += 1
        memory_context = {
            "logical_time": {"session": session_id, "turn": turn_index},
            "user_input": user_input,
            "source_refs": [input_event_id, llm_event_id],
        }
        memory_new = propose_memory_writes(memory_context)
        for record in memory_new:
            expected.append(
                Event.new(
                    event_id,
                    "MEMORY_PROPOSED",
                    {
                        "logical_time": {"turn": turn_index, "step": 7},
                        "memory": record.model_dump(),
                    },
                ).to_json()
            )
            event_id += 1
            memory_records.append(record)
            expected.append(
                Event.new(
                    event_id,
                    "MEMORY_APPENDED",
                    {
                        "logical_time": {"turn": turn_index, "step": 7},
                        "memory_id": record.memory_id,
                        "memory_type": record.type,
                    },
                ).to_json()
            )
            event_id += 1
        plan_id += 1
    # Compare lengths
    if len(expected) != len(lines):
        print(
            f"Mismatch: expected {len(expected)} events, found {len(lines)} in trace."
        )
        return False
    # Compare each event except timestamp differences; ignore timestamp field
    mismatches: List[Tuple[int, str]] = []
    for idx, (exp, act) in enumerate(zip(expected, lines), 1):
        # Compare id, type and content; ignore timestamp
        if exp["id"] != act.get("id") or exp["type"] != act.get("type") or exp["content"] != act.get("content"):
            mismatches.append(
                (
                    idx,
                    f"Event {idx}: expected id {exp['id']}, type {exp['type']}, content {exp['content']} but got id {act.get('id')}, type {act.get('type')}, content {act.get('content')}",
                )
            )
    if mismatches:
        print("Replay failed; mismatches found:\n")
        for idx, msg in mismatches:
            print(msg)
        return False
    memory_path = path.parent / "memory.jsonl"
    if memory_path.exists():
        with memory_path.open("r", encoding="utf-8") as f:
            mem_lines = [ln for ln in f.read().splitlines() if ln.strip()]
        mem_expected = [rec.model_dump() for rec in memory_records]
        if len(mem_lines) != len(mem_expected):
            print(
                f"Replay failed; memory length mismatch: expected {len(mem_expected)}, found {len(mem_lines)}."
            )
            return False
        for idx, (line, exp) in enumerate(zip(mem_lines, mem_expected), 1):
            try:
                obj = json.loads(line)
            except Exception as exc:
                print(f"Replay failed; invalid memory JSON on line {idx}: {exc}")
                return False
            if obj != exp:
                print(f"Replay failed; memory mismatch on line {idx}.")
                return False
    constraints_path = path.parent / "constraints.jsonl"
    if constraints_path.exists():
        with constraints_path.open("r", encoding="utf-8") as f:
            con_lines = [ln for ln in f.read().splitlines() if ln.strip()]
        con_expected = [rec.model_dump() for rec in constraints_records]
        if len(con_lines) != len(con_expected):
            print(
                f"Replay failed; constraints length mismatch: expected {len(con_expected)}, found {len(con_lines)}."
            )
            return False
        for idx, (line, exp) in enumerate(zip(con_lines, con_expected), 1):
            try:
                obj = json.loads(line)
            except Exception as exc:
                print(f"Replay failed; invalid constraints JSON on line {idx}: {exc}")
                return False
            if obj != exp:
                print(f"Replay failed; constraints mismatch on line {idx}.")
                return False
    print("Replay passed; all decisions match expected sequence.")
    return True

# ---------------------------------------------------------------------------
# V3.0 — Session-aware run (persistent Ivy sessions)
# ---------------------------------------------------------------------------

# Tool schema for native function calling (OpenAI format)
_TOOL_SCHEMA: List[Dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "fs_read",
        "description": "Read a file from the sandbox directory",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path inside sandbox"}
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "fs_write",
        "description": "Write content to a file in the sandbox directory",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File path inside sandbox"},
            "content": {"type": "string", "description": "Content to write"},
        }, "required": ["path", "content"]},
    }},
    {"type": "function", "function": {
        "name": "fs_list",
        "description": "List files in a sandbox directory",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Directory path (default: root)"}
        }, "required": []},
    }},
]


def _parse_tool_call(tc: Dict[str, Any], sandbox_root: Path) -> "ToolCall":
    """Convert OpenAI tool_call format to kernel ToolCall contract.

    Path handling:
    - Absolute paths or paths with drive letters (C:\\...) are passed
      as-is; FSCapability.check() will reject them.
    - Truly relative paths get sandbox_root prepended, then validated
      via resolve() + relative_to() for containment.
    """
    from .contracts.tool import ToolCall as _ToolCall

    func = tc.get("function", {})
    name = func.get("name", "")
    try:
        args = json.loads(func.get("arguments", "{}"))
    except (json.JSONDecodeError, TypeError):
        args = {}

    if name.startswith("fs_"):
        operation = name[3:]
        tool_id = "fs_sandbox"
        if "path" in args:
            p = Path(args["path"])
            if p.is_absolute() or p.drive:
                pass  # Leave as-is — FSCapability will reject
            else:
                joined = sandbox_root / p
                resolved = joined.resolve()
                try:
                    resolved.relative_to(sandbox_root.resolve())
                    args["path"] = str(resolved)
                except ValueError:
                    args["path"] = str(joined)
    else:
        tool_id = name
        operation = "execute"

    return _ToolCall(
        call_id=tc.get("id", ""),
        tool_id=tool_id,
        operation=operation,
        params=args,
    )


class TurnProcessor:
    """Processes turns for a session (V6b-E2 extraction).

    Encapsulates all per-session state (budget, killswitch, routines, etc.)
    that was previously captured by the ``_process_turn`` closure inside
    ``run_session``.  This allows the API layer to process individual turns
    without going through stdin.
    """

    def __init__(
        self,
        ctx: "SessionContext",
        llm: str = "fake",
        no_cache: bool = False,
    ) -> None:
        from .session.manager import SessionManager
        from .metabolism import BudgetTracker, calculate_turn_cost
        from .routines import RoutineLibrary
        from .memory.recall import recall as episodic_recall
        from .memory.consolidation import (
            should_consolidate,
            consolidate as consolidate_journal,
        )

        self.ctx = ctx
        self.llm = llm
        self.use_cache = not no_cache
        self.state = ctx.state
        self.event_id = ctx.event_id
        self.turn_index = ctx.turn_count
        self.plan_id = ctx.turn_count + 1
        self.mgr = SessionManager(ctx.session_dir.parent)
        self.crystallizer = CrystallizationEngine()
        self.tools_enabled = True

        # V6b-E4: model info
        self.model_info: Dict[str, Any] = {
            "model_id": os.environ.get("LM_API_MODEL", "qwen3-vl-30b-a3b-instruct"),
            "endpoint": os.environ.get("LM_API_BASE_URL", "http://127.0.0.1:1234/v1"),
            "provider": llm,
            "params": {
                "temperature": float(os.environ.get("LM_API_TEMPERATURE", "0.2")),
                "top_p": float(os.environ.get("LM_API_TOP_P", "0.5")),
            },
        }

        self._speech_compiler = SpeechCompiler()

        # Kill-switch
        self._ks_report_path = ctx.session_dir / "killswitch_report.json"
        self.killswitch = KillSwitchState.load(self._ks_report_path)

        # Metabolism
        self._calculate_turn_cost = calculate_turn_cost
        self._metab_path = ctx.session_dir / "metabolism.json"
        self.budget = BudgetTracker.load(self._metab_path)

        # Routines
        self._routines_path = ctx.session_dir / "routines.json"
        self.routine_lib = RoutineLibrary.load(self._routines_path)

        # Memory
        self._episodic_recall = episodic_recall
        self._should_consolidate = should_consolidate
        self._consolidate_journal = consolidate_journal
        self._consol_meta_path = ctx.session_dir / "consolidation.json"
        self._last_consol_turn = 0
        if self._consol_meta_path.exists():
            _cm = json.loads(self._consol_meta_path.read_text(encoding="utf-8"))
            self._last_consol_turn = int(_cm.get("last_turn", 0))

        # V4 — ControlVector from previous turn
        # After snapshot restore, the engine already has a control_vector
        # from deserialize(). Use it so memory_gate fires on the first
        # post-restore turn (otherwise _prev_cv=None skips the gate).
        self._prev_cv = None
        from .state_engine.registry import StateEngineRegistry as _SER
        _active = _SER.get_active()
        if hasattr(_active, 'control_vector') and _active.control_vector is not None:
            self._prev_cv = _active.control_vector

        # Governed tool use — ToolExecutor setup
        from .contracts.tool import ToolCapability
        from .tools.allowlist import NetworkAllowlist, FSCapability
        from .tools.executor import ToolExecutor

        self._sandbox_root = ctx.session_dir / "sandbox"
        self._sandbox_root.mkdir(exist_ok=True)
        self.tool_executor = ToolExecutor(
            capabilities=[
                ToolCapability(
                    tool_id="fs_sandbox",
                    domain="filesystem",
                    allowed_operations=["read", "write", "list", "mkdir"],
                    requires_confirmation=["delete", "move"],
                ),
            ],
            network_allowlist=NetworkAllowlist([]),
            fs_capability=FSCapability(self._sandbox_root, ["read", "write", "list", "mkdir"]),
            rate_limit_per_turn=3,
            rate_limit_per_session=20,
        )

        # R-STDP reward — governed online learning
        self._rstdp_online = os.environ.get("PIE_RSTDP_ONLINE", "0") == "1"
        if self._rstdp_online:
            if hasattr(_active, 'enable_reward_tracking'):
                _active.enable_reward_tracking()
        # Journal-first: reconstruct pending/applied from journal (crash-safe)
        self._pending_rewards, self._applied_rewards = self._scan_journal_rewards(
            ctx.journal_path
        )

    @staticmethod
    def _scan_journal_rewards(journal_path: Path) -> Tuple[Dict[int, Dict], Set[int]]:
        """Reconstruct reward state from journal. Journal is source of truth."""
        pending: Dict[int, Dict] = {}
        applied: Set[int] = set()
        if not journal_path.exists():
            return pending, applied
        for line in journal_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            evt = json.loads(line)
            etype = evt.get("type", "")
            content = evt.get("content", {})
            if etype == "REWARD_SIGNAL":
                target = content.get("target_turn")
                if target is not None:
                    pending[target] = {
                        "value": content["value"],
                        "source": content["source"],
                        "actor_id": content.get("actor_id", "unknown"),
                    }
            elif etype == "REWARD_APPLIED":
                target = content.get("target_turn")
                if target is not None:
                    applied.add(target)
        return pending, applied

    def submit_reward(
        self,
        target_turn: int,
        value: int,
        source: str = "thumb",
        reason: Optional[str] = None,
        actor_id: str = "local_ui",
    ) -> Dict[str, Any]:
        """Submit a reward signal for a completed turn.

        Writes REWARD_SIGNAL to journal. The reward is applied at the start
        of the NEXT process() call (step 1.5, after INPUT, before STATE_UPDATED).
        Idempotent: one reward per (session, target_turn).
        """
        if value not in (1, -1):
            raise ValueError(f"Reward value must be +1 or -1, got {value}")
        if target_turn < 1 or target_turn > self.turn_index:
            raise ValueError(
                f"target_turn must be 1..{self.turn_index}, got {target_turn}"
            )
        ctx = self.ctx

        # Idempotency check
        if target_turn in self._pending_rewards or target_turn in self._applied_rewards:
            event = Event.new(self.event_id, "REWARD_DUPLICATE_IGNORED", {
                "logical_time": {"turn": target_turn, "step": 0},
                "target_turn": target_turn,
            })
            atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
            self.event_id += 1
            ctx.event_id = self.event_id
            return {"status": "duplicate_ignored", "target_turn": target_turn}

        # Emit REWARD_SIGNAL
        event = Event.new(self.event_id, "REWARD_SIGNAL", {
            "logical_time": {"turn": target_turn, "step": 0},
            "target_turn": target_turn,
            "value": value,
            "source": source,
            "reason": reason,
            "actor_id": actor_id,
        })
        atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
        reward_event_id = self.event_id
        self.event_id += 1
        ctx.event_id = self.event_id

        self._pending_rewards[target_turn] = {
            "value": value,
            "source": source,
            "actor_id": actor_id,
        }

        return {
            "status": "accepted",
            "target_turn": target_turn,
            "event_id": reward_event_id,
        }

    def _handle_tool_calls(self, tool_calls_raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls through governed pipeline.

        Returns tool-result messages (role="tool") for the LLM follow-up.
        Journals TOOL_CALL / TOOL_DENIED / TOOL_RESULT events.
        """
        from .persistence.atomic import atomic_append

        ctx = self.ctx
        results: List[Dict[str, Any]] = []
        self.tool_executor.new_turn()

        for idx, tc in enumerate(tool_calls_raw):
            call = _parse_tool_call(tc, self._sandbox_root)
            call.call_id = f"t{self.turn_index}_c{idx}_{call.operation}"

            # Journal: TOOL_CALL (shape only — no param values)
            event = Event.new(self.event_id, "TOOL_CALL", {
                "logical_time": {"turn": self.turn_index, "step": 6},
                "call_id": call.call_id,
                "tool_id": call.tool_id,
                "operation": call.operation,
                "params_keys": sorted(call.params.keys()),
            })
            atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
            self.event_id += 1

            # Execute through ToolExecutor (allowlist + capability + rate limit)
            tool_result = self.tool_executor.execute(call)
            self._tool_call_count += 1

            if tool_result.status in ("DENIED", "FAILED"):
                event = Event.new(self.event_id, "TOOL_DENIED", {
                    "logical_time": {"turn": self.turn_index, "step": 6},
                    "call_id": call.call_id,
                    "deny_reason": tool_result.deny_reason or "",
                })
                atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
                self.event_id += 1
                results.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", call.call_id),
                    "content": json.dumps({"error": tool_result.deny_reason}),
                })
            else:
                _result_json = json.dumps(tool_result.output or {}, sort_keys=True)
                _result_sha = hashlib.sha256(_result_json.encode()).hexdigest()
                event = Event.new(self.event_id, "TOOL_RESULT", {
                    "logical_time": {"turn": self.turn_index, "step": 6},
                    "call_id": call.call_id,
                    "status": tool_result.status,
                    "result_sha256": _result_sha,
                    "result_size": len(_result_json),
                })
                atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
                self.event_id += 1
                # Truncate result for LLM context (4KB max)
                results.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", call.call_id),
                    "content": _result_json[:4096],
                })

        return results

    def process(self, user_input: str) -> str:
        """Process one turn, return Ivy's response text.

        Appends all events to journal and saves state after each turn
        (transactional: crash-safe).
        """
        from .persistence.atomic import atomic_append

        ctx = self.ctx

        # V3.3 — If kill-switch is active, return minimal response
        if self.killswitch.active:
            self.killswitch.check("input")
            self.turn_index += 1
            event = Event.new(
                self.event_id, "INPUT",
                {"logical_time": {"turn": self.turn_index, "step": 1}, "input": user_input},
            )
            atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
            self.event_id += 1
            ctx.state = self.state
            ctx.event_id = self.event_id
            ctx.turn_count = self.turn_index
            self.mgr.save(ctx, model_info=self.model_info)
            self.killswitch.save(self._ks_report_path)
            return KILL_SWITCH_RESPONSE

        self.turn_index += 1

        # Lazy R-STDP upgrade: ensure tracker is upgraded before spikes build eligibility
        if self._rstdp_online:
            from .state_engine.registry import StateEngineRegistry as _LazyRSER
            _eng_lazy = _LazyRSER.get_active()
            if hasattr(_eng_lazy, 'enable_reward_tracking'):
                _eng_lazy.enable_reward_tracking()

        memory_view = build_memory_view(ctx.memory_store.read_all())

        # RECALL LOOP
        _recall_top_k = 5
        if self._prev_cv is not None:
            _recall_top_k = max(1, min(10, int(self._prev_cv.memory_gate * 10)))
            _gating_ev = gate_memory(self._prev_cv.memory_gate)
            atomic_append(ctx.journal_path, json.dumps(
                {"event_id": self.event_id, **_gating_ev.to_trace_dict(),
                 "logical_time": {"turn": self.turn_index, "step": 0}}
            ) + "\n")
            self.event_id += 1
        recalled_episodes = self._episodic_recall(
            query=user_input,
            journal_path=ctx.journal_path,
            state=self.state.snapshot(),
            top_k=_recall_top_k,
            exclude_turn=self.turn_index,
        )
        memory_view.episodic_recall = [ep.to_dict() for ep in recalled_episodes]

        turn_events: List[dict] = []

        # INPUT
        input_event_id = self.event_id
        event = Event.new(
            self.event_id, "INPUT",
            {"logical_time": {"turn": self.turn_index, "step": 1}, "input": user_input},
        )
        atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
        turn_events.append(event.to_json())
        self.event_id += 1

        # V3.3: kill-switch check before state evolution
        self.killswitch.check("deliberation")
        if self.killswitch.active:
            ctx.state = self.state
            ctx.event_id = self.event_id
            ctx.turn_count = self.turn_index
            self.mgr.save(ctx, model_info=self.model_info)
            self.killswitch.save(self._ks_report_path)
            return KILL_SWITCH_RESPONSE

        # REWARD APPLICATION (step 1.5) — apply all pending rewards
        # Runs ONLY when KS is off (past the early-return above).
        # Weight changes take effect in the STATE_UPDATED that follows.
        _pending_to_apply = sorted(
            t for t in self._pending_rewards if t not in self._applied_rewards
        )
        for _target_turn in _pending_to_apply:
            _reward = self._pending_rewards[_target_turn]
            _applied = False
            if self._rstdp_online:
                from .state_engine.registry import StateEngineRegistry as _RewSER
                _engine_rw = _RewSER.get_active()
                if hasattr(_engine_rw, 'enable_reward_tracking'):
                    _engine_rw.enable_reward_tracking()
                if hasattr(_engine_rw, 'rstdp_tracker') and _engine_rw.rstdp_tracker is not None:
                    _engine_rw.rstdp_tracker.apply_reward(
                        turn=_target_turn,
                        source=_reward["source"],
                        value=float(_reward["value"]),
                    )
                    _applied = True
            event = Event.new(self.event_id, "REWARD_APPLIED", {
                "logical_time": {"turn": self.turn_index, "step": 1},
                "target_turn": _target_turn,
                "value": _reward["value"],
                "source": _reward["source"],
                "actor_id": _reward.get("actor_id", "local_ui"),
                "applied": _applied,
                "online": self._rstdp_online,
            })
            atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
            self.event_id += 1
            self._applied_rewards.add(_target_turn)

        # STATE_UPDATED
        self.state = self.state.update()
        from .state_engine.registry import StateEngineRegistry
        _engine = StateEngineRegistry.get_active()
        snapshot_content = {
            "logical_time": {"turn": self.turn_index, "step": 2},
            "engine_id": _engine.engine_id,
            "engine_version": _engine.version,
            **self.state.snapshot(),
        }
        if hasattr(_engine, "get_artifacts"):
            snapshot_content["engine_metadata"] = _engine.get_artifacts()
        event = Event.new(self.event_id, "STATE_UPDATED", snapshot_content)
        atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
        turn_events.append(event.to_json())
        self.event_id += 1

        # V3.6 — SNN state
        snn_state = None
        _cv = getattr(_engine, 'control_vector', None)
        if hasattr(_engine, 'spike_log') and hasattr(_engine, '_neurons'):
            recent = [s.neuron_id for s in _engine.spike_log[-5:]]
            near = [
                f"{nid}={n.membrane_potential:.2f}"
                for nid, n in sorted(_engine._neurons.items())
                if n.membrane_potential > 0.7
            ]
            snn_state = {"recent_spikes": recent, "near_threshold": near}
            if _cv is not None:
                snn_state["control_vector"] = _cv.to_dict()

        # Metabolism gating
        gating = self.budget.get_gating()

        # Routines
        _matched_routine = self.routine_lib.find_matching(user_input)
        matched_routines = [_matched_routine] if _matched_routine else []

        # GOALS_GENERATED
        goals = _generate_goals(user_input)
        event = Event.new(
            self.event_id, "GOALS_GENERATED",
            {"logical_time": {"turn": self.turn_index, "step": 3}, "goals": goals},
        )
        atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
        turn_events.append(event.to_json())
        self.event_id += 1

        # COUNTERFACTUAL DELIBERATION
        cf_k = gating.max_alternatives
        _cv = getattr(_engine, 'control_vector', None)
        if _cv is not None:
            cf_k = min(cf_k, _cv.cf_k)
            _gating_ev = gate_counterfactual(_cv.cf_k, gating.max_alternatives)
            atomic_append(ctx.journal_path, json.dumps(
                {"event_id": self.event_id, **_gating_ev.to_trace_dict(),
                 "logical_time": {"turn": self.turn_index, "step": 3}}
            ) + "\n")
            self.event_id += 1
        cf_result = deliberate(
            goals, user_input, ctx.constraints_store.query_active(), k=cf_k
        )
        event = Event.new(
            self.event_id, "CF_GENERATED",
            {"logical_time": {"turn": self.turn_index, "step": 3},
             "k": cf_result.k,
             "candidates": [c.to_dict() for c in cf_result.candidates]},
        )
        atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
        self.event_id += 1
        for cand in cf_result.candidates:
            event = Event.new(
                self.event_id, "CF_SCORED",
                {"logical_time": {"turn": self.turn_index, "step": 3},
                 "candidate_id": cand.candidate_id,
                 "scores": cand.scores.to_dict()},
            )
            atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
            self.event_id += 1
        for cand in cf_result.candidates:
            if not cand.is_chosen and cand.reject_reason:
                event = Event.new(
                    self.event_id, "CF_REJECTED",
                    {"logical_time": {"turn": self.turn_index, "step": 3},
                     "candidate_id": cand.candidate_id,
                     "action_id": cand.action.action_id,
                     "reason": cand.reject_reason},
                )
                atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
                self.event_id += 1
        event = Event.new(
            self.event_id, "CF_CHOSEN",
            {"logical_time": {"turn": self.turn_index, "step": 3},
             "candidate_id": cf_result.chosen.candidate_id,
             "action_id": cf_result.chosen.action.action_id,
             "scores": cf_result.chosen.scores.to_dict()},
        )
        atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
        self.event_id += 1

        # CONSTRAINTS — crystallization
        proposed = self.crystallizer.propose_constraints(
            events=turn_events, state=self.state,
            logical_time={"session": 1, "turn": self.turn_index},
            existing_constraints=ctx.constraints_store.read_all(),
        )
        for record in proposed:
            event = Event.new(
                self.event_id, "CONSTRAINT_PROPOSED",
                {"logical_time": {"turn": self.turn_index, "step": 3, "idx": record.logical_time.idx},
                 "constraint": record.model_dump()},
            )
            atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
            self.event_id += 1
            ctx.constraints_store.append(record)
            event = Event.new(
                self.event_id, "CONSTRAINT_APPENDED",
                {"logical_time": {"turn": self.turn_index, "step": 3, "idx": record.logical_time.idx},
                 "constraint_id": record.constraint_id, "status": record.status},
            )
            atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
            self.event_id += 1

        # ACTION_SELECTED
        active_constraints = ctx.constraints_store.query_active()
        action, enforcement_payload = _select_action_with_constraints(
            goals, user_input, active_constraints,
        )
        event = Event.new(
            self.event_id, "CONSTRAINT_ENFORCED",
            {"logical_time": {"turn": self.turn_index, "step": 3}, **enforcement_payload},
        )
        atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
        self.event_id += 1

        event = Event.new(
            self.event_id, "ACTION_SELECTED",
            {"logical_time": {"turn": self.turn_index, "step": 4}, "action": action},
        )
        atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
        self.event_id += 1

        # SPEECHPLAN
        plan, memory_rationale = _build_speech_plan(
            self.plan_id, action, user_input, memory_view, identity=ctx.identity,
        )
        if gating.verbosity == "low" and plan.verbosity != "low":
            plan = plan.model_copy(update={"verbosity": "low", "max_tokens": min(plan.max_tokens, 40)})
        elif gating.verbosity == "med" and plan.verbosity == "high":
            plan = plan.model_copy(update={"verbosity": "med", "max_tokens": min(plan.max_tokens, 80)})

        if _cv is not None:
            if _cv.verbosity_bias < 0.3 and plan.verbosity != "low":
                plan = plan.model_copy(update={"verbosity": "low", "max_tokens": min(plan.max_tokens, 40)})
            elif _cv.verbosity_bias > 0.7 and gating.verbosity != "low" and plan.verbosity == "low":
                plan = plan.model_copy(update={"verbosity": "med", "max_tokens": min(plan.max_tokens, 80)})
            _gating_ev = gate_verbosity(_cv.verbosity_bias)
            atomic_append(ctx.journal_path, json.dumps(
                {"event_id": self.event_id, **_gating_ev.to_trace_dict(),
                 "logical_time": {"turn": self.turn_index, "step": 5}}
            ) + "\n")
            self.event_id += 1

        _trust_total = sum(memory_view.trust_scores.values()) if memory_view.trust_scores else 0.0
        if _trust_total < -0.2 and plan.verbosity != "low":
            plan = plan.model_copy(update={"verbosity": "low", "max_tokens": min(plan.max_tokens, 40)})

        plan_content = {"logical_time": {"turn": self.turn_index, "step": 5}, **plan.model_dump()}
        if memory_rationale:
            plan_content["memory_rationale"] = memory_rationale
        plan_content["metabolism_gating"] = gating.to_dict()
        event = Event.new(self.event_id, "SPEECHPLAN", plan_content)
        atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
        self.event_id += 1

        # V3.3: kill-switch check before voice generation
        self.killswitch.check("voice_generation")
        if self.killswitch.active:
            ctx.state = self.state
            ctx.event_id = self.event_id
            ctx.turn_count = self.turn_index
            self.mgr.save(ctx, model_info=self.model_info)
            self.killswitch.save(self._ks_report_path)
            return KILL_SWITCH_RESPONSE

        # V4: tool_gate — gating BEFORE LLM call
        _tools_available = self.tools_enabled
        _tools_deny_reason: Optional[str] = None

        if not self.tools_enabled:
            _tools_deny_reason = "gated_pre_llm:disabled"

        if _cv is not None:
            if _cv.tool_gate < 0.3:
                _tools_available = False
                _tools_deny_reason = "gated_pre_llm:tool_gate"
            _gating_ev = gate_tools(_cv.tool_gate)
            atomic_append(ctx.journal_path, json.dumps(
                {"event_id": self.event_id, **_gating_ev.to_trace_dict(),
                 "logical_time": {"turn": self.turn_index, "step": 5}}
            ) + "\n")
            self.event_id += 1

        _budget_gating = self.budget.get_gating()
        if not _budget_gating.tools_allowed:
            _tools_available = False
            _tools_deny_reason = "gated_pre_llm:budget"

        # Emit synthetic TOOL_DENIED when pre-LLM gating blocks tools
        if not _tools_available and _tools_deny_reason:
            event = Event.new(self.event_id, "TOOL_DENIED", {
                "logical_time": {"turn": self.turn_index, "step": 5},
                "call_id": f"t{self.turn_index}_pre_gate",
                "deny_reason": _tools_deny_reason,
            })
            atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
            self.event_id += 1

        _tools_for_llm = _TOOL_SCHEMA if _tools_available else None
        self._tool_call_count = 0

        # Kernel context
        _kernel_ctx = _build_kernel_context(
            origin=ctx.identity,
            state=self.state,
            memory_view=memory_view,
            active_constraints=active_constraints,
            cf_result=cf_result,
            metabolism=self.budget,
            matched_routines=matched_routines,
            snn_state=snn_state,
        )
        _kernel_ctx["_compiler"] = self._speech_compiler
        _kernel_ctx["_plan"] = plan
        _kernel_ctx["user_input"] = user_input

        # LLM OUTPUT (with governed tool use)
        result = generate_with_policy(
            plan, provider=self.llm, max_retries=DEFAULT_MAX_RETRIES,
            use_cache=self.use_cache, record_cache=True,
            kernel_context=_kernel_ctx,
            tools=_tools_for_llm,
            tool_handler=self._handle_tool_calls if _tools_available else None,
        )
        for ri, reason in enumerate(result.retry_reasons, start=1):
            event = Event.new(
                self.event_id, "LLM_RETRY",
                {"logical_time": {"turn": self.turn_index, "step": 6, "attempt": ri}, "reason": reason},
            )
            atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
            self.event_id += 1

        final_type = "LLM_FALLBACK" if result.used_fallback else "LLM_OUTPUT"
        final_content = {
            "logical_time": {"turn": self.turn_index, "step": 6, "attempt": result.attempts},
            "text": result.text,
        }
        if result.used_fallback and result.failure_reason:
            final_content["reason"] = result.failure_reason
        event = Event.new(self.event_id, final_type, final_content)
        atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
        llm_event_id = self.event_id
        self.event_id += 1

        # MEMORY
        mem_ctx = {
            "logical_time": {"session": 1, "turn": self.turn_index},
            "user_input": user_input,
            "source_refs": [input_event_id, llm_event_id],
        }
        memory_records = propose_memory_writes(mem_ctx)
        for record in memory_records:
            event = Event.new(
                self.event_id, "MEMORY_PROPOSED",
                {"logical_time": {"turn": self.turn_index, "step": 7}, "memory": record.model_dump()},
            )
            atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
            self.event_id += 1
            ctx.memory_store.append(record)
            event = Event.new(
                self.event_id, "MEMORY_APPENDED",
                {"logical_time": {"turn": self.turn_index, "step": 7},
                 "memory_id": record.memory_id, "memory_type": record.type},
            )
            atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
            self.event_id += 1

        self.plan_id += 1

        # Metabolism
        turn_cost = self._calculate_turn_cost(
            retries=result.retries,
            tool_calls=self._tool_call_count,
            memory_writes=len(memory_records),
        )
        self.budget.consume(turn_cost, self.turn_index)
        self.budget.save(self._metab_path)

        # Routines
        for routine in matched_routines:
            routine.record_use(success=True, cost=turn_cost.total)
        if matched_routines:
            self.routine_lib.save(self._routines_path)

        # Save state (transactional)
        ctx.state = self.state
        ctx.event_id = self.event_id
        ctx.turn_count = self.turn_index
        self.mgr.save(ctx, model_info=self.model_info)
        # save() writes SNAPSHOT_SAVED and increments ctx.event_id — sync back
        self.event_id = ctx.event_id

        # CONSOLIDATION
        snap = self.state.snapshot()
        _consol_urgency = _cv.consolidation_urgency if _cv is not None else 0.0
        if _cv is not None:
            _gating_ev = gate_consolidation(_cv.consolidation_urgency)
            atomic_append(ctx.journal_path, json.dumps(
                {"event_id": self.event_id, **_gating_ev.to_trace_dict(),
                 "logical_time": {"turn": self.turn_index, "step": 8}}
            ) + "\n")
            self.event_id += 1
        if self._should_consolidate(self.turn_index, self._last_consol_turn, snap.get("affect"), urgency=_consol_urgency):
            consol_records = self._consolidate_journal(
                journal_path=ctx.journal_path,
                since_turn=self._last_consol_turn,
                logical_time={"session": 1, "turn": self.turn_index},
                source_refs=[input_event_id, llm_event_id],
            )
            for record in consol_records:
                event = Event.new(
                    self.event_id, "MEMORY_PROPOSED",
                    {"logical_time": {"turn": self.turn_index, "step": 8}, "memory": record.model_dump()},
                )
                atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
                self.event_id += 1
                ctx.memory_store.append(record)
                event = Event.new(
                    self.event_id, "MEMORY_APPENDED",
                    {"logical_time": {"turn": self.turn_index, "step": 8},
                     "memory_id": record.memory_id, "memory_type": record.type},
                )
                atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
                self.event_id += 1
            self._last_consol_turn = self.turn_index
            atomic_write(self._consol_meta_path, json.dumps({"last_turn": self._last_consol_turn}))
            ctx.event_id = self.event_id
            self.mgr.save(ctx, model_info=self.model_info)
            self.event_id = ctx.event_id

        # V4: Save ControlVector for next turn
        self._prev_cv = _cv

        return result.text


def run_session(
    session_ctx: "SessionContext",
    turns: int = 1,
    llm: str = "fake",
    no_cache: bool = False,
    interactive: bool = False,
) -> None:
    """Run the kernel using a persistent session context (V3.0).

    This reuses the same pipeline as ``run()`` (state update, goals,
    action selection, speech plan, LLM voice, memory, constraints) but
    reads/writes from the session store instead of ephemeral artifacts.

    Parameters
    ----------
    session_ctx:
        A ``SessionContext`` obtained from ``SessionManager.create()``
        or ``SessionManager.resume()``.
    turns:
        Number of turns to process.  Ignored when *interactive* is True.
    llm:
        LLM provider (``"fake"`` or ``"real"``).
    no_cache:
        When True, disables LLM cache reads.
    interactive:
        When True, reads input from stdin in a loop until the user
        types ``:quit``.
    """
    from .session.manager import SessionContext, SessionManager
    from .persistence.atomic import atomic_append

    ctx = session_ctx
    tp = TurnProcessor(ctx, llm=llm, no_cache=no_cache)

    if interactive:
        name = ctx.identity.get("identity", {}).get("name", "Ivy")
        print(f"Session {ctx.session_id} — talking to {name}")
        print("Type :quit to exit, :save to save, :state to view state, :whoami for identity")
        while True:
            try:
                user_input = input("\nYou> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                continue
            if user_input == ":quit":
                tp.mgr.save(ctx, model_info=tp.model_info)
                print("Session saved. Goodbye.")
                break
            if user_input == ":save":
                tp.mgr.save(ctx, model_info=tp.model_info)
                print("Session saved.")
                continue
            if user_input == ":state":
                print(json.dumps(tp.state.snapshot(), indent=2))
                continue
            if user_input == ":whoami":
                ident = ctx.identity.get("identity", {})
                print(f"Name: {ident.get('name', '?')}, Alias: {ident.get('alias', '?')}, Creator: {ident.get('creator_anchor', '?')}")
                continue
            if user_input == ":memory":
                mv = build_memory_view(ctx.memory_store.read_all())
                print(json.dumps(mv.to_dict(), indent=2))
                continue
            if user_input == ":tools on":
                tp.tools_enabled = True
                print("Tools enabled.")
                continue
            if user_input == ":tools off":
                tp.tools_enabled = False
                print("Tools disabled.")
                continue
            if user_input == ":killswitch on":
                ks_content = tp.killswitch.activate(phase="interactive")
                event = Event.new(
                    tp.event_id, "KILL_SWITCH_ON",
                    {"logical_time": {"turn": tp.turn_index, "step": 0}, **ks_content},
                )
                atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
                tp.event_id += 1
                tp.killswitch.save(tp._ks_report_path)
                print("Kill-switch ATTIVATO. Tools e memoria disabilitati. Modalità audit.")
                continue
            if user_input == ":killswitch off":
                ks_content = tp.killswitch.deactivate()
                event = Event.new(
                    tp.event_id, "KILL_SWITCH_OFF",
                    {"logical_time": {"turn": tp.turn_index, "step": 0}, **ks_content},
                )
                atomic_append(ctx.journal_path, json.dumps(event.to_json()) + "\n")
                tp.event_id += 1
                tp.killswitch.save(tp._ks_report_path)
                print("Kill-switch DISATTIVATO. Operatività normale ripristinata.")
                continue
            if user_input.startswith(":"):
                print(f"Unknown command: {user_input}")
                print("Available: :quit :save :state :whoami :memory :tools on/off :killswitch on/off")
                continue
            output = tp.process(user_input)
            display = f"\n{name}> {output}"
            import sys as _sys
            _sys.stdout.buffer.write((display + "\n").encode("utf-8", errors="replace"))
            _sys.stdout.buffer.flush()
    else:
        for i in range(turns):
            try:
                user_input = input(f"Turn {tp.turn_index + 1}> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user_input:
                user_input = "..."
            output = tp.process(user_input)
            print(output)


# There is intentionally no Typer CLI defined here because the environment used in
# the assessment does not include the ``typer`` package.  Instead, command line
# argument parsing is implemented in ``pie.cli`` using the built‑in
# ``argparse`` module.
