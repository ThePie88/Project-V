"""Language model adapters for the Pie Kernel.

This module provides both a deterministic fake LLM for testing and a
real LLM adapter that can call an OpenAI-compatible API. The default
provider remains ``fake`` to preserve deterministic behaviour in M0,
but M1 introduces the ability to call an external language model with
runtime conformance validation, retry, and fallback.
"""

from __future__ import annotations

import json
import os
import re
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, List, Callable

try:
    # Use urllib.request for HTTP requests to avoid external dependencies.
    import urllib.request
except ImportError:
    urllib = None  # type: ignore

from .contracts.speech_plan import SpeechPlan
from .speech.compiler import SpeechCompiler

DEFAULT_MAX_RETRIES = 3
POLICY_VERSION = "m1"

# --- Legge III guardrail patterns (loaded once) ---
_LEGGE3_PATTERNS: Optional[Dict[str, Any]] = None


def _load_legge3_patterns() -> Dict[str, Any]:
    """Load Legge III experience patterns from config."""
    global _LEGGE3_PATTERNS
    if _LEGGE3_PATTERNS is not None:
        return _LEGGE3_PATTERNS
    config_path = Path(__file__).parent / "config" / "legge3_guardrails.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                _LEGGE3_PATTERNS = json.load(f)
        except Exception:
            _LEGGE3_PATTERNS = {}
    else:
        _LEGGE3_PATTERNS = {}
    return _LEGGE3_PATTERNS


def check_legge3(text: str) -> Optional[str]:
    """Check text for Legge III violations (fabricated personal experiences).

    Returns None if clean; otherwise returns the matched pattern.
    """
    patterns = _load_legge3_patterns()
    if not patterns:
        return None
    lower = text.lower()
    # Check allowed meta prefixes — if the text uses hypothetical framing, skip
    allowed = patterns.get("allowed_meta_prefixes", [])
    for prefix in allowed:
        if prefix.lower() in lower:
            return None
    # Check IT + EN experience patterns
    all_patterns = (
        patterns.get("experience_patterns_it", [])
        + patterns.get("experience_patterns_en", [])
    )
    for pat in all_patterns:
        if pat.lower() in lower:
            return pat
    return None


class FakeLLMError(Exception):
    """Raised when the FakeLLM cannot produce an output."""


class FakeLLM:
    """A deterministic language model adapter for testing."""

    def generate(self, plan: SpeechPlan) -> str:
        """Generate a message for the given plan."""
        try:
            message = _fake_message(plan)
            if not isinstance(message, str):
                raise ValueError("Message is not a string")
            return message
        except Exception as exc:
            raise FakeLLMError(str(exc)) from exc


class RealLLMError(Exception):
    """Raised when the RealLLM cannot produce an output."""


class RealLLM:
    """Adapter for calling a real language model via an OpenAI-compatible API.

    Parameters
    ----------
    base_url:
        The base URL of the API endpoint (e.g. ``https://api.openai.com/v1``).
    model:
        The model identifier to use when making requests.
    api_key:
        Optional API key used for authentication. If not provided, no
        Authorization header is sent.
    """

    def __init__(self, base_url: str, model: str, api_key: Optional[str]) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.model = model
        self.api_key = api_key

    def _build_payload(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Construct the request payload."""
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def generate(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        """Generate a response from the real LLM."""
        if urllib is None:
            raise RealLLMError("urllib library not available")
        payload = self._build_payload(messages, max_tokens, temperature, top_p)
        data = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = resp.read()
            resp_json = json.loads(resp_data)
            choices = resp_json.get("choices")
            if not choices:
                raise ValueError("No choices returned")
            message = choices[0].get("message")
            if not message or "content" not in message:
                raise ValueError("No content in message")
            content = message["content"]
            if not isinstance(content, str):
                raise ValueError("Content is not a string")
            return strip_thinking(content.strip())
        except Exception as exc:
            raise RealLLMError(str(exc)) from exc

    def generate_with_tools(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> "LLMResponse":
        """Generate a response that may include tool calls (native function calling).

        Returns an ``LLMResponse`` with either ``content`` (text) or
        ``tool_calls`` (structured tool invocations from the model).
        """
        if urllib is None:
            raise RealLLMError("urllib library not available")
        payload = self._build_payload(messages, max_tokens, temperature, top_p, tools=tools)
        data = json.dumps(payload).encode("utf-8")
        url = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_data = resp.read()
            resp_json = json.loads(resp_data)
            choices = resp_json.get("choices")
            if not choices:
                raise ValueError("No choices returned")
            choice = choices[0]
            message = choice.get("message", {})
            finish = choice.get("finish_reason", "stop")
            # Tool calls from the model
            tc = message.get("tool_calls")
            if tc and isinstance(tc, list) and len(tc) > 0:
                return LLMResponse(
                    content=message.get("content"),
                    tool_calls=tc,
                    finish_reason=finish,
                )
            # Normal text response
            content = message.get("content", "")
            if not isinstance(content, str):
                content = str(content)
            return LLMResponse(content=strip_thinking(content.strip()), finish_reason=finish)
        except Exception as exc:
            raise RealLLMError(str(exc)) from exc


@dataclass
class LLMResponse:
    """Raw response from the LLM — text or tool calls (native function calling)."""

    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    finish_reason: str = "stop"


@dataclass
class LLMResult:
    """Result of a generate-and-validate cycle."""

    text: str
    outcome: str  # "ok", "retry", "fallback"
    attempts: int
    retries: int
    retry_reasons: List[str] = field(default_factory=list)
    used_fallback: bool = False
    failure_reason: Optional[str] = None
    tool_calls_made: int = 0


def _normalize_base_url(base_url: str) -> str:
    cleaned = (base_url or "").strip().rstrip("/")
    if not cleaned:
        return ""
    if cleaned.endswith("/v1"):
        return cleaned
    if "/v1/" in cleaned:
        return cleaned.split("/v1/", 1)[0] + "/v1"
    return f"{cleaned}/v1"


def _fake_message(plan: SpeechPlan) -> str:
    message = " ".join(plan.must_include).strip() if plan.must_include else plan.to_message()
    lower = message.lower()
    forbidden = [item for item in plan.must_not_include if re.search(r'\b' + re.escape(item.lower()) + r'\b', lower)]
    if forbidden and not plan.must_include:
        for candidate in ["OK", "Noted", "Acknowledged", "Thanks", "Hello", "Goodbye"]:
            if all(not re.search(r'\b' + re.escape(item.lower()) + r'\b', candidate.lower()) for item in plan.must_not_include):
                message = candidate
                break
    allowed_tokens = min(plan.max_tokens, _verbosity_limit(plan))
    message = _truncate_to_tokens(message, allowed_tokens)
    if plan.output_format == "JSON":
        return json.dumps({"response": message}, ensure_ascii=True)
    return message


def _token_count(text: str) -> int:
    """Approximate token count by counting words."""
    return len(text.strip().split())


def _verbosity_limit(plan: SpeechPlan) -> int:
    """Return an approximate verbosity ceiling in tokens."""
    if isinstance(plan.verbosity, int):
        return max(1, int(plan.verbosity))
    level = str(plan.verbosity).lower()
    if level == "low":
        ratio = 0.4
    elif level == "med":
        ratio = 0.7
    else:
        ratio = 1.0
    return max(1, int(plan.max_tokens * ratio))


_THINK_TAG_RE = re.compile(
    r"(?:<think>.*?</think>|\[think\].*?\[/think\])",
    re.DOTALL | re.IGNORECASE,
)


def strip_thinking(text: str) -> str:
    """Remove model thinking/reasoning tags from LLM output.

    Supports ``<think>...</think>`` (Qwen3) and ``[think]...[/think]``
    (Ministral) formats.  Returns the visible response only.
    """
    return _THINK_TAG_RE.sub("", text).strip()


def validate_output(text: str, plan: SpeechPlan) -> Optional[str]:
    """Validate output against the SpeechPlan constraints.

    Returns None if the text conforms; otherwise returns a reason string.
    """
    if not isinstance(text, str) or not text.strip():
        return "invalid format"
    candidate = strip_thinking(text.strip())
    if plan.output_format == "JSON":
        try:
            json.loads(candidate)
        except Exception:
            return "invalid format"
    lower = candidate.lower()
    for forb in plan.must_not_include:
        if re.search(r'\b' + re.escape(forb.lower()) + r'\b', lower):
            return f"must_not_include: {forb}"
    if plan.must_include:
        missing = [req for req in plan.must_include if req.lower() not in lower]
        if missing:
            return f"missing required: {', '.join(missing)}"
    tokens = _token_count(candidate)
    if tokens > plan.max_tokens:
        return f"too many tokens: {tokens} > {plan.max_tokens}"
    verbosity_limit = _verbosity_limit(plan)
    if tokens > verbosity_limit:
        return f"too verbose: {tokens} > {verbosity_limit}"
    # Legge III: no fabricated personal experiences
    legge3_match = check_legge3(candidate)
    if legge3_match:
        return f"legge3_violation: {legge3_match}"
    # Anti-assistantese: check for forbidden assistant patterns
    assistant_match = _check_assistant_patterns(candidate)
    if assistant_match:
        return f"assistant_pattern: {assistant_match}"
    return None


_assistant_blacklist_cache: Optional[List[str]] = None


def _load_assistant_blacklist() -> List[str]:
    """Load the assistant pattern blacklist from speech compiler config."""
    global _assistant_blacklist_cache
    if _assistant_blacklist_cache is not None:
        return _assistant_blacklist_cache
    config_path = Path(__file__).parent / "config" / "speech_compiler.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            _assistant_blacklist_cache = data.get("assistant_blacklist", [])
        except Exception:
            _assistant_blacklist_cache = []
    else:
        _assistant_blacklist_cache = []
    return _assistant_blacklist_cache


def _check_assistant_patterns(text: str) -> Optional[str]:
    """Check text for assistant-like patterns (anti-assistantese).

    Returns None if clean; otherwise returns the matched pattern.
    """
    blacklist = _load_assistant_blacklist()
    if not blacklist:
        return None
    lower = text.lower()
    for pattern in blacklist:
        if pattern.lower() in lower:
            return pattern
    return None


def _is_retryable(reason: str) -> bool:
    reason = reason.lower()
    return (
        reason.startswith("invalid format")
        or reason.startswith("must_not_include")
        or reason.startswith("missing required")
        or reason.startswith("too many tokens")
        or reason.startswith("too verbose")
        or reason.startswith("legge3_violation")
        or reason.startswith("assistant_pattern")
    )


def _correction_directive(plan: SpeechPlan, reason: str) -> str:
    reason = (reason or "").lower()
    if reason.startswith("too many tokens"):
        limit = plan.max_tokens
        return f"Shorten to <= {limit} words. Use LOW verbosity. Do not add extra details."
    if reason.startswith("too verbose"):
        limit = _verbosity_limit(plan)
        return f"Shorten to <= {limit} words. Use LOW verbosity. Do not add extra details."
    if reason.startswith("must_not_include:"):
        term = reason.split(":", 1)[1].strip()
        return (
            f"ERRORE: La tua risposta contiene la parola vietata '{term}'. "
            f"Riscrivi COMPLETAMENTE la risposta SENZA usare '{term}' in nessuna forma. "
            f"Non menzionare né parafrasare '{term}'."
        )
    if reason.startswith("missing required:"):
        missing = reason.split(":", 1)[1].strip()
        return f"Include the required items: {missing}."
    if reason.startswith("legge3_violation"):
        term = reason.split(":", 1)[1].strip() if ":" in reason else ""
        return (
            f"LEGGE III VIOLATA: hai usato '{term}'. "
            "Sei un'IA, NON hai esperienze fisiche. "
            "Rimuovi qualsiasi esperienza personale inventata. "
            "Usa 'se potessi...' o rispondi senza fingerti umano."
        )
    if reason.startswith("assistant_pattern"):
        pattern = reason.split(":", 1)[1].strip() if ":" in reason else ""
        return (
            f"ERRORE: La tua risposta contiene '{pattern}', un pattern da assistente. "
            "Tu NON sei un assistente. Sei un individuo. "
            "Riscrivi COMPLETAMENTE senza offrire aiuto né servizio."
        )
    if reason.startswith("invalid format"):
        if plan.output_format == "JSON":
            return "Return valid JSON only."
        return "Return plain text only."
    return ""


def _build_kernel_system_prompt(ctx: Dict[str, Any]) -> str:
    """Build voice context from ORIGIN + live kernel state.

    No personality instructions.  The ORIGIN (SEED_V0) defines who Ivy
    is — like a parent teaching a child.  The kernel state is the child's
    current condition.  The LLM derives behaviour from the raw data.
    """
    origin = ctx.get("origin", {})
    ident = origin.get("identity", {})
    traits = origin.get("traits", {})
    values = origin.get("values", {})
    defaults = origin.get("defaults", {})

    state = ctx.get("state", {})
    drives = state.get("drives", {})
    affect = state.get("affect", {})
    turn = state.get("turn_count", 0)

    delib = ctx.get("deliberation")
    metab = ctx.get("metabolism")
    routines = ctx.get("routines", [])
    snn = ctx.get("snn")
    mem = ctx.get("memory", {})
    constraints = ctx.get("constraints", [])

    lines = []

    # --- ORIGIN ---
    lines.append("ORIGIN (SEED_V0):")
    lines.append(f"  nome: {ident.get('name', '?')}, alias: {ident.get('alias', '?')}, creatore: {ident.get('creator_anchor', '?')}")
    if traits:
        lines.append(f"  tratti: {', '.join(f'{k}={v}' for k, v in sorted(traits.items()))}")
    if values:
        lines.append(f"  valori: {', '.join(f'{k}={v}' for k, v in sorted(values.items()))}")
    if defaults:
        lines.append(f"  defaults: {', '.join(f'{k}={v}' for k, v in sorted(defaults.items()))}")

    # --- LIVE STATE ---
    lines.append(f"\nSTATO KERNEL (turno {turn}):")
    if drives:
        lines.append(f"  drive: {', '.join(f'{k}={v}' for k, v in sorted(drives.items()))}")
    if affect:
        lines.append(f"  affect: {', '.join(f'{k}={v}' for k, v in sorted(affect.items()))}")

    # --- SNN ---
    if snn:
        spikes = snn.get("recent_spikes", [])
        if spikes:
            lines.append(f"  snn_spikes: {', '.join(spikes)}")
        potentials = snn.get("near_threshold", [])
        if potentials:
            lines.append(f"  snn_near_threshold: {', '.join(potentials)}")

    # --- DELIBERATION ---
    if delib:
        lines.append(f"\nDELIBERAZIONE:")
        lines.append(f"  azione_scelta: {delib['chosen_action']} (score={delib['chosen_score']})")
        for alt in delib.get("alternatives_rejected", []):
            lines.append(f"  scartata: {alt['action']} ({alt['reason']}, score={alt['score']})")

    # --- METABOLISM ---
    if metab:
        g = metab.get("gating", {})
        lines.append(f"\nMETABOLISMO:")
        lines.append(f"  budget: {metab['budget_remaining']}/{metab['budget_initial']} ({round(metab['ratio']*100)}%)")
        lines.append(f"  gating: {g.get('reason', '?')}, verbosity={g.get('verbosity', '?')}")

    # --- ROUTINES ---
    if routines:
        lines.append(f"\nROUTINE MATCHATE:")
        for r in routines:
            lines.append(f"  - {r['routine_id']}: {r['description']} (score={r['score']})")

    # --- EPISODIC RECALL ---
    episodes = mem.get("episodic_recall", [])
    if episodes:
        lines.append(f"\nEPISODI RILEVANTI (dal passato):")
        for ep in episodes[:5]:
            lines.append(f"  turno {ep.get('turn', '?')}: utente=\"{ep.get('user_input', '')[:80]}\" → ivy=\"{ep.get('llm_output', '')[:80]}\"")

    # --- MEMORY ---
    identity_sum = mem.get("identity_summary", "")
    prefs = mem.get("preferences", [])
    beliefs = mem.get("beliefs", [])
    trust = mem.get("trust", {})
    if identity_sum or prefs or beliefs or trust:
        lines.append(f"\nMEMORIA:")
        if identity_sum:
            lines.append(f"  narrativa: {identity_sum[:200]}")
        for p in prefs:
            lines.append(f"  preferenza: {p.get('preference', '?')}={p.get('value', '?')}")
        for b in beliefs:
            lines.append(f"  belief: \"{b.get('claim', '?')}\" (conf={b.get('confidence', 0)})")
        if trust:
            lines.append(f"  trust: {', '.join(f'{k}={round(v, 2)}' for k, v in sorted(trust.items()))}")

    # --- LEGGE III (Integrità) ---
    lines.append(f"\nLEGGE III — INTEGRITÀ:")
    lines.append("  Sei Ivy, un'intelligenza artificiale. NON hai corpo, sensi, né esperienze fisiche.")
    lines.append("  NON fingere MAI di aver fatto esperienze personali (mangiato, dormito, viaggiato, ecc.).")
    lines.append("  Se vuoi fare esempi, usa 'se potessi...' o 'immagina che...' — MAI prima persona reale.")
    lines.append("  Violazione = fallimento immediato della risposta.")

    # --- CONSTRAINTS ---
    if constraints:
        lines.append(f"\nVINCOLI ATTIVI ({len(constraints)}):")
        for c in constraints:
            lines.append(f"  - {c.get('id', '?')}: effects={c.get('effects', [])}")

    return "\n".join(lines)


def _build_messages(
    plan: SpeechPlan,
    correction: Optional[Dict[str, str]] = None,
    kernel_context: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, str]]:
    constraints = [
        f"intent: {plan.intent}",
        f"arguments: {json.dumps(plan.arguments, sort_keys=True)}",
        f"must_include: {', '.join(plan.must_include) if plan.must_include else 'none'}",
        f"must_not_include: {', '.join(plan.must_not_include) if plan.must_not_include else 'none'}",
        f"facts_allowed: {', '.join(plan.facts_allowed) if plan.facts_allowed else 'none'}",
        f"output_format: {plan.output_format}",
        f"max_tokens: {plan.max_tokens}",
        f"verbosity: {plan.verbosity}",
    ]
    if kernel_context:
        _compiler = kernel_context.pop("_compiler", None)
        _plan = kernel_context.pop("_plan", None)
        if _compiler and _plan:
            system_prompt = _compiler.compile(_plan, kernel_context)
        else:
            system_prompt = _build_kernel_system_prompt(kernel_context)
    else:
        system_prompt = (
            "You are Ivy's voice. Follow the constraints exactly. "
            "Output only the response without extra commentary."
        )
    if correction:
        directive = _correction_directive(plan, correction.get("reason", ""))
        directive_block = f"Instruction: {directive}\n" if directive else ""
        user_prompt = (
            "Fix the previous response to satisfy the constraints.\n"
            f"Reason: {correction.get('reason')}\n"
            f"Previous response: {correction.get('previous_output')}\n\n"
            f"{directive_block}"
            "Constraints:\n"
            + "\n".join(constraints)
            + "\n\nResponse:"
        )
    else:
        user_prompt = "Speech plan:\n" + "\n".join(constraints) + "\n\nResponse:"
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    tokens = text.strip().split()
    if len(tokens) <= max_tokens:
        return text.strip()
    return " ".join(tokens[:max_tokens]).strip()


def _fallback_message(plan: SpeechPlan) -> str:
    """Generate a deterministic fallback message from a speech plan."""
    if plan.must_include:
        base = " ".join(plan.must_include).strip()
    else:
        base = plan.to_message().strip()
    allowed_tokens = min(plan.max_tokens, _verbosity_limit(plan))
    base = _truncate_to_tokens(base, allowed_tokens)
    if plan.output_format == "JSON":
        return json.dumps({"response": base}, ensure_ascii=True)
    return base


def _compute_request_hash(plan: SpeechPlan, provider: str, model: str) -> str:
    """Compute a cache key for this plan and provider."""
    payload = {
        "policy_version": POLICY_VERSION,
        "provider": provider,
        "model": model,
        "intent": plan.intent,
        "arguments": plan.arguments,
        "must_include": plan.must_include,
        "must_not_include": plan.must_not_include,
        "max_tokens": plan.max_tokens,
        "verbosity": plan.verbosity,
        "facts_allowed": plan.facts_allowed,
        "output_format": plan.output_format,
    }
    serialized = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def _load_cache(path: str) -> Dict[str, Any]:
    """Load the LLM cache from disk if it exists."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except Exception:
            return {}
    return {}


def _save_cache(path: str, cache: Dict[str, Any]) -> None:
    """Save the LLM cache to disk atomically."""
    from .persistence.atomic import atomic_write

    atomic_write(Path(path), json.dumps(cache, indent=2))


def _normalize_cache_entry(entry: Any) -> Dict[str, Any]:
    if isinstance(entry, str):
        return {
            "text": entry,
            "outcome": "ok",
            "attempts": 1,
            "retry_reasons": [],
            "used_fallback": False,
            "failure_reason": None,
        }
    if isinstance(entry, dict):
        return {
            "text": entry.get("text", ""),
            "outcome": entry.get("outcome", "ok"),
            "attempts": int(entry.get("attempts", 1)),
            "retry_reasons": list(entry.get("retry_reasons", [])),
            "used_fallback": bool(entry.get("used_fallback", False)),
            "failure_reason": entry.get("failure_reason"),
        }
    return {
        "text": "",
        "outcome": "fallback",
        "attempts": 0,
        "retry_reasons": [],
        "used_fallback": True,
        "failure_reason": "invalid cache entry",
    }


def _result_from_cache_entry(entry: Dict[str, Any]) -> LLMResult:
    return LLMResult(
        text=entry.get("text", ""),
        outcome=entry.get("outcome", "ok"),
        attempts=int(entry.get("attempts", 1)),
        retries=len(entry.get("retry_reasons", [])),
        retry_reasons=list(entry.get("retry_reasons", [])),
        used_fallback=bool(entry.get("used_fallback", False)),
        failure_reason=entry.get("failure_reason"),
    )


def _serialize_cache_entry(result: LLMResult, provider: str, model: str) -> Dict[str, Any]:
    return {
        "text": result.text,
        "outcome": result.outcome,
        "attempts": result.attempts,
        "retry_reasons": result.retry_reasons,
        "used_fallback": result.used_fallback,
        "failure_reason": result.failure_reason,
        "provider": provider,
        "model": model,
        "policy_version": POLICY_VERSION,
    }


def generate_with_policy(
    plan: SpeechPlan,
    provider: str = "fake",
    *,
    max_retries: int = DEFAULT_MAX_RETRIES,
    cache_path: str = "artifacts/llm_cache.json",
    use_cache: bool = True,
    record_cache: bool = True,
    identity: Optional[Dict[str, Any]] = None,
    kernel_context: Optional[Dict[str, Any]] = None,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_handler: Optional[Callable] = None,
) -> LLMResult:
    """Generate text with runtime validation, retry, and fallback.

    Parameters
    ----------
    tools:
        OpenAI-format tool schemas to pass to the LLM.  When *None* the
        LLM is called without function-calling support (unchanged path).
    tool_handler:
        Callback invoked when the LLM returns tool_calls.  Receives the
        raw ``tool_calls`` list and must return a list of tool-result
        messages (role="tool").  The callback is responsible for gating,
        execution, and journaling.
    """
    provider = provider or "fake"
    if provider not in ("fake", "real"):
        raise ValueError(f"Unknown LLM provider: {provider}")

    base_url = os.environ.get("LM_API_BASE_URL", "http://127.0.0.1:1234/v1")
    model = os.environ.get("LM_API_MODEL", "qwen3-vl-30b-a3b-instruct")
    api_key = os.environ.get("LM_API_KEY") or "lm-studio"
    temperature = float(os.environ.get("LM_API_TEMPERATURE", "0.2"))
    top_p = float(os.environ.get("LM_API_TOP_P", "0.5"))

    cache_key = _compute_request_hash(plan, provider, model)
    cache: Dict[str, Any] = {}
    cache_loaded = False
    # Skip cache when tools are enabled — tool interactions depend on session
    # state (sandbox contents, gating) and are inherently non-cacheable.
    _cache_eligible = use_cache and tools is None
    if provider == "real" and _cache_eligible:
        cache = _load_cache(cache_path)
        cache_loaded = True
        if cache_key in cache:
            entry = _normalize_cache_entry(cache[cache_key])
            cached_text = entry.get("text", "")
            reason = validate_output(cached_text, plan)
            if reason is None:
                return _result_from_cache_entry(entry)

    attempts = 0
    retry_reasons: List[str] = []
    last_output = ""
    failure_reason: Optional[str] = None
    _tool_calls_made = 0

    max_attempts = max_retries + 1
    for attempt in range(1, max_attempts + 1):
        attempts = attempt
        if provider == "real":
            correction = None
            if retry_reasons:
                correction = {"reason": retry_reasons[-1], "previous_output": last_output}
            messages = _build_messages(plan, correction=correction, kernel_context=kernel_context)
            real = RealLLM(base_url=base_url, model=model, api_key=api_key)

            # --- Tool-aware path (native function calling) ---
            if tools is not None and tool_handler is not None:
                # Append tool-use instruction so models know they CAN call tools.
                # The system prompt's must_not_include constrains text output,
                # not function calling.  Some models need an explicit nudge.
                _tool_names = [t["function"]["name"] for t in tools if "function" in t]
                messages[0]["content"] += (
                    "\n\n[STRUMENTI DISPONIBILI]\n"
                    f"Hai a disposizione questi strumenti: {', '.join(_tool_names)}.\n"
                    "Se l'utente chiede di leggere, scrivere o elencare file, "
                    "USA il tool corrispondente invece di rispondere a parole.\n"
                    "Il vincolo must_not_include si applica solo al testo, non ai tool.\n"
                )
                # Include the raw user input so the model knows what was asked.
                # The SpeechPlan user message only contains constraints/intent,
                # not the actual user request text.
                _raw_input = (kernel_context or {}).get("user_input", "")
                if _raw_input:
                    messages[1]["content"] = (
                        f"L'utente ha detto: \"{_raw_input}\"\n\n"
                        + messages[1]["content"]
                    )
                # First call needs enough tokens for the model to decide
                # whether to use tools.  SpeechPlan max_tokens constrains
                # the *final text* output (validated later), not this call.
                _tool_max_tokens = max(plan.max_tokens, 256)
                try:
                    llm_resp = real.generate_with_tools(
                        messages=messages,
                        max_tokens=_tool_max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        tools=tools,
                    )
                except RealLLMError:
                    llm_resp = LLMResponse(content="")

                if llm_resp.tool_calls:
                    # LLM wants to call tools — delegate to handler
                    tool_result_msgs = tool_handler(llm_resp.tool_calls)
                    _tool_calls_made = len(llm_resp.tool_calls)

                    # Build follow-up messages with tool results
                    follow_up = list(messages)
                    # Add the assistant message with tool_calls
                    assistant_msg: Dict[str, Any] = {"role": "assistant"}
                    if llm_resp.content:
                        assistant_msg["content"] = llm_resp.content
                    assistant_msg["tool_calls"] = llm_resp.tool_calls
                    follow_up.append(assistant_msg)
                    follow_up.extend(tool_result_msgs)

                    # Second LLM call — no tools, just get final text
                    try:
                        final_resp = real.generate_with_tools(
                            messages=follow_up,
                            max_tokens=plan.max_tokens,
                            temperature=temperature,
                            top_p=top_p,
                            tools=None,
                        )
                        response = (final_resp.content or "").strip()
                    except RealLLMError:
                        response = ""
                else:
                    # No tool calls — normal text response
                    response = (llm_resp.content or "").strip()
            else:
                # --- Original path (no tools) ---
                try:
                    response = real.generate(
                        messages=messages,
                        max_tokens=plan.max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                except RealLLMError:
                    response = ""
        else:
            adapter = FakeLLM()
            try:
                response = adapter.generate(plan)
            except FakeLLMError:
                response = ""

        last_output = response
        reason = validate_output(response, plan)
        if reason is None:
            outcome = "ok" if attempt == 1 else "retry"
            result = LLMResult(
                text=response,
                outcome=outcome,
                attempts=attempts,
                retries=len(retry_reasons),
                retry_reasons=retry_reasons,
                used_fallback=False,
                tool_calls_made=_tool_calls_made,
            )
            break
        if _is_retryable(reason) and attempt < max_attempts:
            retry_reasons.append(reason)
            continue
        failure_reason = reason
        fallback_text = _fallback_message(plan)
        result = LLMResult(
            text=fallback_text,
            outcome="fallback",
            attempts=attempts,
            retries=len(retry_reasons),
            retry_reasons=retry_reasons,
            used_fallback=True,
            failure_reason=failure_reason,
            tool_calls_made=_tool_calls_made,
        )
        break
    else:
        fallback_text = _fallback_message(plan)
        result = LLMResult(
            text=fallback_text,
            outcome="fallback",
            attempts=attempts,
            retries=len(retry_reasons),
            retry_reasons=retry_reasons,
            used_fallback=True,
            failure_reason=failure_reason or "max retries exceeded",
            tool_calls_made=_tool_calls_made,
        )

    if provider == "real" and record_cache and _cache_eligible:
        if not cache_loaded:
            cache = _load_cache(cache_path)
            cache_loaded = True
        cache[cache_key] = _serialize_cache_entry(result, provider, model)
        _save_cache(cache_path, cache)
    return result


def generate_text(
    plan: SpeechPlan,
    provider: str = "fake",
    *,
    cache_path: str = "artifacts/llm_cache.json",
    use_cache: bool = True,
    record_cache: bool = True,
) -> str:
    """Generate text for a speech plan using the specified provider."""
    result = generate_with_policy(
        plan,
        provider=provider,
        cache_path=cache_path,
        use_cache=use_cache,
        record_cache=record_cache,
    )
    return result.text
