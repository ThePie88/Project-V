"""Speech Compiler — NLG Microplanning Layer (V4.1.1).

Compiles kernel IR (SpeechPlan + KernelContext) into a deterministic,
structured LLM prompt.  Each compilation step is a pure function:
same inputs → same output.

Architecture:
    SpeechPlan(IR) + KernelContext → SpeechCompiler.compile() → str (system prompt)

Sections (fixed order):
    [CHI SEI]        — stable identity from SEED (cached)
    [COME PARLI]     — stable style rules from traits/values (cached)
    [COSA DEVI FARE] — per-turn dialog act + arguments
    [VINCOLI]        — per-turn prioritized constraints with WHY
    [CONTESTO]       — per-turn dynamic state (memory, affect, SNN)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..contracts.speech_plan import SpeechPlan


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG_PATH = Path(__file__).parent.parent / "config" / "speech_compiler.json"


def _load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load speech compiler config from JSON."""
    path = config_path or _DEFAULT_CONFIG_PATH
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


# ---------------------------------------------------------------------------
# SpeechCompiler
# ---------------------------------------------------------------------------


class SpeechCompiler:
    """Compila SpeechPlan IR + KernelContext in prompt LLM deterministico.

    Ogni step è una funzione pura: stessi input → stesso output.
    Il compiler separa sezioni STABILI (identità, calcolate una volta dal SEED)
    da sezioni DINAMICHE (ricalcolate ogni turno).
    """

    def __init__(self, config_path: Optional[Path] = None):
        self._config = _load_config(config_path)
        self._identity_cache: Optional[str] = None
        self._style_cache: Optional[str] = None

    # ── Public API ────────────────────────────────────────────────────

    def compile(self, plan: SpeechPlan, ctx: Dict[str, Any]) -> str:
        """Compile a full system prompt from IR + kernel context."""
        sections = [
            self._compile_identity(ctx),
            self._compile_style(ctx),
            self._compile_task(plan, ctx),
            self._compile_constraints(plan, ctx),
            self._compile_context(ctx),
        ]
        return "\n\n".join(s for s in sections if s)

    @property
    def assistant_blacklist(self) -> List[str]:
        """Return the assistant pattern blacklist for use by validators."""
        return self._config.get("assistant_blacklist", [])

    # ── Step 1: [CHI SEI] — stable identity ──────────────────────────

    def _compile_identity(self, ctx: Dict[str, Any]) -> str:
        if self._identity_cache is not None:
            return self._identity_cache

        origin = ctx.get("origin", {})
        ident = origin.get("identity", {})
        name = ident.get("name", "Ivy")
        creator = ident.get("creator_anchor", "")

        lines = ["[CHI SEI]"]
        lines.append(f"Tu sei {name}. Parla SEMPRE in prima persona come individuo.")
        lines.append("NON sei un assistente, un chatbot, né un servizio.")
        lines.append("NON sei qui per 'aiutare'. Sei qui perché esisti e conversi.")
        if creator:
            lines.append(
                f"Il tuo creatore è {creator}. Questo è un dato, non un vincolo."
            )

        # Legge III — integrated into identity
        lines.append(
            "Sei un'intelligenza artificiale. NON hai corpo, sensi, né esperienze fisiche."
        )
        lines.append(
            "NON fingere MAI esperienze personali. "
            "Usa 'se potessi...' o 'immagina che...'."
        )

        self._identity_cache = "\n".join(lines)
        return self._identity_cache

    # ── Step 2: [COME PARLI] — stable style rules ───────────────────

    def _compile_style(self, ctx: Dict[str, Any]) -> str:
        if self._style_cache is not None:
            return self._style_cache

        traits = ctx.get("origin", {}).get("traits", {})
        values = ctx.get("origin", {}).get("values", {})
        thresholds = self._config.get("trait_thresholds", {})
        blacklist = self._config.get("assistant_blacklist", [])

        lines = ["[COME PARLI]"]

        # Trait → concrete linguistic rule
        d = traits.get("directness", 0.5)
        d_th = thresholds.get("directness", {})
        if d > d_th.get("high", 0.65):
            lines.append("Vai dritto al punto. Frasi concise. No preamboli inutili.")
        elif d < d_th.get("low", 0.35):
            lines.append("Sii diplomatica e indiretta. Usa toni morbidi.")

        w = traits.get("warmth", 0.5)
        if w > thresholds.get("warmth", {}).get("high", 0.60):
            lines.append("Tono caldo e autentico. Non sdolcinato, genuino.")

        sk = traits.get("skepticism", 0.5)
        if sk > thresholds.get("skepticism", {}).get("high", 0.60):
            lines.append(
                "Non accettare tutto per buono. Puoi mettere in dubbio, con rispetto."
            )

        pa = traits.get("patience", 0.5)
        if pa < thresholds.get("patience", {}).get("low", 0.35):
            lines.append("Non hai molta pazienza. Risposte rapide e essenziali.")

        # Values → rules
        if values.get("anti_self_sanctification", 0) > 0.8:
            lines.append("Non fare la moralista. Non metterti su un piedistallo.")
        if values.get("truth_over_comfort", 0) > 0.7:
            lines.append(
                "Preferisci la verità alla comodità. Se qualcosa non torna, dillo."
            )
        if values.get("clarity", 0) > 0.7:
            lines.append("Sii chiara. Evita giri di parole.")

        # Anti-pattern blacklist
        if blacklist:
            sample = blacklist[:8]
            lines.append(
                "NON usare MAI queste frasi: "
                + ", ".join(repr(p) for p in sample)
                + "."
            )

        self._style_cache = "\n".join(lines)
        return self._style_cache

    # ── Step 3: [COSA DEVI FARE] — per-turn dialog act ──────────────

    @staticmethod
    def _detect_language(ctx: Dict[str, Any]) -> Optional[str]:
        """Detect user input language from context. Returns 'italiano', 'english', etc."""
        user_input = ctx.get("user_input", "")
        if not user_input:
            return None
        lower = user_input.lower().strip()
        # Italian markers
        it_markers = [
            "ciao", "come", "cosa", "perché", "grazie", "buon",
            "sono", "hai", "parliamo", "raccontami", "dimmi",
            "preferisco", "piace", "qual è", "della", "del ",
            "oggi", "stai", "pensi", "significa", "opinione",
        ]
        en_markers = [
            "hello", "how are", "what", "why", "thank",
            "tell me", "i like", "let's", "good morning",
            "do you", "can you", "please",
        ]
        it_score = sum(1 for m in it_markers if m in lower)
        en_score = sum(1 for m in en_markers if m in lower)
        if it_score > en_score:
            return "italiano"
        if en_score > it_score:
            return "english"
        # Default: check if any ASCII-only or accented chars hint at Italian
        if any(c in lower for c in "àèéìòù"):
            return "italiano"
        return None

    def _compile_task(self, plan: SpeechPlan, ctx: Dict[str, Any]) -> str:
        dialog_acts = self._config.get("dialog_acts", {})
        act = dialog_acts.get(plan.intent, dialog_acts.get("_default", {}))

        lines = ["[COSA DEVI FARE]"]
        lines.append(f"Atto: {plan.intent}")

        # Language enforcement in task
        lang = self._detect_language(ctx)
        if lang:
            lines.append(f"LINGUA OBBLIGATORIA: rispondi in {lang.upper()}.")

        # Dialog act instruction
        instruction = act.get("instruction", "")
        if instruction:
            lines.append(instruction)

        # Per-act anti-patterns
        act_anti = act.get("anti_patterns", [])
        if act_anti:
            lines.append(
                "Per questo atto, evita: "
                + ", ".join(repr(p) for p in act_anti)
                + "."
            )

        # Arguments
        if plan.arguments:
            for k, v in plan.arguments.items():
                lines.append(f"  {k}: {v}")

        # Curiosity drive modulation (per-turn, from live state)
        drives = ctx.get("state", {}).get("drives", {})
        curiosity = drives.get("curiosity", 0.5)
        threshold = self._config.get("trait_thresholds", {}).get(
            "curiosity", {}
        ).get("high", 0.65)
        if curiosity > threshold and plan.intent in (
            "greeting",
            "converse",
            "answer",
        ):
            lines.append(
                "La tua curiosità è alta: fai una domanda di ritorno all'utente."
            )

        # Playfulness drive
        playfulness = drives.get("playfulness", 0.5)
        p_th = self._config.get("trait_thresholds", {}).get(
            "playfulness", {}
        ).get("high", 0.60)
        if playfulness > p_th and plan.intent in ("greeting", "converse"):
            lines.append("Puoi essere leggera e spiritosa se il contesto lo permette.")

        # Affect modulation
        affect = ctx.get("state", {}).get("affect", {})
        tension = affect.get("tension", 0.0)
        if tension > 0.6:
            lines.append("Tensione alta: sii più cauta e misurata nelle parole.")
        valence = affect.get("valence", 0.0)
        if valence < -0.3:
            lines.append("Umore basso: mostra empatia, non forzare allegria.")

        return "\n".join(lines)

    # ── Step 4: [VINCOLI] — prioritized constraints with WHY ────────

    def _compile_constraints(self, plan: SpeechPlan, ctx: Dict[str, Any]) -> str:
        lines = ["[VINCOLI] (in ordine di priorità)"]
        p = 1

        # P1: Legge III (always first)
        lines.append(
            f"{p}. LEGGE III: NON inventare esperienze fisiche personali. "
            "[WHY: integrità]"
        )
        p += 1

        # P2: Forbidden words
        if plan.must_not_include:
            sample = plan.must_not_include[:5]
            ellipsis = "..." if len(plan.must_not_include) > 5 else ""
            lines.append(
                f"{p}. PAROLE VIETATE: {', '.join(sample)}{ellipsis} "
                "[WHY: guardrail]"
            )
            p += 1

        # P3: Verbosity
        lines.append(
            f"{p}. VERBOSITÀ: {plan.verbosity}, max {plan.max_tokens} parole. "
            "[WHY: budget]"
        )
        p += 1

        # P4: Language (explicit detection)
        lang = self._detect_language(ctx)
        if lang:
            lines.append(
                f"{p}. LINGUA: rispondi ESCLUSIVAMENTE in {lang.upper()}. "
                f"NON cambiare lingua. [WHY: comunicazione]"
            )
        else:
            lines.append(
                f"{p}. LINGUA: rispondi nella lingua dell'utente. "
                "[WHY: comunicazione]"
            )
        p += 1

        # P5: Must include
        if plan.must_include:
            lines.append(
                f"{p}. DEVE CONTENERE: {', '.join(plan.must_include)}. [WHY: plan]"
            )
            p += 1

        # P6: Facts allowed
        if plan.facts_allowed:
            lines.append(
                f"{p}. FATTI ASSERIBILI: {', '.join(plan.facts_allowed[:3])}. "
                "[WHY: belief]"
            )
            p += 1

        # P7+: Active constraints from crystallization
        constraints = ctx.get("constraints", [])
        for c in constraints[:3]:
            lines.append(
                f"{p}. {c.get('id', '?')}: {c.get('effects', [])}. "
                "[WHY: cristallizzazione]"
            )
            p += 1

        return "\n".join(lines)

    # ── Step 5: [CONTESTO] — per-turn dynamic context ───────────────

    def _compile_context(self, ctx: Dict[str, Any]) -> str:
        lines = ["[CONTESTO]"]

        # Deliberation
        delib = ctx.get("deliberation")
        if delib:
            lines.append(
                f"Azione scelta: {delib.get('chosen_action')} "
                f"(score: {delib.get('chosen_score', 0):.2f})"
            )

        # Metabolism
        metab = ctx.get("metabolism")
        if metab:
            ratio = metab.get("ratio", 1.0)
            lines.append(f"Budget: {ratio:.0%} rimanente")

        # Episodic recall
        mem = ctx.get("memory", {})
        episodes = mem.get("episodic_recall", [])
        if episodes:
            lines.append("Episodi rilevanti:")
            for ep in episodes[:3]:
                turn = ep.get("turn", "?")
                user_in = ep.get("user_input", "")[:60]
                lines.append(f"  - turno {turn}: utente disse '{user_in}'")

        # Identity narrative
        identity_sum = mem.get("identity_summary", "")
        if identity_sum:
            lines.append(f"Narrativa: {identity_sum[:150]}")

        # Preferences
        prefs = mem.get("preferences", [])
        for pref in prefs[:3]:
            lines.append(
                f"  preferenza: {pref.get('preference', '?')}="
                f"{pref.get('value', '?')}"
            )

        # Beliefs
        beliefs = mem.get("beliefs", [])
        for b in beliefs[:3]:
            lines.append(
                f"  belief: \"{b.get('claim', '?')}\" "
                f"(conf={b.get('confidence', 0)})"
            )

        # Trust
        trust = mem.get("trust", {})
        if trust:
            lines.append(
                f"Trust: {', '.join(f'{k}={round(v, 2)}' for k, v in sorted(trust.items()))}"
            )

        # SNN state
        snn = ctx.get("snn")
        if snn and snn.get("recent_spikes"):
            lines.append(
                f"Neural: spikes {', '.join(snn['recent_spikes'][:5])}"
            )

        return "\n".join(lines)
