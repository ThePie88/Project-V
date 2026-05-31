"""Speech plan contract for the Pie Kernel.

Speech plans describe the content and intent of a response before it is
passed to the language model adapter.  Plans are deterministic in M0
and include enough information for a FakeLLM to render a voice output.

The ``schema_version`` field must be bumped when backwards
incompatible changes are made to the model.
"""

from __future__ import annotations

from typing import Dict, Literal, List, Union

from pydantic import BaseModel, Field


class SpeechPlan(BaseModel):
    """A structured plan for speech output.

    Attributes
    ----------
    schema_version:
        Version of the speech plan schema.  Allows detection of
        incompatible plans in older traces.
    plan_id:
        A monotonically increasing identifier for the plan within a run.
    intent:
        A short string describing the intent of the utterance, e.g.
        ``"greeting"``, ``"answer"`` or ``"farewell"``.
    arguments:
        A dictionary of arguments relevant to the intent.  For example,
        for an ``"answer"`` intent the arguments might include the
        question or topic being answered.
    must_include:
        Substrings that must appear in the output.
    must_not_include:
        Substrings that must never appear in the output.
    max_tokens:
        Maximum length allowed for the output (approximate tokens).
    verbosity:
        Target verbosity level (integer budget or low/med/high).
    facts_allowed:
        Optional list of whitelisted facts the voice may mention.
    output_format:
        Output format expected from the voice.
    """

    schema_version: Literal["0.1"] = Field(
        "0.1",
        description="Schema version for SpeechPlan; bump on backwards incompatible change",
    )
    plan_id: int = Field(..., ge=0, description="Monotonically increasing plan identifier")
    intent: str = Field(..., description="A short label for the speech intent")
    arguments: Dict[str, str] = Field(
        default_factory=dict, description="Intent‑specific arguments for the speech plan"
    )

    must_include: List[str] = Field(
        default_factory=list, description="Required substrings that must appear in output"
    )
    must_not_include: List[str] = Field(
        default_factory=list, description="Forbidden substrings that must not appear in output"
    )
    max_tokens: int = Field(
        64, ge=1, description="Maximum approximate token count for the output"
    )
    verbosity: Union[int, Literal["low", "med", "high"]] = Field(
        "med", description="Verbosity target (integer or low/med/high)"
    )
    facts_allowed: List[str] = Field(
        default_factory=list, description="Whitelisted facts allowed in the output"
    )
    output_format: Literal["TEXT", "JSON"] = Field(
        "TEXT", description="Expected output format"
    )

    def to_message(self) -> str:
        """Convert this speech plan into a deterministic message.

        The conversion is deliberately simple: it constructs a message
        based on the intent and the provided arguments.  In a real
        implementation this method would be part of the LLM adapter and
        call into a language model.  For M0 it returns a fixed format.
        """
        if self.intent == "greeting":
            message = "Hello!"
        elif self.intent == "farewell":
            message = "Goodbye."
        elif self.intent == "answer":
            # Use a naive template with the topic argument.
            topic = self.arguments.get("topic", "").strip()
            if topic:
                if self.verbosity == "low" or (isinstance(self.verbosity, int) and self.verbosity <= 10):
                    message = f"Brief answer about {topic}."
                else:
                    message = f"Here is a short answer about {topic}."
            else:
                if self.verbosity == "low" or (isinstance(self.verbosity, int) and self.verbosity <= 10):
                    message = "Brief answer."
                else:
                    message = "Here is a short answer."
        elif self.intent == "acknowledge":
            message = "I understand."
        else:
            # Fallback for unknown intents
            message = "I'm not sure how to respond."

        if self.must_include:
            lower = message.lower()
            missing = [req for req in self.must_include if req.lower() not in lower]
            if missing:
                message = f"{message} {' '.join(missing)}".strip()
        return message
