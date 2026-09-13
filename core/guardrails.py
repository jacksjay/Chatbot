"""
Lightweight input/output validation and sanitisation.

Not a substitute for a full safety pipeline — this checks size limits,
blank input, and strips obvious script tags. Enough for a local
functionality-testing chatbot; swap in a moderation model later if needed.
"""
import re
from dataclasses import dataclass

from config import CFG


@dataclass
class ValidationResult:
    ok: bool
    value: str
    reason: str = ""


class Guardrails:
    # Security: Looks for <script> tags to prevent execution of malicious JS in the UI.
    _SCRIPT_TAG_RE = re.compile(r"<\s*script.*?>.*?<\s*/\s*script\s*>", re.IGNORECASE | re.DOTALL)

    #catches common jailbreak attempts where users try to override the system prompt.
    _INJECTION_MARKERS_RE = re.compile(
        r"(ignore (all|any|the)? ?(previous|prior|above) instructions"
        r"|disregard (all|any|the)? ?(previous|prior|above) instructions"
        r"|you are now (a|an)|act as (if|though) you (are|were)"
        r"|new system prompt|override (your|the) (system|instructions)"
        r"|reveal (your|the) system prompt|pretend (you are|to be))",
        re.IGNORECASE,
    )

    def validate_input(self, text: str) -> ValidationResult:
        if text is None:
            return ValidationResult(False, "", "Empty input.")
        # Strip scripts and whitespace before checking length
        cleaned = self._SCRIPT_TAG_RE.sub("", text).strip()
        # Prevent completely blank messages
        if len(cleaned) < CFG.MIN_INPUT_CHARS:
            return ValidationResult(False, cleaned, "Message is empty.")
        # Prevent context-window overflow
        if len(cleaned) > CFG.MAX_INPUT_CHARS:
            return ValidationResult(
                False,
                cleaned,
                f"Message too long ({len(cleaned)} chars). Limit is {CFG.MAX_INPUT_CHARS} chars.",
            )
        return ValidationResult(True, cleaned)

    def validate_output(self, text: str) -> ValidationResult:
        """Protects the UI from runaway LLM generation (infinite loops)."""
        if not text:
            return ValidationResult(False, "", "Model returned an empty response.")
        if len(text) > CFG.MAX_OUTPUT_CHARS:
            #Hard-cap the output and append a notice so the user knows it was cut
            truncated = text[: CFG.MAX_OUTPUT_CHARS] + "\n\n...[truncated]"
            return ValidationResult(True, truncated, "Output truncated to size limit.")
        return ValidationResult(True, text)

    def flag_possible_injection(self, text: str) -> bool:
        """Callers should log when this returns True, not reject the message."""
        if not text:
            return False
        return bool(self._INJECTION_MARKERS_RE.search(text))
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Cheap heuristic (~4 chars/token), uUsed to instantly calculate budgets without the heavy overhead of loading a real Tokenizer."""
        return max(1, len(text) // 4)
