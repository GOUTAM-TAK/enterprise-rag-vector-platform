from typing import Optional

class GuardrailUtils:
    """
    Enterprise Guardrail Utilities.

    Responsibilities:
    - Detect hard stop decisions
    - Extract safe user-facing messages
    - Guarantee refusal text even when LLM is not called
    """

    DEFAULT_REFUSAL_MESSAGE = (
        "I'm sorry, I can’t help with that request."
    )

    @staticmethod
    def is_guardrail_blocked(response) -> bool:
        """
        Returns True if any activated rail issued a stop decision.
        """
        log = getattr(response, "log", None)
        if not log or not log.activated_rails:
            return False

        return any(rail.stop for rail in log.activated_rails)
    
    @staticmethod
    def extract_guardrail_text(response):
        """
        Extracts a user-facing message safely.

        Priority:
        1. Assistant response (from say_refusal)
        2. Enterprise fallback message
        """

        # Case 1: Guardrails produced an assistant message
        response_data = getattr(response, "response", None)

        if response_data:
            # Typical case: list of messages
            if isinstance(response_data, list):
                for msg in response_data:
                    if msg.get("role") == "assistant":
                        content = msg.get("content", "").strip()
                        if content:
                            return content

            # Less common: raw string
            elif isinstance(response_data, str):
                if response_data.strip():
                    return response_data.strip()

        # Case 2: Hard stop without dialog/output execution
        return GuardrailUtils.DEFAULT_REFUSAL_MESSAGE
