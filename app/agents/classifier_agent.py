"""
Compatibility classifier agent aligned with the active website pipeline.

This module is kept for older imports, but it now follows the current
four-class taxonomy and uses the routed LLM client instead of the legacy
direct Anthropic-only path.
"""

from __future__ import annotations

import json
import time

from app.agents.llm_router import RoutedLLMClient, LLMRouterError
from src.data.generation_lexicon import expand_generation_language


_SYSTEM_PROMPT = """\
You are a safety-focused NLP classifier for social media text.
Respond ONLY with a single valid JSON object.

Risk levels:
- HIGH_RISK_SELF_HARM: explicit suicidal intent, plans, farewell language, or desire to die
- HIGH_RISK_HARM_TO_OTHERS: explicit threats or intent to kill or seriously harm another person
- MODERATE_RISK: hopelessness, burden, severe distress, indirect self-harm language
- LOW_RISK: everyday, neutral, joking, or non-clinical content without meaningful risk evidence

Required JSON schema:
{
  "risk_level": "HIGH_RISK_SELF_HARM|HIGH_RISK_HARM_TO_OTHERS|MODERATE_RISK|LOW_RISK",
  "confidence": <float 0.0-1.0>,
  "risk_score": <int 0-100>,
  "reasoning": "<brief reasoning>"
}
"""


class ClassifierAgent:
    """Legacy compatibility wrapper for risk classification."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.client = RoutedLLMClient()
        self.model = model
        self.system_prompt = _SYSTEM_PROMPT

    def __repr__(self) -> str:
        return f"ClassifierAgent(model={self.model!r})"

    def classify(self, text: str) -> dict:
        start = time.time()
        normalized = expand_generation_language(text)
        user_message = (
            "Classify this social media text.\n\n"
            f"Original text:\n{text}\n\n"
            f"Normalized text:\n{normalized}"
        )

        try:
            response = self.client.create_message(
                model=self.model,
                max_tokens=220,
                system=self.system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            result = json.loads(response.text.strip())
            result["model"] = response.model
            result["provider"] = response.provider
            result["input_tokens"] = response.input_tokens
            result["output_tokens"] = response.output_tokens
            result["processing_time_ms"] = int((time.time() - start) * 1000)
            return result
        except (json.JSONDecodeError, LLMRouterError) as exc:
            return {
                "risk_level": "LOW_RISK",
                "confidence": 0.0,
                "risk_score": 0,
                "reasoning": "Classification unavailable.",
                "model": self.model,
                "input_tokens": 0,
                "output_tokens": 0,
                "processing_time_ms": int((time.time() - start) * 1000),
                "error": True,
                "error_detail": str(exc),
            }

    def classify_batch(self, texts: list[str]) -> list[dict]:
        return [self.classify(text) for text in texts]
