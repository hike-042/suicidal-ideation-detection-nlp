"""
Compatibility explainer agent aligned with the active website pipeline.
"""

from __future__ import annotations

import json
import time

from app.agents.llm_router import RoutedLLMClient, LLMRouterError


_SYSTEM_PROMPT = """\
You are a linguistic explanation agent for social media safety analysis.
Respond ONLY with a single valid JSON object.

Required JSON schema:
{
  "risk_indicators": ["<phrase>"],
  "protective_factors": ["<phrase>"],
  "linguistic_patterns": {
    "hopelessness": <bool>,
    "social_isolation": <bool>,
    "self_blame": <bool>,
    "finality_language": <bool>,
    "help_seeking": <bool>,
    "emotional_dysregulation": <bool>
  },
  "sentiment_summary": "<brief description>",
  "key_phrases": ["<phrase>"]
}
"""


class ExplainerAgent:
    """Legacy compatibility wrapper for explanation generation."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.client = RoutedLLMClient()
        self.model = model
        self.system_prompt = _SYSTEM_PROMPT

    def __repr__(self) -> str:
        return f"ExplainerAgent(model={self.model!r})"

    def explain(self, text: str, risk_level: str) -> dict:
        start = time.time()
        user_message = (
            f"Text:\n{text}\n\n"
            f"Risk level:\n{risk_level}\n\n"
            "Explain the linguistic markers in the required JSON schema."
        )

        try:
            response = self.client.create_message(
                model=self.model,
                max_tokens=420,
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
                "risk_indicators": [],
                "protective_factors": [],
                "linguistic_patterns": {
                    "hopelessness": False,
                    "social_isolation": False,
                    "self_blame": False,
                    "finality_language": False,
                    "help_seeking": False,
                    "emotional_dysregulation": False,
                },
                "sentiment_summary": "Explanation unavailable.",
                "key_phrases": [],
                "model": self.model,
                "input_tokens": 0,
                "output_tokens": 0,
                "processing_time_ms": int((time.time() - start) * 1000),
                "error": True,
                "error_detail": str(exc),
            }
