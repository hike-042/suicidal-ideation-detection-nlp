"""
Compatibility recommender agent aligned with the active website taxonomy.
"""

from __future__ import annotations

from app.agents.unified_agent import _fallback_recommendations


class RecommenderAgent:
    """
    Legacy compatibility wrapper.

    For older imports, this preserves the previous interface while delegating
    to the current recommendation logic shared by the live pipeline.
    """

    def __init__(self, model: str = "compatibility"):
        self.model = model

    def __repr__(self) -> str:
        return f"RecommenderAgent(model={self.model!r})"

    def recommend(self, text: str, risk_level: str, confidence: float) -> dict:
        result = _fallback_recommendations(risk_level)
        result["model"] = self.model
        result.setdefault("input_tokens", 0)
        result.setdefault("output_tokens", 0)
        result["compatibility_mode"] = True
        return result
