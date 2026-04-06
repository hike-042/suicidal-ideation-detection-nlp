"""
Deterministic normalization agent for noisy social-media text.
"""

from __future__ import annotations

from src.data.generation_lexicon import normalize_generation_language


class NormalizationAgent:
    """Expose normalization as a first-class agent stage."""

    def normalize(self, text: str) -> dict:
        report = normalize_generation_language(text or "")
        normalized = report.get("normalized_text", "")
        original = report.get("original_text", "")
        fixes = report.get("fixes", [])

        return {
            "original_text": original,
            "normalized_text": normalized,
            "fixes": fixes,
            "changed": normalized != original,
            "fix_count": sum(int(item.get("count", 0) or 0) for item in fixes),
        }
