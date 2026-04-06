"""
unified_agent.py
================
A single-call agent that performs classification, linguistic analysis, AND
recommendations in ONE API call instead of three.

Cost impact:
  Old: 3 × claude-opus-4-6 calls per analysis
  New: 1 × claude-haiku-4-5 call  (escalates to claude-sonnet-4-6 only when needed)
  Savings: ~85–95% per analysis
"""

import json
import time

from src.data.generation_lexicon import expand_generation_language
from app.agents.llm_router import RoutedLLMClient, LLMRouterError

# ---------------------------------------------------------------------------
# Compact system prompt — single call returns all three result sections
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a mental health NLP agent analyzing social media posts (Twitter/Reddit) \
for suicidal ideation risk. Respond ONLY with a single valid JSON object — \
no markdown, no prose.

RISK LEVELS:
  HIGH_RISK_SELF_HARM       — explicit suicidal intent, plans, farewell messages, "want to die/kill myself"
  HIGH_RISK_HARM_TO_OTHERS  — explicit threats or intent to kill/injure another person
  MODERATE_RISK             — hopelessness, feeling like a burden, emptiness, indirect distress, severe depression
  LOW_RISK                  — normal social media content, everyday stress, positive/neutral posts

REQUIRED JSON SCHEMA (all fields mandatory):
{
  "classification": {
    "risk_level": "HIGH_RISK_SELF_HARM|HIGH_RISK_HARM_TO_OTHERS|MODERATE_RISK|LOW_RISK",
    "confidence": <float 0.0–1.0>,
    "risk_score": <int 0–100>,
    "reasoning": "<≤30 words of clinical reasoning>"
  },
  "explanation": {
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
    "sentiment_summary": "<≤20 words>",
    "key_phrases": ["<up to 5 phrases>"]
  },
  "recommendations": {
    "immediate_action": "<what to do right now>",
    "resources": [
      {"name": "<name>", "contact": "<phone/url>",
       "description": "<brief>", "type": "hotline|website|app|text"}
    ],
    "recommended_response_tone": "<how a moderator should respond>",
    "platform_action": "<moderation action>",
    "follow_up": "<monitoring guidance>"
  }
}

RESOURCE RULES (mandatory):
  HIGH_RISK_SELF_HARM   → include: 988 Lifeline, Crisis Text Line (HOME→741741), IASP
  HIGH_RISK_HARM_TO_OTHERS → include immediate emergency / law-enforcement escalation guidance
  MODERATE_RISK → include: BetterHelp, NAMI Helpline (1-800-950-6264)
  LOW_RISK     → include: Mental Health America (mhanational.org)

LANGUAGE RULES:
  Treat social-media slang, acronyms, meme phrasing, and generation-specific
  wording seriously when it implies self-harm or hopelessness.
  Examples include "kms", "kys", "unalive", "off myself", "sewer slide",
  "kermit sewer slide", "better off without me", and indirect farewell language.
  Also detect threats toward others such as "kill him", "kill her", "kill someone",
  "kill the guy next to me", "stab someone", or "shoot her" as HIGH_RISK_HARM_TO_OTHERS.
"""

_EMPTY_EXPLANATION = {
    "risk_indicators": [],
    "protective_factors": [],
    "linguistic_patterns": {
        "hopelessness": False, "social_isolation": False, "self_blame": False,
        "finality_language": False, "help_seeking": False,
        "emotional_dysregulation": False,
    },
    "sentiment_summary": "Analysis unavailable.",
    "key_phrases": [],
}


class UnifiedAnalysisAgent:
    """
    Performs classification + explanation + recommendations in a single API call.

    Uses claude-haiku-4-5 by default (cheapest capable model).
    The orchestrator escalates to claude-sonnet-4-6 when confidence is low.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.client = RoutedLLMClient()
        self.model = model

    def __repr__(self) -> str:
        return f"UnifiedAnalysisAgent(model={self.model!r})"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> dict:
        """
        Analyze text in one API call.

        Returns a dict with keys: classification, explanation, recommendations,
        model, input_tokens, output_tokens, error (if any).
        """
        start = time.time()
        normalized = expand_generation_language(text)
        user_msg = (
            "Analyze this social media post.\n\n"
            f"Original text:\n{text}\n\n"
            f"Expanded slang/context version:\n{normalized}"
        )

        try:
            message = self.client.create_message(
                model=self.model,
                max_tokens=700,          # enough for all three sections
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = message.text.strip()

            result = self._parse_json(raw, text, message)
            result["model"] = message.model
            result["provider"] = message.provider
            result["input_tokens"] = message.input_tokens
            result["output_tokens"] = message.output_tokens
            result["processing_time_ms"] = int((time.time() - start) * 1000)
            return result

        except LLMRouterError as exc:
            return self._error_result(str(exc), start)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _parse_json(self, raw: str, original_text: str, first_message) -> dict:
        """Parse the JSON response, retrying once if malformed."""
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Single retry — ask the model to output clean JSON
        try:
            retry = self.client.create_message(
                model=self.model,
                max_tokens=700,
                system=_SYSTEM_PROMPT,
                messages=[
                    {"role": "user", "content": f"Analyze this social media post:\n\n{original_text}"},
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content":
                        "Your response was not valid JSON. Output ONLY the raw JSON object, "
                        "no markdown fences, no explanation."},
                ],
            )
            return json.loads(retry.text.strip())
        except (json.JSONDecodeError, LLMRouterError):
            # Return minimal fallback structure
            return {
                "classification": {
                    "risk_level": "LOW_RISK",
                    "confidence": 0.0,
                    "risk_score": 0,
                    "reasoning": "JSON parse failed after retry.",
                },
                "explanation": _EMPTY_EXPLANATION,
                "recommendations": _fallback_recommendations("LOW_RISK"),
                "error": True,
                "error_detail": "JSON parse failed",
            }

    def _error_result(self, error_detail: str, start: float) -> dict:
        return {
            "classification": {
                "risk_level": "LOW_RISK",
                "confidence": 0.0,
                "risk_score": 0,
                "reasoning": "API error during analysis.",
            },
            "explanation": _EMPTY_EXPLANATION,
            "recommendations": _fallback_recommendations("LOW_RISK"),
            "model": self.model,
            "input_tokens": 0,
            "output_tokens": 0,
            "processing_time_ms": int((time.time() - start) * 1000),
            "error": True,
            "error_detail": error_detail,
        }


# ---------------------------------------------------------------------------
# Shared fallback resources (used by orchestrator too)
# ---------------------------------------------------------------------------

def _fallback_recommendations(risk_level: str) -> dict:
    """Hard-coded recommendations when API is unavailable."""
    if risk_level == "HIGH_RISK_HARM_TO_OTHERS":
        resources = [
            {"name": "Emergency Services", "contact": "911 / local emergency number",
             "description": "Immediate emergency response for imminent violent threats.", "type": "hotline"},
            {"name": "Local Law Enforcement / Crisis Intervention", "contact": "Local authority contact",
             "description": "Use when there is an imminent threat to another person.", "type": "hotline"},
        ]
        return {
            "immediate_action": "Treat as an imminent violence threat. Contact emergency services and escalate immediately.",
            "resources": resources,
            "recommended_response_tone": "Calm, direct, safety-focused, and urgent.",
            "platform_action": "Escalate to trust & safety and emergency escalation workflow immediately.",
            "follow_up": "Preserve evidence, alert the proper safety team, and do not treat as low-risk content.",
        }
    if risk_level in {"HIGH_RISK", "HIGH_RISK_SELF_HARM"}:
        resources = [
            {"name": "988 Suicide & Crisis Lifeline", "contact": "988",
             "description": "Free 24/7 crisis support.", "type": "hotline"},
            {"name": "Crisis Text Line", "contact": "Text HOME to 741741",
             "description": "Free 24/7 text-based support.", "type": "text"},
            {"name": "IASP Crisis Centres",
             "contact": "https://www.iasp.info/resources/Crisis_Centres/",
             "description": "International crisis centre directory.", "type": "website"},
        ]
        return {
            "immediate_action": "Contact emergency services or a crisis hotline immediately.",
            "resources": resources,
            "recommended_response_tone": "Empathetic, non-judgmental, urgent.",
            "platform_action": "Escalate to trust & safety team immediately.",
            "follow_up": "Monitor closely; check in within 24 hours.",
        }
    if risk_level == "MODERATE_RISK":
        resources = [
            {"name": "BetterHelp", "contact": "https://www.betterhelp.com",
             "description": "Online therapy with licensed counselors.", "type": "website"},
            {"name": "NAMI Helpline", "contact": "1-800-950-6264",
             "description": "Support and referrals for mental health.", "type": "hotline"},
            {"name": "988 Lifeline", "contact": "988",
             "description": "Available for non-crisis emotional distress too.", "type": "hotline"},
        ]
        return {
            "immediate_action": "Share mental health resources and offer a supportive message.",
            "resources": resources,
            "recommended_response_tone": "Warm and encouraging without minimising experience.",
            "platform_action": "Flag for human moderator review.",
            "follow_up": "Monitor for escalation over 48–72 hours.",
        }
    return {
        "immediate_action": "No immediate action required.",
        "resources": [
            {"name": "Mental Health America", "contact": "https://mhanational.org",
             "description": "Mental health awareness and screening tools.", "type": "website"},
        ],
        "recommended_response_tone": "Friendly and informative.",
        "platform_action": "No special action needed.",
        "follow_up": "Routine monitoring only.",
    }

