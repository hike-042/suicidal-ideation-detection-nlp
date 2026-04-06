"""
motivation_agent.py
===================
Dedicated agent for generating personalised motivational messages,
coping strategies, and wellness suggestions.

This is a lightweight companion agent invoked AFTER risk classification.
It produces hope-focused, empathetic content tailored to the specific risk
level and the emotional context detected in the text.

Cost model
----------
- Uses claude-haiku-4-5 (cheapest capable model)
- Called only when a real AI analysis ran (not for cache hits — cached
  results already include motivation from the first run)
- Fallback hard-coded responses are used when the API is unavailable or
  when the keyword pre-filter handles a LOW_RISK text
- Typical spend: ~150 input + ~250 output Haiku tokens (~0.02 ¢ per call)
"""

import json
import time
from app.agents.llm_router import RoutedLLMClient

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a compassionate mental health support specialist. Based on the text and
its risk assessment, provide warm, evidence-based motivational support.
Respond ONLY with a single valid JSON object — no markdown, no prose.

REQUIRED JSON SCHEMA (all fields mandatory):
{
  "empathy_message": "<2-3 sentences acknowledging feelings without judgment>",
  "hope_message": "<1-2 sentences of genuine, research-backed hope>",
  "coping_strategies": [
    {
      "technique": "<technique name>",
      "description": "<clear, actionable steps>",
      "duration": "<time needed e.g. '2 minutes'>"
    }
  ],
  "immediate_grounding": "<specific 30-60 second grounding exercise — be concrete>",
  "affirmations": ["<short first-person affirmation>"],
  "daily_wellness_tips": ["<specific actionable wellness habit>"],
  "professional_message": "<warm encouragement to seek professional help>",
  "crisis_reminder": "<crisis line info if HIGH_RISK, else empty string>"
}

TONE & CONTENT RULES:
  HIGH_RISK    → Warm, urgent, non-judgmental. Validate pain. Emphasise immediate help.
                 Include 988 and Crisis Text Line in crisis_reminder.
                 Coping strategies should be immediate (< 5 min each).
  MODERATE_RISK → Supportive, hopeful. Validate the struggle. Encourage therapy.
                  crisis_reminder can be empty or brief.
  LOW_RISK     → Positive, encouraging. Celebrate resilience. Proactive wellness.
                  crisis_reminder = "".

Include 3–4 coping_strategies and 3 affirmations.
Avoid platitudes — be specific and genuine.
"""


class MotivationAgent:
    """
    Generates personalised motivational content and coping strategies.

    Called after risk classification to provide the human-support layer
    of the pipeline. Results are cached at the orchestrator level so this
    agent is only invoked once per unique text.
    """

    def __init__(self, model: str = "claude-haiku-4-5-20251001"):
        self.client = RoutedLLMClient()
        self.model = model

    def __repr__(self) -> str:
        return f"MotivationAgent(model={self.model!r})"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def motivate(self, text: str, risk_level: str, reasoning: str) -> dict:
        """
        Generate motivational support content.

        Parameters
        ----------
        text       : The original text being analysed (truncated to 500 chars)
        risk_level : "HIGH_RISK", "MODERATE_RISK", or "LOW_RISK"
        reasoning  : Short clinical reasoning from the classifier agent

        Returns
        -------
        dict with empathy_message, hope_message, coping_strategies, etc.
        """
        start = time.time()

        user_msg = (
            f"Text: {text[:500]}\n\n"
            f"Risk Level: {risk_level}\n"
            f"Clinical reasoning: {reasoning}\n\n"
            "Generate personalised motivational support."
        )

        try:
            message = self.client.create_message(
                model=self.model,
                max_tokens=650,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = message.text.strip()

            try:
                result = json.loads(raw)
            except json.JSONDecodeError:
                # Single retry asking for clean JSON
                retry = self.client.create_message(
                    model=self.model,
                    max_tokens=650,
                    system=_SYSTEM_PROMPT,
                    messages=[
                        {"role": "user", "content": user_msg},
                        {"role": "assistant", "content": raw},
                        {"role": "user", "content":
                            "Your response was not valid JSON. Output ONLY the raw JSON object — "
                            "no markdown fences, no explanation."},
                    ],
                )
                result = json.loads(retry.text.strip())

            result["model"] = message.model
            result["provider"] = message.provider
            result["input_tokens"] = message.input_tokens
            result["output_tokens"] = message.output_tokens
            result["processing_time_ms"] = int((time.time() - start) * 1000)
            return result

        except Exception:
            return self._fallback(risk_level)

    # ------------------------------------------------------------------
    # Fallback (no API call)
    # ------------------------------------------------------------------

    def _fallback(self, risk_level: str) -> dict:
        """Hard-coded fallback used when the API is unavailable."""
        if risk_level == "HIGH_RISK_HARM_TO_OTHERS":
            return {
                "empathy_message": (
                    "This message suggests a serious risk of harm to another person. "
                    "The priority is immediate safety and urgent escalation."
                ),
                "hope_message": (
                    "A dangerous situation can still be interrupted quickly when the right emergency response happens now."
                ),
                "coping_strategies": [
                    {
                        "technique": "Immediate Escalation",
                        "description": "Contact emergency services or the responsible safety authority right now.",
                        "duration": "Right now",
                    }
                ],
                "immediate_grounding": "Prioritize immediate safety. Do not treat this as a routine wellness case.",
                "affirmations": ["Safety comes first"],
                "daily_wellness_tips": ["Escalate to emergency response immediately"],
                "professional_message": (
                    "This content requires an urgent safety intervention path, not a routine supportive response."
                ),
                "crisis_reminder": "If there is an imminent threat to another person, contact emergency services now.",
                "model": "fallback",
                "input_tokens": 0,
                "output_tokens": 0,
                "processing_time_ms": 0,
            }

        if risk_level in {"HIGH_RISK", "HIGH_RISK_SELF_HARM"}:
            return {
                "empathy_message": (
                    "I hear you, and what you're feeling is completely real. "
                    "Your pain matters deeply, and you don't have to face this alone. "
                    "Reaching out — even in this small way — already shows courage."
                ),
                "hope_message": (
                    "Recovery is genuinely possible. Research shows that most people "
                    "who receive the right support go on to live fulfilling lives — "
                    "your situation is not permanent, even when it feels that way."
                ),
                "coping_strategies": [
                    {
                        "technique": "Box Breathing",
                        "description": "Inhale slowly for 4 counts, hold for 4, exhale for 4, hold for 4. Repeat 4 times.",
                        "duration": "2 minutes",
                    },
                    {
                        "technique": "5-4-3-2-1 Grounding",
                        "description": "Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste.",
                        "duration": "3 minutes",
                    },
                    {
                        "technique": "Reach Out Now",
                        "description": "Call or text 988. A real, trained person will answer immediately. You are not a burden.",
                        "duration": "Right now",
                    },
                    {
                        "technique": "Stay Connected",
                        "description": "Text or call one person you trust — tell them you need company right now.",
                        "duration": "5 minutes",
                    },
                ],
                "immediate_grounding": (
                    "Put both feet flat on the floor. Press them down firmly — feel the ground. "
                    "Take one slow, deep breath in through your nose (count to 4), and out through "
                    "your mouth (count to 6). You are here. You are real. This moment is real. "
                    "Repeat 3 times."
                ),
                "affirmations": [
                    "I deserve support and care",
                    "My pain is real, and help is available",
                    "I am stronger than this moment",
                ],
                "daily_wellness_tips": [
                    "Call 988 — free, confidential, available 24/7 every day",
                    "Stay with someone you trust tonight — don't be alone",
                    "Tell one safe person how you're feeling today",
                ],
                "professional_message": (
                    "Please reach out to a mental health professional or a crisis line right now. "
                    "You deserve specialised, compassionate care — and getting help is a sign of "
                    "strength, not weakness."
                ),
                "crisis_reminder": "988 Suicide & Crisis Lifeline: Call or text 988 (24/7, free). Crisis Text Line: Text HOME to 741741.",
                "model": "fallback",
                "input_tokens": 0,
                "output_tokens": 0,
                "processing_time_ms": 0,
            }

        if risk_level == "MODERATE_RISK":
            return {
                "empathy_message": (
                    "Feeling this way is exhausting, and it makes complete sense you're struggling. "
                    "What you're experiencing is real and valid — many people go through this, "
                    "and you are not alone in this fight."
                ),
                "hope_message": (
                    "With the right support, things can genuinely improve. "
                    "Therapy, connection, and small daily practices have helped countless people "
                    "move from this place to one of real hope and stability."
                ),
                "coping_strategies": [
                    {
                        "technique": "Free Journaling",
                        "description": "Write whatever you feel for 10 minutes without editing. Don't judge what comes out — just let it flow.",
                        "duration": "10 minutes",
                    },
                    {
                        "technique": "Gentle Movement",
                        "description": "A brisk 10-minute walk outside can measurably shift your mental state. Focus on what you see and hear.",
                        "duration": "10 minutes",
                    },
                    {
                        "technique": "Connect with Someone",
                        "description": "Text or call one person you trust today — you don't have to explain everything, just connect.",
                        "duration": "5 minutes",
                    },
                    {
                        "technique": "Limit Negative Inputs",
                        "description": "Pause social media for the rest of today and replace it with music, a podcast, or a book you enjoy.",
                        "duration": "Rest of day",
                    },
                ],
                "immediate_grounding": (
                    "Take 5 slow, intentional breaths. With each exhale, consciously release one "
                    "worry — imagine it leaving your body. You don't need to solve everything right now. "
                    "Just this breath. Just this moment."
                ),
                "affirmations": [
                    "I am doing my best with what I have",
                    "I deserve kindness — especially from myself",
                    "This feeling is temporary, not permanent",
                ],
                "daily_wellness_tips": [
                    "Set one small, achievable goal for today — completing it builds momentum",
                    "Aim for 7–8 hours of sleep — it directly impacts emotional regulation",
                    "Spend 10 minutes in nature or sunlight if possible",
                ],
                "professional_message": (
                    "Talking to a therapist can make a profound difference. BetterHelp connects "
                    "you with licensed counsellors online, and NAMI (1-800-950-6264) can help you "
                    "find local resources. You deserve this support."
                ),
                "crisis_reminder": "",
                "model": "fallback",
                "input_tokens": 0,
                "output_tokens": 0,
                "processing_time_ms": 0,
            }

        # LOW_RISK
        return {
            "empathy_message": (
                "It sounds like you're navigating life with awareness. "
                "Taking time to reflect on your mental state is itself a healthy habit — "
                "it shows you care about your wellbeing."
            ),
            "hope_message": (
                "Building mental wellness practices now creates real resilience for "
                "future challenges. Small daily habits compound into lasting strength."
            ),
            "coping_strategies": [
                {
                    "technique": "Mindful Breathing",
                    "description": "Spend 5 minutes focusing solely on your breath. When your mind wanders, gently bring it back.",
                    "duration": "5 minutes",
                },
                {
                    "technique": "Gratitude Practice",
                    "description": "Write down 3 specific things you're grateful for today, and why each one matters to you.",
                    "duration": "5 minutes",
                },
                {
                    "technique": "Social Nourishment",
                    "description": "Reach out to a friend or family member just to say hello — human connection is foundational to wellbeing.",
                    "duration": "10 minutes",
                },
                {
                    "technique": "Creative Expression",
                    "description": "Spend 15 minutes on something creative — drawing, writing, music, cooking — without any goal except enjoyment.",
                    "duration": "15 minutes",
                },
            ],
            "immediate_grounding": (
                "Pause for a moment and appreciate something beautiful around you — "
                "a colour, a sound, a texture, the warmth of light. "
                "Take three deep breaths and feel gratitude for this present moment."
            ),
            "affirmations": [
                "I am enough exactly as I am",
                "I choose growth and self-compassion",
                "I am resilient and capable",
            ],
            "daily_wellness_tips": [
                "Regular exercise (even 20 min/day) significantly improves mood and cognition",
                "Quality sleep is the single most impactful factor in mental health",
                "Small acts of kindness — to others and yourself — boost long-term wellbeing",
            ],
            "professional_message": (
                "Proactive mental health check-ins are always worthwhile, even when things are going well. "
                "Consider a wellness session with a therapist as you would a routine physical."
            ),
            "crisis_reminder": "",
            "model": "fallback",
            "input_tokens": 0,
            "output_tokens": 0,
            "processing_time_ms": 0,
        }
