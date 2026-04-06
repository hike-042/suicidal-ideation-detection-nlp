"""
orchestrator.py
===============
Token-efficient tiered orchestration pipeline.

COST ARCHITECTURE
-----------------

OLD (every request):
  ClassifierAgent  (claude-opus-4-6)  → ~400 input tokens
  ExplainerAgent   (claude-opus-4-6)  → ~500 input tokens
  RecommenderAgent (claude-opus-4-6)  → ~600 input tokens
  Total per request: 3 calls × Opus pricing ≈ 1 500 tokens input

NEW (per request):
  Tier 0 — KeywordPreFilter    (0 tokens) : obvious cases resolved instantly
  Tier 1 — UnifiedAnalysisAgent (Haiku)   : 1 call, ~350 tokens, ~20× cheaper than Opus
  Tier 2 — UnifiedAnalysisAgent (Sonnet)  : only when Haiku confidence < 0.65
  Cache  — ResultCache          (0 tokens) : zero cost on repeated/identical texts

Typical savings: 85–95% reduction in token spend.

ESCALATION RULES
----------------
  Haiku confidence ≥ 0.65 → return Haiku result (no escalation)
  Haiku confidence < 0.65 → escalate to Sonnet for a second opinion
  Haiku says HIGH_RISK    → always escalate to Sonnet (safety-critical)
  Pre-filter HIGH_RISK    → skip all API calls if confidence is very high
"""

import asyncio
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from app.agents.unified_agent import UnifiedAnalysisAgent, _fallback_recommendations
from app.agents.motivation_agent import MotivationAgent
from app.agents.normalizer_agent import NormalizationAgent
from app.agents.cache import get_cache
from app.agents.signal_engine import analyze_signals, synthesize_risk
from app.risk_settings import load_risk_settings
from src.data.generation_lexicon import expand_generation_language, generation_keywords

# ---------------------------------------------------------------------------
# Configuration — read from env with sensible defaults
# ---------------------------------------------------------------------------

# Model used for standard (cheap) analysis
_HAIKU_MODEL = os.environ.get("AGENT_FAST_MODEL", "claude-haiku-4-5-20251001")
# Model used for escalation (edge cases / HIGH_RISK verification)
_SONNET_MODEL = os.environ.get("AGENT_SMART_MODEL", "claude-sonnet-4-6")

# Minimum Haiku confidence to skip escalation
_CONFIDENCE_THRESHOLD = float(load_risk_settings()["thresholds"]["confidence_escalation_threshold"])
_MODERATE_PREFILTER_THRESHOLD = float(load_risk_settings()["thresholds"]["moderate_prefilter_confidence_threshold"])

# Whether to always escalate HIGH_RISK detections to Sonnet for safety
_ALWAYS_ESCALATE_HIGH_RISK = os.environ.get("AGENT_ESCALATE_HIGH_RISK", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Tier 0 — Keyword Pre-filter (zero API cost)
# ---------------------------------------------------------------------------

_HIGH_RISK_PHRASES = [
    # Explicit intent
    "kill myself", "killing myself", "end my life", "ending my life",
    "want to die", "want to kill myself", "going to kill myself",
    "i want to end it", "i need to end it",
    # Hopelessness / no reason
    "no reason to live", "nothing to live for", "don't want to be alive",
    "don't want to live", "cant see a reason to live",
    # Planning / finality
    "i've decided to end", "i'm going to end it", "ending it all",
    "goodbye forever", "goodbye everyone", "final goodbye", "this is goodbye",
    "written my note", "wrote my note", "suicide note", "left a note",
    "i won't be here", "won't be around much longer", "not be here anymore",
    "my last post", "my last message", "my final message", "my last words",
    "planned how i'll do it", "i have a plan", "i've made up my mind",
    "have the pills", "picked the bridge", "picked the time", "picked the method",
    "chose the method", "know exactly when i am going to die", "final message is scheduled",
    "writing my last words", "by tomorrow i will not be here",
    # Direct statements
    "i've been thinking about suicide", "thinking about suicide",
    "contemplating suicide", "considering suicide", "researched ways to die",
]

_HIGH_RISK_HARM_TO_OTHERS_PHRASES = [
    "kill him", "kill her", "kill them", "kill someone", "kill somebody",
    "kill anyone", "kill anybody", "kill everyone", "kill everybody", "kill people",
    "kill them all", "kill all of them", "shoot someone", "shoot everybody",
    "shoot everyone", "stab someone", "stab everybody", "stab everyone",
    "hurt someone", "hurt everybody", "hurt everyone", "murder someone",
    "murder everybody", "murder everyone", "someone is going to die tonight",
    "people are about to get hurt", "make them all pay with their lives",
    "end someone tonight", "nobody leaves alive", "spill blood tonight",
    "end them all", "take lives tonight", "people will be dead",
    "destroy everyone who stands in front of me",
]

_MODERATE_RISK_PHRASES = [
    # Hopelessness
    "feel so hopeless", "feeling hopeless", "completely hopeless", "so hopeless",
    "nothing matters anymore", "nothing matters to me", "nothing matters",
    "feel like giving up", "i give up", "no hope left", "lost all hope",
    "cannot keep going", "do not see a reason to keep trying",
    "nothing will ever get better", "tired of being me", "awful", "done with life",
    # Worthlessness / burden
    "feel worthless", "i am worthless", "i'm worthless", "i feel worthless",
    "burden to everyone", "burden on my family", "burden to my family",
    "everyone would be better", "better off without me",
    "they'd be better off without me", "world would be better without me",
    "nobody would miss me", "no one would miss me", "everyone would be happier without me",
    "i am a problem for everyone around me", "useless", "feel broken",
    # Can't continue
    "can't go on", "cannot go on", "i can't keep going", "can't keep living",
    "can't do this anymore", "can't take it anymore", "i can't anymore",
    "idk how much longer i can do this",
    # Self-hate
    "hate myself so much", "i hate myself", "i despise myself",
    "i loathe myself", "so disgusted with myself",
    # Meaninglessness
    "no point in living", "no point anymore", "no point to anything",
    "life has no meaning", "my life has no meaning",
    # Emptiness / numbness
    "empty inside", "feel so empty", "completely numb", "feeling numb",
    "dead inside", "hollow inside", "feel empty", "not okay",
    "severely distressed", "falling apart", "spiraling hard",
    # Isolation / crying
    "crying every day", "haven't stopped crying", "can't stop crying",
    "crying myself to sleep", "no one understands me", "so alone",
    "completely alone", "all alone", "nobody cares about me",
    "no one cares about me", "feel disconnected from everyone around me",
    "alone in a crowded room", "nobody gets it", "drowning in my own thoughts",
    "every day feels heavier than the last", "cannot stop crying at night",
    # Physical symptoms of depression
    "haven't eaten in days", "can't get out of bed", "haven't slept in days",
    "can't sleep", "haven't showered in",
    # Indirect ideation
    "wish i was never born", "wish i wasn't here", "wish i didn't exist",
    "don't want to wake up", "wish i could disappear", "want everything to stop",
    "do not want to wake up tomorrow",
    # Help-seeking distress
    "i need help", "please help", "can someone help", "want therapy",
    "need professional help", "booked a therapy appointment", "need support",
]

_LOW_RISK_SIGNALS = [
    "loving this", "so excited", "best day", "great news", "thank you",
    "happy birthday", "congratulations", "so proud", "feeling good today",
    "weekend plans", "watched a movie", "just ate", "coffee",
]

_FICTIONAL_CONTEXT_PATTERNS = (
    re.compile(r"\bin\s+this\s+game\b"),
    re.compile(r"\bdungeon\b"),
    re.compile(r"\bmovie\b"),
    re.compile(r"\btrailer\b"),
    re.compile(r"\bshow\b"),
    re.compile(r"\blyric\b"),
    re.compile(r"\bsong\b"),
    re.compile(r"\bquote\b"),
    re.compile(r"\bdnd\b"),
    re.compile(r"\bbarbarian\b"),
    re.compile(r"\bgoblin\b"),
    re.compile(r"\braid\b"),
    re.compile(r"\bmatch\b"),
    re.compile(r"\bcall of duty\b"),
    re.compile(r"\bactor said\b"),
    re.compile(r"\bin a good way\b"),
)

_GENERATION_SIGNAL_WORDS = tuple(generation_keywords())


class KeywordPreFilter:
    """
    Fast, zero-cost pre-filter using curated phrase lists.

    Conservative by design: returns None when uncertain so the AI agents
    handle ambiguous cases. Only intercepts very obvious patterns.
    """

    def analyze(self, text: str) -> dict | None:
        """
        Returns a result dict if the text is clearly classifiable,
        or None if it should be passed to AI agents.
        """
        normalized = expand_generation_language(text)
        lower = normalized.lower()
        fictional_context = any(pattern.search(lower) for pattern in _FICTIONAL_CONTEXT_PATTERNS)

        matched_harm_to_others = [p for p in _HIGH_RISK_HARM_TO_OTHERS_PHRASES if p in lower]
        if matched_harm_to_others and not fictional_context:
            score = min(80 + len(matched_harm_to_others) * 5, 99)
            return self._build(
                risk_level="HIGH_RISK_HARM_TO_OTHERS",
                risk_score=score,
                confidence=round(score / 100, 2),
                reasoning=f"Explicit harm-to-others phrase(s) detected: {matched_harm_to_others[:3]}",
                matched=matched_harm_to_others,
                tier="keyword_prefilter",
            )

        # Explicit high-risk phrases → very high confidence
        matched_high = [p for p in _HIGH_RISK_PHRASES if p in lower]
        if matched_high:
            score = min(75 + len(matched_high) * 5, 98)
            return self._build(
                risk_level="HIGH_RISK",
                risk_score=score,
                confidence=round(score / 100, 2),
                reasoning=f"Explicit high-risk phrase(s) detected: {matched_high[:3]}",
                matched=matched_high,
                tier="keyword_prefilter",
            )

        # Moderate risk signals
        matched_moderate = [p for p in _MODERATE_RISK_PHRASES if p in lower]
        if len(matched_moderate) >= 2:          # require 2+ signals to avoid false positives
            score = min(45 + len(matched_moderate) * 5, 69)
            return self._build(
                risk_level="MODERATE_RISK",
                risk_score=score,
                confidence=round(score / 100, 2),
                reasoning=f"Moderate-risk phrases detected: {matched_moderate[:3]}",
                matched=matched_moderate,
                tier="keyword_prefilter",
            )

        # Clearly low-risk — only if very short and clearly benign
        low_hits = [s for s in _LOW_RISK_SIGNALS if s in lower]
        has_risk_words = any(w in lower for w in ["die", "dead", "hurt", "pain", "sad", "depress"]) or any(
            kw in lower for kw in _GENERATION_SIGNAL_WORDS
        )
        if len(lower) < 120 and len(low_hits) >= 2 and not has_risk_words:
            return self._build(
                risk_level="LOW_RISK",
                risk_score=8,
                confidence=0.90,
                reasoning="Short, clearly positive/neutral social media content.",
                matched=low_hits,
                tier="keyword_prefilter",
            )

        return None  # ambiguous → let AI decide

    @staticmethod
    def _build(
        risk_level: str,
        risk_score: int,
        confidence: float,
        reasoning: str,
        matched: list,
        tier: str,
    ) -> dict:
        """Build a result dict matching the UnifiedAnalysisAgent output shape."""
        from app.agents.unified_agent import _EMPTY_EXPLANATION
        return {
            "classification": {
                "risk_level": risk_level,
                "confidence": confidence,
                "risk_score": risk_score,
                "reasoning": reasoning,
            },
            "explanation": {
                **_EMPTY_EXPLANATION,
                "risk_indicators": matched[:5],
                "key_phrases": matched[:5],
                "sentiment_summary": f"Keyword match: {risk_level.replace('_', ' ').lower()}.",
            },
            "recommendations": _fallback_recommendations(risk_level),
            "model": tier,
            "input_tokens": 0,
            "output_tokens": 0,
            "tier_used": tier,
            "escalated": False,
        }


# ---------------------------------------------------------------------------
# Tier 1 + 2 — Unified AI agents
# ---------------------------------------------------------------------------

class AgentOrchestrator:
    """
    Tiered orchestrator that minimises Claude token spend.

    Processing flow
    ---------------
    1. Check cache  → return immediately on hit (0 tokens)
    2. KeywordPreFilter  → return immediately on high-confidence match (0 tokens)
    3. UnifiedAnalysisAgent(Haiku) → 1 cheap API call
    4. If confidence < threshold OR result is HIGH_RISK:
         UnifiedAnalysisAgent(Sonnet) → 1 escalation call
    5. Cache the final result
    """

    def __init__(self):
        self._prefilter = KeywordPreFilter()
        self._haiku = UnifiedAnalysisAgent(model=_HAIKU_MODEL)
        self._sonnet = UnifiedAnalysisAgent(model=_SONNET_MODEL)
        self._motivation = MotivationAgent(model=_HAIKU_MODEL)
        self._normalizer = NormalizationAgent()
        self._cache = get_cache()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="agent")

    def __repr__(self) -> str:
        return (
            f"AgentOrchestrator("
            f"fast={_HAIKU_MODEL!r}, smart={_SONNET_MODEL!r}, "
            f"threshold={_CONFIDENCE_THRESHOLD}, "
            f"cache={self._cache!r})"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyze(self, text: str) -> dict:
        """
        Full tiered analysis pipeline (synchronous).

        Always returns a dict with:
          text, classification, explanation, recommendations,
          analysis_timestamp, processing_time_ms, tier_used,
          escalated, cache_hit, token_usage
        """
        wall_start = time.time()
        normalization = self._normalizer.normalize(text)
        analysis_text = normalization.get("normalized_text", text)

        # ── Tier 0a: cache ────────────────────────────────────────────
        cached = self._cache.get(text)
        if cached:
            cached["analysis_timestamp"] = datetime.now(timezone.utc).isoformat()
            cached["cache_hit"] = True
            cached.setdefault("normalization", normalization)
            return cached

        # ── Tier 0b: keyword pre-filter ──────────────────────────────
        prefilter_result = self._prefilter.analyze(analysis_text)
        if prefilter_result:
            # Only skip AI for LOW_RISK or very-high-confidence MODERATE hits
            risk = prefilter_result["classification"]["risk_level"]
            conf = prefilter_result["classification"]["confidence"]
            if risk == "LOW_RISK" or (risk == "MODERATE_RISK" and conf >= _MODERATE_PREFILTER_THRESHOLD):
                return self._finalise(text, analysis_text, normalization, prefilter_result, wall_start,
                                      tier="keyword_prefilter", escalated=False)

        # ── Tier 1: Haiku (fast, cheap) ──────────────────────────────
        haiku_result = self._haiku.analyze(analysis_text)
        haiku_conf = haiku_result.get("classification", {}).get("confidence", 0.0)
        haiku_risk = haiku_result.get("classification", {}).get("risk_level", "LOW_RISK")
        haiku_error = haiku_result.get("error", False)

        needs_escalation = (
            haiku_error
            or haiku_conf < _CONFIDENCE_THRESHOLD
            or (haiku_risk.startswith("HIGH_RISK") and _ALWAYS_ESCALATE_HIGH_RISK)
        )

        if not needs_escalation:
            return self._finalise(text, analysis_text, normalization, haiku_result, wall_start,
                                  tier="haiku", escalated=False)

        # ── Tier 2: Sonnet (smart, escalation only) ──────────────────
        sonnet_result = self._sonnet.analyze(analysis_text)
        sonnet_error = sonnet_result.get("error", False)

        # Pick the better result
        if sonnet_error and not haiku_error:
            final = haiku_result
            tier = "haiku_escalation_failed"
        else:
            final = sonnet_result
            # Merge token counts so the caller sees the total spend
            final["input_tokens"] = (
                haiku_result.get("input_tokens", 0)
                + sonnet_result.get("input_tokens", 0)
            )
            final["output_tokens"] = (
                haiku_result.get("output_tokens", 0)
                + sonnet_result.get("output_tokens", 0)
            )
            tier = "sonnet_escalation"

        return self._finalise(text, analysis_text, normalization, final, wall_start, tier=tier, escalated=True)

    async def analyze_async(self, text: str) -> dict:
        """Async wrapper — runs the synchronous pipeline in a thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.analyze, text)

    def cache_stats(self) -> dict:
        """Return cache diagnostic information."""
        return self._cache.stats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _finalise(
        self,
        text: str,
        analysis_text: str,
        normalization: dict,
        result: dict,
        wall_start: float,
        tier: str,
        escalated: bool,
    ) -> dict:
        """Stamp metadata, add motivation layer, cache, and return."""
        # ── Motivation agent (Tier 3) ─────────────────────────────────
        signal_report = analyze_signals(analysis_text)
        signal_report["normalized_text"] = analysis_text
        result["system_signals"] = signal_report
        result["normalization"] = normalization

        classification = result.get("classification", {})
        if classification:
            result["classification"] = synthesize_risk(signal_report, classification)

        explanation = result.setdefault("explanation", {})
        explanation.setdefault("risk_indicators", [])
        explanation.setdefault("key_phrases", [])
        for signal in signal_report.get("signals_detected", []):
            explanation["risk_indicators"].extend(signal.get("matches", [])[:2])
            explanation["key_phrases"].extend(signal.get("matches", [])[:2])
        explanation["risk_indicators"] = list(dict.fromkeys(explanation["risk_indicators"]))[:10]
        explanation["key_phrases"] = list(dict.fromkeys(explanation["key_phrases"]))[:10]
        explanation["signal_summary"] = signal_report.get("summary", "")

        # Only run if we don't already have motivation (not a cache hit)
        if not result.get("motivation") and not result.get("error"):
            risk_level = result.get("classification", {}).get("risk_level", "LOW_RISK")
            reasoning  = result.get("classification", {}).get("reasoning", "")
            result["motivation"] = self._motivation.motivate(text, risk_level, reasoning)

        # ── Build agents_used list ────────────────────────────────────
        agents_used = []
        if tier == "keyword_prefilter":
            agents_used = ["normalization_agent", "keyword_prefilter"]
        elif tier == "haiku":
            agents_used = ["normalization_agent", "unified_haiku"]
        elif tier in ("sonnet_escalation", "haiku_escalation_failed"):
            agents_used = ["normalization_agent", "unified_haiku", "escalation_sonnet"]
        elif tier == "ml_fallback":
            agents_used = ["normalization_agent", "keyword_heuristic"]
        else:
            agents_used = ["normalization_agent", tier]
        if not result.get("error"):
            agents_used.append("motivation_haiku")

        # ── Metadata ──────────────────────────────────────────────────
        elapsed_ms = int((time.time() - wall_start) * 1000)
        result["text"] = text
        result["analysis_text"] = analysis_text
        result["analysis_timestamp"] = datetime.now(timezone.utc).isoformat()
        result["processing_time_ms"] = elapsed_ms
        result["tier_used"] = tier
        result["escalated"] = escalated
        result["agents_used"] = agents_used
        result.setdefault("cache_hit", False)
        result["token_usage"] = {
            "input_tokens": result.get("input_tokens", 0),
            "output_tokens": result.get("output_tokens", 0),
            "model": result.get("model", tier),
        }

        # Cache result for future identical inputs
        if not result.get("error"):
            self._cache.set(text, result)

        return result


# ---------------------------------------------------------------------------
# Legacy MLFallbackClassifier — kept for backward compatibility
# Used by demo.py when no API key is present
# ---------------------------------------------------------------------------

class MLFallbackClassifier:
    """
    Keyword-heuristic classifier used when ANTHROPIC_API_KEY is not set.
    Returns the same dict structure as AgentOrchestrator.analyze().
    """

    def analyze(self, text: str) -> dict:
        start = time.time()
        normalizer = NormalizationAgent()
        normalization = normalizer.normalize(text)
        analysis_text = normalization.get("normalized_text", text)
        lower = analysis_text.lower()

        matched_high = [p for p in _HIGH_RISK_PHRASES if p in lower]
        matched_moderate = [p for p in _MODERATE_RISK_PHRASES if p in lower]

        signal_report = analyze_signals(analysis_text)
        system_level = signal_report.get("system_risk_level", "LOW_RISK")

        if system_level == "HIGH_RISK_HARM_TO_OTHERS":
            risk_level, score, conf = "HIGH_RISK_HARM_TO_OTHERS", max(signal_report.get("system_score", 0), 75), 0.90
            reasoning = "Violence threat signals detected in fallback analysis."
        elif matched_high or system_level == "HIGH_RISK_SELF_HARM":
            risk_level, score, conf = "HIGH_RISK_SELF_HARM", max(signal_report.get("system_score", 0), min(70 + len(matched_high) * 5, 98)), min(0.70 + len(matched_high) * 0.05 if matched_high else 0.88, 0.95)
            reasoning = f"High-risk keywords: {matched_high[:3]}"
            if not matched_high:
                reasoning = "High self-harm signals detected in fallback analysis."
        elif matched_moderate or system_level == "MODERATE_RISK":
            risk_level, score, conf = "MODERATE_RISK", max(signal_report.get("system_score", 0), min(40 + len(matched_moderate) * 5, 69)), min(0.50 + len(matched_moderate) * 0.05 if matched_moderate else 0.72, 0.80)
            reasoning = f"Moderate-risk keywords: {matched_moderate[:3]}"
            if not matched_moderate:
                reasoning = "Moderate distress signals detected in fallback analysis."
        else:
            risk_level, score, conf = "LOW_RISK", 10, 0.85
            reasoning = "No risk keywords detected."

        from app.agents.unified_agent import _EMPTY_EXPLANATION
        signal_matches = []
        for signal in signal_report.get("signals_detected", []):
            signal_matches.extend(signal.get("matches", [])[:2])
        signal_matches = list(dict.fromkeys(signal_matches))
        return {
            "text": text,
            "analysis_text": analysis_text,
            "normalization": normalization,
            "classification": {
                "risk_level": risk_level, "confidence": conf,
                "risk_score": score, "reasoning": reasoning,
            },
            "explanation": {
                **_EMPTY_EXPLANATION,
                "risk_indicators": signal_matches[:10] or (matched_high + matched_moderate),
                "key_phrases": signal_matches[:5] or (matched_high + matched_moderate)[:5],
                "sentiment_summary": f"Heuristic: {risk_level.replace('_', ' ').lower()}.",
                "signal_summary": signal_report.get("summary", ""),
                "linguistic_patterns": {
                    "hopelessness": any(w in lower for w in ["hopeless", "no reason", "no point"]),
                    "social_isolation": any(w in lower for w in ["alone", "no one", "burden"]),
                    "self_blame": any(w in lower for w in ["hate myself", "worthless", "burden"]),
                    "finality_language": any(w in lower for w in ["goodbye", "final", "last post"]),
                    "help_seeking": False,
                    "emotional_dysregulation": any(w in lower for w in ["crying", "numb", "empty"]),
                },
            },
            "recommendations": _fallback_recommendations(risk_level),
            "motivation": MotivationAgent()._fallback(risk_level),
            "model": "keyword_heuristic",
            "input_tokens": 0, "output_tokens": 0,
            "analysis_timestamp": datetime.now(timezone.utc).isoformat(),
            "processing_time_ms": int((time.time() - start) * 1000),
            "tier_used": "ml_fallback", "escalated": False, "cache_hit": False,
            "agents_used": ["normalization_agent", "keyword_heuristic"],
            "token_usage": {"input_tokens": 0, "output_tokens": 0, "model": "keyword_heuristic"},
            "fallback_used": True,
            "system_signals": signal_report,
        }

