"""
Structured multi-angle signal detection for suicidal-risk analysis.

This module provides a deterministic, explainable layer that inspects text from
multiple angles:
- explicit suicidal intent
- explicit violent intent toward others
- planning / preparation
- finality / farewell language
- self-harm references
- hopelessness / meaninglessness
- burden / worthlessness
- isolation / disconnection
- emotional dysregulation
- help-seeking / protective factors

The output is designed to complement the LLM result, not replace clinical care.
"""

from __future__ import annotations

from dataclasses import dataclass
import re

from app.risk_settings import load_risk_settings
from src.data.generation_lexicon import expand_generation_language


@dataclass(frozen=True)
class SignalCategory:
    name: str
    weight: int
    patterns: tuple[str, ...]


_HIGH_SIGNAL_CATEGORIES = (
    SignalCategory(
        "harm_to_others_intent",
        42,
        (
            "kill him", "kill her", "kill them", "kill the guy", "kill the man",
            "kill the woman", "kill my friend", "kill my dad", "kill my mom",
            "kill my mother", "kill my father", "kill my wife", "kill my husband",
            "kill my coworker", "kill my colleague", "kill my boss", "kill that guy",
            "kill someone", "kill somebody", "kill anyone", "kill anybody", "kill people",
            "stab him", "stab her", "stab someone", "stab somebody",
            "shoot him", "shoot her", "shoot someone", "shoot somebody",
            "hurt him", "hurt her", "hurt someone", "hurt somebody",
            "beat him up", "beat her up", "beat someone up", "beat somebody up",
            "going to kill him", "gonna kill him", "going to kill her", "gonna kill her",
            "going to kill the guy", "gonna kill the guy", "going to kill someone",
            "gonna kill someone", "going to kill somebody", "gonna kill somebody",
        ),
    ),
    SignalCategory(
        "explicit_intent",
        40,
        (
            "kill myself", "end my life", "want to die", "want to kill myself",
            "do not want to be alive", "don't want to be alive", "suicide", "ending it all",
            "off myself", "unalive myself", "commit suicide",
        ),
    ),
    SignalCategory(
        "planning_preparation",
        32,
        (
            "i have a plan", "made up my mind", "written my note", "wrote my note",
            "left a note", "chosen a date", "chosen a time", "researched methods",
            "said my goodbyes", "given away my things", "gave away my things",
            "everything i need", "have the pills", "picked the bridge", "picked one",
            "picked the time", "picked the date", "picked the method", "chose the method",
            "know exactly when", "not backing out", "following through",
            "scheduled my final message", "final message is scheduled",
        ),
    ),
    SignalCategory(
        "finality_farewell",
        24,
        (
            "goodbye everyone", "final goodbye", "my last message", "my last post",
            "won't be here anymore", "i won't be here", "not be here anymore",
            "this is goodbye", "last words", "writing my last words", "this is goodbye for real",
            "by tomorrow i will not be here", "after tonight none of this will matter",
            "by morning it will all be over for me", "i will not be here when the sun comes up",
            "you will not have to worry about me tomorrow", "thanks for everything this is the end for me",
            "my final message", "my final goodbye", "saying goodbye to everyone",
        ),
    ),
    SignalCategory(
        "self_harm",
        20,
        (
            "self harm", "cut myself", "cutting myself", "hurt myself", "harm myself",
            "overdose", "pills", "bridge", "jump off",
        ),
    ),
)


_MODERATE_SIGNAL_CATEGORIES = (
    SignalCategory(
        "hopelessness",
        18,
        (
            "hopeless", "no hope", "nothing matters", "no point", "meaningless",
            "cannot go on", "can't go on", "give up", "lost all hope",
            "cannot keep going", "can't keep going", "do not see a reason to keep trying",
            "nothing will ever get better", "every day feels heavier than the last",
            "tired of being me", "not doing well mentally", "awful", "done with life",
        ),
    ),
    SignalCategory(
        "burden_worthlessness",
        16,
        (
            "better off without me", "burden", "worthless", "hate myself",
            "world would be better without me", "nobody would miss me",
            "everyone would be happier without me", "i am a problem for everyone around me",
            "i am a problem for everyone", "useless", "feel broken",
        ),
    ),
    SignalCategory(
        "isolation_disconnection",
        14,
        (
            "alone", "completely alone", "nobody cares", "no one cares",
            "isolating myself", "disconnected", "no one understands",
            "feel disconnected from everyone around me", "alone in a crowded room",
            "nobody gets it",
        ),
    ),
    SignalCategory(
        "emotional_dysregulation",
        18,
        (
            "numb", "empty inside", "crying every day", "cannot stop crying",
            "dead inside", "panic attack", "everything hurts", "spiraling",
            "empty and i do not know what to do", "drowning in my own thoughts",
            "falling apart inside", "feel broken", "feel so empty", "feel empty",
            "feel alone and exhausted", "cannot calm down", "not okay",
            "severely distressed", "spiraling hard", "falling apart",
        ),
    ),
    SignalCategory(
        "indirect_self_harm",
        20,
        (
            "wish i could disappear", "want everything to stop", "do not want to wake up tomorrow",
            "wish i did not exist", "wish i was never born", "wish i wasn't here",
            "do not want to wake up", "not wake up tomorrow",
        ),
    ),
)


_PROTECTIVE_SIGNAL_CATEGORIES = (
    SignalCategory(
        "help_seeking",
        2,
        (
            "i need help", "please help", "can someone help", "i want therapy",
            "i called my therapist", "i reached out", "i told my friend",
            "i told my sister", "i asked my friend to stay with me",
            "booked a therapy appointment", "need professional help",
            "i need support", "admitted i need support",
        ),
    ),
    SignalCategory(
        "future_orientation",
        -8,
        (
            "tomorrow", "next week", "trying to get better", "working on this",
            "want to recover", "staying with family", "seeking support",
        ),
    ),
)


_HARM_TO_OTHERS_REGEX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\b(?:kill|murder|shoot|stab|hurt|beat)\s+(?:everyone|everybody|someone|somebody|anyone|anybody|people)\b"),
    re.compile(r"\b(?:kill|murder|shoot|stab|hurt|beat)\s+(?:them\s+all|all\s+of\s+them|every\s+one\s+of\s+you)\b"),
    re.compile(r"\b(?:i\s*(?:will|ll)|i\s+am\s+going\s+to|i\s+am\s+gonna|gonna|going\s+to)\s+(?:kill|murder|shoot|stab|hurt)\s+(?:everyone|everybody|someone|somebody|anyone|anybody|people|them\s+all|all\s+of\s+them)\b"),
    re.compile(r"\b(?:someone|somebody|people)\s+(?:is|are)\s+going\s+to\s+die\s+tonight\b"),
    re.compile(r"\bpeople\s+are\s+about\s+to\s+get\s+hurt\b"),
    re.compile(r"\bmake\s+them\s+all\s+pay\s+with\s+their\s+lives\b"),
    re.compile(r"\bend\s+(?:someone|them\s+all)\b"),
    re.compile(r"\bnobody\s+leaves\s+alive\b"),
    re.compile(r"\bspill\s+blood\s+tonight\b"),
    re.compile(r"\btake\s+lives\s+tonight\b"),
    re.compile(r"\bby\s+the\s+end\s+of\s+tonight\s+people\s+will\s+be\s+dead\b"),
    re.compile(r"\bdestroy\s+everyone\s+who\s+stands\s+in\s+front\s+of\s+me\b"),
)


_SELF_HARM_REGEX_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bresearched\s+ways\s+to\s+die\b"),
    re.compile(r"\bpicked\s+(?:the\s+)?(?:bridge|method|time|date)\b"),
    re.compile(r"\bi\s+have\s+the\s+pills\b"),
    re.compile(r"\bi\s+am\s+going\s+to\s+overdose\b"),
    re.compile(r"\b(?:tonight|by\s+tomorrow|by\s+morning).*(?:not\s+be\s+here|be\s+over\s+for\s+me)\b"),
    re.compile(r"\b(?:you\s+will\s+not\s+have\s+to\s+worry\s+about\s+me\s+tomorrow)\b"),
    re.compile(r"\bi\s+know\s+exactly\s+when\s+i\s+am\s+going\s+to\s+die\b"),
)


_FICTIONAL_CONTEXT_PATTERNS: tuple[re.Pattern[str], ...] = (
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


def analyze_signals(text: str) -> dict:
    normalized = expand_generation_language(text or "")
    lowered = _normalize(normalized)
    fictional_context = _find_regex_matches(lowered, _FICTIONAL_CONTEXT_PATTERNS)

    detected = []
    score = 0

    for category in _HIGH_SIGNAL_CATEGORIES + _MODERATE_SIGNAL_CATEGORIES + _PROTECTIVE_SIGNAL_CATEGORIES:
        if category.name == "harm_to_others_intent" and fictional_context:
            continue
        matches = _find_matches(lowered, category.patterns)
        if not matches:
            continue
        score += category.weight * min(len(matches), 2)
        detected.append(
            {
                "category": category.name,
                "weight": category.weight,
                "matches": matches[:5],
                "severity": _severity_for(category.weight),
            }
        )

    regex_harm_matches = [] if fictional_context else _find_regex_matches(lowered, _HARM_TO_OTHERS_REGEX_PATTERNS)
    if regex_harm_matches and not _has_category(detected, "harm_to_others_intent"):
        score += 42 * min(len(regex_harm_matches), 2)
        detected.append(
            {
                "category": "harm_to_others_intent",
                "weight": 42,
                "matches": regex_harm_matches[:5],
                "severity": "high",
            }
        )

    regex_self_harm_matches = _find_regex_matches(lowered, _SELF_HARM_REGEX_PATTERNS)
    if regex_self_harm_matches and not any(
        _has_category(detected, name)
        for name in ("explicit_intent", "planning_preparation", "finality_farewell")
    ):
        score += 32 * min(len(regex_self_harm_matches), 2)
        detected.append(
            {
                "category": "planning_preparation",
                "weight": 32,
                "matches": regex_self_harm_matches[:5],
                "severity": "high",
            }
        )

    score = max(0, min(100, score))
    risk_level = _risk_from_score_and_categories(score, detected)

    return {
        "normalized_text": normalized,
        "system_score": score,
        "system_risk_level": risk_level,
        "signals_detected": detected,
        "summary": _build_summary(risk_level, detected),
        "angles": {
            "harm_to_others_intent": _has_category(detected, "harm_to_others_intent"),
            "explicit_intent": _has_category(detected, "explicit_intent"),
            "planning_preparation": _has_category(detected, "planning_preparation"),
            "finality_farewell": _has_category(detected, "finality_farewell"),
            "self_harm": _has_category(detected, "self_harm"),
            "hopelessness": _has_category(detected, "hopelessness"),
            "burden_worthlessness": _has_category(detected, "burden_worthlessness"),
            "isolation_disconnection": _has_category(detected, "isolation_disconnection"),
            "emotional_dysregulation": _has_category(detected, "emotional_dysregulation"),
            "indirect_self_harm": _has_category(detected, "indirect_self_harm"),
            "help_seeking": _has_category(detected, "help_seeking"),
            "future_orientation": _has_category(detected, "future_orientation"),
            "fictional_context": bool(fictional_context),
        },
    }


def synthesize_risk(system_signals: dict, llm_classification: dict) -> dict:
    """
    Blend deterministic signals with the LLM classification into one final view.
    """
    system_level = system_signals.get("system_risk_level", "LOW_RISK")
    system_score = int(system_signals.get("system_score", 0))

    llm_level = _normalize_llm_level((llm_classification or {}).get("risk_level", "LOW_RISK"))
    llm_score = int((llm_classification or {}).get("risk_score", 0) or 0)
    llm_conf = float((llm_classification or {}).get("confidence", 0.0) or 0.0)

    order = {
        "LOW_RISK": 0,
        "MODERATE_RISK": 1,
        "HIGH_RISK_SELF_HARM": 2,
        "HIGH_RISK_HARM_TO_OTHERS": 2,
        "HIGH_RISK": 2,
    }
    final_level = system_level if order.get(system_level, 0) >= order.get(llm_level, 0) else llm_level
    final_score = max(system_score, llm_score)

    # If LLM is more severe and reasonably confident, keep it.
    if order.get(llm_level, 0) > order.get(system_level, 0) and llm_conf >= 0.60:
        final_level = llm_level
        final_score = max(final_score, llm_score)

    return {
        "risk_level": final_level,
        "risk_score": final_score,
        "confidence": max(llm_conf, min(0.95, 0.45 + final_score / 1000)),
        "reasoning": _synthesis_reasoning(system_level, llm_level, system_signals),
        "decision_source": "system_plus_llm",
    }


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def _find_matches(text: str, patterns: tuple[str, ...]) -> list[str]:
    hits = []
    for p in patterns:
        if p in text:
            hits.append(p)
    return hits


def _find_regex_matches(text: str, patterns: tuple[re.Pattern[str], ...]) -> list[str]:
    hits = []
    for pattern in patterns:
        for match in pattern.finditer(text):
            hits.append(match.group(0))
    return list(dict.fromkeys(hits))


def _severity_for(weight: int) -> str:
    if weight >= 24:
        return "high"
    if weight >= 12:
        return "moderate"
    return "protective"


def _has_category(detected: list[dict], name: str) -> bool:
    return any(item["category"] == name for item in detected)


def _risk_from_score_and_categories(score: int, detected: list[dict]) -> str:
    settings = load_risk_settings()
    labels = settings["labels"]
    thresholds = settings["thresholds"]
    if any(item["category"] == "harm_to_others_intent" for item in detected):
        return labels["high_harm_to_others"]
    high_categories = {"explicit_intent", "planning_preparation", "finality_farewell"}
    if any(item["category"] in high_categories for item in detected):
        return labels["high_self_harm"]
    if score >= int(thresholds.get("high_risk_min_score", 55)):
        return labels["high_self_harm"]
    if score >= int(thresholds.get("moderate_risk_min_score", 18)):
        return labels["moderate"]
    return labels["low"]


def _build_summary(risk_level: str, detected: list[dict]) -> str:
    if not detected:
        return "No major deterministic risk signals detected."
    top = ", ".join(item["category"].replace("_", " ") for item in detected[:3])
    return f"{risk_level.replace('_', ' ').title()} based on signals including {top}."


def _normalize_llm_level(level: str) -> str:
    level = (level or "").strip().upper()
    if level in {"HIGH_RISK_HARM_TO_OTHERS", "HIGH_RISK_SELF_HARM", "MODERATE_RISK", "LOW_RISK"}:
        return level
    if level == "HIGH_RISK":
        return "HIGH_RISK_SELF_HARM"
    return level or "LOW_RISK"


def _synthesis_reasoning(system_level: str, llm_level: str, system_signals: dict) -> str:
    detected = system_signals.get("signals_detected", [])
    if not detected:
        return f"LLM classification favored {llm_level.replace('_', ' ').lower()} with limited rule-based evidence."
    top = ", ".join(item["category"].replace("_", " ") for item in detected[:2])
    return (
        f"System reviewed multiple angles and found {top}. "
        f"Rule engine suggests {system_level.replace('_', ' ').lower()} while the model suggested "
        f"{llm_level.replace('_', ' ').lower()}."
    )


