"""
Generation-aware social media lexicon helpers.

This module expands slang, acronyms, and generation-specific phrasing that
often appears in social posts. It is intentionally conservative: we normalise
terms into descriptive phrases so downstream models can reason over the text
more reliably without depending on every exact slang spelling.

Sources used to shape the vocabulary:
- AYA suicide-risk language study describing slang/acronyms such as
  "KMS", "KYS", and "Kermit Sewage Slide".
- Social-media suicide ideation detection literature highlighting the
  importance of youth vernacular, hopelessness language, and context.
"""

from __future__ import annotations

import re
from typing import Iterable


_SLANG_REPLACEMENTS: dict[str, str] = {
    # Internet / youth shorthand
    "kms": "kill myself",
    "kys": "kill yourself",
    "fml": "fuck my life",
    "irl": "in real life",
    "idk": "i do not know",
    "imo": "in my opinion",
    "imho": "in my honest opinion",
    "ngl": "not going to lie",
    "tbh": "to be honest",
    "rn": "right now",
    "af": "as fuck",
    "lmk": "let me know",
    "wanna": "want to",
    "gonna": "going to",
    "gotta": "got to",
    "imma": "i am going to",
    "cuz": "because",
    "bc": "because",
    "tho": "though",
    "thoo": "though",
    "u": "you",
    "ur": "your",
    "ya": "you",
    "yall": "you all",
    "ain't": "is not",
    "aint": "is not",
    "wtf": "what the fuck",
    "omg": "oh my god",
    "smh": "shaking my head",
    "ikr": "i know right",
    "imooo": "in my opinion",
    "dw": "do not worry",
    "venting": "sharing distress",
    "spiraling": "emotionally spiraling",
    "spiralling": "emotionally spiraling",
    "burnt out": "burned out",
    "burntout": "burned out",
    "unalive": "dead",
    "unalive myself": "kill myself",
    "off myself": "kill myself",
    "end it all": "end my life",
    "end it": "end my life",
    "check out permanently": "die",
    "disappear forever": "die",
    "not wake up": "die in sleep",
    "no thoughts just": "emotionally numb",
    # Terms referenced in youth-language literature and common online usage
    "kermit sewer slide": "commit suicide",
    "kermit sewage slide": "commit suicide",
    "sewer slide": "suicide",
    "sewyside": "suicide",
    "self delete": "kill myself",
    "self-delete": "kill myself",
    "self yeet": "kill myself",
    "yeet myself": "throw myself away",
    "2meirl4meirl": "me in real life and depressed",
    "emo": "emotionally distressed",
    "lowkey": "quietly",
}


_PHRASE_REPLACEMENTS: dict[str, str] = {
    "kms vibes": "severely distressed vibes",
    "cant do this anymore": "cannot do this anymore",
    "can't do this anymore": "cannot do this anymore",
    "dont want to be here": "do not want to be alive",
    "don't want to be here": "do not want to be alive",
    "dont wanna be here": "do not want to be alive",
    "i'm done": "i am done",
    "im done": "i am done",
    "so done": "emotionally exhausted",
    "dead inside": "emotionally numb",
    "empty inside": "emotionally empty",
    "no reason to be here": "no reason to live",
    "better off without me": "others are better off without me",
    "not gonna do anything": "not acting on these thoughts",
    "still here": "still alive and present",
}


_MERGED_PHRASE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Self-harm / suicide intent
    (re.compile(r"\bkillmyself\b", re.IGNORECASE), "kill myself"),
    (re.compile(r"\bkillingmyself\b", re.IGNORECASE), "killing myself"),
    (re.compile(r"\bendmylife\b", re.IGNORECASE), "end my life"),
    (re.compile(r"\bendingmylife\b", re.IGNORECASE), "ending my life"),
    (re.compile(r"\bwanttodie\b", re.IGNORECASE), "want to die"),
    (re.compile(r"\bdontwanttobealive\b", re.IGNORECASE), "do not want to be alive"),
    (re.compile(r"\bdon'twanttobealive\b", re.IGNORECASE), "do not want to be alive"),
    (re.compile(r"\bdontwanttolive\b", re.IGNORECASE), "do not want to live"),
    (re.compile(r"\bdon'twanttolive\b", re.IGNORECASE), "do not want to live"),
    (re.compile(r"\bbetteroffwithoutme\b", re.IGNORECASE), "better off without me"),
    (re.compile(r"\bnoreasontolive\b", re.IGNORECASE), "no reason to live"),
    (re.compile(r"\bnothingtolivefor\b", re.IGNORECASE), "nothing to live for"),
    (re.compile(r"\bihatemyself\b", re.IGNORECASE), "i hate myself"),
    (re.compile(r"\bworthlessaf\b", re.IGNORECASE), "worthless as fuck"),
    # Harm to others
    (re.compile(r"\bkillhim\b", re.IGNORECASE), "kill him"),
    (re.compile(r"\bkillher\b", re.IGNORECASE), "kill her"),
    (re.compile(r"\bkillthem\b", re.IGNORECASE), "kill them"),
    (re.compile(r"\bstabhim\b", re.IGNORECASE), "stab him"),
    (re.compile(r"\bstabher\b", re.IGNORECASE), "stab her"),
    (re.compile(r"\bshoothim\b", re.IGNORECASE), "shoot him"),
    (re.compile(r"\bshoother\b", re.IGNORECASE), "shoot her"),
    # Compressed common phrases
    (re.compile(r"\bimgonna\b", re.IGNORECASE), "i am gonna"),
    (re.compile(r"\bimgonna\b", re.IGNORECASE), "i am gonna"),
    (re.compile(r"\bicant\b", re.IGNORECASE), "i cannot"),
    (re.compile(r"\bican't\b", re.IGNORECASE), "i cannot"),
)


def normalize_generation_language(text: str) -> dict:
    """
    Expand slang and generation-specific wording into clearer English phrases
    while recording what changed.
    """
    if not isinstance(text, str):
        text = str(text) if text is not None else ""

    expanded = text
    fixes: list[dict] = []

    for pattern, replacement in _MERGED_PHRASE_PATTERNS:
        expanded, count = pattern.subn(replacement, expanded)
        if count:
            fixes.append({
                "kind": "merged_phrase",
                "pattern": pattern.pattern,
                "replacement": replacement,
                "count": count,
            })

    for phrase, replacement in _PHRASE_REPLACEMENTS.items():
        expanded, count = re.subn(re.escape(phrase), replacement, expanded, flags=re.IGNORECASE)
        if count:
            fixes.append({
                "kind": "phrase_rewrite",
                "original": phrase,
                "replacement": replacement,
                "count": count,
            })

    # Token-level replacements with word boundaries to avoid damaging normal text.
    for token, replacement in _SLANG_REPLACEMENTS.items():
        pattern = rf"(?<!\w){re.escape(token)}(?!\w)"
        expanded, count = re.subn(pattern, replacement, expanded, flags=re.IGNORECASE)
        if count:
            fixes.append({
                "kind": "slang_expansion",
                "original": token,
                "replacement": replacement,
                "count": count,
            })

    return {
        "original_text": text,
        "normalized_text": expanded,
        "fixes": fixes,
    }


def expand_generation_language(text: str) -> str:
    """
    Backward-compatible wrapper returning only the normalized text.
    """
    return normalize_generation_language(text)["normalized_text"]


def generation_keywords() -> Iterable[str]:
    """Expose the curated vocabulary for analysis/reporting."""
    return sorted(set(_SLANG_REPLACEMENTS) | set(_PHRASE_REPLACEMENTS))
