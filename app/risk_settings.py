"""
Shared runtime settings for the active website risk pipeline.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path


_DEFAULT_SETTINGS = {
    "thresholds": {
        "confidence_escalation_threshold": 0.65,
        "moderate_prefilter_confidence_threshold": 0.70,
        "high_risk_min_score": 55,
        "moderate_risk_min_score": 18,
    },
    "labels": {
        "low": "LOW_RISK",
        "moderate": "MODERATE_RISK",
        "high_self_harm": "HIGH_RISK_SELF_HARM",
        "high_harm_to_others": "HIGH_RISK_HARM_TO_OTHERS",
    },
}


@lru_cache(maxsize=1)
def load_risk_settings() -> dict:
    settings = json.loads(json.dumps(_DEFAULT_SETTINGS))
    config_path = Path(__file__).with_name("risk_settings.json")

    if config_path.exists():
        try:
            user_settings = json.loads(config_path.read_text(encoding="utf-8"))
            _deep_update(settings, user_settings)
        except Exception:
            pass

    thresholds = settings.setdefault("thresholds", {})
    thresholds["confidence_escalation_threshold"] = float(
        os.environ.get(
            "AGENT_CONFIDENCE_THRESHOLD",
            thresholds.get("confidence_escalation_threshold", 0.65),
        )
    )
    return settings


def _deep_update(target: dict, source: dict) -> dict:
    for key, value in source.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target
