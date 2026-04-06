import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).parent.parent
APP_DIR = Path(__file__).parent

# API Configuration
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-6")

# Agent Configuration
CLASSIFIER_MAX_TOKENS = 256
EXPLAINER_MAX_TOKENS = 512
RECOMMENDER_MAX_TOKENS = 512
AGENT_TIMEOUT_SECONDS = 30

# App Configuration
APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("APP_PORT", "8000"))
DEBUG = os.environ.get("DEBUG", "false").lower() == "true"

# Analysis limits
MAX_TEXT_LENGTH = 2000
MAX_BATCH_SIZE = 10
RATE_LIMIT_PER_MINUTE = 60

# Risk levels
RISK_LEVELS = [
    "LOW_RISK",
    "MODERATE_RISK",
    "HIGH_RISK_SELF_HARM",
    "HIGH_RISK_HARM_TO_OTHERS",
]
RISK_COLORS = {
    "LOW_RISK": "#22c55e",
    "MODERATE_RISK": "#f59e0b",
    "HIGH_RISK_SELF_HARM": "#ef4444",
    "HIGH_RISK_HARM_TO_OTHERS": "#ef4444",
}
RISK_LABELS = {
    "LOW_RISK": "Low Risk",
    "MODERATE_RISK": "Moderate Risk",
    "HIGH_RISK_SELF_HARM": "High Risk - Self Harm",
    "HIGH_RISK_HARM_TO_OTHERS": "High Risk - Harm To Others",
}

# Crisis Resources (always shown for high-risk self-harm classifications)
CRISIS_RESOURCES = [
    {"name": "988 Suicide & Crisis Lifeline", "contact": "988", "type": "hotline", "description": "Call or text 988 (US)", "available": "24/7"},
    {"name": "Crisis Text Line", "contact": "Text HOME to 741741", "type": "text", "description": "Text-based crisis support (US)", "available": "24/7"},
    {"name": "International Association for Suicide Prevention", "contact": "https://www.iasp.info/resources/Crisis_Centres/", "type": "website", "description": "Global crisis center directory", "available": "Always"},
    {"name": "Samaritans (UK)", "contact": "116 123", "type": "hotline", "description": "Free call, UK and Ireland", "available": "24/7"},
]
