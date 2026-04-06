import time
import uuid
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field, field_validator

from app.agents.orchestrator import AgentOrchestrator, MLFallbackClassifier

router = APIRouter()

# ---------------------------------------------------------------------------
# In-memory stats store
# ---------------------------------------------------------------------------
_stats: dict[str, Any] = {
    "total_analyses": 0,
    "total_processing_time_ms": 0,
    "risk_level_counts": {
        "HIGH_RISK": 0,
        "HIGH_RISK_SELF_HARM": 0,
        "HIGH_RISK_HARM_TO_OTHERS": 0,
        "MODERATE_RISK": 0,
        "LOW_RISK": 0,
    },
    "feedback_count": 0,
    "tier_counts": {
        "keyword_prefilter": 0,
        "haiku": 0,
        "sonnet_escalation": 0,
        "ml_fallback": 0,
        "cache_hit": 0,
    },
}

# Feedback store keyed by analysis_id
_feedback_store: dict[str, dict] = {}

# Cached agent instances (created once per process)
_orchestrator: AgentOrchestrator | None = None
_fallback: MLFallbackClassifier | None = None


def _get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator


def _get_fallback() -> MLFallbackClassifier:
    global _fallback
    if _fallback is None:
        _fallback = MLFallbackClassifier()
    return _fallback


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Social media post text to analyse.")
    use_ml_fallback: bool = Field(
        False,
        description="Use keyword-heuristic fallback instead of the Claude API.",
    )


class BatchAnalysisRequest(BaseModel):
    texts: list[str] = Field(
        ...,
        min_length=1,
        description="List of social media posts to analyse (max 10).",
    )
    use_ml_fallback: bool = Field(False)

    @field_validator("texts")
    @classmethod
    def check_batch_size(cls, v: list[str]) -> list[str]:
        if len(v) > 10:
            raise ValueError("Batch size must not exceed 10 texts.")
        if any(not t.strip() for t in v):
            raise ValueError("Each text in the batch must be non-empty.")
        return v


class FeedbackRequest(BaseModel):
    analysis_id: str = Field(..., description="ID of the analysis result being corrected.")
    correct_label: str = Field(
        ...,
        description="The correct risk label.",
    )
    user_comment: str = Field("", description="Optional free-text comment from the reviewer.")

    @field_validator("correct_label")
    @classmethod
    def validate_label(cls, v: str) -> str:
        valid = {
            "HIGH_RISK",
            "HIGH_RISK_SELF_HARM",
            "HIGH_RISK_HARM_TO_OTHERS",
            "MODERATE_RISK",
            "LOW_RISK",
        }
        if v not in valid:
            raise ValueError(f"correct_label must be one of {sorted(valid)}")
        return v


# ---------------------------------------------------------------------------
# Background task helpers
# ---------------------------------------------------------------------------

def _record_analysis(result: dict) -> None:
    """Update in-memory stats after an analysis completes."""
    _stats["total_analyses"] += 1
    _stats["total_processing_time_ms"] += result.get("processing_time_ms", 0)

    risk_level = result.get("classification", {}).get("risk_level", "LOW_RISK")
    if risk_level in _stats["risk_level_counts"]:
        _stats["risk_level_counts"][risk_level] += 1
        if risk_level in {"HIGH_RISK_SELF_HARM", "HIGH_RISK_HARM_TO_OTHERS"}:
            _stats["risk_level_counts"]["HIGH_RISK"] += 1

    # Track which tier handled this request
    tier = result.get("tier_used", "")
    if result.get("cache_hit"):
        _stats["tier_counts"]["cache_hit"] += 1
    elif tier in _stats["tier_counts"]:
        _stats["tier_counts"][tier] += 1


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/health", tags=["Meta"])
async def health_check() -> dict:
    """Return service health status."""
    return {
        "status": "ok",
        "agents": [
            "normalization_agent - repairs merged words and slang before analysis",
            "keyword_prefilter (0 tokens)",
            "unified_haiku - classify + explain + recommend",
            "escalation_sonnet - edge cases and high-risk verification",
            "motivation_haiku - personalised coping and support",
        ],
        "architecture": "tiered: Tier0 keyword -> Tier1 Haiku -> Tier2 Sonnet -> Tier3 Motivation",
        "cost_model": "~90% cheaper than 3-call Opus architecture",
    }


@router.get("/stats", tags=["Meta"])
async def get_stats() -> dict:
    """Return in-memory analysis statistics including cache and tier breakdown."""
    total = _stats["total_analyses"]
    avg_ms = _stats["total_processing_time_ms"] / total if total > 0 else 0

    # Pull live cache stats from the orchestrator if available
    cache_info: dict = {}
    try:
        orch = _get_orchestrator()
        cache_info = orch.cache_stats()
    except Exception:
        pass

    return {
        "total_analyses": total,
        "average_processing_time_ms": round(avg_ms, 2),
        "risk_level_counts": _stats["risk_level_counts"],
        "feedback_count": _stats["feedback_count"],
        "tier_counts": _stats.get("tier_counts", {}),
        "cache": cache_info,
    }


@router.get("/cache/stats", tags=["Meta"])
async def get_cache_stats() -> dict:
    """Return detailed cache statistics."""
    try:
        return _get_orchestrator().cache_stats()
    except Exception as exc:
        return {"error": str(exc)}


@router.post("/cache/clear", tags=["Meta"])
async def clear_cache() -> dict:
    """Clear the result cache (admin endpoint)."""
    try:
        from app.agents.cache import get_cache
        get_cache().clear()
        return {"status": "cleared", "message": "Result cache cleared successfully."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/analyze", tags=["Analysis"])
async def analyze_text(
    request: AnalysisRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Analyse a single social media post for suicidal ideation risk.

    Uses all three Claude agents (classifier → explainer + recommender in
    parallel) unless ``use_ml_fallback`` is True, in which case a
    keyword-heuristic classifier is used instead.
    """
    analysis_id = str(uuid.uuid4())

    try:
        if request.use_ml_fallback:
            result = _get_fallback().analyze(request.text)
        else:
            result = _get_orchestrator().analyze(request.text)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analysis failed: {exc}",
        ) from exc

    result["analysis_id"] = analysis_id
    background_tasks.add_task(_record_analysis, result)
    return result


@router.post("/analyze/batch", tags=["Analysis"])
async def analyze_batch(
    request: BatchAnalysisRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Analyse up to 10 social media posts in a single request.

    Results are returned in the same order as the input texts.
    Each result includes its own ``analysis_id``.
    """
    results = []
    errors = []

    agent = _get_fallback() if request.use_ml_fallback else _get_orchestrator()

    for idx, text in enumerate(request.texts):
        analysis_id = str(uuid.uuid4())
        try:
            if request.use_ml_fallback:
                result = _get_fallback().analyze(text)
            else:
                result = _get_orchestrator().analyze(text)
            result["analysis_id"] = analysis_id
            results.append(result)
            background_tasks.add_task(_record_analysis, result)
        except Exception as exc:  # noqa: BLE001
            error_entry = {
                "index": idx,
                "text": text[:100],
                "error": str(exc),
                "analysis_id": analysis_id,
            }
            errors.append(error_entry)
            results.append({
                "analysis_id": analysis_id,
                "text": text,
                "error": True,
                "error_detail": str(exc),
            })

    return {
        "total": len(request.texts),
        "succeeded": len(results) - len(errors),
        "failed": len(errors),
        "results": results,
        "errors": errors if errors else None,
    }


@router.post("/feedback", tags=["Feedback"])
async def submit_feedback(
    request: FeedbackRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Submit a correction or annotation for a previous analysis result.

    Feedback is stored in-memory and contributes to the ``feedback_count``
    statistic.  In a production system this would be persisted to a database.
    """
    entry = {
        "analysis_id": request.analysis_id,
        "correct_label": request.correct_label,
        "user_comment": request.user_comment,
        "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    _feedback_store[request.analysis_id] = entry

    def _increment_feedback() -> None:
        _stats["feedback_count"] += 1

    background_tasks.add_task(_increment_feedback)

    return {
        "status": "accepted",
        "analysis_id": request.analysis_id,
        "message": "Feedback recorded. Thank you for helping improve the model.",
    }

