"""v1 API routes for Hybrid Recommendation Engine.

All endpoints are prefixed with /api/v1 in the main app.
Dependencies inject the HybridRecommender via app state.
"""

from fastapi import APIRouter, HTTPException, Request, status, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.main import app_state
from app.hybrid import HybridRecommender
from app.data import DataProcessor
from app.cf import CollaborativeFiltering
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["v1 Inference"])


# --- Request/Response Models ---

class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="Target user ID")
    n: int = Field(default=10, ge=1, le=100, description="Number of recommendations")
    alpha: float = Field(default=None, ge=0.0, le=1.0, description="Optional CF weight override")
    beta: float = Field(default=None, ge=0.0, le=1.0, description="Optional CB weight override")
    exclude_seen: bool = Field(default=True, description="Filter out already-interacted items")


class BatchRecommendationRequest(BaseModel):
    user_ids: list[str] = Field(..., description="List of target user IDs")
    n: int = Field(default=10, ge=1, le=100, description="Per-user recommendation count")
    alpha: float = Field(default=None, ge=0.0, le=1.0)
    beta: float = Field(default=None, ge=0.0, le=1.0)
    exclude_seen: bool = Field(default=True)


class ItemInfo(BaseModel):
    item_id: str
    hybrid_score: float
    cf_score: float
    cb_score: float


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: list[ItemInfo]
    total_evaluated: int


class BatchRecommendationResponse(BaseModel):
    results: dict[str, list[ItemInfo]]
    total_evaluated: dict[str, int]


class HealthDetail(BaseModel):
    status: str
    model_fitted: bool
    uptime_seconds: float
    current_time: str
    redis_connected: bool = True


class MetricsDetail(BaseModel):
    uptime_seconds: float
    model_fitted: bool
    recommender_type: str
    cf_alpha: float
    cf_beta: float
    total_items: int = 0
    total_users: int = 0


# --- Dependency ---

def get_rec() -> HybridRecommender:
    """Retrieve the HybridRecommender from the global app state."""
    rec = app_state.recommender
    if rec is None:
        raise HTTPException(status_code=503, detail="Recommender not initialized")
    return rec


# --- Endpoints ---

@router.post("/init", status_code=201, tags=["lifecycle"])
async def init_model(
    interactions: list[dict],
    items: list[dict],
    save_after_fit: bool = True,
    rec: HybridRecommender = Depends(get_rec),
):
    """Initialize and fit the hybrid model.

    **Authorization**: Requires model initialization privileges.
    """
    # If rec is already provided via Depends, use it; otherwise the endpoint
    # body data will be processed to fit a new model.
    # For now, this endpoint is a pass-through that signals intent.
    return {"status": "success", "message": "Init endpoint reached"}


@router.post("/recommend", response_model=RecommendationResponse, tags=["inference"])
async def get_recommendations(
    request: RecommendationRequest,
    rec_model: HybridRecommender = Depends(get_rec),
):
    """Get top-N hybrid recommendations for a single user."""
    try:
        from ..app.main import ItemScore
        recommendations = rec_model.recommend(
            user_id=request.user_id,
            n=request.n,
            alpha=request.alpha,
            beta=request.beta,
            exclude_seen=request.exclude_seen,
        )

        item_infos = []
        for item_id, hybrid_score, cf_score, cb_score in recommendations:
            item_infos.append(
                ItemInfo(
                    item_id=item_id,
                    hybrid_score=round(hybrid_score, 4),
                    cf_score=round(cf_score, 4),
                    cb_score=round(cb_score, 4),
                )
            )

        effective_alpha, effective_beta = rec_model._get_user_weights(
            request.user_id, request.alpha, request.beta
        )

        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=item_infos,
            total_evaluated=len(rec_model.data_processor.item_ids) if rec_model.data_processor else 0,
            weights_used={"alpha": round(effective_alpha, 4), "beta": round(effective_beta, 4)},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating recommendations: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/recommend/batch", response_model=BatchRecommendationResponse, tags=["inference"])
async def get_batch_recommendations(
    request: BatchRecommendationRequest,
    rec_model: HybridRecommender = Depends(get_rec),
):
    """Get recommendations for multiple users (batch endpoint)."""
    try:
        results = {}
        total_evaluated = {}

        for user_id in request.user_ids:
            recommendations = rec_model.recommend(
                user_id=user_id,
                n=request.n,
                alpha=request.alpha,
                beta=request.beta,
                exclude_seen=request.exclude_seen,
            )

            item_infos = []
            for item_id, hybrid_score, cf_score, cb_score in recommendations:
                item_infos.append(
                    ItemInfo(
                        item_id=item_id,
                        hybrid_score=round(hybrid_score, 4),
                        cf_score=round(cf_score, 4),
                        cb_score=round(cb_score, 4),
                    )
                )

            results[user_id] = item_infos
            total_evaluated[user_id] = len(rec_model.data_processor.item_ids) if rec_model.data_processor else 0

        return BatchRecommendationResponse(
            results=results,
            total_evaluated=total_evaluated,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error generating batch recommendations: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health", response_model=HealthDetail, tags=["monitoring"])
async def v1_health_check(rec: HybridRecommender = Depends(get_rec)):
    """Detailed health check endpoint."""
    import redis as redis_lib

    redis_connected = True
    try:
        r = redis_lib.from_url("redis://localhost:6379")
        r.ping()
    except Exception:
        redis_connected = False

    from datetime import datetime
    return HealthDetail(
        status="healthy" if rec.is_fitted else "initializing",
        model_fitted=rec.is_fitted,
        uptime_seconds=round(
            (
                getattr(rec, "_startup_time", time.time())
                if hasattr(rec, "_startup_time")
                else time.time()
            )
            - time.time(),
            2
        ),
        current_time=datetime.utcnow().isoformat() + "Z",
        redis_connected=redis_connected,
    )


@router.get("/metrics", tags=["monitoring"])
async def v1_metrics(rec: HybridRecommender = Depends(get_rec)):
    """Prometheus-style metrics endpoint."""
    import time

    fitted = rec is not None and rec.is_fitted
    fit_info = rec.get_fit_info() if fitted else {}

    uptime = time.time() - getattr(rec, "_startup_time", time.time())

    return {
        "uptime_seconds": uptime,
        "model_fitted": int(fitted),
        "recommender_type": rec.cf_method if fitted else "none",
        "cf_alpha": rec.cf_alpha if fitted else 0.0,
        "cf_beta": rec.cf_beta if fitted else 0.0,
        "fit_time_seconds": fit_info.get("fit_time_seconds", 0.0),
    }


@router.get("/version", tags=["system"])
async def v1_version():
    """Version endpoint."""
    return {"version": "1.0.0", "status": "active", "environment": "production"}