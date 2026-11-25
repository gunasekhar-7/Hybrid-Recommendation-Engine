"""Production FastAPI Application for Hybrid Recommendation Engine.

Exposes RESTful endpoints for single/batch recommendations, item similarity,
health/readiness probes, metrics, and on-demand model fitting.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .cf import CollaborativeFiltering
from .data import DataProcessor
from .hybrid import HybridRecommender

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Production application settings loaded from environment variables."""

    REDIS_URL: str = "redis://localhost:6379"
    MODEL_DIR: str = "artifacts/model"
    CORS_ORIGINS: List[str] = ["*"]
    DEFAULT_K: int = 10
    MAX_K: int = 100
    LOG_LEVEL: str = "INFO"
    SBERT_BATCH_SIZE: int = 64
    EMBEDDING_TTL_SECONDS: int = 86400

    model_config = SettingsConfigDict(
        env_prefix="REC_",
        case_sensitive=False,
    )


settings = Settings()

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared Application State
# ---------------------------------------------------------------------------

class AppState:
    """Encapsulates global application dependencies and state."""

    def __init__(self) -> None:
        self.recommender: Optional[HybridRecommender] = None
        self.redis_client: Optional[Any] = None
        self.startup_time: float = time.time()
        self.is_ready: bool = False


app_state = AppState()


# ---------------------------------------------------------------------------
# Lifespan Management
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup and shutdown lifecycle hooks."""
    app_state.startup_time = time.time()
    logger.info("Initializing Hybrid Recommendation Engine service...")

    # 1. Connect Redis (optional caching layer)
    try:
        import redis
        client = redis.from_url(settings.REDIS_URL, decode_responses=False)
        client.ping()
        app_state.redis_client = client
        logger.info("Connected to Redis cache at %s", settings.REDIS_URL)
    except Exception as e:
        logger.warning("Redis is unavailable (%s). Running with in-memory caching.", e)
        app_state.redis_client = None

    # 2. Attempt to load pre-trained model artifact if available
    if app_state.recommender is None or not app_state.recommender.is_fitted:
        model_dir = Path(settings.MODEL_DIR)
        if (model_dir / "metadata.pkl").exists():
            try:
                logger.info("Loading pre-trained model artifact from %s...", model_dir)
                app_state.recommender = HybridRecommender.load(
                    directory=model_dir,
                    redis_client=app_state.redis_client,
                )
                app_state.is_ready = True
                logger.info("Model loaded successfully. Ready for inference.")
            except Exception as e:
                logger.error("Failed to load model from %s: %s", model_dir, e)
                app_state.recommender = None
        else:
            logger.info("No saved model found at %s. Service awaiting /init payload.", model_dir)
            if app_state.recommender is None:
                app_state.recommender = HybridRecommender(
                    redis_client=app_state.redis_client,
                    embedding_ttl=settings.EMBEDDING_TTL_SECONDS,
                )

    yield

    # Shutdown logic
    logger.info("Shutting down Hybrid Recommendation Engine service...")
    if app_state.redis_client is not None:
        try:
            app_state.redis_client.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

# Optional API key authentication via header: X-API-Key
# Set REC_API_KEY env var to enforce authentication; leave empty to allow open access
_API_KEY = os.getenv("REC_API_KEY")

if _API_KEY:
    @app.middleware("http")
    async def api_key_auth_middleware(request: Request, call_next):
        """Enforce API key authentication if REC_API_KEY is configured."""
        provided_key = request.headers.get("X-API-Key")
        if provided_key != _API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )
        response = await call_next(request)
        return response
else:
    logger.info("API key authentication disabled (REC_API_KEY not set)")

app = FastAPI(
    title="Hybrid Recommendation Engine API",
    description="High-performance recommender fusing Collaborative Filtering with SBERT embeddings.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS Middleware - restrict origins in production
# Default allows localhost:3000 for development; override via REC_CORS_ORIGINS env var
cors_origins = settings.CORS_ORIGINS if settings.CORS_ORIGINS != ["*"] else ["http://localhost:3000", "http://localhost:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Process-Time", "X-Request-ID"],
)

# Include API versioned router
from .api.v1 import router as v1_router
app.include_router(v1_router, prefix="/api/v1")

# Optional API key authentication via header: X-API-Key
# Set REC_API_KEY env var to enforce authentication; leave empty to allow open access
_API_KEY = os.getenv("REC_API_KEY")

if _API_KEY:
    @app.middleware("http")
    async def api_key_auth_middleware(request: Request, call_next):
        """Enforce API key authentication if REC_API_KEY is configured."""
        provided_key = request.headers.get("X-API-Key")
        if provided_key != _API_KEY:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing API key",
            )
        response = await call_next(request)
        return response
else:
    logger.info("API key authentication disabled (REC_API_KEY not set)")


# Root-level health (non-versioned, for k8s liveness/readiness probes)


@app.middleware("http")
async def request_timing_middleware(request: Request, call_next):
    """Add correlation ID and processing time headers to every response."""
    req_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    start_time = time.time()

    response = await call_next(request)

    latency = time.time() - start_time
    response.headers["X-Request-ID"] = req_id
    response.headers["X-Process-Time"] = f"{latency:.4f}s"

    if request.url.path not in ("/health", "/metrics"):
        logger.info(
            "[%s] %s %s status=%d latency=%.4fs",
            req_id,
            request.method,
            request.url.path,
            response.status_code,
            latency,
        )
    return response


# ---------------------------------------------------------------------------
# Request & Response Schemas
# ---------------------------------------------------------------------------

class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="Target user identifier", examples=["user_42"])
    n: int = Field(default=10, ge=1, le=100, description="Number of recommendations to retrieve")
    alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional CF weight override")
    beta: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Optional CB weight override")
    exclude_seen: bool = Field(default=True, description="Filter out items user already interacted with")

    @field_validator("n")
    @classmethod
    def validate_n(cls, v: int) -> int:
        if v > settings.MAX_K:
            raise ValueError(f"n cannot exceed {settings.MAX_K}")
        return v


class BatchRecommendationRequest(BaseModel):
    user_ids: List[str] = Field(..., min_length=1, max_length=1000, description="List of user IDs")
    n: int = Field(default=10, ge=1, le=100, description="Number of recommendations per user")
    alpha: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    beta: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    exclude_seen: bool = Field(default=True)


class SimilarItemsRequest(BaseModel):
    item_id: str = Field(..., description="Target item identifier", examples=["item_101"])
    n: int = Field(default=10, ge=1, le=50, description="Number of similar items")


class ItemScore(BaseModel):
    item_id: str
    hybrid_score: float
    cf_score: float
    cb_score: float


class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[ItemScore]
    total_evaluated: int
    weights_used: Dict[str, float]


class BatchRecommendationResponse(BaseModel):
    results: Dict[str, List[ItemScore]]
    users_processed: int
    total_items_in_catalog: int


class SimilarItemScore(BaseModel):
    item_id: str
    similarity_score: float


class SimilarItemsResponse(BaseModel):
    item_id: str
    similar_items: List[SimilarItemScore]


class InitRequest(BaseModel):
    interactions: List[Dict[str, Any]] = Field(
        ...,
        description="List of interaction dicts containing 'user_id', 'item_id', and 'rating'",
    )
    items: List[Dict[str, Any]] = Field(
        ...,
        description="List of catalog item dicts containing 'item_id', 'title', and optional 'description'",
    )
    save_after_fit: bool = Field(default=True, description="Save fitted model artifact to disk")


class HealthResponse(BaseModel):
    status: str
    is_ready: bool
    model_fitted: bool
    uptime_seconds: float
    timestamp: str
    redis_connected: bool
    catalog_items: int
    active_users: int


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check():
    """Liveness and readiness health check probe."""
    redis_ok = False
    if app_state.redis_client is not None:
        try:
            app_state.redis_client.ping()
            redis_ok = True
        except Exception:
            redis_ok = False

    fitted = bool(app_state.recommender and app_state.recommender.is_fitted)
    n_items = len(app_state.recommender.data_processor.item_ids) if fitted and app_state.recommender.data_processor else 0
    n_users = len(app_state.recommender.data_processor.user_ids) if fitted and app_state.recommender.data_processor else 0

    return HealthResponse(
        status="healthy" if fitted else "degraded (unfitted)",
        is_ready=fitted,
        model_fitted=fitted,
        uptime_seconds=round(time.time() - app_state.startup_time, 2),
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        redis_connected=redis_ok,
        catalog_items=n_items,
        active_users=n_users,
    )


@app.get("/metrics", tags=["Monitoring"])
def metrics_endpoint():
    """Prometheus-compatible monitoring endpoint."""
    fitted = bool(app_state.recommender and app_state.recommender.is_fitted)
    fit_info = app_state.recommender.get_fit_info() if fitted and app_state.recommender else {}

    return {
        "service_uptime_seconds": time.time() - app_state.startup_time,
        "model_fitted": int(fitted),
        "catalog_size": fit_info.get("total_items", 0),
        "user_count": fit_info.get("total_users", 0),
        "cf_alpha": fit_info.get("cf_alpha", 0.5),
        "cf_beta": fit_info.get("cf_beta", 0.5),
        "fit_time_seconds": fit_info.get("fit_time_seconds", 0.0),
    }


@app.get("/version", tags=["System"])
def version_endpoint():
    """Get service version information."""
    return {"version": "1.0.0", "status": "active", "environment": "production"}


@app.post("/recommend", response_model=RecommendationResponse, tags=["Inference"])
def recommend_endpoint(request: RecommendationRequest):
    """Retrieve top-N personalized hybrid recommendations for a single user."""
    rec = app_state.recommender
    if rec is None or not rec.is_fitted:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation model is not yet fitted. Call /init first.",
        )

    try:
        raw_recs = rec.recommend(
            user_id=request.user_id,
            n=request.n,
            alpha=request.alpha,
            beta=request.beta,
            exclude_seen=request.exclude_seen,
        )

        item_scores = [
            ItemScore(
                item_id=iid,
                hybrid_score=round(float(h), 4),
                cf_score=round(float(c), 4),
                cb_score=round(float(b), 4),
            )
            for iid, h, c, b in raw_recs
        ]

        effective_alpha, effective_beta = rec._get_user_weights(
            request.user_id, request.alpha, request.beta
        )

        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=item_scores,
            total_evaluated=len(rec.data_processor.item_ids) if rec.data_processor else 0,
            weights_used={"alpha": round(effective_alpha, 4), "beta": round(effective_beta, 4)},
        )
    except Exception as e:
        logger.exception("Inference error for user %s: %s", request.user_id, e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/recommend/batch", response_model=BatchRecommendationResponse, tags=["Inference"])
def batch_recommend_endpoint(request: BatchRecommendationRequest):
    """Retrieve top-N hybrid recommendations for a batch of users."""
    rec = app_state.recommender
    if rec is None or not rec.is_fitted:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Recommendation model is not yet fitted. Call /init first.",
        )

    try:
        batch_results = rec.recommend_batch(
            user_ids=request.user_ids,
            n=request.n,
            alpha=request.alpha,
            beta=request.beta,
            exclude_seen=request.exclude_seen,
        )

        formatted_results: Dict[str, List[ItemScore]] = {}
        for uid, rec_list in batch_results.items():
            formatted_results[uid] = [
                ItemScore(
                    item_id=iid,
                    hybrid_score=round(float(h), 4),
                    cf_score=round(float(c), 4),
                    cb_score=round(float(b), 4),
                )
                for iid, h, c, b in rec_list
            ]

        catalog_len = len(rec.data_processor.item_ids) if rec.data_processor else 0
        return BatchRecommendationResponse(
            results=formatted_results,
            users_processed=len(request.user_ids),
            total_items_in_catalog=catalog_len,
        )
    except Exception as e:
        logger.exception("Batch recommendation error: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/similar-items", response_model=SimilarItemsResponse, tags=["Inference"])
def similar_items_endpoint(request: SimilarItemsRequest):
    """Find catalog items semantically similar to target item."""
    rec = app_state.recommender
    if rec is None or not rec.is_fitted:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not fitted. Call /init first.",
        )

    try:
        sims = rec.recommend_similar_items(item_id=request.item_id, n=request.n)
        return SimilarItemsResponse(
            item_id=request.item_id,
            similar_items=[
                SimilarItemScore(item_id=iid, similarity_score=round(score, 4))
                for iid, score in sims
            ],
        )
    except Exception as e:
        logger.exception("Similar items lookup failed: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.post("/init", status_code=status.HTTP_201_CREATED, tags=["Lifecycle"])
def initialize_model_endpoint(request: InitRequest):
    """Train or refit the recommendation model with provided interactions and catalog data."""
    try:
        interactions_df = pd.DataFrame(request.interactions)
        item_features_df = pd.DataFrame(request.items)

        if interactions_df.empty or item_features_df.empty:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Interactions and items datasets cannot be empty.",
            )

        for col in ["user_id", "item_id", "rating"]:
            if col not in interactions_df.columns:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Interactions missing required column: '{col}'",
                )

        if "item_id" not in item_features_df.columns:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Item features missing required column: 'item_id'",
            )

        logger.info(
            "Fitting new HybridRecommender on %d interactions and %d items...",
            len(interactions_df),
            len(item_features_df),
        )

        # Create new model instance
        new_recommender = HybridRecommender(
            redis_client=app_state.redis_client,
            embedding_ttl=settings.EMBEDDING_TTL_SECONDS,
        )
        new_recommender.fit(
            interactions_df=interactions_df,
            item_features_df=item_features_df,
            sbert_batch_size=settings.SBERT_BATCH_SIZE,
        )

        # Atomically swap active model
        app_state.recommender = new_recommender
        app_state.is_ready = True

        if request.save_after_fit:
            model_dir = Path(settings.MODEL_DIR)
            new_recommender.save(model_dir)

        fit_info = new_recommender.get_fit_info()
        return {
            "status": "success",
            "message": "Model fitted successfully and deployed for inference.",
            "fit_info": fit_info,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error fitting model via /init: %s", e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
