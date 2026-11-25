"""Versioned API router for Hybrid Recommendation Engine v1.

FastAPI instance is created in main app and router is included via:
    from .api.v1.router import router
    app.include_router(router, prefix="/api/v1")
"""
from .router import router

__all__ = ["router"]