"""Health, readiness, and service-metadata endpoints."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from app.api.deps import get_manager
from app.config import Settings, get_settings
from app.core.model_manager import ModelManager
from app.schemas import HealthResponse, ModelStatus

router = APIRouter(tags=["health"])


@router.get("/api", summary="Service metadata")
async def root(settings: Settings = Depends(get_settings)) -> dict:
    """Return basic identifying information about the running service."""
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "docs": None if settings.is_production else "/docs",
        "health": "/health",
    }


@router.get("/health", response_model=HealthResponse, summary="Health check")
async def health(
    settings: Settings = Depends(get_settings),
    manager: ModelManager = Depends(get_manager),
) -> HealthResponse:
    """Report liveness plus a snapshot of every model in the cache."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        timestamp=datetime.now(timezone.utc).isoformat(),
        cached_models=[ModelStatus(**s) for s in manager.all_statuses()],
    )
