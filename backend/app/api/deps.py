"""Reusable FastAPI dependencies.

These wire the request-handling layer to the singleton :class:`ModelManager`
(created once at startup and stored on ``app.state``) and provide a convenient
dependency for resolving the loaded model a request operates on.
"""

from __future__ import annotations

from fastapi import Depends, Query, Request

from app.config import Settings, get_settings
from app.core.model_manager import ModelManager
from app.core.visualizer import LoadedModel


def get_manager(request: Request) -> ModelManager:
    """Return the application-wide :class:`ModelManager`.

    The manager is constructed in the lifespan handler (see ``app.main``) and
    attached to ``app.state`` so it is shared across all requests and workers
    within a process.
    """
    return request.app.state.manager


def get_loaded_model(
    model: str = Query(
        ...,
        min_length=1,
        description="Hugging Face id of the (already loaded) model to query.",
        examples=["gpt2"],
    ),
    manager: ModelManager = Depends(get_manager),
) -> LoadedModel:
    """Resolve the loaded model named by the ``model`` query parameter.

    Raises :class:`~app.core.exceptions.ModelNotLoadedError` (mapped to HTTP 409)
    if the model has not been loaded yet, prompting the client to load it first.
    """
    return manager.get(model)


def settings_dep() -> Settings:
    """Expose application settings to routes that need configuration values."""
    return get_settings()
