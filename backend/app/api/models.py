"""Endpoints for discovering, loading, inspecting, and unloading models."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_manager
from app.config import Settings, get_settings
from app.core.model_manager import ModelManager
from app.schemas import (
    AvailableModels,
    LoadModelRequest,
    LoadModelResponse,
    ModelInfo,
    ModelStatus,
    PresetModel,
)

router = APIRouter(prefix="/api/models", tags=["models"])

# Curated, lightweight models that are quick to download and illustrative for
# exploring embedding geometry. Custom Hugging Face ids are also accepted unless
# an allow-list is configured.
_PRESETS: list[PresetModel] = [
    PresetModel(id="distilgpt2", name="DistilGPT-2", family="GPT-2", params="82M"),
    PresetModel(id="gpt2", name="GPT-2", family="GPT-2", params="124M"),
    PresetModel(id="gpt2-medium", name="GPT-2 Medium", family="GPT-2", params="355M"),
    PresetModel(
        id="distilbert-base-uncased", name="DistilBERT", family="BERT", params="66M"
    ),
    PresetModel(
        id="bert-base-uncased", name="BERT Base", family="BERT", params="110M"
    ),
    PresetModel(id="roberta-base", name="RoBERTa Base", family="RoBERTa", params="125M"),
]


@router.get("", response_model=AvailableModels, summary="List available models")
async def list_models(settings: Settings = Depends(get_settings)) -> AvailableModels:
    """Return curated presets and the server's loading capabilities."""
    presets = _PRESETS
    if settings.allowed_models:
        # When an allow-list is set, only advertise presets that are permitted.
        allowed = set(settings.allowed_models)
        presets = [p for p in _PRESETS if p.id in allowed]
    return AvailableModels(
        presets=presets,
        supports_custom_models=not settings.allowed_models,
        load_timeout_seconds=settings.model_load_timeout_seconds,
    )


@router.post("/load", response_model=LoadModelResponse, summary="Load a model")
async def load_model(
    request: LoadModelRequest,
    manager: ModelManager = Depends(get_manager),
) -> LoadModelResponse:
    """Load a model into the cache (idempotent; shared across users).

    Loading is bounded by the configured timeout. Because models are cached by
    name, repeated calls for an already-loaded model return immediately. Errors
    (timeout, disallowed model, download failure) surface as typed HTTP errors.
    """
    state = await manager.ensure_loaded(request.model)
    return LoadModelResponse(
        model=request.model,
        state=state,
        message=f"Model '{request.model}' is {state.value}.",
    )


@router.get("/status", response_model=ModelStatus, summary="Model loading status")
async def model_status(
    model: str = Query(..., min_length=1, description="Model id to inspect."),
    manager: ModelManager = Depends(get_manager),
) -> ModelStatus:
    """Return the current cache/loading state for a single model."""
    return ModelStatus(**manager.status(model))


@router.get("/info", response_model=ModelInfo, summary="Loaded model details")
async def model_info(
    model: str = Query(..., min_length=1, description="Loaded model id."),
    manager: ModelManager = Depends(get_manager),
) -> ModelInfo:
    """Return structural facts about a loaded model's embedding space."""
    loaded = manager.get(model)  # raises 409 if not loaded
    return ModelInfo(
        model=loaded.name,
        state=manager.status(model)["state"],
        vocabulary_size=loaded.vocabulary_size,
        embedding_dimension=loaded.embedding_dimension,
        tokens_loaded=loaded.token_count,
    )


@router.delete("", summary="Unload a model")
async def unload_model(
    model: str = Query(..., min_length=1, description="Model id to evict."),
    manager: ModelManager = Depends(get_manager),
) -> dict:
    """Evict a model from the cache to free memory."""
    removed = await manager.unload(model)
    return {
        "success": True,
        "removed": removed,
        "message": (
            f"Model '{model}' unloaded." if removed else f"Model '{model}' was not cached."
        ),
    }
