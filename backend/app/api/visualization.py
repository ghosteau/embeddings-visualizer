"""Endpoint for building UMAP projections of a model's token space."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_manager
from app.core.model_manager import ModelManager
from app.schemas import (
    VisualizationConfig,
    VisualizationData,
    VisualizationStatistics,
)

router = APIRouter(prefix="/api/visualization", tags=["visualization"])


@router.post("", response_model=VisualizationData, summary="Create a projection")
async def create_visualization(
    config: VisualizationConfig,
    model: str = Query(..., min_length=1, description="Loaded model to project."),
    manager: ModelManager = Depends(get_manager),
) -> VisualizationData:
    """Reduce a model's token embeddings to 2D/3D with UMAP.

    The whole prepared token set is projected so that token indices line up
    across the projection, neighbor search, and detail queries. Identical
    configs are served from a per-model cache, so re-requesting a projection
    (e.g. toggling back to a previous setting) is instant. How many of the
    resulting points are *displayed* is a client-side concern.
    """
    loaded = manager.get(model)  # raises 409 if not loaded

    projection = loaded.reduce_dimensions(config)

    statistics = VisualizationStatistics(
        total_tokens=loaded.token_count,
        original_dimension=loaded.embedding_dimension,
        reduced_dimension=config.n_components,
        type_distribution=loaded.type_distribution(),
    )

    return VisualizationData(
        model=loaded.name,
        coordinates=projection.tolist(),
        tokens=loaded.tokens,
        metadata={
            "types": [t.value for t in loaded.token_types],
            "lengths": loaded.lengths,
            "embedding_norm": [float(n) for n in loaded.embedding_norms],
            "frequency_rank": loaded.selected_indices,
            "has_special_chars": loaded.has_special,
            "is_uppercase": loaded.is_upper,
            "is_digit": loaded.is_digit,
        },
        config=config,
        statistics=statistics,
    )
