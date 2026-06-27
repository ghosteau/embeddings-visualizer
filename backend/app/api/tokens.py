"""Endpoints for exploring individual tokens and their neighborhoods."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import get_loaded_model
from app.core.exceptions import TokenNotFoundError
from app.core.visualizer import LoadedModel
from app.schemas import (
    DistanceMetric,
    SearchResult,
    TokenDetails,
    TokenNeighbor,
    TokenWithNeighbors,
)

router = APIRouter(prefix="/api/tokens", tags=["tokens"])


def _safe_details(model: LoadedModel, index: int, **kwargs) -> dict:
    """Fetch token details, translating an out-of-range index to a 404."""
    try:
        return model.get_token_details(index, **kwargs)
    except IndexError as exc:
        raise TokenNotFoundError(str(exc)) from exc


@router.get("/search", response_model=list[SearchResult], summary="Search tokens")
async def search_tokens(
    query: str = Query(..., min_length=1, description="Substring to search for."),
    max_results: int = Query(50, ge=1, le=1000),
    model: LoadedModel = Depends(get_loaded_model),
) -> list[SearchResult]:
    """Find tokens whose text contains ``query`` (exact matches ranked first)."""
    return [SearchResult(**r) for r in model.search_tokens(query, max_results)]


@router.get("/{index}", response_model=TokenDetails, summary="Token details")
async def token_details(
    index: int,
    include_distance: bool = Query(
        False, description="Include the token's distance from the origin."
    ),
    metric: DistanceMetric = DistanceMetric.euclidean,
    model: LoadedModel = Depends(get_loaded_model),
) -> TokenDetails:
    """Return metadata (and optional origin distance) for a single token."""
    return TokenDetails(
        **_safe_details(model, index, include_distance=include_distance, metric=metric)
    )


@router.get(
    "/{index}/neighbors",
    response_model=list[TokenNeighbor],
    summary="Nearest neighbors",
)
async def token_neighbors(
    index: int,
    n_neighbors: int = Query(10, ge=1, le=200),
    metric: DistanceMetric = DistanceMetric.euclidean,
    model: LoadedModel = Depends(get_loaded_model),
) -> list[TokenNeighbor]:
    """Return the nearest tokens to ``index`` under the chosen metric."""
    try:
        neighbors = model.find_neighbors(index, n_neighbors, metric)
    except IndexError as exc:
        raise TokenNotFoundError(str(exc)) from exc
    return [TokenNeighbor(**n) for n in neighbors]


@router.get(
    "/{index}/full",
    response_model=TokenWithNeighbors,
    summary="Token details with neighbors",
)
async def token_full(
    index: int,
    n_neighbors: int = Query(15, ge=1, le=200),
    metric: DistanceMetric = DistanceMetric.euclidean,
    include_embedding: bool = Query(
        False, description="Include the raw embedding vector (large)."
    ),
    model: LoadedModel = Depends(get_loaded_model),
) -> TokenWithNeighbors:
    """Return a token's details and neighbors together (one round-trip).

    Details and neighbors use the *same* metric so the numbers shown side by
    side in the UI are directly comparable.
    """
    details = _safe_details(model, index, include_distance=True, metric=metric)
    neighbors = model.find_neighbors(index, n_neighbors, metric)
    embedding = model.get_embedding_vector(index) if include_embedding else None
    return TokenWithNeighbors(
        details=TokenDetails(**details),
        neighbors=[TokenNeighbor(**n) for n in neighbors],
        embedding_vector=embedding,
    )
