"""Endpoints for comparing tokens and summarising a model's token space."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from app.api.deps import get_loaded_model
from app.core.exceptions import InvalidRequestError, TokenNotFoundError
from app.core.visualizer import LoadedModel
from app.schemas import (
    AnalysisStatistics,
    BatchAnalysisRequest,
    BatchAnalysisResult,
    CompareByIdRequest,
    CompareByNameRequest,
    ComparisonResult,
)

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


@router.post("/compare", response_model=ComparisonResult, summary="Compare by name")
async def compare_by_name(
    request: CompareByNameRequest,
    model: LoadedModel = Depends(get_loaded_model),
) -> ComparisonResult:
    """Compare two tokens identified by their text."""
    try:
        result = model.compare_by_name(request.token1, request.token2)
    except KeyError as exc:
        raise TokenNotFoundError(f"Token '{exc.args[0]}' not found.") from exc
    return ComparisonResult(**result)


@router.post(
    "/compare/by-id", response_model=ComparisonResult, summary="Compare by index"
)
async def compare_by_id(
    request: CompareByIdRequest,
    model: LoadedModel = Depends(get_loaded_model),
) -> ComparisonResult:
    """Compare two tokens identified by their analysis index."""
    try:
        result = model.compare_by_index(request.token1_index, request.token2_index)
    except IndexError as exc:
        raise TokenNotFoundError(str(exc)) from exc
    return ComparisonResult(**result)


@router.post(
    "/batch", response_model=BatchAnalysisResult, summary="Pairwise similarity matrix"
)
async def batch_analysis(
    request: BatchAnalysisRequest,
    model: LoadedModel = Depends(get_loaded_model),
) -> BatchAnalysisResult:
    """Return a cosine-similarity matrix over a set of tokens."""
    try:
        result = model.batch_similarity(request.tokens)
    except ValueError as exc:
        raise InvalidRequestError(str(exc)) from exc
    return BatchAnalysisResult(**result)


@router.get(
    "/statistics", response_model=AnalysisStatistics, summary="Token-space statistics"
)
async def statistics(
    model: LoadedModel = Depends(get_loaded_model),
) -> AnalysisStatistics:
    """Return aggregate statistics describing the loaded token space."""
    return AnalysisStatistics(**model.statistics())
