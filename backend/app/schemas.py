"""Pydantic models describing the API's request and response contracts.

These models are the single source of truth for the shape of data crossing the
HTTP boundary. They drive request validation, response serialization, and the
auto-generated OpenAPI schema consumed by the frontend's typed client.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------
class DistanceMetric(str, Enum):
    """Supported distance/similarity metrics for token-space queries."""

    cosine = "cosine"
    euclidean = "euclidean"


class TokenType(str, Enum):
    """Coarse lexical classification assigned to each token."""

    word = "word"
    number = "number"
    special = "special"
    mixed = "mixed"
    unknown = "unknown"


class LoadState(str, Enum):
    """Lifecycle state of a model within the manager's cache."""

    not_loaded = "not_loaded"
    loading = "loading"
    loaded = "loaded"
    error = "error"


# ---------------------------------------------------------------------------
# Model discovery & lifecycle
# ---------------------------------------------------------------------------
class PresetModel(BaseModel):
    """A curated model offered to users in the UI."""

    id: str = Field(..., description="Hugging Face model identifier.")
    name: str = Field(..., description="Human-friendly display name.")
    family: str = Field(..., description="Architecture family, e.g. 'GPT-2', 'BERT'.")
    params: str = Field(..., description="Approximate parameter count, e.g. '124M'.")


class AvailableModels(BaseModel):
    """Response listing curated presets and current capabilities."""

    presets: list[PresetModel]
    supports_custom_models: bool = Field(
        ..., description="Whether arbitrary Hugging Face ids may be loaded."
    )
    load_timeout_seconds: int


class LoadModelRequest(BaseModel):
    """Request body for triggering a model load."""

    model: str = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Hugging Face model id to load, e.g. 'gpt2'.",
        examples=["gpt2"],
    )

    @field_validator("model")
    @classmethod
    def validate_hugging_face_id(cls, value: str) -> str:
        """Accept Hub-style repository ids, never URLs or filesystem paths."""
        cleaned = value.strip()
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*(?:/[A-Za-z0-9][A-Za-z0-9._-]*)?",
            cleaned,
        ):
            raise ValueError("Use a Hugging Face model id such as 'gpt2' or 'owner/model'.")
        return cleaned


class LoadModelResponse(BaseModel):
    """Acknowledgement that a load was started (or was already complete)."""

    model: str
    state: LoadState
    message: str


class ModelStatus(BaseModel):
    """Current cache/loading state for a single model."""

    model: str
    state: LoadState
    progress: Optional[str] = Field(
        None, description="Human-readable progress detail while loading."
    )
    error: Optional[str] = Field(None, description="Error message if state is 'error'.")


class ModelInfo(BaseModel):
    """Structural facts about a loaded model's embedding space."""

    model: str
    state: LoadState
    vocabulary_size: int = Field(..., description="Total tokens in the model vocabulary.")
    embedding_dimension: int = Field(..., description="Raw embedding vector length.")
    tokens_loaded: int = Field(..., description="Number of tokens prepared for analysis.")


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------
class VisualizationConfig(BaseModel):
    """UMAP projection parameters and token-selection size.

    The defaults are sensible for a first look; the frontend exposes them so
    users can explore how the projection changes with different settings.
    """

    n_components: int = Field(
        3, ge=2, le=3, description="Output dimensionality (2D or 3D)."
    )
    n_neighbors: int = Field(
        15, ge=2, le=200, description="UMAP local-neighborhood size."
    )
    min_dist: float = Field(
        0.1, ge=0.0, le=0.99, description="UMAP minimum point separation."
    )
    metric: DistanceMetric = Field(
        DistanceMetric.cosine, description="Distance metric UMAP optimises against."
    )


class VisualizationStatistics(BaseModel):
    """Summary statistics returned alongside a projection."""

    total_tokens: int
    original_dimension: int
    reduced_dimension: int
    type_distribution: dict[str, int]


class VisualizationData(BaseModel):
    """A complete projection ready for rendering by the frontend.

    ``coordinates[i]`` are the reduced coordinates for ``tokens[i]``; the
    parallel ``metadata`` arrays describe each token. Keeping these as parallel
    arrays (rather than a list of objects) keeps the payload compact for the
    thousands of points sent to the 3D scene.
    """

    model: str
    coordinates: list[list[float]]
    tokens: list[str]
    metadata: dict[str, list[Any]]
    config: VisualizationConfig
    statistics: VisualizationStatistics


# ---------------------------------------------------------------------------
# Token exploration
# ---------------------------------------------------------------------------
class TokenDetails(BaseModel):
    """Per-token metadata, optionally including geometric facts."""

    token: str
    index: int
    length: int
    type: TokenType
    frequency_rank: int = Field(
        ..., description="Rank in the vocabulary by frequency (0 = most frequent)."
    )
    embedding_norm: float = Field(..., description="L2 norm of the raw embedding vector.")
    has_special_chars: bool
    is_uppercase: bool
    is_digit: bool
    # Projected coordinates, present only once a visualization has been built.
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    # Distance from the origin, present only when distances were requested.
    distance_to_origin: Optional[float] = None
    distance_metric: Optional[DistanceMetric] = None


class TokenNeighbor(BaseModel):
    """A single nearest-neighbor result for a token query."""

    token: str
    index: int
    distance: float = Field(..., description="Distance under the requested metric.")
    similarity: float = Field(..., description="Cosine similarity, always provided.")


class TokenWithNeighbors(BaseModel):
    """A token's details bundled with its nearest neighbors."""

    details: TokenDetails
    neighbors: list[TokenNeighbor]
    embedding_vector: Optional[list[float]] = Field(
        None, description="The raw embedding vector, included on request."
    )


class SearchResult(BaseModel):
    """A token matching a text search query."""

    token: str
    index: int
    match_type: str = Field(..., description="'exact' or 'contains'.")


# ---------------------------------------------------------------------------
# Comparison & batch analysis
# ---------------------------------------------------------------------------
class CompareByNameRequest(BaseModel):
    """Compare two tokens identified by their text."""

    token1: str
    token2: str


class CompareByIdRequest(BaseModel):
    """Compare two tokens identified by their analysis index."""

    token1_index: int = Field(..., ge=0)
    token2_index: int = Field(..., ge=0)


class ComparisonResult(BaseModel):
    """Pairwise similarity/distance between two tokens (raw embeddings)."""

    token1: str
    token2: str
    token1_index: int
    token2_index: int
    cosine_similarity: float
    euclidean_distance: float


class BatchAnalysisRequest(BaseModel):
    """Request a pairwise similarity matrix for a set of tokens."""

    tokens: list[str] = Field(..., min_length=2, description="Tokens to compare.")


class BatchAnalysisResult(BaseModel):
    """A cosine-similarity matrix over the resolved tokens."""

    tokens: list[str] = Field(..., description="Tokens that were found and compared.")
    similarity_matrix: list[list[float]]
    not_found: list[str] = Field(..., description="Requested tokens that were absent.")


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
class AnalysisStatistics(BaseModel):
    """Aggregate statistics describing a loaded model's token space."""

    model_info: dict[str, Any]
    token_distribution: dict[str, Any]
    embedding_statistics: dict[str, Any]
    special_characteristics: dict[str, int]


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------
class HealthResponse(BaseModel):
    """Liveness/readiness signal plus a snapshot of the model cache."""

    status: str
    version: str
    timestamp: str
    cached_models: list[ModelStatus]
