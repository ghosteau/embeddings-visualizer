# =====================================================================================
# PYDANTIC MODELS FOR API
# =====================================================================================
from typing import Optional, List, Any, Dict

from pydantic import BaseModel


class TokenDetails(BaseModel):
    token: str
    index: int
    length: int
    type: str
    frequency_rank: int
    embedding_norm: float
    has_special_chars: bool
    is_uppercase: bool
    is_digit: bool
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None


class TokenNeighbor(BaseModel):
    token: str
    distance: float
    similarity: Optional[float] = None  # None for euclidean distance
    index: int


class TokenWithNeighbors(BaseModel):
    details: TokenDetails
    neighbors: List[TokenNeighbor]
    embedding_vector: Optional[List[float]] = None


class SearchResult(BaseModel):
    token: str
    index: int
    match_type: str
    similarity: Optional[float] = None


class ModelInfo(BaseModel):
    name: str
    vocabulary_size: int
    embedding_dimension: int
    is_loaded: bool
    token_count: int
    loading_status: Optional[Dict] = None


class VisualizationConfig(BaseModel):
    method: str = "umap"
    n_components: int = 3
    n_neighbors: int = 15
    min_dist: float = 0.1
    metric: str = "cosine"
    top_n: int = 3000


class ComparisonRequest(BaseModel):
    token1: str
    token2: str


class ComparisonByIdRequest(BaseModel):
    token1_index: int
    token2_index: int


class ComparisonResult(BaseModel):
    token1: str
    token2: str
    token1_index: int
    token2_index: int
    cosine_similarity: float
    euclidean_distance: float


class BatchAnalysisRequest(BaseModel):
    tokens: List[str]


class VisualizationData(BaseModel):
    coordinates: List[List[float]]
    tokens: List[str]
    metadata: Dict[str, List[Any]]
    config: VisualizationConfig
    statistics: Dict[str, Any]


class LoadModelRequest(BaseModel):
    model_name: str


class ExportRequest(BaseModel):
    token_index: int
    include_neighbors: bool = True
    n_neighbors: int = 15


class LoadingStatus(BaseModel):
    is_loading: bool
    model_name: Optional[str] = None
    progress: Optional[str] = None
    error: Optional[str] = None