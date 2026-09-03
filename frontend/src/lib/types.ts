/**
 * TypeScript mirrors of the backend's Pydantic schemas (see backend/app/schemas.py).
 * Keeping these in one place gives the whole UI a single, typed contract for the API.
 */

export type DistanceMetric = "cosine" | "euclidean";
export type TokenType = "word" | "number" | "special" | "mixed" | "unknown";
export type LoadState = "not_loaded" | "loading" | "loaded" | "error";

export interface PresetModel {
  id: string;
  name: string;
  family: string;
  params: string;
}

export interface AvailableModels {
  presets: PresetModel[];
  supports_custom_models: boolean;
  load_timeout_seconds: number;
}

export interface ModelStatus {
  model: string;
  state: LoadState;
  progress: string | null;
  error: string | null;
}

export interface ModelInfo {
  model: string;
  state: LoadState;
  vocabulary_size: number;
  embedding_dimension: number;
  tokens_loaded: number;
}

export interface VisualizationConfig {
  n_components: 2 | 3;
  n_neighbors: number;
  min_dist: number;
  metric: DistanceMetric;
}

export interface VisualizationStatistics {
  total_tokens: number;
  original_dimension: number;
  reduced_dimension: number;
  type_distribution: Record<string, number>;
}

/** Parallel-array payload: coordinates[i] describes tokens[i]. */
export interface VisualizationData {
  model: string;
  coordinates: number[][];
  tokens: string[];
  metadata: {
    types: TokenType[];
    lengths: number[];
    embedding_norm: number[];
    frequency_rank: number[];
    has_special_chars: boolean[];
    is_uppercase: boolean[];
    is_digit: boolean[];
  };
  config: VisualizationConfig;
  statistics: VisualizationStatistics;
}

export interface TokenDetails {
  token: string;
  index: number;
  length: number;
  type: TokenType;
  frequency_rank: number;
  embedding_norm: number;
  has_special_chars: boolean;
  is_uppercase: boolean;
  is_digit: boolean;
  x?: number | null;
  y?: number | null;
  z?: number | null;
  distance_to_origin?: number | null;
  distance_metric?: DistanceMetric | null;
}

export interface TokenNeighbor {
  token: string;
  index: number;
  distance: number;
  similarity: number;
}

export interface TokenWithNeighbors {
  details: TokenDetails;
  neighbors: TokenNeighbor[];
  embedding_vector?: number[] | null;
}

export interface SearchResult {
  token: string;
  index: number;
  match_type: "exact" | "contains";
}

export interface ComparisonResult {
  token1: string;
  token2: string;
  token1_index: number;
  token2_index: number;
  cosine_similarity: number;
  euclidean_distance: number;
}

export interface BatchAnalysisResult {
  tokens: string[];
  similarity_matrix: number[][];
  not_found: string[];
}

export interface AnalysisStatistics {
  model_info: Record<string, unknown>;
  token_distribution: {
    by_type: Record<string, number>;
    by_length: { min: number; max: number; mean: number; median: number };
  };
  embedding_statistics: {
    norm: { min: number; max: number; mean: number; std: number };
  };
  special_characteristics: Record<string, number>;
}
