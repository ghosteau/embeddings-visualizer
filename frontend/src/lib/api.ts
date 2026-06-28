/**
 * Typed client for the Embeddings Visualizer backend.
 *
 * Every query endpoint is model-keyed (the backend has no notion of a single
 * "current model"), so most calls take the model id and forward it as a `model`
 * query parameter. Errors are normalised into {@link ApiError} so the UI can
 * surface the backend's `detail` message verbatim.
 */

import type {
  AnalysisStatistics,
  AvailableModels,
  BatchAnalysisResult,
  ComparisonResult,
  DistanceMetric,
  ModelInfo,
  ModelStatus,
  SearchResult,
  TokenDetails,
  TokenWithNeighbors,
  VisualizationConfig,
  VisualizationData,
} from "./types";

const BASE_URL =
  (import.meta.env.VITE_API_URL as string | undefined)?.replace(/\/$/, "") ||
  "http://localhost:8000";

/** Error carrying the HTTP status and the backend's human-readable detail. */
export class ApiError extends Error {
  status: number;
  errorType?: string;

  constructor(status: number, message: string, errorType?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.errorType = errorType;
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE_URL}${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    // Network-level failure (server down, CORS, offline).
    throw new ApiError(0, `Cannot reach the API at ${BASE_URL}. Is the backend running?`);
  }

  if (!res.ok) {
    let detail = res.statusText;
    let errorType: string | undefined;
    try {
      const body = await res.json();
      detail = body.detail ?? detail;
      errorType = body.error;
    } catch {
      /* response had no JSON body */
    }
    throw new ApiError(res.status, detail, errorType);
  }

  return res.status === 204 ? (undefined as T) : ((await res.json()) as T);
}

const q = (params: Record<string, string | number | boolean>) =>
  "?" +
  Object.entries(params)
    .map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(String(v))}`)
    .join("&");

export const api = {
  // ----------------------------------------------------------------- models --
  listModels: () => request<AvailableModels>("/api/models"),

  loadModel: (model: string) =>
    request<{ model: string; state: string; message: string }>("/api/models/load", {
      method: "POST",
      body: JSON.stringify({ model }),
    }),

  modelStatus: (model: string) =>
    request<ModelStatus>(`/api/models/status${q({ model })}`),

  modelInfo: (model: string) => request<ModelInfo>(`/api/models/info${q({ model })}`),

  unloadModel: (model: string) =>
    request<{ success: boolean }>(`/api/models${q({ model })}`, { method: "DELETE" }),

  // ---------------------------------------------------------- visualization --
  createVisualization: (model: string, config: VisualizationConfig) =>
    request<VisualizationData>(`/api/visualization${q({ model })}`, {
      method: "POST",
      body: JSON.stringify(config),
    }),

  // ----------------------------------------------------------------- tokens --
  tokenFull: (
    model: string,
    index: number,
    nNeighbors = 15,
    metric: DistanceMetric = "cosine",
    includeEmbedding = false,
  ) =>
    request<TokenWithNeighbors>(
      `/api/tokens/${index}/full${q({
        model,
        n_neighbors: nNeighbors,
        metric,
        include_embedding: includeEmbedding,
      })}`,
    ),

  tokenDetails: (model: string, index: number) =>
    request<TokenDetails>(`/api/tokens/${index}${q({ model })}`),

  search: (model: string, query: string, maxResults = 50) =>
    request<SearchResult[]>(`/api/tokens/search${q({ model, query, max_results: maxResults })}`),

  // --------------------------------------------------------------- analysis --
  compareByName: (model: string, token1: string, token2: string) =>
    request<ComparisonResult>(`/api/analysis/compare${q({ model })}`, {
      method: "POST",
      body: JSON.stringify({ token1, token2 }),
    }),

  batch: (model: string, tokens: string[]) =>
    request<BatchAnalysisResult>(`/api/analysis/batch${q({ model })}`, {
      method: "POST",
      body: JSON.stringify({ tokens }),
    }),

  statistics: (model: string) =>
    request<AnalysisStatistics>(`/api/analysis/statistics${q({ model })}`),
};

export { BASE_URL };
