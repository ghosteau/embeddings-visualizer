/**
 * Central application state (Zustand).
 *
 * Holds the model-loading lifecycle, the active visualization payload, the
 * current selection/neighbors, search, comparison, and transient toasts.
 * Async actions orchestrate the API calls and keep the UI declarative.
 */

import { create } from "zustand";
import { api, ApiError } from "../lib/api";
import { normalizeCoordinates } from "../lib/layout";
import { applyModelTheme, themeForModel } from "../lib/modelTheme";
import type {
  AnalysisStatistics,
  AvailableModels,
  ComparisonResult,
  DistanceMetric,
  SearchResult,
  TokenWithNeighbors,
  VisualizationConfig,
  VisualizationData,
} from "../lib/types";

export interface Toast {
  id: number;
  type: "info" | "success" | "error";
  message: string;
}

/** A request to fly the camera to a point; the nonce re-triggers identical targets. */
export interface FocusRequest {
  index: number;
  nonce: number;
}

interface AppState {
  // Model discovery & loading lifecycle.
  models: AvailableModels | null;
  selectedModelId: string;
  loadedModel: string | null;
  loadState: "idle" | "loading" | "visualizing" | "ready" | "error";
  loadProgress: string | null;

  // Visualization.
  config: VisualizationConfig;
  vizData: VisualizationData | null;
  // Normalised point positions for the active projection (shared by the point
  // cloud and the camera rig so coordinates never diverge).
  positions: Float32Array | null;
  // How many of the projected points to render (client-side declutter). The
  // full projection is always kept so selection/neighbor markers stay valid.
  displayCount: number;

  // Selection & exploration.
  selectedIndex: number | null;
  hoveredIndex: number | null;
  tokenDetail: TokenWithNeighbors | null;
  detailLoading: boolean;
  neighborIndices: number[];
  neighborMetric: DistanceMetric;
  focus: FocusRequest | null;
  // Hex of the current model's accent glow, used by the 3D scene (halos).
  accentGlowHex: string;
  // Short label describing the active theme, e.g. "OpenAI / GPT".
  themeLabel: string | null;

  // Search.
  searchQuery: string;
  searchResults: SearchResult[];

  // Comparison & statistics.
  comparison: ComparisonResult | null;
  comparisonError: string | null;
  statistics: AnalysisStatistics | null;

  toasts: Toast[];

  // Actions.
  init: () => Promise<void>;
  setSelectedModel: (id: string) => void;
  setConfig: (patch: Partial<VisualizationConfig>) => void;
  setDisplayCount: (n: number) => void;
  loadAndVisualize: () => Promise<void>;
  regenerate: () => Promise<void>;
  selectToken: (index: number) => Promise<void>;
  focusOn: (index: number) => Promise<void>;
  setHovered: (index: number | null) => void;
  clearSelection: () => void;
  setNeighborMetric: (m: DistanceMetric) => Promise<void>;
  runSearch: (query: string) => Promise<void>;
  compareTokens: (a: string, b: string) => Promise<void>;
  pushToast: (type: Toast["type"], message: string) => void;
  dismissToast: (id: number) => void;
}

const DEFAULT_CONFIG: VisualizationConfig = {
  n_components: 3,
  n_neighbors: 15,
  min_dist: 0.1,
  metric: "cosine",
};

let toastSeq = 0;
let focusSeq = 0;

export const useStore = create<AppState>((set, get) => ({
  models: null,
  selectedModelId: "distilgpt2",
  loadedModel: null,
  loadState: "idle",
  loadProgress: null,
  config: DEFAULT_CONFIG,
  vizData: null,
  positions: null,
  displayCount: 0,
  selectedIndex: null,
  hoveredIndex: null,
  tokenDetail: null,
  detailLoading: false,
  neighborIndices: [],
  neighborMetric: "cosine",
  focus: null,
  accentGlowHex: "#f0a184",
  themeLabel: null,
  searchQuery: "",
  searchResults: [],
  comparison: null,
  comparisonError: null,
  statistics: null,
  toasts: [],

  init: async () => {
    try {
      const models = await api.listModels();
      set({ models });
      if (models.presets.length && !models.presets.some((p) => p.id === get().selectedModelId)) {
        set({ selectedModelId: models.presets[0].id });
      }
    } catch (e) {
      get().pushToast("error", e instanceof Error ? e.message : "Failed to reach API");
    }
  },

  setSelectedModel: (id) => set({ selectedModelId: id }),

  setConfig: (patch) => set({ config: { ...get().config, ...patch } }),

  setDisplayCount: (n) => set({ displayCount: n }),

  loadAndVisualize: async () => {
    const model = get().selectedModelId.trim();
    if (!model) return;

    set({ loadState: "loading", loadProgress: "Requesting model…", comparison: null });

    // Poll the status endpoint for live progress while /load runs server-side.
    // The load happens in a worker thread, so the event loop keeps serving
    // /status requests and we can surface the backend's stage messages.
    const poll = setInterval(async () => {
      try {
        const status = await api.modelStatus(model);
        if (status.progress && get().loadState === "loading") {
          set({ loadProgress: status.progress });
        }
      } catch {
        /* transient; ignore */
      }
    }, 800);

    try {
      await api.loadModel(model);
      clearInterval(poll);
      // Re-theme the UI/scene accent based on the model family (a bit of fun).
      const family = get().models?.presets.find((p) => p.id === model)?.family;
      const theme = themeForModel(model, family);
      applyModelTheme(theme);
      set({
        loadedModel: model,
        loadState: "visualizing",
        loadProgress: "Projecting embeddings… (first projection can take ~30s)",
        accentGlowHex: theme.glowHex,
        themeLabel: theme.label,
      });
      await get().regenerate();
      // Statistics are non-critical; fetch in the background.
      api
        .statistics(model)
        .then((statistics) => set({ statistics }))
        .catch(() => undefined);
      set({ loadState: "ready" });
      get().pushToast("success", `${model} loaded — ${get().vizData?.tokens.length ?? 0} tokens projected.`);
    } catch (e) {
      clearInterval(poll);
      const msg = e instanceof ApiError ? e.message : "Failed to load model";
      set({ loadState: "error", loadProgress: msg });
      get().pushToast("error", msg);
    }
  },

  regenerate: async () => {
    const model = get().loadedModel;
    if (!model) return;
    const wasReady = get().loadState === "ready";
    if (wasReady) set({ loadState: "visualizing", loadProgress: "Re-projecting…" });
    try {
      const vizData = await api.createVisualization(model, get().config);
      set({
        vizData,
        positions: normalizeCoordinates(vizData.coordinates),
        // Render a lighter subset by default for smoothness; every token stays
        // searchable/inspectable, and the "Visible points" slider goes to max.
        displayCount: Math.min(3000, vizData.tokens.length),
        selectedIndex: null,
        tokenDetail: null,
        neighborIndices: [],
      });
      if (wasReady) set({ loadState: "ready" });
    } catch (e) {
      const msg = e instanceof ApiError ? e.message : "Projection failed";
      get().pushToast("error", msg);
      if (wasReady) set({ loadState: "ready" });
      throw e;
    }
  },

  selectToken: async (index) => {
    const model = get().loadedModel;
    if (!model) return;
    set({ selectedIndex: index, detailLoading: true });
    try {
      const detail = await api.tokenFull(model, index, 20, get().neighborMetric);
      set({
        tokenDetail: detail,
        neighborIndices: detail.neighbors.map((n) => n.index),
        detailLoading: false,
      });
    } catch (e) {
      set({ detailLoading: false });
      get().pushToast("error", e instanceof Error ? e.message : "Failed to load token");
    }
  },

  // Select a token AND fly the camera to it — used when inspecting a token that
  // may be off-screen (e.g. chosen from search or comparison results).
  focusOn: async (index) => {
    set({ focus: { index, nonce: ++focusSeq } });
    await get().selectToken(index);
  },

  setHovered: (index) => set({ hoveredIndex: index }),

  clearSelection: () => set({ selectedIndex: null, tokenDetail: null, neighborIndices: [] }),

  setNeighborMetric: async (m) => {
    set({ neighborMetric: m });
    const idx = get().selectedIndex;
    if (idx != null) await get().selectToken(idx);
  },

  runSearch: async (query) => {
    set({ searchQuery: query });
    const model = get().loadedModel;
    if (!model || query.trim().length === 0) {
      set({ searchResults: [] });
      return;
    }
    try {
      const searchResults = await api.search(model, query.trim(), 30);
      set({ searchResults });
    } catch {
      set({ searchResults: [] });
    }
  },

  compareTokens: async (a, b) => {
    const model = get().loadedModel;
    if (!model) return;
    set({ comparison: null, comparisonError: null });
    try {
      const comparison = await api.compareByName(model, a, b);
      set({ comparison });
    } catch (e) {
      set({ comparisonError: e instanceof ApiError ? e.message : "Comparison failed" });
    }
  },

  pushToast: (type, message) => {
    const id = ++toastSeq;
    set({ toasts: [...get().toasts, { id, type, message }] });
    setTimeout(() => get().dismissToast(id), 5000);
  },

  dismissToast: (id) => set({ toasts: get().toasts.filter((t) => t.id !== id) }),
}));
