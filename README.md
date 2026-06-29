# Embeddings Visualizer

An interactive platform for **exploring token embeddings** from transformer language models — GPT-2, BERT, RoBERTa, and beyond. Load a model, project its 50k+ tokens to 3D space with UMAP, and inspect how the model represents meaning through an interactive 3D point cloud.

Built for researchers, students, and developers curious about how language models encode semantic meaning.

---

## Features

### 3D Token Explorer
- **Interactive 3D point cloud** — drag to orbit, scroll to zoom, click to inspect.
- **Smooth camera fly-to** — search for a token and the camera smoothly flies to its position in 3D space.
- **Live neighbor highlighting** — select a token and see its semantically similar neighbors highlighted with glowing halos.
- **Vocabulary access** — search and inspect any of the top 6,000 most-frequent tokens, even if the visible cloud is lighter.

### Search & Comparison
- **Fast token search** — type a word; jump to it with keyboard shortcut (`/`).
- **Token comparison** — compare any two tokens side-by-side: cosine similarity, euclidean distance, and shared semantics.
- **Neighbor analysis** — drill into the 20 nearest semantic neighbors for any token.

### Researcher Tools
- **Export JSON** — download a token's metadata, neighbors, and raw embedding vector for downstream analysis.
- **Copy token** — quickly copy token text to clipboard.
- **Vocabulary size display** — see the model's total vocabulary alongside the projected subset.
- **Keyboard shortcuts** — `/` to search, `Esc` to deselect.

### Per-Model Theming
- **Auto re-themes UI per model family** — GPT/OpenAI models are blue, BERT is Google blue, RoBERTa is violet, custom models are terracotta.
- **Accent colors** theme the entire UI, buttons, ambient glow, and in-scene neighbor halos.

### Performance & Smoothness
- **Capped DPR rendering** — bloom effect optimized for smooth 60 FPS on high-resolution displays.
- **UMAP caching** — first projection of a config takes ~20–40s; repeats are instant.
- **Model caching** — first model download takes ~1–5 min; repeats are instant.
- **Graceful custom models** — load any Hugging Face model; models without embeddings fail clearly.

---

## Architecture

```
embeddings-visualizer/
├── backend/           # FastAPI service
│   ├── app/
│   │   ├── main.py              # FastAPI app factory, lifespan, exception handlers
│   │   ├── config.py            # Settings (pydantic-settings): env, cache, allowed models
│   │   ├── schemas.py           # Pydantic models (API contracts)
│   │   ├── core/
│   │   │   ├── model_manager.py # LRU model cache, async loads, per-model locks
│   │   │   ├── visualizer.py    # LoadedModel: vectorized neighbors, UMAP, statistics
│   │   │   └── exceptions.py    # Typed domain errors → HTTP statuses
│   │   └── api/                 # Router endpoints
│   │       ├── models.py        # POST /api/models/load, GET /api/models/status
│   │       ├── visualization.py # POST /api/visualizations
│   │       ├── tokens.py        # GET /api/tokens/search, /api/tokens/{idx}/full
│   │       ├── analysis.py      # POST /api/analysis/compare, /api/analysis/statistics
│   │       └── health.py        # GET /health
│   ├── tests/                   # 33 offline pytest tests
│   ├── requirements-dev.txt
│   ├── Dockerfile
│   └── .env.example
│
├── frontend/          # React + Vite + Three.js
│   ├── src/
│   │   ├── App.tsx               # Root component, keyboard shortcuts
│   │   ├── index.css             # Tailwind + custom CSS (instrument panels, vignette)
│   │   ├── components/
│   │   │   ├── scene/
│   │   │   │   ├── EmbeddingCanvas.tsx  # WebGL canvas, post-processing bloom, camera rig
│   │   │   │   └── PointCloud.tsx       # Three.js point geometry, selection, halos
│   │   │   ├── ControlRail.tsx          # Left sidebar: model picker, UMAP settings, search
│   │   │   ├── DetailPanel.tsx          # Right sidebar: token metadata, neighbors, export
│   │   │   ├── SearchPanel.tsx          # Search input + results
│   │   │   ├── ComparePanel.tsx         # Token comparison inputs
│   │   │   ├── Legend.tsx               # Stats: vocab size, dims, metric
│   │   │   ├── Logo.tsx                 # Custom SVG constellation mark
│   │   │   ├── WelcomeOverlay.tsx       # Loading screen
│   │   │   └── ...
│   │   ├── lib/
│   │   │   ├── api.ts                   # Typed API client
│   │   │   ├── types.ts                 # TS mirrors of backend Pydantic schemas
│   │   │   ├── modelTheme.ts            # Per-model accent color map & applier
│   │   │   ├── layout.ts                # normalizeCoordinates, pointAt
│   │   │   └── tokenColors.ts           # Token-type color palette
│   │   └── store/
│   │       └── useStore.ts              # Zustand: app state, async workflows, focus/search
│   ├── package.json
│   ├── tailwind.config.js
│   └── vite.config.ts
│
├── RUNNING.md         # Local + PyCharm setup guide
├── README.md          # This file
└── .env.example       # Environment variables template
```

### Backend: Model-Keyed LRU Cache

Every API request names the model it operates on (`?model=distilgpt2`). Models are
loaded into a **memory-capped LRU cache** (default 2 models) shared across all users.
This is safe, efficient, and fits well with multi-user hosting. See
[`backend/README.md`](backend/README.md) for full API docs and design details.

**Key safeguards:**
- Per-model **async locks** prevent concurrent loads of the same model.
- **Real asyncio.wait_for timeouts** (not threading tricks) kill slow/hung downloads.
- **UMAP projection cache** (per config) avoids redundant computation.
- **Typed domain exceptions** map to clear HTTP errors (400, 404, 503, etc.).

### Frontend: React + Three.js Explorer

A **full-screen WebGL canvas** (Three.js + react-three-fiber) sits behind a layer of
floating glass panels. The 3D scene is always interactive (drag/zoom); panels overlay
with `pointer-events` controls. Camera smoothly flies to tokens on selection; neighbor
halos re-theme per model. State lives in Zustand; async API calls are orchestrated
there, not in React.

---

## Quick Start

### One-time setup

```bash
# From the project root:

# 1) Python deps
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt

# 2) Frontend deps
cd frontend
npm install
cd ..
```

### Start both servers

**Terminal 1 — Backend (FastAPI on :8000)**
```bash
.\.venv\Scripts\Activate.ps1      # PowerShell: activate venv
cd backend
python -m app.main
```

Or in **Git Bash / Linux**:
```bash
source .venv/Scripts/activate      # Git Bash: activate venv
cd backend
python -m app.main
```

**Terminal 2 — Frontend (Vite on :5173)**
```bash
cd frontend
npm run dev
```

Open **http://localhost:5173**, pick a model, click **Load & visualize**.

**See [`RUNNING.md`](RUNNING.md) for PyCharm setup, Docker, and troubleshooting.**

---

## Usage

### Exploring the point cloud
- **Drag** to orbit the camera.
- **Scroll** to zoom in/out.
- **Click a point** to inspect its token, see neighbors, and export.
- **Click empty space** to deselect (or press `Esc`).

### Finding tokens
- Press `/` (or click the search box) and type a token.
- Results list semantically similar tokens; click one to fly the camera to it.

### Comparing tokens
- Type two token names in the "Compare" section (bottom-left).
- See their cosine similarity, euclidean distance, and shared neighbors.
- Click a result chip to inspect it.

### Exporting for research
- Select a token (click it in the point cloud or via search).
- In the detail panel (right sidebar), click **Export JSON**.
- Downloads `{model}_{token}.json` with metadata, neighbors (up to 50), and raw embedding vector.
- Also **Copy token** to clipboard.

### Changing the visible density
- Drag the **Visible points** slider (left panel) to show fewer points for clarity.
- All 6,000 tokens remain searchable; the slider only affects rendering.

### Adjusting the projection
- **UMAP metric**: toggle between `cosine` (default) and `euclidean` distance.
- **Neighbors per token** (slider): re-computes the UMAP graph; projection cached.
- **Re-project** button: force a fresh projection (useful if you changed settings).

---

## API Endpoints

All requests are **model-keyed** (`?model=gpt2`, `?model=bert-base-uncased`, etc.).

### Model Management
- `POST /api/models/load?model={model_id}` — Load a model (async, polling `/status` for progress).
- `GET /api/models/status?model={model_id}` — Get load state, progress, or error.
- `GET /api/models/list` — List available presets and custom model slot.

### Visualization
- `POST /api/visualizations?model={model_id}` (JSON body: `{n_components, n_neighbors, min_dist, metric}`) — Compute or retrieve cached UMAP projection.

### Tokens
- `GET /api/tokens/search?model={model_id}&query={text}&max_results={n}` — Search for tokens by substring/prefix.
- `GET /api/tokens/{index}/full?model={model_id}&n_neighbors=20&metric=cosine&include_embedding=true` — Get token details + neighbors + optionally the raw embedding vector.

### Analysis
- `POST /api/analysis/compare?model={model_id}` (JSON: `{token1, token2}`) — Compare two tokens.
- `GET /api/analysis/statistics?model={model_id}` — Get model vocab size and other stats.

### Health
- `GET /health` — Service status.

**Full API docs** (when running locally): http://localhost:8000/docs (Swagger UI).

---

## Development

### Run tests (backend)
```bash
cd backend
python -m pytest -v          # verbose
python -m pytest --cov       # with coverage
```

All tests are **offline** (no network, no model downloads). We use synthetic data and
mocked Hugging Face calls.

### TypeScript compilation (frontend)
```bash
cd frontend
npx tsc -b                   # type-check
npm run build                # production build → dist/
```

### Code style
- **Backend:** Black-formatted, type-hinted with Pydantic, FastAPI best practices.
- **Frontend:** Prettier, ESLint, Tailwind CSS, TypeScript strict mode.

### Common dev tasks
| Task | Command |
| --- | --- |
| Backend with auto-reload | `python -m app.main` (already enabled via Uvicorn) |
| Frontend hot-reload | `npm run dev` (Vite HMR) |
| Type-check frontend only | `cd frontend && npx tsc --noEmit` |
| Rebuild frontend CSS | `npm run build` (if Tailwind config changed, restart dev server) |
| Lint Python | `cd backend && black . && isort .` |
| Format TypeScript | `cd frontend && npx prettier --write src/` |

---

## Design & Theming

### Per-Model Colors

The UI accent is **CSS-variable driven** and re-themes on model load:

| Model Family | Accent | Hex |
| --- | --- | --- |
| GPT / OpenAI | Blue | `#3b82f6` |
| BERT | Blue-gray | `#4284f4` |
| RoBERTa | Violet | `#7c61ff` |
| Custom / Other | Terracotta | `#d8623a` |

See [`frontend/src/lib/modelTheme.ts`](frontend/src/lib/modelTheme.ts) to add more families.

### Instrument-Style UI

Panels use crisp box-shadows, hairline borders, and subtle tick-mark labels. No frosted
glass or gradients — clean, minimal, distinctive.

---

## Deployment

### Docker (Backend)

```bash
cd backend
docker build -t embeddings-viz .
docker run -p 8000:8000 \
  -e CORS_ORIGINS=https://yourdomain.com \
  -e ALLOWED_MODELS=gpt2,distilgpt2,bert-base-uncased \
  embeddings-viz
```

### Static Frontend (Production Build)

```bash
cd frontend
npm run build                # outputs dist/
# Serve dist/ via any static host (Vercel, Netlify, S3 + CloudFront, etc.)
```

### Environment Variables

See [`.env.example`](.env.example) for backend config:
- `DEFAULT_TOP_N` — token cap (default 6000).
- `MAX_CACHED_MODELS` — LRU cache size (default 2).
- `MODEL_LOAD_TIMEOUT_SECONDS` — hard timeout for downloads (default 180s).
- `ALLOWED_MODELS` — optional allow-list (leave empty to allow any Hugging Face model).
- `CORS_ORIGINS` — frontend URL(s).

---

## Understanding the Code

### Why per-model caching?

When hosting for many users, a single global model would be a bottleneck (one user
waiting for a slow load blocks everyone). An LRU cache lets us hold 2–3 models at a
time, with older models swapped out. Each model has a per-slot async lock, so
concurrent `/load` requests don't download twice.

### Why UMAP on the backend?

UMAP is expensive (~1 min for 6000 tokens). We compute it server-side once, cache
the result (keyed by config), and serve the pre-computed coordinates to all clients.
The frontend only downloads the coords and renders; no client-side recomputation.

### Why Zustand for state?

Zustand keeps the async model-loading workflow and search/selection state in one
place, declarative and easy to debug. All API calls flow through actions, so the UI
is a pure function of `store.getState()`.

### Why Three.js + react-three-fiber?

Three.js is battle-tested for large point clouds (3D-6D scenes, smooth interaction).
react-three-fiber (r3f) lets us compose 3D scenes declaratively in React, bridging
the procedural Three.js world and React's component model.

---

## Contributing

Contributions welcome! Areas for exploration:

- **2D UMAP mode** — an alternative to 3D for performance or preference.
- **Custom color schemes** — expose theme config to the UI.
- **Batch export** — download multiple tokens' embeddings at once.
- **Token analogies** — "king is to man as queen is to ?" interface.
- **API authentication** — for multi-user deployments.
- **Embedding arithmetic** — visualize the result of `embedding_a - embedding_b + embedding_c`.

---

## License & Attribution

- **Manny McGrail** — Project direction, backend API, ML module.
- **Claude (Anthropic)** — Backend remaster (model-keyed LRU, async timeouts, typed errors), full frontend (React/Three.js, UMAP caching, keyboard shortcuts, theming, export/search/compare workflows).

---

## Related Work

- **Hugging Face Transformers** — model loading and tokenization.
- **UMAP** — dimensionality reduction.
- **react-three-fiber** — declarative Three.js.
- **Zustand** — lightweight state management.

---

## FAQ

**Q: Can I load my own model?**
A: Yes. Type any Hugging Face model ID (e.g., `sentence-transformers/all-MiniLM-L6-v2`) in the "custom" field. If the model exports a token embedding table, it works. Vision/audio/other architectures fail clearly.

**Q: How long does the first load take?**
A: Model download: 1–5 min (cached after). UMAP projection (6000 tokens): 20–40s (cached after). Repeats are instant.

**Q: Can I use this offline?**
A: The backend requires a Hugging Face download (unless you use `HF_HUB_OFFLINE=1` with pre-cached models). The frontend works offline once loaded.

**Q: Can I compare tokens that aren't in the top 6000?**
A: Not directly — the projection is built from the top 6000. To include more, raise `DEFAULT_TOP_N` in `.env` (will slow projection).

**Q: Why does the theme color not change immediately when I switch models?**
A: Real browsers re-resolve `var(--accent)` instantly. If you're in the Claude Code preview (headless), there's a browser caching quirk — the accent *is* being set correctly (you'll see it in the real browser).

---

## Questions?

Open an issue on GitHub or reach out to the maintainers. Enjoy exploring!
