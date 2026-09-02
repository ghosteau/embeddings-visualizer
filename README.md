# Embeddings Visualizer

Embeddings Visualizer is an interactive research tool for inspecting the token
embedding spaces of transformer language models. It loads a model from Hugging
Face, reduces a representative vocabulary to two or three dimensions with UMAP,
and renders the result as a searchable Three.js point cloud.

The project is designed for researchers, students, and engineers who want to
understand what a language model places near a token, compare token vectors, or
export embedding data without first writing an analysis notebook.

The interface shares the visual language of
[mannymcgrail.com](https://mannymcgrail.com): dark navy surfaces, editorial serif
type, compact technical labels, cyan accents, and restrained motion. Model
families apply a secondary accent, such as blue for GPT models.

## What the application does

- Loads curated GPT-2, BERT, and RoBERTa models or a compatible Hugging Face
  repository ID.
- Extracts and caches a model's input embedding table.
- Projects up to a configured number of tokens with UMAP using cosine or
  Euclidean distance.
- Renders a responsive 3D point cloud with orbit, zoom, token selection, and
  animated camera focus.
- Searches the full analyzed subset, including points currently hidden by the
  visible-density control.
- Inspects token metadata and nearest neighbors.
- Compares two tokens using cosine similarity and Euclidean distance in the
  original embedding space.
- Exports a selected token as JSON and the current projection as CSV.
- Changes the interface accent by model family while preserving a consistent
  visual system.

Whitespace is significant in many subword tokenizers. The application preserves
the raw token text and renders spaces, tabs, and newlines with visible glyphs so
researchers can distinguish `word` from ` word`.

## Research workflow

1. Select a curated model or enter a Hugging Face repository ID.
2. Load the model. The first request downloads model assets; later requests use
   the Hugging Face cache.
3. Generate a UMAP projection. The first projection for a parameter set is
   computed once and cached in memory.
4. Orbit the scene, search for a token, or select a point.
5. Inspect nearest neighbors, compare tokens, and export the relevant data.

The visible-point slider only changes how many points WebGL draws. It does not
discard analyzed tokens, so the complete prepared subset remains searchable and
comparable.

## Architecture

```text
Browser
  React + TypeScript + Zustand
  react-three-fiber + Three.js
          |
          | typed JSON over HTTP
          v
FastAPI application
  model lifecycle and validation
  token search and vector analysis
  UMAP projection worker
          |
          v
Model-keyed LRU cache
  Hugging Face tokenizer
  input embedding matrix
  cached projection configurations
```

### Frontend

The frontend is a Vite application built with React, strict TypeScript,
Tailwind CSS, Zustand, Three.js, and react-three-fiber.

- The WebGL scene is loaded as a separate JavaScript chunk, so the application
  shell can become interactive before Three.js finishes loading.
- Rendering uses an on-demand frame loop and a capped device-pixel ratio to
  avoid spending GPU time on unchanged frames or excessive high-DPI pixels.
- Neighbor markers are instanced, reducing them to one draw call.
- Request sequence guards prevent stale search and token-detail responses from
  replacing newer user intent.
- Mobile and lower-concurrency devices start with fewer visible points while
  retaining access to the full analyzed token set.
- Production API calls default to the current origin. Development defaults to
  `http://localhost:8000` and can be overridden with `VITE_API_URL`.

See [frontend/README.md](frontend/README.md) for frontend-specific notes.

### Backend

The backend is a FastAPI service with validated Pydantic contracts and a
model-keyed LRU cache.

- A per-model asynchronous lock collapses concurrent requests for the same
  model into one load.
- Model loading runs outside the event loop and is bounded by a real timeout.
- CPU-bound UMAP work runs in a worker thread, so health and status requests
  remain responsive during projection.
- Normalized embeddings are computed once and reused for cosine operations.
- Neighbor search and batch similarity use vectorized NumPy operations.
- Projection results are cached by model and UMAP configuration.
- GZip compresses large projection payloads.
- Request IDs, process timing, content-type protection, and referrer-policy
  headers are attached to responses.
- Domain errors map to stable JSON error responses instead of leaking internal
  exceptions.

See [backend/README.md](backend/README.md) for backend internals and the complete
API table.

## Repository layout

```text
embeddings-visualizer/
|-- backend/
|   |-- app/
|   |   |-- api/                 FastAPI route modules
|   |   |-- core/                Model manager and embedding analysis
|   |   |-- config.py            Environment-driven settings
|   |   |-- main.py              Application factory and entry point
|   |   `-- schemas.py           Request and response contracts
|   |-- tests/                   Offline API and numerical tests
|   `-- requirements*.txt
|-- frontend/
|   |-- public/                  Static assets
|   |-- src/
|   |   |-- components/          Workbench UI and Three.js scene
|   |   |-- lib/                 API, types, layout, and model themes
|   |   `-- store/               Zustand application state
|   `-- package.json
|-- deploy/
|   `-- Caddyfile.example        TLS reverse-proxy example
|-- Dockerfile                   Combined production image
|-- docker-compose.yml           Single-host deployment definition
|-- RUNNING.md                   Local and PyCharm command sheet
`-- README.md
```

## Prerequisites

- Python 3.12
- Node.js 24 and npm
- Git
- Internet access for the first download of each Hugging Face model

Docker is optional for local development and recommended for the production
configuration included in this repository.

## Quick start

Run the backend and frontend in separate terminals from the repository root.
The commands below deliberately use the virtual environment's Python executable
directly, so shell activation is optional.

### PowerShell

One-time setup:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
Set-Location frontend
npm install
Set-Location ..
```

Backend terminal:

```powershell
Set-Location C:\path\to\embeddings-visualizer\backend
..\.venv\Scripts\python.exe -m app.main
```

Frontend terminal:

```powershell
Set-Location C:\path\to\embeddings-visualizer\frontend
npm run dev
```

### Git Bash on Windows

One-time setup:

```bash
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install -r backend/requirements-dev.txt
cd frontend
npm install
cd ..
```

Backend terminal:

```bash
cd /c/path/to/embeddings-visualizer/backend
../.venv/Scripts/python.exe -m app.main
```

Frontend terminal:

```bash
cd /c/path/to/embeddings-visualizer/frontend
npm run dev
```

Open [http://localhost:5173](http://localhost:5173). The API is available at
[http://localhost:8000](http://localhost:8000), with interactive documentation
at [http://localhost:8000/docs](http://localhost:8000/docs) in development.

For shell activation, PyCharm run configurations, Docker commands, and common
errors, use the comprehensive [RUNNING.md](RUNNING.md) command sheet.

## Configuration

Copy `backend/.env.example` to `backend/.env` when defaults need to change.
The backend reads this file from its working directory.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ENVIRONMENT` | `development` | Enables reload and development API docs; use `production` when deployed. |
| `HOST` | `0.0.0.0` | API bind address. |
| `PORT` | `8000` | API port. |
| `STATIC_DIR` | empty | Compiled frontend directory for same-origin serving. |
| `GZIP_MINIMUM_SIZE` | `1000` | Minimum response size to compress. |
| `CORS_ORIGINS` | local Vite origins | Comma-separated allowed frontend origins. |
| `MAX_CACHED_MODELS` | `2` | Maximum resident models before LRU eviction. |
| `MODEL_LOAD_TIMEOUT_SECONDS` | `180` | Hard limit for one model load. |
| `DEFAULT_TOP_N` | `6000` | Number of vocabulary entries prepared for analysis. |
| `MAX_CACHED_PROJECTIONS` | `8` | Cached UMAP configurations per model. |
| `ALLOWED_MODELS` | empty | Optional comma-separated public deployment allow-list. |

For a split frontend/API deployment, copy `frontend/.env.example` to
`frontend/.env` and set `VITE_API_URL` to the public API origin. The included
production image does not require this because it serves both layers from the
same origin.

## Custom model compatibility

Local development accepts a Hugging Face repository ID such as `gpt2` or
`organization/model-name`. URLs and filesystem paths are rejected.

A model is compatible when Transformers can load both its tokenizer and model,
and `get_input_embeddings()` returns a two-dimensional token embedding table.
Standard text transformer models generally satisfy this contract. Models built
only for vision, audio, or another modality may not expose token embeddings.
Those cases return a clear load error and do not crash the server or replace an
already cached model.

`trust_remote_code` is disabled. Models that require executing repository code
are intentionally unsupported by the public-facing loader.

For a public deployment, set `ALLOWED_MODELS`. Leaving arbitrary model loading
enabled lets visitors initiate large downloads and memory allocations.

## API overview

Every analysis route identifies the loaded model with `?model=<repository-id>`.

| Method | Route | Purpose |
| --- | --- | --- |
| `GET` | `/api` | Service metadata and documentation location. |
| `GET` | `/health` | Service health and cache status. |
| `GET` | `/api/models` | Curated presets and custom-model capability. |
| `POST` | `/api/models/load` | Load or reuse a model. Body: `{ "model": "gpt2" }`. |
| `GET` | `/api/models/status?model=gpt2` | Poll model load state. |
| `GET` | `/api/models/info?model=gpt2` | Inspect loaded model dimensions. |
| `DELETE` | `/api/models?model=gpt2` | Evict a cached model. |
| `POST` | `/api/visualization?model=gpt2` | Build or retrieve a UMAP projection. |
| `GET` | `/api/tokens/search?model=gpt2&query=king` | Search prepared token text. |
| `GET` | `/api/tokens/{index}?model=gpt2` | Read token metadata. |
| `GET` | `/api/tokens/{index}/neighbors?model=gpt2` | Read nearest neighbors. |
| `GET` | `/api/tokens/{index}/full?model=gpt2` | Read metadata, neighbors, and optional vector. |
| `POST` | `/api/analysis/compare?model=gpt2` | Compare two tokens by text. |
| `POST` | `/api/analysis/compare/by-id?model=gpt2` | Compare two tokens by analysis index. |
| `POST` | `/api/analysis/batch?model=gpt2` | Build a pairwise cosine-similarity matrix. |
| `GET` | `/api/analysis/statistics?model=gpt2` | Summarize the prepared token space. |

## Verification

Backend tests use synthetic models and run without network access:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m pytest
```

Frontend verification:

```powershell
Set-Location frontend
npm run lint
npm run build
```

The production build runs strict TypeScript compilation before Vite bundles the
application.

## Production deployment on mannymcgrail.com

The cleanest deployment is a dedicated subdomain such as
`embeddings.mannymcgrail.com`. The root Dockerfile builds the frontend and then
serves both the static application and FastAPI routes from one container and one
origin. This avoids cross-origin configuration and keeps the deployment simple.

The included Compose configuration assumes:

- a Linux host with Docker Compose;
- at least 8 GB of memory for the curated model set;
- an `A` or `AAAA` DNS record for `embeddings.mannymcgrail.com` pointing at the
  host;
- Caddy on the host for TLS termination and reverse proxying.

Build and start the application from the repository root:

```bash
git checkout dev
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f embeddings-visualizer
```

Verify the service on the host before configuring the proxy:

```bash
curl http://127.0.0.1:8000/health
```

Use `deploy/Caddyfile.example` as the site block for Caddy, then validate and
reload Caddy:

```bash
sudo caddy validate --config /etc/caddy/Caddyfile
sudo systemctl reload caddy
curl https://embeddings.mannymcgrail.com/health
```

The Compose service binds FastAPI to `127.0.0.1:8000`, so it is not directly
exposed to the public internet. Caddy is the public entry point and provisions
TLS certificates. The Hugging Face cache is stored in a named Docker volume and
survives application image rebuilds.

Before a public launch:

- keep `ALLOWED_MODELS` limited to models that fit the host;
- keep `MAX_CACHED_MODELS=1` unless memory measurements justify more;
- place rate limiting or access control at the reverse proxy if traffic is
  untrusted;
- monitor memory, disk use, model-download time, and projection latency;
- review each model's license before making it available;
- back up deployment configuration, not the disposable model cache.

No production deployment is performed automatically by this repository.

## Operational characteristics and limitations

- The first model load requires a download and can take several minutes.
- The first UMAP projection for a model/configuration is CPU intensive. Cached
  repeats are much faster.
- In-memory model and projection caches are process-local. Multiple API workers
  do not share them, so the supplied container intentionally uses one worker.
- `DEFAULT_TOP_N` controls the analyzed vocabulary subset, not merely the number
  of points shown. Increasing it raises memory, CPU, and payload costs.
- Token search is lexical substring search. Semantic relationships come from
  nearest-neighbor and comparison operations, not the search ranking.
- UMAP is an exploratory projection. Apparent 3D distance is not a substitute
  for measurement in the original embedding space; the inspector and compare
  tools report original-space metrics for that reason.

## Project

Designed and maintained by [Emmanuel McGrail](https://mannymcgrail.com).

The project does not currently declare an open-source license. Add one before
inviting unrestricted redistribution or outside contributions.
