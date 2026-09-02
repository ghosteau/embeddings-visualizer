# Embeddings Visualizer backend

The backend is a FastAPI service for loading transformer token embedding tables,
projecting prepared vocabularies with UMAP, and exposing token-level analysis over
HTTP. It is model-keyed: every analysis request identifies its model rather than
mutating one global active-model singleton.

## Design

`ModelManager` maintains a bounded least-recently-used cache keyed by Hugging Face
model ID. A per-model asynchronous lock prevents duplicate concurrent loads, and
each load runs outside the event loop under a configurable timeout. Eviction
releases model resources and triggers garbage collection.

`LoadedModel` owns the prepared token subset and numerical operations:

- tokenizer batch decoding with a compatibility fallback;
- preservation of raw subword whitespace;
- precomputed token metadata and normalized embeddings;
- vectorized cosine and Euclidean nearest-neighbor search;
- vectorized pairwise cosine similarity;
- bounded, configuration-keyed UMAP projection caching;
- serialized per-model projection computation to avoid duplicate expensive fits.

Projection work runs in a worker thread from the API route. The ASGI event loop
therefore remains available for health, status, and lightweight token requests.

## Layout

```text
backend/
|-- app/
|   |-- api/
|   |   |-- analysis.py
|   |   |-- health.py
|   |   |-- models.py
|   |   |-- tokens.py
|   |   `-- visualization.py
|   |-- core/
|   |   |-- exceptions.py
|   |   |-- model_manager.py
|   |   `-- visualizer.py
|   |-- config.py
|   |-- main.py
|   `-- schemas.py
|-- tests/
|-- .env.example
|-- requirements.txt
`-- requirements-dev.txt
```

## Local setup

From the repository root in PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
Set-Location backend
..\.venv\Scripts\python.exe -m app.main
```

From the repository root in Git Bash:

```bash
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install -r backend/requirements-dev.txt
cd backend
../.venv/Scripts/python.exe -m app.main
```

The API listens on `http://localhost:8000`. Development OpenAPI documentation is
available at `http://localhost:8000/docs`.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api` | Service metadata and documentation location. |
| `GET` | `/health` | Liveness and cache snapshot. |
| `GET` | `/api/models` | Curated presets and server capabilities. |
| `POST` | `/api/models/load` | Load or reuse a model. |
| `GET` | `/api/models/status?model=` | Read load progress or failure detail. |
| `GET` | `/api/models/info?model=` | Read vocabulary and embedding dimensions. |
| `DELETE` | `/api/models?model=` | Evict a cached model. |
| `POST` | `/api/visualization?model=` | Build or retrieve a UMAP projection. |
| `GET` | `/api/tokens/search?model=&query=` | Lexically search prepared token text. |
| `GET` | `/api/tokens/{index}?model=` | Read token metadata. |
| `GET` | `/api/tokens/{index}/neighbors?model=` | Find nearest neighbors. |
| `GET` | `/api/tokens/{index}/full?model=` | Read metadata, neighbors, and optional vector. |
| `POST` | `/api/analysis/compare?model=` | Compare tokens by text. |
| `POST` | `/api/analysis/compare/by-id?model=` | Compare tokens by analysis index. |
| `POST` | `/api/analysis/batch?model=` | Build a pairwise similarity matrix. |
| `GET` | `/api/analysis/statistics?model=` | Summarize the prepared embedding space. |

Example model load:

```bash
curl -X POST http://localhost:8000/api/models/load \
  -H "Content-Type: application/json" \
  -d '{"model":"distilgpt2"}'
```

Example projection:

```bash
curl -X POST "http://localhost:8000/api/visualization?model=distilgpt2" \
  -H "Content-Type: application/json" \
  -d '{"n_components":3,"n_neighbors":15,"min_dist":0.1,"metric":"cosine"}'
```

## Configuration

Copy `.env.example` to `.env` in this directory when overrides are needed.

| Variable | Default | Purpose |
| --- | --- | --- |
| `ENVIRONMENT` | `development` | Controls reload and API documentation exposure. |
| `HOST` | `0.0.0.0` | Bind address. |
| `PORT` | `8000` | Bind port. |
| `STATIC_DIR` | empty | Optional compiled frontend directory. |
| `GZIP_MINIMUM_SIZE` | `1000` | Response compression threshold in bytes. |
| `CORS_ORIGINS` | local Vite origins | Comma-separated allowed origins. |
| `MAX_CACHED_MODELS` | `2` | Resident model limit. |
| `MODEL_LOAD_TIMEOUT_SECONDS` | `180` | Load timeout in seconds. |
| `DEFAULT_TOP_N` | `6000` | Prepared vocabulary size. |
| `MAX_CACHED_PROJECTIONS` | `8` | Projection-cache size per model. |
| `ALLOWED_MODELS` | empty | Optional model repository allow-list. |

Public deployments should set `ALLOWED_MODELS` and size
`MAX_CACHED_MODELS` according to measured host memory. Arbitrary model loading
is useful locally but is not an appropriate unrestricted public default.

## Model compatibility and safety

The load request accepts only Hub-style IDs such as `gpt2` or `owner/model`.
URLs and local paths are rejected. Transformers loads with
`trust_remote_code=False`, and a compatible model must provide a two-dimensional
input token embedding table through `get_input_embeddings()`.

Unsupported model architectures return typed API errors. A failed load does not
replace or corrupt another cached model.

## Tests

The suite uses synthetic embeddings and mocked model loaders; it does not
download model assets.

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m pytest
```

Coverage includes numerical operations, whitespace-preserving token handling,
search and comparison behavior, model LRU eviction, load serialization, timeouts,
allow-list enforcement, request validation, API routes, and operational headers.

## Production behavior

When `STATIC_DIR` points to a compiled Vite directory, FastAPI serves the SPA
after registering every API route. This is how the root production Dockerfile
provides a same-origin deployment.

Production mode hides Swagger and ReDoc. Responses include request and processing
metadata headers, large JSON payloads are compressed, and the supplied container
runs as a non-root user. Use one API worker because model and projection caches
are process-local.

The backend-only `Dockerfile` remains available for split deployments. The root
`Dockerfile` is preferred when the frontend and API should ship together.
