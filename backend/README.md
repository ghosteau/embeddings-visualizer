# Embeddings Visualizer — Backend

A FastAPI service for loading transformer language models, extracting their
token embedding spaces, projecting them to 2D/3D with UMAP, and exploring the
geometry of those spaces (nearest neighbors, similarity, comparisons,
statistics).

## Why it's structured this way

The service is **model-keyed and stateless per request**. Instead of one global
"currently loaded model" that all visitors share and overwrite, a
`ModelManager` keeps an **LRU cache of models keyed by name**:

- Two users requesting `gpt2` share a single in-memory copy.
- At most `MAX_CACHED_MODELS` are resident at once; the least-recently-used is
  evicted, capping memory on modest hosts.
- A per-model async lock collapses simultaneous load requests into one load.
- Loads run in a worker thread under a real `asyncio.wait_for` timeout, so a
  slow download can't hang the server.

Every query endpoint names the model it operates on (`?model=gpt2`), so
concurrent users never interfere with each other.

## Layout

```
backend/
  app/
    main.py            # app factory, CORS, lifespan, exception handlers
    config.py          # env-driven settings (pydantic-settings)
    schemas.py         # Pydantic request/response contracts
    core/
      exceptions.py    # typed domain errors -> HTTP statuses
      visualizer.py    # LoadedModel: vectorized embedding math + projection cache
      model_manager.py # LRU model cache, async load w/ timeout
    api/
      health.py models.py visualization.py tokens.py analysis.py deps.py
  tests/               # offline pytest suite (synthetic data, no downloads)
  requirements.txt  requirements-dev.txt  Dockerfile  .env.example
```

## Running locally

```bash
cd backend
python -m venv .venv && source .venv/Scripts/activate   # Windows Git Bash
pip install -r requirements-dev.txt
cp .env.example .env            # optional; defaults are sensible
python -m app.main              # serves http://localhost:8000  (docs at /docs)
```

## Tests

```bash
cd backend
pytest                          # 31 tests, fully offline
```

## Configuration

All settings are environment variables (see `.env.example`). Notably:

| Variable | Default | Purpose |
| --- | --- | --- |
| `CORS_ORIGINS` | `localhost:5173` | Allowed frontend origins |
| `MAX_CACHED_MODELS` | `2` | LRU model cache size |
| `MODEL_LOAD_TIMEOUT_SECONDS` | `180` | Hard load timeout |
| `DEFAULT_TOP_N` | `3000` | Tokens analysed per model |
| `ALLOWED_MODELS` | *(empty)* | Allow-list for public deploys |

For a public deployment, set `ALLOWED_MODELS` to a curated list so visitors
cannot trigger arbitrary multi-gigabyte downloads.

## API overview

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health` | Liveness + cache snapshot |
| `GET` | `/api/models` | List curated presets |
| `POST` | `/api/models/load` | Load a model (cached, idempotent) |
| `GET` | `/api/models/status?model=` | Loading status |
| `GET` | `/api/models/info?model=` | Loaded model details |
| `DELETE` | `/api/models?model=` | Evict a model |
| `POST` | `/api/visualization?model=` | Build a UMAP projection |
| `GET` | `/api/tokens/{i}?model=` | Token details |
| `GET` | `/api/tokens/{i}/neighbors?model=` | Nearest neighbors |
| `GET` | `/api/tokens/{i}/full?model=` | Details + neighbors |
| `GET` | `/api/tokens/search?model=&query=` | Search tokens |
| `POST` | `/api/analysis/compare?model=` | Compare two tokens by name |
| `POST` | `/api/analysis/compare/by-id?model=` | Compare two tokens by index |
| `POST` | `/api/analysis/batch?model=` | Pairwise similarity matrix |
| `GET` | `/api/analysis/statistics?model=` | Token-space statistics |

Interactive docs (OpenAPI/Swagger) are served at `/docs` in development.

## Docker

```bash
cd backend
docker build -t embeddings-visualizer-backend .
docker run -p 8000:8000 -v hf-cache:/cache/huggingface embeddings-visualizer-backend
```

The volume persists the Hugging Face model cache across container restarts.
