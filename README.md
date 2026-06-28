# Embeddings Visualizer for Language Models

An interactive platform for **exploring and understanding token embeddings** from
transformer-based language models such as GPT-2, BERT, DistilBERT, and RoBERTa.
Load a model, project its embedding space to 2D/3D with UMAP, and inspect how the
model represents meaning, similarity, and structure in high-dimensional space.

Built for researchers, students, and developers curious about how language models
"see" tokens.

---

## Status

| Component | State |
| --- | --- |
| **Backend** (FastAPI) | v1.0 — remastered: model-keyed LRU cache, vectorized math, typed errors, tested, Dockerized |
| **Frontend** (React + Three.js) |

---

## Architecture

```
embeddings-visualizer/
  backend/      FastAPI service — model loading, UMAP projection, token analysis
  frontend/     React + Vite + react-three-fiber 3D explorer (in progress)
```

The backend is **model-keyed**: every request names the model it operates on, and
models are shared across users via a memory-capped LRU cache. This makes the
service safe to host for multiple concurrent visitors. See
[`backend/README.md`](backend/README.md) for the full design and API reference.

## Quick start (backend)

```bash
cd backend
python -m venv .venv && source .venv/Scripts/activate
pip install -r requirements-dev.txt
python -m app.main          # http://localhost:8000  (interactive docs at /docs)
pytest                      # run the offline test suite
```

## Tech stack

**Backend:** Python · FastAPI · Pydantic · Hugging Face Transformers · PyTorch ·
UMAP · scikit-learn · NumPy

**Frontend:** JavaScript/TypeScript · React · Vite · react-three-fiber / Three.js

## Contributors

- **Manny McGrail** — Project management, backend API, and ML module design.
