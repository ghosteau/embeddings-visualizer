# Running the Embeddings Visualizer locally

The app is **two processes**:

| Process | Folder | URL | Command |
| --- | --- | --- | --- |
| Backend API (FastAPI) | `backend/` | http://localhost:8000 (docs at `/docs`) | `python -m app.main` |
| Frontend (Vite/React) | `frontend/` | http://localhost:5173 | `npm run dev` |

Run them in **two separate terminals** (or two PyCharm tabs). Start the backend
first so the first model load works.

---

## One-time setup

```powershell
# from the project root
# 1) Python deps into the existing .venv (Python 3.12)
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt

# 2) Frontend deps
cd frontend
npm install
cd ..
```

> Optional: `copy backend\.env.example backend\.env` and edit it to change the
> port, CORS origins, token cap (`DEFAULT_TOP_N`), cache size, or an
> `ALLOWED_MODELS` allow-list. Defaults work without a `.env`.

---

## Day-to-day: start both servers

**Terminal 1 — backend**
```powershell
.\.venv\Scripts\Activate.ps1     # activate the venv (Git Bash: source .venv/Scripts/activate)
cd backend
python -m app.main               # http://localhost:8000  (auto-reloads on code changes)
```

**Terminal 2 — frontend**
```powershell
cd frontend
npm run dev                      # http://localhost:5173
```

Open **http://localhost:5173**, pick a model, click **Load & visualize**.
Stop either server with **Ctrl+C**.

---

## Running from PyCharm

### Backend — as a Run Configuration (recommended)
1. **Run → Edit Configurations… → + → Python**.
2. Set:
   - **Name:** `backend`
   - **Module name** (toggle the "Script path" dropdown to "Module name"): `app.main`
   - **Working directory:** the `backend/` folder
   - **Python interpreter:** the project `.venv` (Python 3.12)
3. **OK**, then press ▶ (Run) or 🐞 (Debug — breakpoints work).

> Equivalent uvicorn config if you prefer: Module name `uvicorn`,
> Parameters `app.main:app --reload --host 127.0.0.1 --port 8000`, working dir `backend/`.

### Frontend — two options
- **npm Run Configuration** (needs the bundled *Node.js* plugin):
  **+ → npm** → **package.json:** `frontend/package.json` → **Command:** `run` →
  **Scripts:** `dev` → Run.
- **Or just use the PyCharm Terminal:** `cd frontend` then `npm run dev`.

### Tip: launch both at once
Create a **Compound** run configuration (**+ → Compound**) that bundles the
`backend` and `frontend` configs, so one ▶ starts the whole app.

---

## Good to know
- **First load of a model downloads it from Hugging Face** (cached afterward).
  `gpt2`, `distilgpt2`, `bert-base-uncased`, etc. are quick once cached.
- The **first projection per model runs UMAP over ~6000 tokens (~20–40s)**;
  repeat projections of the same model/settings are served from cache instantly.
- The UI **re-themes per model** (GPT→blue, BERT→Google blue-ish, RoBERTa→violet,
  custom→terracotta) and shows the family in the control rail.
- **Custom models:** type any Hugging Face id in the "custom" box. Models without
  a token-embedding table (vision/audio/etc.) fail gracefully with a clear message.
- **Shortcuts:** `/` focuses search, `Esc` deselects. Inspect any token via search
  (even ones hidden by the "Visible points" slider); **Export JSON** in the detail
  panel saves its metadata + neighbors + embedding vector.

## Troubleshooting
- **Port already in use** (`5173`/`8000`): stop the other process, or change the
  port (`--port` for uvicorn; `npm run dev -- --port 5174`). If you change the
  frontend port, add it to `CORS_ORIGINS` in `backend/.env`.
- **"Cannot reach the API"** in the UI: the backend isn't running, or it's on a
  different host/port than `VITE_API_URL` (defaults to `http://localhost:8000`).
- **Edited `frontend/tailwind.config.js` and styles look stale:** restart
  `npm run dev` (config changes aren't hot-reloaded).
- **Production build:** `cd frontend && npm run build` outputs static files to
  `frontend/dist/`; the backend ships with a `Dockerfile` for container hosting.
