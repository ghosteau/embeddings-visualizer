# Local development and command sheet

Embeddings Visualizer runs as two development processes:

| Process | Working directory | Address | Command |
| --- | --- | --- | --- |
| FastAPI backend | `backend` | `http://localhost:8000` | `python -m app.main` |
| Vite frontend | `frontend` | `http://localhost:5173` | `npm run dev` |

Start the backend first, then the frontend. Keep both terminals open while using
the application.

## Prerequisites

Check the installed tools from any terminal:

```powershell
py --version
node --version
npm --version
git --version
```

Use Python 3.12 and Node.js 24. On Windows, the repository virtual environment
is expected at `.venv` in the project root, not inside `backend`.

## One-time setup in PowerShell

Run these commands from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
Set-Location frontend
npm install
Set-Location ..
```

Optional backend configuration:

```powershell
Copy-Item backend\.env.example backend\.env
```

The defaults are correct for local development, so `.env` is not required.

## One-time setup in Git Bash

Run these commands from the repository root:

```bash
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install -r backend/requirements-dev.txt
cd frontend
npm install
cd ..
```

Optional backend configuration:

```bash
cp backend/.env.example backend/.env
```

PowerShell activation commands do not work in Git Bash. In particular,
`.\.venv\Scripts\Activate.ps1` is PowerShell syntax. Either use the virtual
environment's Python executable directly, as shown above, or activate it with:

```bash
source .venv/Scripts/activate
```

## Start the application in PowerShell

Terminal 1, from the repository root:

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m app.main
```

Terminal 2, from the repository root:

```powershell
Set-Location frontend
npm run dev
```

Open `http://localhost:5173`. Stop either process with `Ctrl+C`.

Activation is optional. If you prefer an activated PowerShell environment:

```powershell
.\.venv\Scripts\Activate.ps1
Set-Location backend
python -m app.main
```

If PowerShell blocks the activation script, the direct executable commands
above still work and do not require an execution-policy change.

## Start the application in Git Bash

Terminal 1, from the repository root:

```bash
cd backend
../.venv/Scripts/python.exe -m app.main
```

Terminal 2, from the repository root:

```bash
cd frontend
npm run dev
```

If your prompt already ends in `/embeddings-visualizer/backend`, do not run
`cd backend` again. Start the backend with:

```bash
../.venv/Scripts/python.exe -m app.main
```

## PyCharm configuration

Open the repository root as the PyCharm project.

### Select the Python interpreter

1. Open `Settings` or `Preferences`.
2. Go to `Project: embeddings-visualizer` and then `Python Interpreter`.
3. Select `Add Interpreter`, then `Add Local Interpreter`.
4. Choose `Existing` and select:

```text
C:\path\to\embeddings-visualizer\.venv\Scripts\python.exe
```

If `.venv` does not exist, complete the one-time setup first.

### Backend run configuration

1. Open `Run` and then `Edit Configurations`.
2. Add a `Python` configuration.
3. Set the name to `Backend`.
4. Change `Script path` to `Module name`.
5. Set the module name to `app.main`.
6. Set the working directory to the repository's `backend` folder.
7. Select the project `.venv` interpreter.
8. Leave parameters empty and run the configuration.

Equivalent Uvicorn configuration:

| Field | Value |
| --- | --- |
| Module name | `uvicorn` |
| Parameters | `app.main:app --reload --host 127.0.0.1 --port 8000` |
| Working directory | `C:\path\to\embeddings-visualizer\backend` |
| Interpreter | `C:\path\to\embeddings-visualizer\.venv\Scripts\python.exe` |

### Frontend run configuration

1. Add an `npm` run configuration.
2. Set `package.json` to `frontend/package.json`.
3. Set command to `run`.
4. Set script to `dev`.
5. Name the configuration `Frontend` and run it.

If PyCharm does not offer an npm configuration, enable its Node.js support or
use the integrated terminal:

```powershell
Set-Location frontend
npm run dev
```

### Start both from one PyCharm action

Add a `Compound` run configuration and include `Backend` and `Frontend`. Running
the compound configuration starts both processes. PyCharm's Services window then
shows both logs and lets you stop or restart either process independently.

## Day-to-day commands

All examples start at the repository root.

### Backend

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m app.main
```

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m pytest
```

```powershell
Set-Location backend
..\.venv\Scripts\python.exe -m pytest -v
```

### Frontend

```powershell
Set-Location frontend
npm run dev
```

```powershell
Set-Location frontend
npm run lint
npm run build
```

```powershell
Set-Location frontend
npm run preview
```

`npm run preview` serves the already built frontend for a production-bundle
smoke test. The API must still be running separately.

### Git

```powershell
git status
git branch --show-current
git log --oneline -10
```

The active integration branch for the current work is `dev`.

## Local Docker build

Docker runs the compiled frontend and API from one container at
`http://localhost:8000`:

```powershell
docker compose build
docker compose up -d
docker compose ps
docker compose logs -f embeddings-visualizer
```

Stop the application without deleting the model cache:

```powershell
docker compose down
```

Rebuild after code changes:

```powershell
docker compose up -d --build
```

The supplied Compose file binds only to `127.0.0.1:8000`. This is intentional:
a public server should place Caddy or another reverse proxy in front of it.

## Configuration examples

Analyze fewer tokens for faster local projections:

```dotenv
DEFAULT_TOP_N=3000
```

Allow only selected models:

```dotenv
ALLOWED_MODELS=distilgpt2,gpt2,distilbert-base-uncased
```

Run the frontend on a different port:

```powershell
Set-Location frontend
npm run dev -- --port 5174
```

When changing the frontend origin, update `CORS_ORIGINS` in `backend/.env`:

```dotenv
CORS_ORIGINS=http://localhost:5174
```

When the API is on another origin, create `frontend/.env`:

```dotenv
VITE_API_URL=http://localhost:8001
```

Restart Vite after changing a frontend environment variable.

## Troubleshooting

### `ModuleNotFoundError: No module named 'pydantic_settings'`

The backend dependencies are not installed in the Python interpreter currently
running the server. From the repository root, run:

```bash
./.venv/Scripts/python.exe -m pip install -r backend/requirements-dev.txt
cd backend
../.venv/Scripts/python.exe -m app.main
```

In PowerShell, use backslashes:

```powershell
.\.venv\Scripts\python.exe -m pip install -r backend\requirements-dev.txt
Set-Location backend
..\.venv\Scripts\python.exe -m app.main
```

### `cd: backend: No such file or directory`

Your terminal is already in the `backend` directory. Confirm with `pwd`, then
run the backend command without changing directory:

```bash
../.venv/Scripts/python.exe -m app.main
```

### `Activate.ps1: command not found` in Git Bash

Use Git Bash syntax:

```bash
source ../.venv/Scripts/activate
```

That path assumes the terminal is already in `backend`. Activation is not
required if you invoke `../.venv/Scripts/python.exe` directly.

### The frontend says it cannot reach the API

- Confirm `http://localhost:8000/health` returns JSON.
- Confirm the backend terminal is still running.
- Confirm `VITE_API_URL` is unset or points to the correct API origin.
- Restart Vite after editing `.env`.
- If the frontend port changed, add that origin to backend `CORS_ORIGINS`.

### Port 8000 or 5173 is already in use

Stop the previous process with `Ctrl+C`, or identify it in PowerShell:

```powershell
Get-NetTCPConnection -LocalPort 8000,5173 -ErrorAction SilentlyContinue
```

Use a different frontend port with `npm run dev -- --port 5174`. Change backend
`PORT` in `backend/.env` if the API needs another port.

### A model fails to load

- Verify the machine has internet access on the first load.
- Use a Hugging Face repository ID, not a URL or filesystem path.
- Confirm the model exposes an input token embedding table.
- Models that require `trust_remote_code=True` are not supported.
- Check whether `ALLOWED_MODELS` excludes the requested model.
- Review the backend terminal for the preserved error detail.

The server reports unsupported embedding architectures as a controlled API error
and remains available for another model request.

### Projection feels slow

The first UMAP run is CPU intensive. Repeating the same model and settings uses
the in-memory projection cache. For local iteration, reduce `DEFAULT_TOP_N` in
`backend/.env`, restart the backend, and reload the model.

### The scene feels slow

Reduce `Visible points` in the application. Search, inspect, and comparison still
operate over every analyzed token. Closing other GPU-heavy browser tabs can also
help on integrated graphics.

### Tailwind or environment changes do not appear

Restart `npm run dev`. Vite does not apply every configuration-file change
through hot module replacement.

### Reset local frontend dependencies

Prefer a clean, lockfile-driven install:

```powershell
Set-Location frontend
npm ci
```

Do not delete the Hugging Face cache unless disk recovery is intentional; model
downloads can be large and are reusable across runs.
