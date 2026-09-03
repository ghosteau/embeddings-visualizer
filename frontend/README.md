# Embeddings Visualizer frontend

The frontend is a responsive React and Three.js research workbench for exploring
the projection and analysis APIs exposed by the FastAPI backend.

## Stack

- React 19 and strict TypeScript
- Vite 8
- Zustand for application and request state
- Three.js, react-three-fiber, and Drei for the point-cloud scene
- Tailwind CSS plus a project-specific component layer
- Oxlint

## Development

From the repository root:

```powershell
Set-Location frontend
npm install
npm run dev
```

The development server listens on `http://localhost:5173` and calls
`http://localhost:8000` by default. Start the backend separately before loading a
model.

## Commands

| Command | Purpose |
| --- | --- |
| `npm run dev` | Start the Vite development server. |
| `npm run lint` | Run Oxlint. |
| `npm run build` | Run TypeScript project compilation and create `dist`. |
| `npm run preview` | Serve the production bundle locally. |

Use `npm ci` in CI or when reproducing the lockfile exactly.

## API configuration

Development uses `http://localhost:8000` when `VITE_API_URL` is unset.
Production uses same-origin requests when it is unset, matching the root Docker
image.

For a split deployment, create `.env` in this directory:

```dotenv
VITE_API_URL=https://api.example.com
```

Vite embeds this value at build time. Restart the development server or rebuild
after changing it.

## Interface structure

- `App.tsx` assembles the top bar, controls, visualization surface, and inspector.
- `components/scene` contains the lazily loaded WebGL scene and point cloud.
- `components/ControlRail.tsx` contains model, projection, search, and comparison
  workflows.
- `components/DetailPanel.tsx` contains token inspection and export.
- `store/useStore.ts` coordinates model load polling, projection, selection,
  search, comparison, and stale-request protection.
- `lib/types.ts` mirrors backend Pydantic response contracts.
- `lib/modelTheme.ts` maps model families to accent variables.

The scene uses an on-demand frame loop, capped device-pixel ratio, instanced
neighbor markers, and a separate JavaScript chunk. The visible point count is a
rendering control only; search and analysis continue to cover the complete
backend-prepared token subset.
