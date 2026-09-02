/** First-use state for the empty research canvas. */

import { useStore } from "../store/useStore";

export function WelcomeOverlay() {
  const loadState = useStore((s) => s.loadState);
  const loadProgress = useStore((s) => s.loadProgress);
  const models = useStore((s) => s.models);
  const busy = loadState === "loading" || loadState === "visualizing";

  return (
    <div className="welcome-surface">
      <div className="welcome-copy">
        <div className="eyebrow">TOKEN SPACE / INTERACTIVE ATLAS</div>
        <h2>See what the model puts near each other.</h2>
        <p>
          Project a transformer vocabulary into a navigable field. Inspect raw embedding
          neighborhoods, compare exact tokens, and export the data behind the view.
        </p>

        {busy ? (
          <div className="welcome-progress" aria-live="polite">
            <span className="status-dot status-dot--visualizing" />
            <div>
              <div className="eyebrow">CURRENT OPERATION</div>
              <div className="mt-1 font-mono text-xs text-paper">{loadProgress ?? "Working…"}</div>
            </div>
          </div>
        ) : (
          <div className="welcome-steps">
            <div>
              <span>01</span>
              <p>Choose a curated model or enter a Hugging Face repository.</p>
            </div>
            <div>
              <span>02</span>
              <p>Load the embedding table and compute a reproducible UMAP projection.</p>
            </div>
            <div>
              <span>03</span>
              <p>Search, inspect, compare, and export without leaving the workspace.</p>
            </div>
          </div>
        )}
      </div>

      <div className="welcome-readout" aria-label="Workspace capabilities">
        <div className="eyebrow">SYSTEM / READY</div>
        <dl>
          <div>
            <dt>Model sources</dt>
            <dd>{models?.presets.length ?? "—"} curated + custom</dd>
          </div>
          <div>
            <dt>Projection</dt>
            <dd>UMAP / 2D or 3D</dd>
          </div>
          <div>
            <dt>Similarity</dt>
            <dd>cosine + euclidean</dd>
          </div>
          <div>
            <dt>Export</dt>
            <dd>token JSON + projection CSV</dd>
          </div>
        </dl>
      </div>
    </div>
  );
}
