/** UMAP projection controls; re-projects on demand once a model is loaded. */

import { useStore } from "../store/useStore";
import { Field, SegToggle, Slider } from "./ui";
import type { DistanceMetric } from "../lib/types";

export function VizControls() {
  const config = useStore((s) => s.config);
  const setConfig = useStore((s) => s.setConfig);
  const regenerate = useStore((s) => s.regenerate);
  const loadedModel = useStore((s) => s.loadedModel);
  const loadState = useStore((s) => s.loadState);
  const vizData = useStore((s) => s.vizData);
  const displayCount = useStore((s) => s.displayCount);
  const setDisplayCount = useStore((s) => s.setDisplayCount);

  const busy = loadState === "visualizing" || loadState === "loading";
  const disabled = !loadedModel || busy;
  const totalPoints = vizData?.tokens.length ?? 0;
  const dirty = Boolean(
    vizData &&
      (vizData.config.n_components !== config.n_components ||
        vizData.config.metric !== config.metric ||
        vizData.config.n_neighbors !== config.n_neighbors ||
        vizData.config.min_dist !== config.min_dist),
  );

  return (
    <div className="space-y-4">
      <div className="data-strip">
        <span>UMAP</span>
        <span>{dirty ? "changes pending" : vizData ? "projection current" : "not computed"}</span>
      </div>
      <Field label="Dimensions">
        <SegToggle
          value={String(config.n_components)}
          onChange={(v) => setConfig({ n_components: Number(v) as 2 | 3 })}
          options={[
            { value: "2", label: "2D" },
            { value: "3", label: "3D" },
          ]}
        />
      </Field>

      <Field label="UMAP metric">
        <SegToggle<DistanceMetric>
          value={config.metric}
          onChange={(v) => setConfig({ metric: v })}
          options={[
            { value: "cosine", label: "Cosine" },
            { value: "euclidean", label: "Euclidean" },
          ]}
        />
      </Field>

      <Field label="Neighbors" hint={String(config.n_neighbors)}>
        <Slider
          value={config.n_neighbors}
          min={2}
          max={100}
          step={1}
          onChange={(v) => setConfig({ n_neighbors: v })}
        />
      </Field>

      <Field label="Min distance" hint={config.min_dist.toFixed(2)}>
        <Slider
          value={config.min_dist}
          min={0}
          max={0.99}
          step={0.01}
          onChange={(v) => setConfig({ min_dist: v })}
        />
      </Field>

      <button
        className="btn-ghost w-full"
        disabled={disabled}
        onClick={() => void regenerate().catch(() => undefined)}
      >
        {loadState === "visualizing" ? "Computing…" : dirty ? "Apply projection" : "Recompute"}
      </button>

      {/* Visible-points is a purely client-side declutter — no re-projection.
          The full projection is always retained so any token can still be
          searched, inspected, and compared. */}
      {totalPoints > 0 && (
        <Field
          label="Visible points"
          hint={`${displayCount.toLocaleString()} / ${totalPoints.toLocaleString()}`}
        >
          <Slider
            value={displayCount}
            min={Math.min(100, totalPoints)}
            max={totalPoints}
            step={50}
            onChange={(v) => setDisplayCount(v)}
          />
        </Field>
      )}

      <p className="help-copy">
        Projection controls change the spatial layout. Visible points only changes rendering and
        keeps every analysed token searchable.
      </p>
    </div>
  );
}
