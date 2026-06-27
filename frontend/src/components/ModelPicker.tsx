/** Model selection + load trigger, with live loading progress. */

import { useStore } from "../store/useStore";
import { Field } from "./ui";

export function ModelPicker() {
  const models = useStore((s) => s.models);
  const selectedModelId = useStore((s) => s.selectedModelId);
  const setSelectedModel = useStore((s) => s.setSelectedModel);
  const loadAndVisualize = useStore((s) => s.loadAndVisualize);
  const loadState = useStore((s) => s.loadState);
  const loadProgress = useStore((s) => s.loadProgress);
  const loadedModel = useStore((s) => s.loadedModel);

  const busy = loadState === "loading" || loadState === "visualizing";
  const supportsCustom = models?.supports_custom_models ?? true;

  return (
    <div className="space-y-3">
      <Field label="Model">
        <select
          className="input"
          value={selectedModelId}
          disabled={busy}
          onChange={(e) => setSelectedModel(e.target.value)}
        >
          {models?.presets.map((p) => (
            <option key={p.id} value={p.id} className="bg-ink-800">
              {p.name} · {p.params}
            </option>
          ))}
        </select>
      </Field>

      {supportsCustom && (
        <Field label="…or a custom Hugging Face id" hint="optional">
          <input
            className="input font-mono"
            placeholder="e.g. sentence-transformers/all-MiniLM-L6-v2"
            disabled={busy}
            onChange={(e) => setSelectedModel(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !busy) loadAndVisualize();
            }}
          />
        </Field>
      )}

      <button className="btn-primary w-full" disabled={busy} onClick={() => loadAndVisualize()}>
        {busy ? "Loading…" : loadedModel ? "Reload model" : "Load & visualize"}
      </button>

      {busy && (
        <div className="space-y-2 animate-fade-in">
          <div className="relative h-1 overflow-hidden rounded-full bg-white/10">
            <div className="absolute inset-y-0 w-1/3 animate-sweep rounded-full bg-gradient-to-r from-transparent via-accent to-transparent" />
          </div>
          <p className="font-mono text-xs text-muted">{loadProgress}</p>
          <p className="text-[11px] text-faint">
            First load downloads the model (cached afterwards) — this can take a minute.
          </p>
        </div>
      )}
    </div>
  );
}
