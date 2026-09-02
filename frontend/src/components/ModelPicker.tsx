/** Model selection and loading with a deliberate custom-model path. */

import { useEffect, useMemo, useState } from "react";
import type { PresetModel } from "../lib/types";
import { useStore } from "../store/useStore";
import { Field, SegToggle } from "./ui";

type ModelSource = "preset" | "custom";
const EMPTY_PRESETS: PresetModel[] = [];

export function ModelPicker() {
  const models = useStore((s) => s.models);
  const selectedModelId = useStore((s) => s.selectedModelId);
  const setSelectedModel = useStore((s) => s.setSelectedModel);
  const loadAndVisualize = useStore((s) => s.loadAndVisualize);
  const loadState = useStore((s) => s.loadState);
  const loadProgress = useStore((s) => s.loadProgress);
  const loadedModel = useStore((s) => s.loadedModel);
  const [source, setSource] = useState<ModelSource>("preset");
  const [customId, setCustomId] = useState("");

  const busy = loadState === "loading" || loadState === "visualizing";
  const supportsCustom = models?.supports_custom_models ?? true;
  const presets = models?.presets ?? EMPTY_PRESETS;
  const currentPreset = useMemo(
    () => presets.find((model) => model.id === selectedModelId),
    [presets, selectedModelId],
  );

  useEffect(() => {
    if (!supportsCustom && source === "custom") setSource("preset");
  }, [source, supportsCustom]);

  const changeSource = (next: ModelSource) => {
    setSource(next);
    if (next === "preset") {
      const preset = currentPreset ?? presets[0];
      if (preset) setSelectedModel(preset.id);
    } else {
      setSelectedModel(customId.trim());
    }
  };

  const canLoad = Boolean(selectedModelId.trim()) && !busy;

  return (
    <div className="space-y-4">
      {supportsCustom && (
        <SegToggle<ModelSource>
          value={source}
          onChange={changeSource}
          options={[
            { value: "preset", label: "Curated" },
            { value: "custom", label: "Hugging Face" },
          ]}
        />
      )}

      {source === "preset" || !supportsCustom ? (
        <Field label="Model">
          <select
            className="input"
            value={selectedModelId}
            disabled={busy || presets.length === 0}
            onChange={(event) => setSelectedModel(event.target.value)}
          >
            {presets.map((model) => (
              <option key={model.id} value={model.id} className="bg-ink-900">
                {model.name} / {model.params}
              </option>
            ))}
          </select>
        </Field>
      ) : (
        <Field label="Repository id" hint="owner/model">
          <input
            className="input font-mono"
            value={customId}
            placeholder="sentence-transformers/all-MiniLM-L6-v2"
            autoComplete="off"
            spellCheck={false}
            disabled={busy}
            onChange={(event) => {
              const value = event.target.value;
              setCustomId(value);
              setSelectedModel(value.trim());
            }}
            onKeyDown={(event) => {
              if (event.key === "Enter" && canLoad) loadAndVisualize();
            }}
          />
        </Field>
      )}

      {currentPreset && source === "preset" && (
        <div className="data-strip">
          <span>{currentPreset.family}</span>
          <span>{currentPreset.params} parameters</span>
        </div>
      )}

      <button className="btn-primary w-full" disabled={!canLoad} onClick={loadAndVisualize}>
        {busy ? "Working…" : loadedModel === selectedModelId ? "Reload projection" : "Load model"}
      </button>

      {busy ? (
        <div className="space-y-2" aria-live="polite">
          <div className="progress-track">
            <div className="progress-sweep" />
          </div>
          <p className="font-mono text-[11px] leading-relaxed text-muted">{loadProgress}</p>
        </div>
      ) : (
        <p className="help-copy">
          Standard text models load directly. Gated repositories and architectures without token
          embeddings return a clear error instead of interrupting the workspace.
        </p>
      )}
    </div>
  );
}
