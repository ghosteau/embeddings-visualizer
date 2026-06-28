/**
 * Right-hand inspector for the selected token: metadata, a neighbor list
 * (click to fly to it), and a metric toggle. Empty state nudges the user to
 * pick or search a token when nothing is selected.
 */

import { useState } from "react";
import { useStore } from "../store/useStore";
import { api } from "../lib/api";
import { TOKEN_COLORS } from "../lib/tokenColors";
import { SegToggle, Stat } from "./ui";
import { Logo } from "./Logo";
import type { DistanceMetric } from "../lib/types";

export function DetailPanel() {
  const detail = useStore((s) => s.tokenDetail);
  const detailLoading = useStore((s) => s.detailLoading);
  const focusOn = useStore((s) => s.focusOn);
  const neighborMetric = useStore((s) => s.neighborMetric);
  const setNeighborMetric = useStore((s) => s.setNeighborMetric);
  const clearSelection = useStore((s) => s.clearSelection);
  const loadedModel = useStore((s) => s.loadedModel);
  const pushToast = useStore((s) => s.pushToast);
  const [exporting, setExporting] = useState(false);

  if (!detail) {
    return (
      <div className="flex h-full flex-col items-center justify-center px-8 text-center">
        <Logo size={34} className="mb-4 text-faint opacity-50" />
        <p className="text-sm leading-relaxed text-muted">
          Click a point — or search a token — to inspect its metadata and nearest
          neighbors.
        </p>
      </div>
    );
  }

  const d = detail.details;

  const copyToken = async () => {
    try {
      await navigator.clipboard.writeText(d.token);
      pushToast("success", "Token copied to clipboard.");
    } catch {
      pushToast("error", "Clipboard not available.");
    }
  };

  // Export the full record for this token — metadata, neighbors, and the raw
  // embedding vector — as JSON, for downstream analysis.
  const exportJson = async () => {
    if (!loadedModel) return;
    setExporting(true);
    try {
      const full = await api.tokenFull(loadedModel, d.index, 50, neighborMetric, true);
      const payload = {
        model: loadedModel,
        metric: neighborMetric,
        exported_at: new Date().toISOString(),
        ...full,
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const safe = (d.token.trim() || `token_${d.index}`).replace(/[^a-z0-9_-]+/gi, "_");
      a.href = url;
      a.download = `${loadedModel.replace(/[^a-z0-9]+/gi, "_")}_${safe}.json`;
      a.click();
      URL.revokeObjectURL(url);
      pushToast("success", "Exported token JSON.");
    } catch (e) {
      pushToast("error", e instanceof Error ? e.message : "Export failed.");
    } finally {
      setExporting(false);
    }
  };

  return (
    <div className="scroll-thin flex h-full flex-col gap-4 overflow-y-auto p-4">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="panel-header px-0 pt-0">Selected token</div>
          <div className="break-all font-mono text-2xl font-semibold text-paper">
            {d.token.trim() === "" ? JSON.stringify(d.token) : d.token}
          </div>
        </div>
        <button onClick={clearSelection} className="btn-ghost shrink-0 px-2 py-1 text-xs">
          Clear
        </button>
      </div>

      <div className="flex flex-wrap gap-1.5">
        <span
          className="chip"
          style={{ borderColor: `${TOKEN_COLORS[d.type]}55`, color: TOKEN_COLORS[d.type] }}
        >
          {d.type}
        </span>
        {d.is_uppercase && <span className="chip">upper</span>}
        {d.is_digit && <span className="chip">digit</span>}
        {d.has_special_chars && <span className="chip">special</span>}
      </div>

      <div className="flex gap-2">
        <button className="btn-ghost flex-1 px-2 py-1.5 text-xs" onClick={copyToken}>
          Copy token
        </button>
        <button
          className="btn-ghost flex-1 px-2 py-1.5 text-xs"
          onClick={exportJson}
          disabled={exporting}
        >
          {exporting ? "Exporting…" : "Export JSON"}
        </button>
      </div>

      <div className="grid grid-cols-2 gap-2">
        <Stat label="Index" value={d.index} />
        <Stat label="Freq. rank" value={d.frequency_rank.toLocaleString()} />
        <Stat label="Length" value={d.length} />
        <Stat label="norm" value={d.embedding_norm.toFixed(3)} />
      </div>

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="font-mono text-[10px] font-medium uppercase tracking-widest2 text-faint">
            Nearest neighbors
          </span>
          {detailLoading && <span className="font-mono text-[10px] text-faint">updating…</span>}
        </div>
        <SegToggle<DistanceMetric>
          value={neighborMetric}
          onChange={(v) => setNeighborMetric(v)}
          options={[
            { value: "cosine", label: "Cosine" },
            { value: "euclidean", label: "Euclidean" },
          ]}
        />
        <ul className="space-y-0.5">
          {detail.neighbors.map((n, i) => (
            <li key={n.index}>
              <button
                onClick={() => focusOn(n.index)}
                className="group flex w-full items-center gap-2 rounded px-2 py-1.5 text-left hover:bg-white/[0.07]"
              >
                <span className="w-4 text-right font-mono text-xs text-faint">{i + 1}</span>
                <span className="flex-1 truncate font-mono text-sm text-paper">
                  {n.token.trim() === "" ? JSON.stringify(n.token) : n.token}
                </span>
                <span className="font-mono text-xs text-accent-glow">
                  {neighborMetric === "cosine" ? n.similarity.toFixed(3) : n.distance.toFixed(2)}
                </span>
              </button>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
