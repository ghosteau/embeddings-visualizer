/**
 * Right-hand inspector for the selected token: metadata, a neighbor list
 * (click to fly to it), and a metric toggle. Empty state nudges the user to
 * pick or search a token when nothing is selected.
 */

import { useState } from "react";
import { useStore } from "../store/useStore";
import { api } from "../lib/api";
import { formatToken, tokenSlug } from "../lib/tokenFormat";
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
      <div className="flex h-full flex-col">
        <div className="inspector-heading">
          <span className="eyebrow">INSPECTOR / IDLE</span>
          <h2 className="mt-2 font-display text-2xl text-paper">Read the local structure.</h2>
        </div>
        <div className="flex flex-1 flex-col items-center justify-center px-8 text-center">
          <Logo size={32} className="mb-5 text-faint opacity-45" />
          <p className="max-w-[240px] text-sm leading-relaxed text-muted">
            Select a point or search a token to inspect its vocabulary metadata and nearest
            neighbors.
          </p>
          <div className="mt-6 grid w-full max-w-[240px] grid-cols-2 gap-2 text-left">
            <Stat label="Select" value="point" />
            <Stat label="Search" value="/ key" />
          </div>
        </div>
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
      const safe = tokenSlug(d.token, `token_${d.index}`);
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
    <div className="scroll-thin flex h-full flex-col gap-5 overflow-y-auto p-5">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="eyebrow">TOKEN / {String(d.index).padStart(4, "0")}</div>
          <div className="mt-2 break-all font-mono text-2xl font-semibold text-paper">
            {formatToken(d.token)}
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
        <Stat label="Analysis index" value={d.index} />
        <Stat label="Vocabulary id" value={d.frequency_rank.toLocaleString()} />
        <Stat label="Length" value={d.length} />
        <Stat label="L2 norm" value={d.embedding_norm.toFixed(3)} />
      </div>

      {d.x != null && d.y != null && (
        <div>
          <div className="eyebrow mb-2">ACTIVE PROJECTION</div>
          <div className="grid grid-cols-3 gap-2">
            <Stat label="X" value={d.x.toFixed(2)} />
            <Stat label="Y" value={d.y.toFixed(2)} />
            <Stat label="Z" value={(d.z ?? 0).toFixed(2)} />
          </div>
        </div>
      )}

      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <span className="eyebrow">NEAREST NEIGHBORS</span>
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
                className="result-row group"
              >
                <span className="w-4 text-right font-mono text-xs text-faint">{i + 1}</span>
                <span className="flex-1 truncate font-mono text-sm text-paper">
                  {formatToken(n.token)}
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
