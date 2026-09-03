/** Projection legend and client-side dataset export. */

import { useStore } from "../store/useStore";
import { TOKEN_COLORS, TOKEN_TYPE_LABELS } from "../lib/tokenColors";
import type { TokenType } from "../lib/types";

function csvCell(value: string | number | boolean): string {
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function Legend() {
  const vizData = useStore((s) => s.vizData);
  const statistics = useStore((s) => s.statistics);
  const displayCount = useStore((s) => s.displayCount);
  const loadedModel = useStore((s) => s.loadedModel);
  if (!vizData) return null;

  const vocab = statistics?.model_info?.vocabulary_size as number | undefined;
  const distribution = vizData.statistics.type_distribution;
  const types = (Object.keys(TOKEN_COLORS) as TokenType[]).filter(
    (type) => (distribution[type] ?? 0) > 0,
  );

  const exportProjection = () => {
    const header = ["token", "analysis_index", "vocabulary_id", "type", "x", "y", "z", "embedding_norm"];
    const rows = vizData.tokens.map((token, index) => {
      const coordinate = vizData.coordinates[index];
      return [
        token,
        index,
        vizData.metadata.frequency_rank[index],
        vizData.metadata.types[index],
        coordinate[0],
        coordinate[1],
        coordinate[2] ?? 0,
        vizData.metadata.embedding_norm[index],
      ].map(csvCell).join(",");
    });

    const blob = new Blob([[header.join(","), ...rows].join("\n")], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${(loadedModel ?? "model").replace(/[^a-z0-9]+/gi, "_")}_umap.csv`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="legend-panel">
      <div className="mb-3 flex items-start justify-between gap-4">
        <div>
          <div className="eyebrow">PROJECTION / ACTIVE</div>
          <div className="mt-1 font-mono text-[11px] text-muted">
            {displayCount.toLocaleString()} visible / {vizData.tokens.length.toLocaleString()} analyzed
          </div>
        </div>
        <button className="text-link text-[11px]" onClick={exportProjection}>
          Export CSV
        </button>
      </div>

      <ul className="grid grid-cols-2 gap-x-4 gap-y-1.5">
        {types.map((type) => (
          <li key={type} className="flex items-center justify-between gap-3">
            <span className="flex min-w-0 items-center gap-2">
              <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: TOKEN_COLORS[type] }} />
              <span className="truncate text-[11px] text-muted">{TOKEN_TYPE_LABELS[type]}</span>
            </span>
            <span className="font-mono text-[10px] text-faint">{distribution[type].toLocaleString()}</span>
          </li>
        ))}
      </ul>

      <div className="mt-3 border-t border-line pt-2 font-mono text-[10px] leading-relaxed text-faint">
        {vocab ? `${vocab.toLocaleString()} vocabulary / ` : ""}
        {vizData.statistics.original_dimension}D → {vizData.statistics.reduced_dimension}D / UMAP {vizData.config.metric}
      </div>
    </div>
  );
}
