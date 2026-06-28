/** Floating legend + at-a-glance projection statistics (bottom-left overlay). */

import { useStore } from "../store/useStore";
import { TOKEN_COLORS, TOKEN_TYPE_LABELS } from "../lib/tokenColors";
import type { TokenType } from "../lib/types";

export function Legend() {
  const vizData = useStore((s) => s.vizData);
  const statistics = useStore((s) => s.statistics);
  if (!vizData) return null;

  const vocab = statistics?.model_info?.vocabulary_size as number | undefined;

  const dist = vizData.statistics.type_distribution;
  const types = (Object.keys(TOKEN_COLORS) as TokenType[]).filter((t) => (dist[t] ?? 0) > 0);

  return (
    <div className="panel pointer-events-auto w-56 p-3 text-xs">
      <div className="mb-2 flex items-center justify-between">
        <span className="font-mono font-medium uppercase tracking-widest2 text-faint">Token types</span>
        <span className="font-mono text-muted">{vizData.tokens.length.toLocaleString()} pts</span>
      </div>
      <ul className="space-y-1.5">
        {types.map((t) => (
          <li key={t} className="flex items-center justify-between">
            <span className="flex items-center gap-2">
              <span className="h-2.5 w-2.5 rounded-full" style={{ background: TOKEN_COLORS[t], boxShadow: `0 0 8px ${TOKEN_COLORS[t]}` }} />
              <span className="text-muted">{TOKEN_TYPE_LABELS[t]}</span>
            </span>
            <span className="font-mono text-faint">{dist[t].toLocaleString()}</span>
          </li>
        ))}
      </ul>
      <div className="mt-3 border-t border-white/10 pt-2 font-mono text-[11px] text-faint">
        {vocab ? `${vocab.toLocaleString()} vocab · ` : ""}
        {vizData.statistics.original_dimension}D → {vizData.statistics.reduced_dimension}D · UMAP/
        {vizData.config.metric}
      </div>
    </div>
  );
}
