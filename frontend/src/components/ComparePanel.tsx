/**
 * Compare two specific tokens by name. Because the point cloud is capped, this
 * (and search) is how users reach any token in the loaded set: type two tokens,
 * see their cosine similarity + euclidean distance, then click either result to
 * fly to and inspect it.
 */

import { useState } from "react";
import { formatToken } from "../lib/tokenFormat";
import { useStore } from "../store/useStore";

export function ComparePanel() {
  const [a, setA] = useState("man");
  const [b, setB] = useState("woman");
  const comparison = useStore((s) => s.comparison);
  const comparisonError = useStore((s) => s.comparisonError);
  const compareTokens = useStore((s) => s.compareTokens);
  const focusOn = useStore((s) => s.focusOn);
  const loadedModel = useStore((s) => s.loadedModel);

  const submit = () => {
    if (loadedModel && a && b) compareTokens(a, b);
  };

  // Map cosine similarity (-1..1) to a 0..1 bar fill.
  const fill = comparison ? Math.max(0, Math.min(1, (comparison.cosine_similarity + 1) / 2)) : 0;

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2">
        <input
          className="input font-mono"
          value={a}
          onChange={(e) => setA(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder="token A"
        />
        <input
          className="input font-mono"
          value={b}
          onChange={(e) => setB(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && submit()}
          placeholder="token B"
        />
      </div>
      <button className="btn-ghost w-full" disabled={!loadedModel || !a || !b} onClick={submit}>
        Run comparison
      </button>

      {comparisonError && <p className="error-copy">{comparisonError}</p>}

      {comparison && (
        <div className="comparison-card animate-fade-in space-y-3">
          {/* Clickable tokens → inspect in the scene. */}
          <div className="flex items-center justify-between gap-2">
            <button
              onClick={() => focusOn(comparison.token1_index)}
              className="chip max-w-[45%] truncate hover:border-accent/50 hover:text-accent-glow"
              title="Inspect this token"
            >
              {formatToken(comparison.token1)}
            </button>
            <span className="font-mono text-[10px] text-faint">vs</span>
            <button
              onClick={() => focusOn(comparison.token2_index)}
              className="chip max-w-[45%] truncate hover:border-accent/50 hover:text-accent-glow"
              title="Inspect this token"
            >
              {formatToken(comparison.token2)}
            </button>
          </div>

          <div className="space-y-1">
            <div className="flex items-center justify-between text-sm">
              <span className="text-muted">cosine similarity</span>
              <span className="font-mono text-accent-glow">
                {comparison.cosine_similarity.toFixed(4)}
              </span>
            </div>
            <div className="h-1 overflow-hidden rounded-full bg-white/10">
              <div className="h-full rounded-full bg-accent" style={{ width: `${fill * 100}%` }} />
            </div>
          </div>

          <div className="flex items-center justify-between text-sm">
            <span className="text-muted">euclidean distance</span>
            <span className="font-mono text-paper">{comparison.euclidean_distance.toFixed(3)}</span>
          </div>
        </div>
      )}

      {!comparison && !comparisonError && (
        <p className="help-copy">
          Values are calculated from the original embedding vectors, not from the UMAP projection.
        </p>
      )}
    </div>
  );
}
