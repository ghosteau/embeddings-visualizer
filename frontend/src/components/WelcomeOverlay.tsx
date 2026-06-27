/**
 * Centered hero shown before the first visualization exists. Doubles as the
 * loading state once a model load is in flight.
 */

import { Logo } from "./Logo";
import { useStore } from "../store/useStore";

export function WelcomeOverlay() {
  const loadState = useStore((s) => s.loadState);
  const vizData = useStore((s) => s.vizData);
  const loadProgress = useStore((s) => s.loadProgress);

  // Once a projection exists, the scene takes over.
  if (vizData) return null;

  const busy = loadState === "loading" || loadState === "visualizing";

  return (
    <div className="pointer-events-none absolute inset-0 z-10 flex flex-col items-center justify-center text-center">
      <div className="max-w-xl px-6">
        <Logo size={52} className="mx-auto mb-6 text-paper" />

        <h1 className="font-display text-5xl font-bold tracking-tight text-paper">
          Embeddings Visualizer
        </h1>
        <div className="mx-auto mt-4 h-px w-24 bg-accent" />
        <p className="mx-auto mt-5 max-w-md text-[15px] leading-relaxed text-muted">
          A spatial reader for how transformer language models represent meaning.
          Choose a model, project its token embeddings, and move through the space.
        </p>

        {busy ? (
          <div className="mt-8 inline-flex items-center gap-3 rounded-md border border-white/10 bg-ink-900/80 px-4 py-2.5">
            <span className="relative block h-3.5 w-3.5">
              <span className="absolute inset-0 animate-ping rounded-full bg-accent/60" />
              <span className="absolute inset-0.5 rounded-full bg-accent" />
            </span>
            <span className="font-mono text-xs text-paper">{loadProgress ?? "Working…"}</span>
          </div>
        ) : (
          <p className="mt-8 font-mono text-[11px] uppercase tracking-widest2 text-faint">
            Load a model to begin &nbsp;←
          </p>
        )}
      </div>
    </div>
  );
}
