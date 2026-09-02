/** Main application shell for the research workbench. */

import { lazy, Suspense, useEffect, useState } from "react";
import { ControlRail } from "./components/ControlRail";
import { DetailPanel } from "./components/DetailPanel";
import { Legend } from "./components/Legend";
import { Toasts } from "./components/Toasts";
import { TopBar } from "./components/TopBar";
import { WelcomeOverlay } from "./components/WelcomeOverlay";
import { useStore } from "./store/useStore";

// Three.js is by far the heaviest part of the client. It is not needed until a
// projection exists, so keep it out of the initial application bundle.
const EmbeddingCanvas = lazy(() =>
  import("./components/scene/EmbeddingCanvas").then((module) => ({
    default: module.EmbeddingCanvas,
  })),
);

export default function App() {
  const init = useStore((s) => s.init);
  const vizData = useStore((s) => s.vizData);
  const tokenDetail = useStore((s) => s.tokenDetail);
  const clearSelection = useStore((s) => s.clearSelection);
  const [controlsOpen, setControlsOpen] = useState(false);

  useEffect(() => {
    init();
  }, [init]);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const active = document.activeElement;
      const typing = active instanceof HTMLInputElement || active instanceof HTMLTextAreaElement;

      if (event.key === "/" && !typing) {
        event.preventDefault();
        document.getElementById("token-search")?.focus();
      } else if (event.key === "Escape") {
        if (typing) (active as HTMLElement).blur();
        else if (controlsOpen) setControlsOpen(false);
        else clearSelection();
      }
    };

    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [clearSelection, controlsOpen]);

  return (
    <div className="app-shell">
      <TopBar
        controlsOpen={controlsOpen}
        onToggleControls={() => setControlsOpen((open) => !open)}
      />

      <main className="workspace">
        <div
          id="control-rail"
          className={`control-drawer ${controlsOpen ? "control-drawer--open" : ""}`}
        >
          <ControlRail onClose={() => setControlsOpen(false)} />
        </div>

        <section className="canvas-stage" aria-label="Embedding projection">
          <div className="canvas-grid" aria-hidden="true" />
          {vizData ? (
            <Suspense fallback={<div className="canvas-loading">Preparing renderer…</div>}>
              <EmbeddingCanvas />
            </Suspense>
          ) : (
            <WelcomeOverlay />
          )}

          <div className="canvas-legend">
            <Legend />
          </div>

          {vizData && (
            <div className="canvas-hint">
              drag / orbit&nbsp;&nbsp;·&nbsp;&nbsp;scroll / zoom&nbsp;&nbsp;·&nbsp;&nbsp;click / inspect
            </div>
          )}
        </section>

        <aside className="inspector-column" aria-label="Token inspector">
          <DetailPanel />
        </aside>
      </main>

      {tokenDetail && (
        <aside className="mobile-inspector" aria-label="Selected token details">
          <DetailPanel />
        </aside>
      )}

      {controlsOpen && (
        <button
          className="drawer-scrim"
          aria-label="Close controls"
          onClick={() => setControlsOpen(false)}
        />
      )}

      <Toasts />
    </div>
  );
}
