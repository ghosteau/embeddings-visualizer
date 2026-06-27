/**
 * Application shell.
 *
 * A full-bleed WebGL canvas sits behind a layer of floating glass panels. The
 * overlay layer is pointer-events-none so the 3D scene stays fully interactive,
 * while individual panels re-enable pointer events for their own controls.
 */

import { useEffect } from "react";
import { EmbeddingCanvas } from "./components/scene/EmbeddingCanvas";
import { ControlRail } from "./components/ControlRail";
import { DetailPanel } from "./components/DetailPanel";
import { Legend } from "./components/Legend";
import { Toasts } from "./components/Toasts";
import { WelcomeOverlay } from "./components/WelcomeOverlay";
import { useStore } from "./store/useStore";

export default function App() {
  const init = useStore((s) => s.init);
  const vizData = useStore((s) => s.vizData);

  useEffect(() => {
    init();
  }, [init]);

  return (
    <div className="relative h-screen w-screen overflow-hidden">
      {/* Background WebGL scene. */}
      <div className="absolute inset-0">
        <EmbeddingCanvas />
      </div>

      <WelcomeOverlay />

      {/* Floating UI layer. */}
      <div className="pointer-events-none absolute inset-0 flex justify-between gap-4 p-4">
        <ControlRail />

        {vizData && (
          <div className="panel pointer-events-auto hidden h-[calc(100vh-2rem)] w-80 shrink-0 lg:block">
            <DetailPanel />
          </div>
        )}
      </div>

      {/* Bottom-left legend / stats. */}
      <div className="pointer-events-none absolute bottom-4 left-4">
        <Legend />
      </div>

      {/* Controls hint, bottom center. */}
      {vizData && (
        <div className="pointer-events-none absolute bottom-4 left-1/2 -translate-x-1/2 font-mono text-[11px] text-faint">
          drag to orbit · scroll to zoom · click a point to inspect · click empty space to deselect
        </div>
      )}

      <Toasts />
    </div>
  );
}
