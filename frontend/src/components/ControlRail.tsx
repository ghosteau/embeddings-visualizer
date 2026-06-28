/** Left control rail: model loading, projection controls, search, comparison. */

import type { ReactNode } from "react";
import { Logo } from "./Logo";
import { ModelPicker } from "./ModelPicker";
import { VizControls } from "./VizControls";
import { SearchPanel } from "./SearchPanel";
import { ComparePanel } from "./ComparePanel";
import { useStore } from "../store/useStore";

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section>
      <div className="panel-header">{title}</div>
      <div className="px-4 pb-4">{children}</div>
    </section>
  );
}

export function ControlRail() {
  const themeLabel = useStore((s) => s.themeLabel);
  const loadedModel = useStore((s) => s.loadedModel);

  return (
    <aside className="panel scroll-thin pointer-events-auto flex max-h-[calc(100vh-2rem)] w-80 flex-col divide-y divide-white/5 overflow-y-auto">
      <div className="flex items-center gap-2.5 px-4 pb-4 pt-4">
        <Logo size={24} className="text-paper" />
        <div className="min-w-0 leading-tight">
          <h1 className="font-display text-[15px] font-bold tracking-tight text-paper">
            Embeddings Visualizer
          </h1>
          {loadedModel && themeLabel ? (
            <p className="flex items-center gap-1.5 font-mono text-[10px] uppercase tracking-widest2 text-faint">
              <span className="inline-block h-2 w-2 rounded-full bg-accent" />
              {themeLabel}
            </p>
          ) : (
            <p className="font-mono text-[10px] uppercase tracking-widest2 text-faint">
              token-space explorer
            </p>
          )}
        </div>
      </div>
      <Section title="Model">
        <ModelPicker />
      </Section>
      <Section title="Projection">
        <VizControls />
      </Section>
      <Section title="Find token">
        <SearchPanel />
      </Section>
      <Section title="Compare">
        <ComparePanel />
      </Section>
    </aside>
  );
}
