/** Left control rail: model loading, projection controls, search, comparison. */

import type { ReactNode } from "react";
import { Logo } from "./Logo";
import { ModelPicker } from "./ModelPicker";
import { VizControls } from "./VizControls";
import { SearchPanel } from "./SearchPanel";
import { ComparePanel } from "./ComparePanel";

function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section>
      <div className="panel-header">{title}</div>
      <div className="px-4 pb-4">{children}</div>
    </section>
  );
}

export function ControlRail() {
  return (
    <aside className="panel scroll-thin pointer-events-auto flex max-h-[calc(100vh-2rem)] w-80 flex-col divide-y divide-white/5 overflow-y-auto">
      <div className="flex items-center gap-2.5 px-4 pb-4 pt-4">
        <Logo size={24} className="text-paper" />
        <div className="leading-tight">
          <h1 className="font-display text-[15px] font-bold tracking-tight text-paper">
            Embeddings Visualizer
          </h1>
          <p className="font-mono text-[10px] uppercase tracking-widest2 text-faint">
            token-space explorer
          </p>
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
