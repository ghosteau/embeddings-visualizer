/** Controls for model loading, projection, search, and pairwise analysis. */

import type { ReactNode } from "react";
import { ComparePanel } from "./ComparePanel";
import { ModelPicker } from "./ModelPicker";
import { SearchPanel } from "./SearchPanel";
import { VizControls } from "./VizControls";

function Section({ index, title, children }: { index: string; title: string; children: ReactNode }) {
  return (
    <section className="rail-section">
      <div className="panel-header">
        <span>{index}</span>
        <span>/</span>
        <span>{title}</span>
      </div>
      <div className="px-4 pb-5">{children}</div>
    </section>
  );
}

export function ControlRail({ onClose }: { onClose?: () => void }) {
  return (
    <aside className="control-rail">
      <div className="flex items-start justify-between gap-4 border-b border-line px-4 py-4 lg:hidden">
        <div>
          <div className="eyebrow">CONTROL SURFACE</div>
          <p className="mt-1 font-display text-xl text-paper">Configure the atlas.</p>
        </div>
        <button type="button" className="btn-ghost px-2.5 py-1.5 text-xs" onClick={onClose}>
          Close
        </button>
      </div>

      <Section index="01" title="Model source">
        <ModelPicker />
      </Section>
      <Section index="02" title="Projection">
        <VizControls />
      </Section>
      <Section index="03" title="Token lookup">
        <SearchPanel />
      </Section>
      <Section index="04" title="Pairwise comparison">
        <ComparePanel />
      </Section>
    </aside>
  );
}
