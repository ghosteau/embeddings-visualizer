import { useStore } from "../store/useStore";
import { Logo } from "./Logo";

interface TopBarProps {
  controlsOpen: boolean;
  onToggleControls: () => void;
}

export function TopBar({ controlsOpen, onToggleControls }: TopBarProps) {
  const loadedModel = useStore((s) => s.loadedModel);
  const loadState = useStore((s) => s.loadState);
  const vizData = useStore((s) => s.vizData);
  const themeLabel = useStore((s) => s.themeLabel);

  const status =
    loadState === "loading"
      ? "loading model"
      : loadState === "visualizing"
        ? "computing projection"
        : loadState === "ready"
          ? "ready"
          : loadState === "error"
            ? "attention needed"
            : "standby";

  return (
    <header className="topbar">
      <div className="flex min-w-0 items-center gap-3">
        <div className="brand-mark" aria-hidden="true">
          <Logo size={22} />
        </div>
        <div className="min-w-0">
          <div className="eyebrow">EM / RESEARCH TOOL</div>
          <div className="truncate font-display text-[17px] font-semibold leading-tight text-paper">
            Embeddings Visualizer
          </div>
        </div>
      </div>

      <div className="hidden min-w-0 flex-1 items-center justify-center gap-6 md:flex">
        <div className="status-cluster">
          <span className={`status-dot status-dot--${loadState}`} />
          <span>{status}</span>
        </div>
        {loadedModel && (
          <div className="min-w-0 truncate font-mono text-[11px] text-muted">
            {loadedModel}
            {themeLabel ? ` / ${themeLabel}` : ""}
            {vizData ? ` / ${vizData.tokens.length.toLocaleString()} tokens` : ""}
          </div>
        )}
      </div>

      <div className="flex items-center gap-2">
        <button
          type="button"
          className="btn-ghost px-3 py-1.5 text-xs lg:hidden"
          aria-expanded={controlsOpen}
          aria-controls="control-rail"
          onClick={onToggleControls}
        >
          {controlsOpen ? "Close controls" : "Controls"}
        </button>
        <a
          className="portfolio-link hidden sm:inline-flex"
          href="https://mannymcgrail.com"
          target="_blank"
          rel="noreferrer"
        >
          mannymcgrail.com <span aria-hidden="true">↗</span>
        </a>
      </div>
    </header>
  );
}
