/** Small reusable presentational primitives for the control panels. */

import type { ReactNode } from "react";

export function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <label className="block space-y-1.5">
      <div className="flex items-baseline justify-between">
        <span className="text-xs font-medium text-muted">{label}</span>
        {hint && <span className="font-mono text-xs text-faint">{hint}</span>}
      </div>
      {children}
    </label>
  );
}

export function Slider({
  value,
  min,
  max,
  step,
  onChange,
  disabled,
}: {
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  disabled?: boolean;
}) {
  return (
    <input
      type="range"
      value={value}
      min={min}
      max={max}
      step={step}
      disabled={disabled}
      onChange={(e) => onChange(Number(e.target.value))}
      className="h-1.5 w-full cursor-pointer appearance-none rounded-full bg-white/10
                 accent-accent disabled:cursor-not-allowed disabled:opacity-40"
    />
  );
}

export function SegToggle<T extends string>({
  options,
  value,
  onChange,
}: {
  options: { value: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
}) {
  return (
    <div className="flex rounded-lg border border-white/10 bg-ink-900/60 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={`flex-1 rounded px-2 py-1 text-xs font-medium transition ${
            value === o.value
              ? "bg-accent font-semibold text-accent-ink"
              : "text-muted hover:text-paper"
          }`}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Stat({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="rounded-lg border border-white/5 bg-white/5 px-3 py-2">
      <div className="font-mono text-[10px] uppercase tracking-wider text-faint">{label}</div>
      <div className="font-mono text-sm text-paper">{value}</div>
    </div>
  );
}
