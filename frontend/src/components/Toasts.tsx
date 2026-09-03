/** Transient bottom-right notifications driven by the store. */

import { useStore } from "../store/useStore";

const STYLES = {
  info: "border-line-strong bg-ink-800 text-paper",
  success: "border-cyan-400/35 bg-ink-800 text-paper",
  error: "border-red-400/40 bg-ink-800 text-paper",
} as const;

export function Toasts() {
  const toasts = useStore((s) => s.toasts);
  const dismiss = useStore((s) => s.dismissToast);

  return (
    <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex w-80 flex-col gap-2">
      {toasts.map((t) => (
        <div
          key={t.id}
          onClick={() => dismiss(t.id)}
          className={`pointer-events-auto animate-fade-in cursor-pointer rounded-xl border px-4 py-3 text-sm shadow-panel ${STYLES[t.type]}`}
        >
          {t.message}
        </div>
      ))}
    </div>
  );
}
