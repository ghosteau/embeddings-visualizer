/** Transient bottom-right notifications driven by the store. */

import { useStore } from "../store/useStore";

const STYLES = {
  info: "border-white/15 bg-ink-700/90",
  success: "border-emerald-400/30 bg-emerald-500/15 text-emerald-100",
  error: "border-rose-400/30 bg-rose-500/15 text-rose-100",
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
          className={`pointer-events-auto animate-fade-in cursor-pointer rounded-xl border px-4 py-3 text-sm shadow-panel backdrop-blur-xl ${STYLES[t.type]}`}
        >
          {t.message}
        </div>
      ))}
    </div>
  );
}
