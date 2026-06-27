/**
 * Token search. Searches the entire loaded token set (not just the visible
 * points), so any capped/hidden token can still be found, inspected, and flown
 * to in the scene.
 */

import { useStore } from "../store/useStore";

export function SearchPanel() {
  const searchQuery = useStore((s) => s.searchQuery);
  const searchResults = useStore((s) => s.searchResults);
  const runSearch = useStore((s) => s.runSearch);
  const focusOn = useStore((s) => s.focusOn);
  const loadedModel = useStore((s) => s.loadedModel);

  return (
    <div className="space-y-2">
      <input
        className="input font-mono"
        placeholder="Search any token…"
        value={searchQuery}
        disabled={!loadedModel}
        onChange={(e) => runSearch(e.target.value)}
      />
      {searchResults.length > 0 && (
        <ul className="scroll-thin max-h-44 space-y-0.5 overflow-y-auto pr-1">
          {searchResults.map((r) => (
            <li key={r.index}>
              <button
                onClick={() => focusOn(r.index)}
                className="flex w-full items-center justify-between rounded px-2.5 py-1.5 text-left text-sm
                           hover:bg-white/[0.07]"
              >
                <span className="truncate font-mono text-paper">{display(r.token)}</span>
                {r.match_type === "exact" && (
                  <span className="chip border-accent/40 text-accent-glow">exact</span>
                )}
              </button>
            </li>
          ))}
        </ul>
      )}
      {loadedModel && searchQuery && searchResults.length === 0 && (
        <p className="px-1 font-mono text-[11px] text-faint">No tokens match.</p>
      )}
    </div>
  );
}

function display(token: string): string {
  return token.trim() === "" ? JSON.stringify(token) : token;
}
