/** Debounced token lookup across the complete analyzed vocabulary subset. */

import { useEffect, useState } from "react";
import { formatToken } from "../lib/tokenFormat";
import { useStore } from "../store/useStore";

export function SearchPanel() {
  const searchResults = useStore((s) => s.searchResults);
  const runSearch = useStore((s) => s.runSearch);
  const focusOn = useStore((s) => s.focusOn);
  const loadedModel = useStore((s) => s.loadedModel);
  const [query, setQuery] = useState("");

  useEffect(() => {
    setQuery("");
    runSearch("");
  }, [loadedModel, runSearch]);

  useEffect(() => {
    const timer = window.setTimeout(() => runSearch(query), 180);
    return () => window.clearTimeout(timer);
  }, [query, runSearch]);

  return (
    <div className="space-y-2">
      <div className="relative">
        <input
          id="token-search"
          className="input pr-10 font-mono"
          placeholder="Search token text"
          value={query}
          disabled={!loadedModel}
          autoComplete="off"
          spellCheck={false}
          onChange={(event) => setQuery(event.target.value)}
        />
        <kbd className="key-hint">/</kbd>
      </div>

      {searchResults.length > 0 && (
        <ul className="scroll-thin max-h-48 space-y-0.5 overflow-y-auto pr-1">
          {searchResults.map((result) => (
            <li key={result.index}>
              <button
                onClick={() => focusOn(result.index)}
                className="result-row"
                title={`Vocabulary index ${result.index}`}
              >
                <span className="min-w-0 truncate font-mono text-paper">
                  {formatToken(result.token)}
                </span>
                <span className="flex shrink-0 items-center gap-2">
                  {result.match_type === "exact" && <span className="exact-mark">exact</span>}
                  <span className="font-mono text-[10px] text-faint">#{result.index}</span>
                </span>
              </button>
            </li>
          ))}
        </ul>
      )}

      {loadedModel && query.trim() && searchResults.length === 0 && (
        <p className="help-copy px-1">No analyzed token matches this text.</p>
      )}
      {!loadedModel && <p className="help-copy">Load a model to query its token vocabulary.</p>}
    </div>
  );
}
