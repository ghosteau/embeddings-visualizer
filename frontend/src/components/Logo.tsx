/**
 * Custom wordmark glyph — a small "constellation" of embedding points with a
 * connected nearest-neighbor edge. Hand-built SVG (no stock/emoji art) so the
 * brand reads as bespoke rather than generic.
 */

export function Logo({ className = "", size = 22 }: { className?: string; size?: number }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 32 32"
      fill="none"
      className={className}
      aria-hidden="true"
    >
      {/* neighbor edges */}
      <path
        d="M16 16 L7 8 M16 16 L25 7 M16 16 L9 25 M16 16 L26 22"
        stroke="currentColor"
        strokeWidth="1"
        strokeOpacity="0.35"
        strokeLinecap="round"
      />
      {/* satellite points (token-type hues) */}
      <circle cx="7" cy="8" r="2" fill="#41c7f0" />
      <circle cx="25" cy="7" r="2" fill="#f0b429" />
      <circle cx="9" cy="25" r="2" fill="#ee6fb0" />
      <circle cx="26" cy="22" r="2" fill="#9b8cf0" />
      {/* central selected node */}
      <circle cx="16" cy="16" r="3.4" fill="#d8623a" />
      <circle cx="16" cy="16" r="3.4" stroke="#f0a184" strokeWidth="1" />
    </svg>
  );
}
