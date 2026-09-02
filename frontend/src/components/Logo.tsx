/**
 * Compact embedding-neighborhood mark used as the product identifier.
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
      <path
        d="M16 16 L7 8 M16 16 L25 7 M16 16 L9 25 M16 16 L26 22"
        stroke="currentColor"
        strokeWidth="1"
        strokeOpacity="0.35"
        strokeLinecap="round"
      />
      <circle cx="7" cy="8" r="2" fill="#65d4ee" />
      <circle cx="25" cy="7" r="2" fill="#e3b55d" />
      <circle cx="9" cy="25" r="2" fill="#e27da8" />
      <circle cx="26" cy="22" r="2" fill="#a99aef" />
      <circle cx="16" cy="16" r="3.4" fill="rgb(var(--accent))" />
      <circle cx="16" cy="16" r="3.4" stroke="rgb(var(--accent-glow))" strokeWidth="1" />
    </svg>
  );
}
