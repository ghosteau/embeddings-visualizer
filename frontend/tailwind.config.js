/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // Warm-neutral "observatory at night" surfaces — not the usual cool
        // blue-black. Slightly warm so the off-white text reads like paper.
        ink: {
          950: "#070708",
          900: "#0b0b0d",
          800: "#121214",
          700: "#1a1a1e",
          600: "#26262c",
          500: "#34343c",
        },
        // Off-white primary text + muted greys (warm-tinted).
        paper: "#ece8e1",
        muted: "#9b958c",
        faint: "#6a655e",
        // Signature accent: terracotta/clay. Deliberately chosen to sit apart
        // from the token-type palette (cyan/amber/pink/violet) so UI chrome and
        // data are never confused.
        accent: {
          DEFAULT: "#d8623a",
          soft: "#e87f5a",
          glow: "#f0a184",
          ink: "#1a0e08", // text color to sit on top of the accent fill
        },
        // Token-type colors, kept in sync with src/lib/tokenColors.ts.
        token: {
          word: "#41c7f0",
          number: "#f0b429",
          special: "#ee6fb0",
          mixed: "#9b8cf0",
          unknown: "#6b7280",
        },
      },
      fontFamily: {
        // Distinctive geometric display face for the wordmark and headings.
        display: ["'Space Grotesk'", "system-ui", "sans-serif"],
        sans: ["'IBM Plex Sans'", "system-ui", "sans-serif"],
        mono: ["'IBM Plex Mono'", "ui-monospace", "monospace"],
      },
      letterSpacing: {
        widest2: "0.22em",
      },
      boxShadow: {
        // Crisp, low-spread shadows + a hairline top highlight for "panel" depth.
        panel: "0 1px 0 0 rgba(255,255,255,0.04) inset, 0 16px 40px -24px rgba(0,0,0,0.9)",
        accent: "0 0 0 1px rgba(216,98,58,0.4), 0 6px 20px -8px rgba(216,98,58,0.5)",
      },
      keyframes: {
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(4px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        sweep: {
          "0%": { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(300%)" },
        },
      },
      animation: {
        "fade-in": "fade-in 0.25s ease-out",
        sweep: "sweep 1.3s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
