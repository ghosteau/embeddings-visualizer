/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        // The navy system mirrors the visual language of mannymcgrail.com.
        ink: {
          950: "#040b14",
          900: "#07121f",
          800: "#0a1928",
          700: "#102235",
          600: "#183149",
          500: "#27465e",
        },
        paper: "#f5f7f8",
        muted: "#9cabb8",
        faint: "#607487",
        line: {
          DEFAULT: "rgba(137, 182, 207, 0.14)",
          strong: "rgba(137, 182, 207, 0.25)",
        },
        // Model-specific accent channels. Runtime values live in modelTheme.ts.
        accent: {
          DEFAULT: "rgb(var(--accent) / <alpha-value>)",
          soft: "rgb(var(--accent-soft) / <alpha-value>)",
          glow: "rgb(var(--accent-glow) / <alpha-value>)",
          ink: "#021019",
        },
        token: {
          word: "#65d4ee",
          number: "#e3b55d",
          special: "#e27da8",
          mixed: "#a99aef",
          unknown: "#71879a",
        },
      },
      fontFamily: {
        display: ["Georgia", "'Times New Roman'", "serif"],
        sans: ["Inter", "ui-sans-serif", "system-ui", "-apple-system", "BlinkMacSystemFont", "'Segoe UI'", "sans-serif"],
        mono: ["'SFMono-Regular'", "Consolas", "'Liberation Mono'", "monospace"],
      },
      letterSpacing: {
        widest2: "0.18em",
      },
      boxShadow: {
        panel: "0 1px 0 rgba(255,255,255,0.025) inset, 0 22px 70px -42px rgba(0,0,0,0.95)",
        accent: "0 0 0 1px rgb(var(--accent) / 0.28), 0 10px 28px -14px rgb(var(--accent) / 0.55)",
      },
      keyframes: {
        "fade-in": {
          "0%": { opacity: "0", transform: "translateY(5px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        sweep: {
          "0%": { transform: "translateX(-110%)" },
          "100%": { transform: "translateX(310%)" },
        },
      },
      animation: {
        "fade-in": "fade-in 180ms ease-out",
        sweep: "sweep 1.2s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
