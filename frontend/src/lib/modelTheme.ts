/**
 * Per-model accent theming.
 *
 * Each model family gets a signature accent (loosely brand-inspired) that
 * re-themes the whole UI — buttons, sliders, the wordmark tick, the ambient
 * background glow, and the in-scene neighbor halos. Colors are chosen to stay
 * clear of the token-type palette (cyan/amber/pink/violet) so chrome and data
 * never blur together.
 */

export interface ModelTheme {
  /** Accent channels as "R G B" strings, fed to the CSS variables. */
  accent: string;
  accentSoft: string;
  accentGlow: string;
  /** Hex of the glow accent, used directly by the 3D scene (neighbor halos). */
  glowHex: string;
  /** Short label shown in the UI to explain the current tint. */
  label: string;
}

const THEMES: Record<string, ModelTheme> = {
  // Blue for the GPT / OpenAI family.
  gpt: {
    accent: "59 130 246",
    accentSoft: "96 165 250",
    accentGlow: "147 197 253",
    glowHex: "#93c5fd",
    label: "OpenAI / GPT",
  },
  // Google blue for the BERT family.
  bert: {
    accent: "66 133 244",
    accentSoft: "104 161 247",
    accentGlow: "150 193 252",
    glowHex: "#96c1fc",
    label: "Google / BERT",
  },
  // Meta-leaning violet for RoBERTa.
  roberta: {
    accent: "124 97 255",
    accentSoft: "150 128 255",
    accentGlow: "181 165 255",
    glowHex: "#b5a5ff",
    label: "Meta / RoBERTa",
  },
  // Signature terracotta fallback for everything else (incl. custom models).
  default: {
    accent: "216 98 58",
    accentSoft: "232 127 90",
    accentGlow: "240 161 132",
    glowHex: "#f0a184",
    label: "Custom",
  },
};

/** Pick a theme from a model's family label and/or its Hugging Face id. */
export function themeForModel(modelId: string, family?: string): ModelTheme {
  const key = `${family ?? ""} ${modelId}`.toLowerCase();
  // Order matters: "roberta" contains the substring "bert", so test it first.
  if (key.includes("roberta")) return THEMES.roberta;
  if (key.includes("gpt")) return THEMES.gpt;
  if (key.includes("bert")) return THEMES.bert;
  return THEMES.default;
}

/** Apply a theme's accent to the document via CSS variables. */
export function applyModelTheme(theme: ModelTheme): void {
  const root = document.documentElement;
  root.style.setProperty("--accent", theme.accent);
  root.style.setProperty("--accent-soft", theme.accentSoft);
  root.style.setProperty("--accent-glow", theme.accentGlow);
}
