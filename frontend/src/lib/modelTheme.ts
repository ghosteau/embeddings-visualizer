/**
 * Per-model accent theming.
 *
 * Each family gets a restrained signature accent. The navy application shell
 * stays consistent with the parent portfolio while controls and selected
 * neighbors pick up the active model's identity.
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
  // Warm gold differentiates bidirectional BERT models from GPT models.
  bert: {
    accent: "209 162 76",
    accentSoft: "229 188 105",
    accentGlow: "240 210 151",
    glowHex: "#f0d297",
    label: "BERT family",
  },
  // Meta-leaning violet for RoBERTa.
  roberta: {
    accent: "124 97 255",
    accentSoft: "150 128 255",
    accentGlow: "181 165 255",
    glowHex: "#b5a5ff",
    label: "Meta / RoBERTa",
  },
  // Portfolio cyan is the neutral/default identity for custom architectures.
  default: {
    accent: "82 198 229",
    accentSoft: "121 216 239",
    accentGlow: "154 229 246",
    glowHex: "#9ae5f6",
    label: "Custom model",
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
