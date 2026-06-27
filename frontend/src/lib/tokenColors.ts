/**
 * Color mapping for token types, shared by the 3D scene and the legend/UI.
 * Values are kept in sync with the `token.*` palette in tailwind.config.js.
 */

import { Color } from "three";
import type { TokenType } from "./types";

export const TOKEN_COLORS: Record<TokenType, string> = {
  word: "#41c7f0",
  number: "#f0b429",
  special: "#ee6fb0",
  mixed: "#9b8cf0",
  unknown: "#6b7280",
};

export const TOKEN_TYPE_LABELS: Record<TokenType, string> = {
  word: "Words",
  number: "Numbers",
  special: "Special",
  mixed: "Mixed",
  unknown: "Unknown",
};

/** Pre-built THREE.Color instances, reused to avoid per-frame allocation. */
export const TOKEN_THREE_COLORS: Record<TokenType, Color> = Object.fromEntries(
  Object.entries(TOKEN_COLORS).map(([k, v]) => [k, new Color(v)]),
) as Record<TokenType, Color>;
