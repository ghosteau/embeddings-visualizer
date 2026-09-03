/**
 * Color mapping for token types, shared by the 3D scene and the legend/UI.
 * Values are kept in sync with the `token.*` palette in tailwind.config.js.
 */

import type { TokenType } from "./types";

export const TOKEN_COLORS: Record<TokenType, string> = {
  word: "#65d4ee",
  number: "#e3b55d",
  special: "#e27da8",
  mixed: "#a99aef",
  unknown: "#71879a",
};

export const TOKEN_TYPE_LABELS: Record<TokenType, string> = {
  word: "Words",
  number: "Numbers",
  special: "Special",
  mixed: "Mixed",
  unknown: "Unknown",
};
