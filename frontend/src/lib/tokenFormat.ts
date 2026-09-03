/**
 * Render tokenizer output without hiding significant whitespace.
 *
 * Byte-pair tokenizers frequently encode a leading space as part of a token.
 * Collapsing it in HTML makes distinct vocabulary entries look identical, so
 * the interface uses compact visible control characters instead.
 *
 * Substitutes are chosen for font coverage, not for semantic precision. The
 * "correct" glyphs here would be U+2420 SYMBOL FOR SPACE and U+2423 OPEN BOX,
 * but neither ships in the monospace faces this app falls back to (Consolas
 * among them). A missing glyph does not degrade gracefully: U+2420 renders as
 * the literal letters "SP", so " March" displayed as "SPMarch" and read as a
 * typo rather than as a space.
 *
 * Every replacement below was measured against the rendered advance width of
 * the mono stack -- a glyph the face lacks is drawn by a fallback font and so
 * breaks the fixed advance. U+00B7, U+2192 and U+00B6 all render natively;
 * U+2420, U+2423, U+21E5 and U+21B5 do not, and were the earlier choices.
 */
export function formatToken(token: string): string {
  if (token.length === 0) return "<empty>";

  return token
    .replace(/ /g, "·")
    .replace(/\t/g, "→")
    .replace(/\r/g, "\\r")
    .replace(/\n/g, "¶");
}

/** A filename-safe token label for exports. */
export function tokenSlug(token: string, fallback: string): string {
  return (token.trim() || fallback).replace(/[^a-z0-9_-]+/gi, "_");
}
