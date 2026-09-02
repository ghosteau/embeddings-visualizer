/**
 * Render tokenizer output without hiding significant whitespace.
 *
 * Byte-pair tokenizers frequently encode a leading space as part of a token.
 * Collapsing it in HTML makes distinct vocabulary entries look identical, so
 * the interface uses compact visible control characters instead.
 */
export function formatToken(token: string): string {
  if (token.length === 0) return "<empty>";

  return token
    .replace(/ /g, "␠")
    .replace(/\t/g, "⇥")
    .replace(/\r/g, "\\r")
    .replace(/\n/g, "↵");
}

/** A filename-safe token label for exports. */
export function tokenSlug(token: string, fallback: string): string {
  return (token.trim() || fallback).replace(/[^a-z0-9_-]+/gi, "_");
}
