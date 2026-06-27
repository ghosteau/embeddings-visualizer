/**
 * Geometry helpers for laying out the projected token cloud.
 *
 * UMAP output lives in an arbitrary coordinate range, so we normalise it into a
 * fixed, centered cube. Computing this once (in the store, when a projection
 * arrives) lets both the point cloud and the camera rig share identical
 * coordinates without recomputing.
 */

const RADIUS = 11;

/** Normalise raw projection coordinates into a centered cube of fixed radius. */
export function normalizeCoordinates(coordinates: number[][]): Float32Array {
  const n = coordinates.length;
  const out = new Float32Array(n * 3);
  if (n === 0) return out;

  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (const c of coordinates) {
    for (let d = 0; d < 3; d++) {
      const v = c[d] ?? 0;
      if (v < min[d]) min[d] = v;
      if (v > max[d]) max[d] = v;
    }
  }
  // Uniform scale across axes preserves the projection's aspect ratio.
  const span = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2]) || 1;
  const scale = (RADIUS * 2) / span;
  const center = [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2];

  for (let i = 0; i < n; i++) {
    out[i * 3] = ((coordinates[i][0] ?? 0) - center[0]) * scale;
    out[i * 3 + 1] = ((coordinates[i][1] ?? 0) - center[1]) * scale;
    out[i * 3 + 2] = ((coordinates[i][2] ?? 0) - center[2]) * scale;
  }
  return out;
}

/** Read a single point's [x, y, z] from a flat normalized positions buffer. */
export function pointAt(positions: Float32Array, i: number): [number, number, number] {
  return [positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]];
}
