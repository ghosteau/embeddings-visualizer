/**
 * The interactive token point cloud.
 *
 * Renders the visible tokens as a single GPU-friendly THREE.Points object (one
 * draw call). Normalised coordinates are shared via the store so the camera rig
 * uses identical positions. Hover/click use r3f's points raycasting
 * (`event.index`). The current selection and its neighbors are emphasised with
 * an overlay that reads from the *full* position buffer — so a token can be
 * highlighted even when it sits outside the visible-points cutoff.
 */

import { useMemo } from "react";
import { AdditiveBlending, CanvasTexture, Color } from "three";
import { Html } from "@react-three/drei";
import { useStore } from "../../store/useStore";
import { pointAt } from "../../lib/layout";
import { TOKEN_THREE_COLORS } from "../../lib/tokenColors";
import type { VisualizationData } from "../../lib/types";

/** Build a soft radial sprite so points render as glowing discs, not squares. */
function useSpriteTexture(): CanvasTexture {
  return useMemo(() => {
    const size = 64;
    const canvas = document.createElement("canvas");
    canvas.width = canvas.height = size;
    const ctx = canvas.getContext("2d")!;
    const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
    g.addColorStop(0, "rgba(255,255,255,1)");
    g.addColorStop(0.4, "rgba(255,255,255,0.85)");
    g.addColorStop(1, "rgba(255,255,255,0)");
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, size, size);
    return new CanvasTexture(canvas);
  }, []);
}

const DIM = new Color("#0b0b0d");

export function PointCloud({ data }: { data: VisualizationData }) {
  const sprite = useSpriteTexture();

  const positions = useStore((s) => s.positions);
  const displayCount = useStore((s) => s.displayCount);
  const selectedIndex = useStore((s) => s.selectedIndex);
  const hoveredIndex = useStore((s) => s.hoveredIndex);
  const neighborIndices = useStore((s) => s.neighborIndices);
  const selectToken = useStore((s) => s.selectToken);
  const setHovered = useStore((s) => s.setHovered);

  const count = Math.min(displayCount || data.tokens.length, data.tokens.length);

  // Visible slice of positions (the base cloud). Sliced rather than reusing the
  // full buffer so raycasting only hits points the user can actually see.
  const visiblePositions = useMemo(
    () => (positions ? positions.slice(0, count * 3) : new Float32Array(0)),
    [positions, count],
  );

  // Per-point colors; unrelated points fade toward the background when a token
  // is selected, drawing the eye to the selection and its neighborhood.
  const colors = useMemo(() => {
    const arr = new Float32Array(count * 3);
    const hasFocus = selectedIndex != null;
    const neighborSet = new Set(neighborIndices);
    const tmp = new Color();
    for (let i = 0; i < count; i++) {
      const base = TOKEN_THREE_COLORS[data.metadata.types[i]] ?? TOKEN_THREE_COLORS.unknown;
      tmp.copy(base);
      if (hasFocus && i !== selectedIndex && !neighborSet.has(i)) {
        tmp.lerp(DIM, 0.84);
      }
      arr[i * 3] = tmp.r;
      arr[i * 3 + 1] = tmp.g;
      arr[i * 3 + 2] = tmp.b;
    }
    return arr;
  }, [data, count, selectedIndex, neighborIndices]);

  if (!positions) return null;

  const posOf = (i: number) => pointAt(positions, i);

  return (
    <group>
      <points
        onPointerMove={(e) => {
          e.stopPropagation();
          if (e.index != null) setHovered(e.index);
        }}
        onPointerOut={() => setHovered(null)}
        onClick={(e) => {
          e.stopPropagation();
          if (e.index != null) selectToken(e.index);
        }}
      >
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[visiblePositions, 3]} />
          <bufferAttribute attach="attributes-color" args={[colors, 3]} />
        </bufferGeometry>
        <pointsMaterial
          size={0.42}
          map={sprite}
          vertexColors
          transparent
          alphaTest={0.02}
          depthWrite={false}
          sizeAttenuation
          blending={AdditiveBlending}
        />
      </points>

      {/* Neighbor halos: warm accent discs, drawn from the full position set. */}
      {neighborIndices.map((i) => (
        <mesh key={`nb-${i}`} position={posOf(i)}>
          <sphereGeometry args={[0.24, 12, 12]} />
          <meshBasicMaterial color="#f0a184" transparent opacity={0.55} />
        </mesh>
      ))}

      {/* Selected token: bright marker + persistent label. */}
      {selectedIndex != null && (
        <group position={posOf(selectedIndex)}>
          <mesh>
            <sphereGeometry args={[0.32, 16, 16]} />
            <meshBasicMaterial color="#ffffff" />
          </mesh>
          <Html center distanceFactor={26} zIndexRange={[20, 0]}>
            <div className="pointer-events-none -translate-y-8 whitespace-nowrap rounded border border-accent/40 bg-ink-950/95 px-2 py-0.5 font-mono text-xs font-semibold text-paper">
              {formatToken(data.tokens[selectedIndex])}
            </div>
          </Html>
        </group>
      )}

      {/* Hover label: lightweight, follows the cursor's point. */}
      {hoveredIndex != null && hoveredIndex !== selectedIndex && (
        <Html position={posOf(hoveredIndex)} center distanceFactor={26} zIndexRange={[10, 0]}>
          <div className="pointer-events-none -translate-y-7 whitespace-nowrap rounded border border-white/10 bg-ink-950/90 px-2 py-0.5 font-mono text-xs text-paper">
            {formatToken(data.tokens[hoveredIndex])}
          </div>
        </Html>
      )}
    </group>
  );
}

/** Make whitespace/empty tokens legible in labels. */
function formatToken(token: string): string {
  if (token === " ") return "␣ (space)";
  if (token.trim() === "") return JSON.stringify(token);
  return token;
}
