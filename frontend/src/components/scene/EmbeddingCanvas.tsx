/**
 * The WebGL stage: camera, controls, post-processing bloom, and the point cloud.
 *
 * Performance notes:
 * - DPR is capped at 1.5 so bloom (a full-screen, multi-pass effect) doesn't
 *   have to shade up to 4x the pixels on HiDPI displays — the single biggest
 *   smoothness win here.
 * - Bloom uses mipmap blur (cheaper than a large kernel) at a modest intensity.
 * - We keep a continuous render loop (so orbit damping glides and external
 *   capture/RAF integrations keep working).
 */

import { useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Bloom, EffectComposer } from "@react-three/postprocessing";
import { Vector3 } from "three";
import { PointCloud } from "./PointCloud";
import { pointAt } from "../../lib/layout";
import { useStore } from "../../store/useStore";

/** Smoothly flies the camera/target to the most recent focus request. */
function CameraRig() {
  const focus = useStore((s) => s.focus);
  const positions = useStore((s) => s.positions);
  const controls = useThree((s) => s.controls) as
    | { target: Vector3; update: () => void }
    | null;
  const camera = useThree((s) => s.camera);
  const goal = useRef<Vector3 | null>(null);
  const lastNonce = useRef<number>(-1);

  useFrame(() => {
    // Pick up a new focus request (compared by nonce so repeats re-trigger).
    if (focus && positions && focus.nonce !== lastNonce.current) {
      lastNonce.current = focus.nonce;
      const [x, y, z] = pointAt(positions, focus.index);
      goal.current = new Vector3(x, y, z);
    }
    if (!goal.current || !controls) return;
    const t = 0.12;
    controls.target.lerp(goal.current, t);
    const dir = camera.position.clone().sub(controls.target).normalize();
    const dist = Math.min(Math.max(camera.position.distanceTo(controls.target), 7), 13);
    camera.position.lerp(goal.current.clone().add(dir.multiplyScalar(dist)), t);
    if (controls.target.distanceTo(goal.current) < 0.04) goal.current = null;
  });

  return null;
}

export function EmbeddingCanvas() {
  const vizData = useStore((s) => s.vizData);
  const clearSelection = useStore((s) => s.clearSelection);

  return (
    <Canvas
      dpr={[1, 1.5]}
      // MSAA off: bloom + additive points hide aliasing, and skipping it is a
      // big fill-rate win. Prefer the discrete GPU when one is available.
      gl={{ antialias: false, powerPreference: "high-performance" }}
      camera={{ position: [16, 12, 22], fov: 55, near: 0.1, far: 400 }}
      // Widen the points raycast threshold so the small glowing points are easy
      // to hover/click. Cast: r3f types want a full RaycasterParameters object.
      raycaster={{ params: { Points: { threshold: 0.45 } } as never }}
      onPointerMissed={() => clearSelection()}
    >
      {/* Deep, warm-neutral backdrop + soft distance fog for depth perception. */}
      <color attach="background" args={["#07070a"]} />
      <fog attach="fog" args={["#07070a", 42, 95]} />

      {vizData && <PointCloud data={vizData} />}

      <OrbitControls
        enableDamping
        dampingFactor={0.1}
        rotateSpeed={0.55}
        zoomSpeed={0.85}
        minDistance={4}
        maxDistance={120}
        makeDefault
      />
      <CameraRig />

      <EffectComposer>
        <Bloom intensity={0.55} luminanceThreshold={0.22} luminanceSmoothing={0.5} mipmapBlur />
      </EffectComposer>
    </Canvas>
  );
}
