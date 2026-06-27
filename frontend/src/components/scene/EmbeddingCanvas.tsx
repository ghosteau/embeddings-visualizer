/**
 * The WebGL stage: camera, controls, post-processing bloom, and the point cloud.
 *
 * The raycaster's Points threshold is widened so the small glowing points are
 * easy to hover/click. Bloom gives the additive points their characteristic
 * glow. A CameraRig smoothly recenters the view on a focused token.
 */

import { useEffect, useRef } from "react";
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
  const desired = useRef<Vector3 | null>(null);

  useEffect(() => {
    if (!focus || !positions) return;
    const [x, y, z] = pointAt(positions, focus.index);
    desired.current = new Vector3(x, y, z);
  }, [focus, positions]);

  useFrame(() => {
    const goal = desired.current;
    if (!goal || !controls) return;
    const t = 0.1;
    controls.target.lerp(goal, t);
    // Keep a comfortable framing distance while approaching the point.
    const dir = camera.position.clone().sub(controls.target).normalize();
    const dist = Math.min(Math.max(camera.position.distanceTo(controls.target), 7), 13);
    camera.position.lerp(goal.clone().add(dir.multiplyScalar(dist)), t);
    controls.update();
    if (controls.target.distanceTo(goal) < 0.05) desired.current = null;
  });

  return null;
}

export function EmbeddingCanvas() {
  const vizData = useStore((s) => s.vizData);
  const clearSelection = useStore((s) => s.clearSelection);

  return (
    <Canvas
      dpr={[1, 2]}
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
        dampingFactor={0.08}
        rotateSpeed={0.6}
        zoomSpeed={0.8}
        minDistance={4}
        maxDistance={120}
        makeDefault
      />
      <CameraRig />

      <EffectComposer>
        <Bloom
          intensity={0.65}
          luminanceThreshold={0.18}
          luminanceSmoothing={0.5}
          mipmapBlur
        />
      </EffectComposer>
    </Canvas>
  );
}
