/** GPU stage for the interactive embedding projection. */

import { useEffect, useRef } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { Vector3 } from "three";
import { pointAt } from "../../lib/layout";
import { useStore } from "../../store/useStore";
import { PointCloud } from "./PointCloud";

/** Smoothly flies the camera and orbit target to a searched/selected token. */
function CameraRig() {
  const focus = useStore((s) => s.focus);
  const positions = useStore((s) => s.positions);
  const controls = useThree((state) => state.controls) as
    | { target: Vector3; update: () => void }
    | null;
  const camera = useThree((state) => state.camera);
  const invalidate = useThree((state) => state.invalidate);
  const goal = useRef<Vector3 | null>(null);

  useEffect(() => {
    if (!focus || !positions) return;
    const [x, y, z] = pointAt(positions, focus.index);
    goal.current = new Vector3(x, y, z);
    invalidate();
  }, [focus, invalidate, positions]);

  useFrame((_, delta) => {
    if (!goal.current || !controls) return;

    const amount = 1 - Math.exp(-8 * delta);
    controls.target.lerp(goal.current, amount);
    const direction = camera.position.clone().sub(controls.target).normalize();
    const distance = Math.min(Math.max(camera.position.distanceTo(controls.target), 7), 13);
    camera.position.lerp(goal.current.clone().add(direction.multiplyScalar(distance)), amount);
    controls.update();

    if (controls.target.distanceTo(goal.current) < 0.025) goal.current = null;
    else invalidate();
  });

  return null;
}

export function EmbeddingCanvas() {
  const vizData = useStore((s) => s.vizData);
  const clearSelection = useStore((s) => s.clearSelection);

  return (
    <Canvas
      dpr={[1, 1.35]}
      frameloop="demand"
      gl={{ antialias: false, powerPreference: "high-performance", alpha: false }}
      camera={{ position: [16, 12, 22], fov: 53, near: 0.1, far: 400 }}
      raycaster={{ params: { Points: { threshold: 0.45 } } as never }}
      onPointerMissed={clearSelection}
    >
      <color attach="background" args={["#050d17"]} />
      <fog attach="fog" args={["#050d17", 42, 95]} />

      {vizData && <PointCloud data={vizData} />}

      <OrbitControls
        enableDamping
        dampingFactor={0.08}
        rotateSpeed={0.5}
        zoomSpeed={0.8}
        minDistance={4}
        maxDistance={120}
        makeDefault
      />
      <CameraRig />
    </Canvas>
  );
}
