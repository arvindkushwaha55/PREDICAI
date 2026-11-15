import { Suspense, useMemo } from "react";
import { Canvas } from "@react-three/fiber";
import { MachineModel } from "./MachineModel";
import type { MachineModelProps } from "./MachineModel";

export type HeroSceneProps = Omit<MachineModelProps, "particleCount">;

/**
 * The test-cell viewport: a dark graphite-navy scene with electric cyan
 * and blue lighting around the procedural turbofan. Loaded lazily (React
 * .lazy) by HeroPanel and TestMachine so three.js stays out of the
 * initial bundle. Only one instance is mounted at a time (the hero
 * unmounts its canvas while the diagnostic panel is open), so only one
 * WebGL context is alive.
 */
export default function HeroScene(props: HeroSceneProps) {
  const particleCount = useMemo(
    () =>
      typeof window !== "undefined" && window.innerWidth < 640 ? 100 : 250,
    [],
  );

  return (
    <Canvas
      dpr={[1, 1.75]}
      camera={{ position: [2.7, 1.15, 3.5], fov: 42, near: 0.1, far: 30 }}
      frameloop={props.reducedMotion ? "never" : "always"}
      gl={{ antialias: true }}
      onCreated={({ camera }) => camera.lookAt(0, -0.05, 0)}
    >
      <color attach="background" args={["#05090f"]} />
      <fog attach="fog" args={["#05090f", 7, 13]} />
      <ambientLight intensity={0.35} />
      {/* Electric cyan key light, blue rim light, soft top fill. */}
      <pointLight position={[3, 2.5, 3]} intensity={30} color="#22d3ee" />
      <pointLight position={[-3.5, 1, -2.5]} intensity={22} color="#2563eb" />
      <directionalLight position={[0, 4, 2]} intensity={0.5} color="#cfe8ff" />
      <Suspense fallback={null}>
        <MachineModel {...props} particleCount={particleCount} />
      </Suspense>
    </Canvas>
  );
}
