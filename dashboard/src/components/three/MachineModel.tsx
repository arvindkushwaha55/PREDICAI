import { useEffect, useMemo, useRef } from "react";
import { useFrame, useThree } from "@react-three/fiber";
import * as THREE from "three";

export interface MachineModelProps {
  /** "idle" for the hero turntable, "scanning" while a diagnostic runs. */
  mode: "idle" | "scanning";
  /** Diagnostic walk progress 0..1; drives the scan ring in scanning mode. */
  progress: number;
  /** Index of the sensor node to highlight (-1 = none). */
  activeSensor: number;
  /** When true, the model holds a single static pose. */
  reducedMotion: boolean;
  particleCount?: number;
}

const SENSOR_COUNT = 14;
const AXIS_MIN = -0.68;
const AXIS_MAX = 0.68;

/** Sensor node positions: two rings around the casing (8 front, 6 rear). */
function sensorPositions(): [number, number, number][] {
  const out: [number, number, number][] = [];
  for (let i = 0; i < 8; i++) {
    const a = (i / 8) * Math.PI * 2 + 0.2;
    out.push([Math.cos(a) * 0.92, Math.sin(a) * 0.92, 0.42]);
  }
  for (let i = 0; i < 6; i++) {
    const a = (i / 6) * Math.PI * 2 + 0.55;
    out.push([Math.cos(a) * 0.92, Math.sin(a) * 0.92, -0.42]);
  }
  return out;
}

/**
 * Stylised turbofan built entirely from three.js primitives (no external
 * models or textures): casing, fan with 14 blades, shaft, housing blocks,
 * plinth, 14 glowing sensor nodes, a scanning ring and floating particles.
 * It is a visualisation of the FD001 simulated benchmark engine, not a
 * model of a real machine.
 */
export function MachineModel({
  mode,
  progress,
  activeSensor,
  reducedMotion,
  particleCount = 250,
}: MachineModelProps) {
  const turntable = useRef<THREE.Group>(null);
  const fan = useRef<THREE.Group>(null);
  const scanRing = useRef<THREE.Group>(null);
  const particles = useRef<THREE.Points>(null);
  const sensors = useRef<(THREE.Mesh | null)[]>([]);
  const invalidate = useThree((s) => s.invalidate);

  const nodes = useMemo(sensorPositions, []);
  const particlePositions = useMemo(() => {
    const arr = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      arr[i * 3] = (Math.random() - 0.5) * 4.8;
      arr[i * 3 + 1] = (Math.random() - 0.5) * 3.0;
      arr[i * 3 + 2] = (Math.random() - 0.5) * 3.4;
    }
    return arr;
  }, [particleCount]);

  const applyFrame = (t: number, delta: number) => {
    if (turntable.current) {
      turntable.current.rotation.y +=
        delta * (mode === "idle" ? 0.15 : 0.05);
    }
    if (fan.current) {
      fan.current.rotation.z -= delta * (mode === "idle" ? 0.6 : 3.2);
    }
    if (scanRing.current) {
      scanRing.current.position.z =
        mode === "scanning"
          ? AXIS_MIN + progress * (AXIS_MAX - AXIS_MIN)
          : Math.sin(t * 0.5) * 0.6;
    }
    for (let i = 0; i < SENSOR_COUNT; i++) {
      const mesh = sensors.current[i];
      if (!mesh) continue;
      const mat = mesh.material as THREE.MeshStandardMaterial;
      if (i === activeSensor) {
        mat.emissiveIntensity = 2.6;
        mesh.scale.setScalar(1.5);
      } else if (reducedMotion) {
        mat.emissiveIntensity = 0.7;
        mesh.scale.setScalar(1);
      } else {
        // Gentle sequential pulse travelling around the casing.
        mat.emissiveIntensity =
          0.55 + 0.45 * (0.5 + 0.5 * Math.sin(t * 2 + i * 0.7));
        mesh.scale.setScalar(1);
      }
    }
    if (particles.current) {
      particles.current.rotation.y += delta * 0.02;
      particles.current.position.y = Math.sin(t * 0.3) * 0.04;
    }
  };

  useFrame((state, delta) => {
    if (reducedMotion) return;
    applyFrame(state.clock.elapsedTime, delta);
  });

  // Static pose for frameloop="never" (reduced motion): apply the
  // prop-driven state once per change and request a single frame.
  useEffect(() => {
    if (!reducedMotion) return;
    applyFrame(0, 0);
    invalidate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, progress, activeSensor, reducedMotion]);

  const blades = Array.from({ length: 14 }, (_, i) => (i / 14) * Math.PI * 2);

  return (
    <group>
      <points ref={particles}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            args={[particlePositions, 3]}
          />
        </bufferGeometry>
        <pointsMaterial
          size={0.022}
          color="#67e8f9"
          transparent
          opacity={0.55}
          depthWrite={false}
          sizeAttenuation
        />
      </points>

      <group ref={turntable}>
        {/* Base plinth and supports */}
        <mesh position={[0, -1.14, 0]}>
          <boxGeometry args={[2.6, 0.18, 1.7]} />
          <meshStandardMaterial color="#141c26" metalness={0.6} roughness={0.5} />
        </mesh>
        {[-0.45, 0.45].map((x) => (
          <mesh key={x} position={[x, -0.8, 0]}>
            <boxGeometry args={[0.18, 0.55, 0.5]} />
            <meshStandardMaterial color="#1a2431" metalness={0.6} roughness={0.45} />
          </mesh>
        ))}

        {/* Side housing blocks with a lit status strip */}
        {[-1.08, 1.08].map((x) => (
          <group key={x} position={[x, -0.25, 0]}>
            <mesh>
              <boxGeometry args={[0.28, 0.95, 1.15]} />
              <meshStandardMaterial color="#1c2733" metalness={0.55} roughness={0.5} />
            </mesh>
            <mesh position={[Math.sign(x) * 0.15, 0.1, 0]}>
              <boxGeometry args={[0.02, 0.6, 0.06]} />
              <meshStandardMaterial
                color="#06202b"
                emissive="#22d3ee"
                emissiveIntensity={0.9}
              />
            </mesh>
          </group>
        ))}

        {/* Outer casing (open cylinder) and rim rings */}
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.85, 0.85, 1.5, 40, 1, true]} />
          <meshStandardMaterial
            color="#232f3e"
            metalness={0.65}
            roughness={0.42}
            side={THREE.DoubleSide}
          />
        </mesh>
        {[0.75, -0.75].map((z) => (
          <mesh key={z} position={[0, 0, z]}>
            <torusGeometry args={[0.85, 0.05, 12, 44]} />
            <meshStandardMaterial color="#2e3d4f" metalness={0.7} roughness={0.35} />
          </mesh>
        ))}

        {/* Rear core and shaft */}
        <mesh position={[0, 0, -0.15]} rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.46, 0.46, 1.15, 28]} />
          <meshStandardMaterial color="#1a2431" metalness={0.6} roughness={0.45} />
        </mesh>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[0.085, 0.085, 2.0, 14]} />
          <meshStandardMaterial color="#3d4f63" metalness={0.8} roughness={0.3} />
        </mesh>

        {/* Fan: hub, spinner cone and 14 pitched blades */}
        <group ref={fan} position={[0, 0, 0.52]}>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.17, 0.17, 0.1, 24]} />
            <meshStandardMaterial color="#2e3d4f" metalness={0.75} roughness={0.3} />
          </mesh>
          <mesh position={[0, 0, 0.16]} rotation={[Math.PI / 2, 0, 0]}>
            <coneGeometry args={[0.15, 0.4, 24]} />
            <meshStandardMaterial color="#3d4f63" metalness={0.8} roughness={0.28} />
          </mesh>
          {blades.map((a) => (
            <group key={a} rotation={[0, 0, a]}>
              <mesh position={[0, 0.38, 0]} rotation={[0, 0.55, 0]}>
                <boxGeometry args={[0.1, 0.44, 0.025]} />
                <meshStandardMaterial color="#54687e" metalness={0.8} roughness={0.3} />
              </mesh>
            </group>
          ))}
        </group>

        {/* Sensor nodes: 14 glowing points around the casing */}
        {nodes.map((pos, i) => (
          <mesh
            key={i}
            position={pos}
            ref={(m) => {
              sensors.current[i] = m;
            }}
          >
            <sphereGeometry args={[0.05, 12, 12]} />
            <meshStandardMaterial
              color="#062a38"
              emissive="#22d3ee"
              emissiveIntensity={0.7}
            />
          </mesh>
        ))}

        {/* Scanning ring plus its translucent scan plane */}
        <group ref={scanRing} position={[0, 0, 0]}>
          <mesh>
            <torusGeometry args={[0.97, 0.02, 10, 48]} />
            <meshBasicMaterial color="#22d3ee" transparent opacity={0.8} />
          </mesh>
          <mesh>
            <ringGeometry args={[0.12, 0.97, 48]} />
            <meshBasicMaterial
              color="#0ea5c9"
              transparent
              opacity={0.07}
              side={THREE.DoubleSide}
              depthWrite={false}
            />
          </mesh>
        </group>
      </group>
    </group>
  );
}
