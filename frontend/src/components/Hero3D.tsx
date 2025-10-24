"use client";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stars } from "@react-three/drei";
import React, { useRef } from "react";
import { useFrame } from "@react-three/fiber";

function RotatingKnot() {
  const ref = useRef<any>();
  useFrame((state, delta) => {
    if (!ref.current) return;
    ref.current.rotation.y += delta * 0.15;
    ref.current.rotation.x += delta * 0.05;
  });
  return (
    <mesh ref={ref} scale={1.6} position={[0, 0, 0]}>
      <torusKnotGeometry args={[0.8, 0.22, 128, 16]} />
      <meshStandardMaterial color="#2563eb" metalness={0.7} roughness={0.2} wireframe opacity={0.35} transparent />
    </mesh>
  );
}

export default function Hero3D() {
  return (
    <Canvas camera={{ position: [0, 0, 4], fov: 50 }}>
      <ambientLight intensity={0.9} />
      <directionalLight intensity={0.6} position={[3, 3, 2]} />
      <Stars radius={12} depth={40} count={2500} factor={3} saturation={0} fade speed={0.6} />
      <RotatingKnot />
      <OrbitControls enableZoom={false} enablePan={false} autoRotate autoRotateSpeed={0.5} />
    </Canvas>
  );
}

