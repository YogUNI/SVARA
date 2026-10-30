import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { RotateCcw } from 'lucide-react';
import { DeviceState } from './FloorPlan';
import {
  createWoodFloorTexture,
  createMarbleTexture,
  createFabricRugTexture,
  createZenGrassTexture,
} from './fpp/ProceduralTextures';

interface Home3DViewProps {
  devices: DeviceState;
  activeLocation: string | null;
  onToggleDevice: (key: keyof DeviceState) => void;
}

const ROOM_TARGETS: Record<string, [number, number]> = {
  kitchen: [-7, -5.5],
  bedroom: [7, -5.5],
  washroom: [-7, 6],
  hall: [5.5, 4.5],
  none: [0, 7],
};

export const Home3DView: React.FC<Home3DViewProps> = ({ devices, activeLocation }) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const controlsRef = useRef<OrbitControls | null>(null);
  const avatarRef = useRef<THREE.Group | null>(null);
  const avatarPartsRef = useRef<{
    leftLeg: THREE.Object3D;
    rightLeg: THREE.Object3D;
    leftArm: THREE.Object3D;
    rightArm: THREE.Object3D;
    body: THREE.Object3D;
  } | null>(null);
  const markerRef = useRef<THREE.Mesh | null>(null);
  const targetPosRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 8.5));
  const lightsRef = useRef<Record<string, { light: THREE.PointLight; bulb: THREE.Mesh }>>({});
  const cooktopRingsRef = useRef<THREE.Mesh[]>([]);
  const eqBarsRef = useRef<THREE.Mesh[]>([]);
  const acParticlesRef = useRef<THREE.Points | null>(null);
  const devicesRef = useRef(devices);
  const heatRef = useRef({
    kitchen: devices.kitchen_heat,
    bedroom: devices.bedroom_heat,
    washroom: devices.washroom_heat,
  });

  devicesRef.current = devices;
  heatRef.current = {
    kitchen: devices.kitchen_heat,
    bedroom: devices.bedroom_heat,
    washroom: devices.washroom_heat,
  };

  useEffect(() => {
    const mount = mountRef.current;
    if (!mount) return;

    const width = mount.clientWidth || 800;
    const height = mount.clientHeight || 520;

    /* ----------------------------- renderer ----------------------------- */
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x151119);
    scene.fog = new THREE.Fog(0x151119, 62, 130);

    const camera = new THREE.PerspectiveCamera(31, width / height, 0.1, 250);
    camera.position.set(25, 23, 29);
    camera.lookAt(0, 0.8, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.5));
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.08;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.domElement.style.display = 'block';
    mount.appendChild(renderer.domElement);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(0, 0.7, 0);
    controls.enableDamping = true;
    controls.dampingFactor = 0.075;
    controls.enablePan = false;
    controls.minPolarAngle = 0.36;
    controls.maxPolarAngle = 1.16;
    controls.minDistance = 24;
    controls.maxDistance = 58;
    controls.rotateSpeed = 0.7;
    controls.zoomSpeed = 0.8;
    controls.update();
    controls.saveState();
    controlsRef.current = controls;

    /* --------------------------- environment --------------------------- */
    scene.add(new THREE.AmbientLight(0xFFD9AE, 0.55));
    scene.add(new THREE.HemisphereLight(0xFFE4BE, 0x2A1D18, 0.7));

    const keyLight = new THREE.DirectionalLight(0xFFD2A0, 1.35);
    keyLight.position.set(18, 30, 16);
    keyLight.castShadow = true;
    keyLight.shadow.mapSize.set(1024, 1024);
    keyLight.shadow.camera.left = -24;
    keyLight.shadow.camera.right = 24;
    keyLight.shadow.camera.top = 24;
    keyLight.shadow.camera.bottom = -24;
    keyLight.shadow.camera.near = 1;
    keyLight.shadow.camera.far = 80;
    keyLight.shadow.bias = -0.0006;
    scene.add(keyLight);

    const warmRim = new THREE.DirectionalLight(0xFF8C5A, 0.3);
    warmRim.position.set(-22, 14, -18);
    scene.add(warmRim);

    /* ------------------------------ base ------------------------------ */
    const woodFloorTex = createWoodFloorTexture();
    woodFloorTex.repeat.set(3, 3);
    woodFloorTex.wrapS = THREE.RepeatWrapping;
    woodFloorTex.wrapT = THREE.RepeatWrapping;
    woodFloorTex.anisotropy = 8;

    const marbleTex = createMarbleTexture();
    marbleTex.wrapS = THREE.RepeatWrapping;
    marbleTex.wrapT = THREE.RepeatWrapping;

    const rugTex = createFabricRugTexture();
    rugTex.wrapS = THREE.RepeatWrapping;
    rugTex.wrapT = THREE.RepeatWrapping;

    const zenGrassTex = createZenGrassTexture();
    zenGrassTex.repeat.set(3, 3);
    zenGrassTex.wrapS = THREE.RepeatWrapping;
    zenGrassTex.wrapT = THREE.RepeatWrapping;

    const base = new THREE.Mesh(
      new THREE.BoxGeometry(36, 1.4, 32),
      new THREE.MeshStandardMaterial({ color: 0x1E1A16, roughness: 0.5, metalness: 0.4 }),
    );
    base.position.y = -0.75;
    base.receiveShadow = true;
    scene.add(base);

    const lawn = new THREE.Mesh(
      new THREE.PlaneGeometry(35.4, 31.4),
      new THREE.MeshStandardMaterial({ map: zenGrassTex, color: 0x9ac0a0, roughness: 0.95, metalness: 0 }),
    );
    lawn.rotation.x = -Math.PI / 2;
    lawn.position.y = 0.005;
    lawn.receiveShadow = true;
    scene.add(lawn);

    const interiorFloor = new THREE.Mesh(
      new THREE.PlaneGeometry(28, 24),
      new THREE.MeshStandardMaterial({ map: woodFloorTex, roughness: 0.42, metalness: 0.08 }),
    );
    interiorFloor.rotation.x = -Math.PI / 2;
    interiorFloor.position.y = 0.02;
    interiorFloor.receiveShadow = true;
    scene.add(interiorFloor);

    /* ----------------------------- helpers ---------------------------- */
    const material = (
      color: number,
      roughness = 0.6,
      metalness = 0.05,
      extra: THREE.MeshStandardMaterialParameters = {},
    ) => new THREE.MeshStandardMaterial({ color, roughness, metalness, ...extra });

    const addBox = (
      w: number,
      h: number,
      d: number,
      x: number,
      y: number,
      z: number,
      mat: THREE.Material,
      shadows = true,
    ) => {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat);
      mesh.position.set(x, y, z);
      mesh.castShadow = shadows;
      mesh.receiveShadow = shadows;
      scene.add(mesh);
      return mesh;
    };

    const addCylinder = (
      rt: number,
      rb: number,
      h: number,
      x: number,
      y: number,
      z: number,
      mat: THREE.Material,
      segments = 24,
    ) => {
      const mesh = new THREE.Mesh(new THREE.CylinderGeometry(rt, rb, h, segments), mat);
      mesh.position.set(x, y, z);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      scene.add(mesh);
      return mesh;
    };

    /* ------------------------------ walls ----------------------------- */
    const wallMat = material(0xb9c4d4, 0.78, 0.03);
    const wallTrimMat = material(0x1e293b, 0.5, 0.15);
    const WALL_H = 2.85;

    // Outer shell
    addBox(0.35, WALL_H, 24.35, -14, WALL_H / 2, 0, wallMat);
    addBox(0.35, WALL_H, 24.35, 14, WALL_H / 2, 0, wallMat);
    addBox(28.35, WALL_H, 0.35, 0, WALL_H / 2, -12, wallMat);
    addBox(12.2, WALL_H, 0.35, -7.9, WALL_H / 2, 12, wallMat);
    addBox(12.2, WALL_H, 0.35, 7.9, WALL_H / 2, 12, wallMat);
    addBox(3.6, 0.55, 0.35, 0, WALL_H - 0.275, 12, wallMat);

    // Interior partitions with door openings
    addBox(0.3, WALL_H - 0.1, 8, 0, (WALL_H - 0.1) / 2, -8, wallMat);
    addBox(0.3, WALL_H - 0.1, 5, 0, (WALL_H - 0.1) / 2, 1.5, wallMat);
    addBox(0.3, WALL_H - 0.1, 5, 0, (WALL_H - 0.1) / 2, 9.5, wallMat);
    addBox(4, WALL_H - 0.1, 0.3, -12, (WALL_H - 0.1) / 2, 0, wallMat);
    addBox(4, WALL_H - 0.1, 0.3, -5, (WALL_H - 0.1) / 2, 0, wallMat);
    addBox(4, WALL_H - 0.1, 0.3, 5, (WALL_H - 0.1) / 2, 0, wallMat);
    addBox(4, WALL_H - 0.1, 0.3, 12, (WALL_H - 0.1) / 2, 0, wallMat);

    // Skirting strips for architectural detail
    addBox(0.38, 0.16, 24.3, -13.98, 0.08, 0, wallTrimMat, false);
    addBox(0.38, 0.16, 24.3, 13.98, 0.08, 0, wallTrimMat, false);
    addBox(28.3, 0.16, 0.38, 0, 0.08, -11.98, wallTrimMat, false);

    /* ------------------------- room floor accents ---------------------- */
    const kitchenFloor = new THREE.Mesh(
      new THREE.PlaneGeometry(13.5, 11.4),
      new THREE.MeshStandardMaterial({ map: marbleTex, color: 0x7c8ea6, roughness: 0.3, metalness: 0.2 }),
    );
    kitchenFloor.rotation.x = -Math.PI / 2;
    kitchenFloor.position.set(-6.9, 0.032, -6);
    kitchenFloor.receiveShadow = true;
    scene.add(kitchenFloor);

    const washroomFloor = new THREE.Mesh(
      new THREE.PlaneGeometry(13.5, 11.4),
      new THREE.MeshStandardMaterial({ color: 0x18242f, roughness: 0.24, metalness: 0.22 }),
    );
    washroomFloor.rotation.x = -Math.PI / 2;
    washroomFloor.position.set(-6.9, 0.032, 6);
    washroomFloor.receiveShadow = true;
    scene.add(washroomFloor);

    /* ----------------------------- kitchen ---------------------------- */
    const cabinetMat = material(0x1e293b, 0.62, 0.08);
    const counterMat = new THREE.MeshStandardMaterial({
      map: marbleTex,
      color: 0xe7edf5,
      roughness: 0.16,
      metalness: 0.18,
    });

    addBox(0.95, 1.9, 7, -13.25, 0.95, -6, cabinetMat);
    addBox(1.1, 0.12, 7.25, -13.2, 1.96, -6, counterMat);
    addBox(0.55, 0.85, 7, -13.75, 2.4, -6, material(0x243244, 0.6, 0.1));

    // Island + marble top
    addBox(5.2, 0.95, 2.4, -7, 0.48, -5, material(0x172131, 0.5, 0.12));
    addBox(5.55, 0.14, 2.75, -7, 1.02, -5, counterMat);

    // Sink + tap
    addBox(1.25, 0.05, 0.85, -8.35, 1.1, -5, material(0x0b1220, 0.15, 0.85));
    addCylinder(0.055, 0.055, 0.55, -8.35, 1.36, -4.55, material(0x94a3b8, 0.2, 0.9), 12);
    addCylinder(0.05, 0.05, 0.32, -8.35, 1.62, -4.7, material(0x94a3b8, 0.2, 0.9), 12).rotation.x = Math.PI / 2;

    // Cooktop + burner rings
    addBox(1.7, 0.035, 1.05, -5.9, 1.11, -5, material(0x05070a, 0.12, 0.6));
    [-0.5, 0.05].forEach((dx) => {
      [-0.25, 0.25].forEach((dz) => {
        const ring = new THREE.Mesh(
          new THREE.TorusGeometry(0.17, 0.024, 10, 28),
          new THREE.MeshStandardMaterial({ color: 0x334155, emissive: 0x000000, emissiveIntensity: 0, roughness: 0.4 }),
        );
        ring.rotation.x = Math.PI / 2;
        ring.position.set(-5.9 + dx, 1.135, -5 + dz);
        scene.add(ring);
        cooktopRingsRef.current.push(ring);
      });
    });

    // Fridge
    addBox(1.8, 2.9, 1.6, -12.35, 1.45, -10.2, material(0x9aa6b2, 0.24, 0.85));
    addBox(0.04, 2.6, 1.5, -11.44, 1.45, -10.2, material(0x64748b, 0.3, 0.8), false);

    // Dining table + chairs
    addCylinder(0.08, 0.08, 0.78, -4.2, 0.39, -8.7, material(0x334155, 0.4, 0.4), 14);
    addCylinder(1.5, 1.5, 0.09, -4.2, 0.82, -8.7, material(0x3f2e22, 0.32, 0.1), 40);
    [
      [-4.2, -7.1],
      [-4.2, -10.3],
      [-2.7, -8.7],
      [-5.7, -8.7],
    ].forEach(([cx, cz]) => {
      addBox(0.55, 0.08, 0.55, cx, 0.48, cz, material(0x1f2a3a, 0.5, 0.1));
      addBox(0.5, 0.55, 0.08, cx, 0.78, cz + (cz < -8.7 ? -0.24 : 0.24), material(0x1f2a3a, 0.5, 0.1));
    });

    // Pendant lights above the island
    [-0.9, 0, 0.9].forEach((dx) => {
      addCylinder(0.09, 0.26, 0.34, -7 + dx, 2.55, -5, material(0x111827, 0.35, 0.5), 18);
      const glow = new THREE.Mesh(
        new THREE.SphereGeometry(0.11, 14, 14),
        new THREE.MeshStandardMaterial({ color: 0xfbbf24, emissive: 0xf59e0b, emissiveIntensity: 1.6 }),
      );
      glow.position.set(-7 + dx, 2.35, -5);
      scene.add(glow);
    });

    /* ----------------------------- bedroom ---------------------------- */
    const bedRug = new THREE.Mesh(
      new THREE.PlaneGeometry(7.2, 7.2),
      new THREE.MeshStandardMaterial({ map: rugTex, color: 0x9aa8bd, roughness: 0.92 }),
    );
    bedRug.rotation.x = -Math.PI / 2;
    bedRug.position.set(7, 0.035, -6.4);
    bedRug.receiveShadow = true;
    scene.add(bedRug);

    addBox(5, 0.42, 6.2, 7, 0.21, -6.8, material(0x231a14, 0.6, 0.05));
    addBox(4.65, 0.5, 5.8, 7, 0.66, -6.8, material(0xe8edf2, 0.88, 0.02));
    // Duvet + folded throw
    addBox(4.7, 0.18, 3.9, 7, 0.94, -5.55, material(0x2b3b55, 0.82, 0.03));
    addBox(4.72, 0.12, 0.55, 7, 1.02, -7.7, material(0x1e293b, 0.85, 0.03));
    // Headboard
    addBox(5.25, 1.5, 0.28, 7, 0.85, -9.85, material(0x1a2433, 0.55, 0.12));
    // Pillows
    addBox(1.45, 0.24, 0.95, 5.65, 1.02, -9.05, material(0xf1f5f9, 0.9, 0.01));
    addBox(1.45, 0.24, 0.95, 8.35, 1.02, -9.05, material(0xf1f5f9, 0.9, 0.01));

    // Nightstands + bedside lamp
    addBox(1, 0.88, 1, 3.95, 0.44, -9.2, material(0x2a1c14, 0.55, 0.05));
    addBox(1, 0.88, 1, 10.05, 0.44, -9.2, material(0x2a1c14, 0.55, 0.05));
    const lampShade = addCylinder(0.18, 0.3, 0.5, 3.95, 1.16, -9.2, material(0xf8fafc, 0.35, 0.05), 20);
    addCylinder(0.025, 0.025, 0.32, 3.95, 0.92, -9.2, material(0x64748b, 0.3, 0.7), 10);

    // Wardrobe + AC + window
    addBox(1.15, 2.7, 4.2, 13.2, 1.35, -5.4, material(0x1c2637, 0.5, 0.12));
    addBox(0.05, 2.4, 0.04, 12.6, 1.35, -5.4, material(0x94a3b8, 0.3, 0.7), false);
    addBox(2.3, 0.6, 0.42, 7, 2.55, -11.55, material(0xe2e8f0, 0.42, 0.1));
    addBox(2.1, 0.06, 0.06, 7, 2.33, -11.32, material(0x64748b, 0.4, 0.3), false);

    const windowGlass = new THREE.Mesh(
      new THREE.PlaneGeometry(6, 2.2),
      new THREE.MeshStandardMaterial({
        color: 0x0b1b2a,
        emissive: 0x0ea5e9,
        emissiveIntensity: 0.16,
        roughness: 0.08,
        metalness: 0.9,
      }),
    );
    windowGlass.position.set(7, 1.9, -11.78);
    scene.add(windowGlass);

    // AC particles (temperature feedback)
    const particleCount = 42;
    const particlePositions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount; i++) {
      particlePositions[i * 3] = 7 + (Math.random() - 0.5) * 2.2;
      particlePositions[i * 3 + 1] = 2.1 + Math.random() * 1.2;
      particlePositions[i * 3 + 2] = -11.2 + (Math.random() - 0.5) * 0.8;
    }
    const particleGeo = new THREE.BufferGeometry();
    particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePositions, 3));
    const particleMat = new THREE.PointsMaterial({
      color: 0xef4444,
      size: 0.08,
      transparent: true,
      opacity: 0,
      depthWrite: false,
      blending: THREE.AdditiveBlending,
    });
    const acParticles = new THREE.Points(particleGeo, particleMat);
    scene.add(acParticles);
    acParticlesRef.current = acParticles;

    /* ---------------------------- washroom --------------------------- */
    addBox(3.5, 1, 5.3, -10.6, 0.5, 6.6, material(0xe6eef3, 0.28, 0.06));
    const water = addBox(2.9, 0.06, 4.7, -10.6, 0.98, 6.6, material(0x0e7490, 0.08, 0.5));
    water.castShadow = false;

    // Shower tray + glass partition
    addBox(2.8, 0.1, 2.8, -6.2, 0.05, 9.5, material(0x1b2736, 0.3, 0.2));
    const glass = new THREE.Mesh(
      new THREE.BoxGeometry(0.04, 2.4, 2.8),
      new THREE.MeshPhysicalMaterial({
        color: 0xbfe6f2,
        transparent: true,
        opacity: 0.16,
        roughness: 0.04,
        metalness: 0,
        side: THREE.DoubleSide,
      }),
    );
    glass.position.set(-4.85, 1.2, 9.5);
    scene.add(glass);
    addCylinder(0.06, 0.06, 0.5, -5.6, 2.3, 9.5, material(0x94a3b8, 0.2, 0.9), 10).rotation.z = Math.PI / 4;

    // Vanity + basin + mirror
    addBox(4.1, 0.86, 1.15, -7.6, 0.43, 11.15, material(0x1b2635, 0.5, 0.15));
    addBox(4.3, 0.1, 1.3, -7.6, 0.91, 11.15, counterMat);
    addCylinder(0.58, 0.66, 0.24, -8.55, 1.03, 11.15, material(0xf1f5f9, 0.15, 0.1), 26);
    addCylinder(0.03, 0.03, 0.4, -8.55, 1.32, 11.55, material(0x94a3b8, 0.2, 0.9), 8);
    addBox(3.7, 1.85, 0.06, -7.6, 2.05, 11.85, material(0x9fb3c8, 0.03, 1));

    // Toilet
    addBox(0.75, 0.62, 0.32, -3.6, 0.72, 9.05, material(0xe6eef3, 0.3, 0.05));
    addCylinder(0.46, 0.5, 0.4, -3.6, 0.22, 9.55, material(0xe6eef3, 0.3, 0.05), 22);

    /* ------------------------------ hall ----------------------------- */
    const hallRug = new THREE.Mesh(
      new THREE.PlaneGeometry(8.4, 6.2),
      new THREE.MeshStandardMaterial({ map: rugTex, color: 0x8fa0b8, roughness: 0.92 }),
    );
    hallRug.rotation.x = -Math.PI / 2;
    hallRug.position.set(7, 0.035, 6.4);
    hallRug.receiveShadow = true;
    scene.add(hallRug);

    // Sofa
    addBox(6, 0.75, 2.5, 7, 0.38, 8.5, material(0x1f2d3f, 0.75, 0.05));
    addBox(6, 1.15, 0.5, 7, 0.95, 9.6, material(0x263447, 0.7, 0.06));
    addBox(0.45, 0.85, 2.5, 4.05, 0.55, 8.5, material(0x263447, 0.7, 0.06));
    addBox(0.45, 0.85, 2.5, 9.95, 0.55, 8.5, material(0x263447, 0.7, 0.06));
    [5.2, 7, 8.8].forEach((cx) => {
      addBox(1.7, 0.2, 1.7, cx, 0.86, 8.2, material(0x334155, 0.8, 0.03), false);
    });

    // Coffee table (glass)
    addBox(3.3, 0.1, 1.7, 7, 0.55, 5.5, material(0x7dd3fc, 0.08, 0.4, { transparent: true, opacity: 0.55 }));
    [
      [5.5, 4.8],
      [8.5, 4.8],
      [5.5, 6.2],
      [8.5, 6.2],
    ].forEach(([lx, lz]) => addCylinder(0.04, 0.04, 0.52, lx, 0.26, lz, material(0x64748b, 0.3, 0.7), 8));

    // TV console + screen
    addBox(0.55, 0.55, 5, 13.35, 0.28, 5.8, material(0x111a28, 0.5, 0.2));
    const tv = addBox(0.09, 2.15, 4, 13.6, 1.7, 5.8, material(0x05080d, 0.15, 0.4));
    const tvMat = tv.material as THREE.MeshStandardMaterial;
    tvMat.emissive = new THREE.Color(0x0ea5e9);
    tvMat.emissiveIntensity = 0.32;

    // Speaker with equalizer bars
    addBox(1.3, 2, 1.3, 11.5, 1, 10.2, material(0x0f172a, 0.45, 0.25));
    for (let i = 0; i < 4; i++) {
      const barGeo = new THREE.BoxGeometry(0.12, 0.3, 0.06);
      barGeo.translate(0, 0.15, 0);
      const bar = new THREE.Mesh(
        barGeo,
        new THREE.MeshStandardMaterial({ color: 0x06b6d4, emissive: 0x06b6d4, emissiveIntensity: 0.8 }),
      );
      bar.position.set(11.05 + i * 0.3, 1.0, 9.53);
      scene.add(bar);
      eqBarsRef.current.push(bar);
    }

    // Plant
    addCylinder(0.34, 0.42, 0.55, 11.6, 0.28, 2.4, material(0x1f2937, 0.6, 0.1), 16);
    const foliageMat = material(0x14532d, 0.75, 0.02);
    [[0.35, 0.85, 0], [-0.3, 0.95, 0.15], [0.05, 1.2, -0.2]].forEach(([dx, dy, dz]) => {
      const leaf = new THREE.Mesh(new THREE.IcosahedronGeometry(0.36, 1), foliageMat);
      leaf.position.set(11.6 + dx, 0.55 + dy, 2.4 + dz);
      leaf.castShadow = true;
      scene.add(leaf);
    });

    /* --------------------------- room lights -------------------------- */
    const setupRoomLight = (key: string, x: number, z: number) => {
      const fixture = new THREE.Mesh(
        new THREE.CylinderGeometry(0.3, 0.42, 0.1, 24),
        material(0x0f172a, 0.4, 0.4),
      );
      fixture.position.set(x, 2.76, z);
      scene.add(fixture);

      const bulb = new THREE.Mesh(
        new THREE.SphereGeometry(0.17, 18, 18),
        new THREE.MeshStandardMaterial({
          color: 0x334155,
          emissive: 0xf59e0b,
          emissiveIntensity: 0,
          roughness: 0.4,
        }),
      );
      bulb.position.set(x, 2.62, z);
      scene.add(bulb);

      const light = new THREE.PointLight(0xf59e0b, 0, 13, 2);
      light.position.set(x, 2.45, z);
      light.castShadow = true;
      light.shadow.mapSize.set(512, 512);
      light.shadow.bias = -0.002;
      scene.add(light);

      lightsRef.current[key] = { light, bulb };
    };

    setupRoomLight('kitchen_lights', -7, -5.5);
    setupRoomLight('bedroom_lights', 7, -5.5);
    setupRoomLight('washroom_lights', -7, 5.5);

    const bedsideLight = new THREE.PointLight(0xfbbf24, 0, 8, 2);
    bedsideLight.position.set(3.95, 1.35, -9.2);
    scene.add(bedsideLight);
    lightsRef.current['bedroom_lamp'] = { light: bedsideLight, bulb: lampShade };

    /* ----------------------------- avatar ---------------------------- */
    const avatar = new THREE.Group();
    const skinMat = material(0xf3c6a5, 0.8, 0.02);
    const shirtMat = material(0x6366f1, 0.62, 0.05);
    const pantsMat = material(0x1e293b, 0.7, 0.05);
    const shoeMat = material(0x0b1220, 0.5, 0.2);

    const body = new THREE.Mesh(new THREE.CapsuleGeometry(0.3, 0.62, 6, 16), shirtMat);
    body.position.y = 1.16;
    body.castShadow = true;
    avatar.add(body);

    const head = new THREE.Mesh(new THREE.SphereGeometry(0.27, 22, 22), skinMat);
    head.position.y = 1.83;
    head.castShadow = true;
    avatar.add(head);

    const hair = new THREE.Mesh(
      new THREE.SphereGeometry(0.285, 22, 22, 0, Math.PI * 2, 0, Math.PI * 0.55),
      material(0x14202e, 0.8, 0.02),
    );
    hair.position.y = 1.86;
    avatar.add(hair);

    const makeLimb = (x: number, y: number, h: number, mat: THREE.Material, radius = 0.09) => {
      const pivot = new THREE.Group();
      pivot.position.set(x, y, 0);
      const limb = new THREE.Mesh(new THREE.CapsuleGeometry(radius, h, 6, 12), mat);
      limb.position.y = -h / 2;
      limb.castShadow = true;
      pivot.add(limb);
      avatar.add(pivot);
      return pivot;
    };

    const leftArm = makeLimb(-0.4, 1.48, 0.5, shirtMat);
    const rightArm = makeLimb(0.4, 1.48, 0.5, shirtMat);
    const leftLeg = makeLimb(-0.17, 0.82, 0.55, pantsMat, 0.11);
    const rightLeg = makeLimb(0.17, 0.82, 0.55, pantsMat, 0.11);

    // Shoes
    [-0.17, 0.17].forEach((x) => {
      const shoe = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.11, 0.3), shoeMat);
      shoe.position.set(x, 0.06, 0.06);
      shoe.castShadow = true;
      avatar.add(shoe);
    });

    const marker = new THREE.Mesh(
      new THREE.TorusGeometry(0.3, 0.035, 8, 32),
      new THREE.MeshStandardMaterial({ color: 0xfbbf24, emissive: 0xf59e0b, emissiveIntensity: 1.4 }),
    );
    marker.rotation.x = Math.PI / 2;
    marker.position.y = 2.42;
    avatar.add(marker);
    markerRef.current = marker;

    avatar.position.set(0, 0, 8.5);
    scene.add(avatar);
    avatarRef.current = avatar;
    avatarPartsRef.current = { leftLeg, rightLeg, leftArm, rightArm, body };

    /* ---------------------------- animation --------------------------- */
    const clock = new THREE.Clock();
    let raf = 0;
    let walkPhase = 0;
    let isWalking = false;

    const animate = () => {
      raf = requestAnimationFrame(animate);
      const dt = Math.min(clock.getDelta(), 0.05);
      const t = clock.elapsedTime;

      controls.update();

      // Avatar navigation
      const av = avatarRef.current;
      const parts = avatarPartsRef.current;
      if (av && parts) {
        const target = targetPosRef.current;
        const dx = target.x - av.position.x;
        const dz = target.z - av.position.z;
        const dist = Math.hypot(dx, dz);

        if (dist > 0.12) {
          const step = Math.min(1, dt * 2.2);
          av.position.x += dx * step;
          av.position.z += dz * step;
          const desired = Math.atan2(dx, dz);
          let diff = desired - av.rotation.y;
          diff = Math.atan2(Math.sin(diff), Math.cos(diff));
          av.rotation.y += diff * Math.min(1, dt * 8);
          isWalking = true;
          walkPhase += dt * 10;
        } else {
          isWalking = false;
        }

        const swing = isWalking ? Math.sin(walkPhase) : 0;
        parts.leftLeg.rotation.x = swing * 0.55;
        parts.rightLeg.rotation.x = -swing * 0.55;
        parts.leftArm.rotation.x = -swing * 0.45;
        parts.rightArm.rotation.x = swing * 0.45;
        parts.body.position.y = 1.16 + (isWalking ? Math.abs(Math.sin(walkPhase)) * 0.045 : Math.sin(t * 1.8) * 0.012);
        av.position.y = isWalking ? Math.abs(Math.sin(walkPhase)) * 0.03 : 0;
      }

      if (markerRef.current) {
        markerRef.current.rotation.z += dt * 1.6;
        markerRef.current.position.y = 2.42 + Math.sin(t * 2.2) * 0.05;
      }

      // Speaker equalizer
      const musicOn = devicesRef.current.hall_music;
      eqBarsRef.current.forEach((bar, i) => {
        const target = musicOn ? 0.6 + Math.abs(Math.sin(t * 5 + i * 1.3)) * 1.6 : 0.25;
        bar.scale.y += (target - bar.scale.y) * Math.min(1, dt * 12);
      });

      // AC particles drift
      const particles = acParticlesRef.current;
      if (particles) {
        const pos = particles.geometry.getAttribute('position') as THREE.BufferAttribute;
        const heat = heatRef.current.bedroom;
        const mat = particles.material as THREE.PointsMaterial;
        mat.opacity += ((heat !== 20 ? 0.75 : 0) - mat.opacity) * Math.min(1, dt * 3);
        mat.color.setHex(heat > 20 ? 0xef4444 : 0x38bdf8);
        for (let i = 0; i < pos.count; i++) {
          let y = pos.getY(i) - dt * 0.5;
          if (y < 1.9) y = 2.9;
          pos.setY(i, y);
          pos.setX(i, pos.getX(i) + Math.sin(t * 2 + i) * dt * 0.06);
        }
        pos.needsUpdate = true;
      }

      renderer.render(scene, camera);
    };
    animate();

    /* ----------------------------- resize ---------------------------- */
    const observer = new ResizeObserver(() => {
      const w = mount.clientWidth;
      const h = mount.clientHeight;
      if (!w || !h) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    observer.observe(mount);

    /* ---------------------------- cleanup ---------------------------- */
    return () => {
      cancelAnimationFrame(raf);
      observer.disconnect();
      controls.dispose();
      scene.traverse((obj) => {
        const mesh = obj as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else if (mat) mat.dispose();
      });
      renderer.dispose();
      if (renderer.domElement.parentElement === mount) {
        mount.removeChild(renderer.domElement);
      }
      lightsRef.current = {};
      cooktopRingsRef.current = [];
      eqBarsRef.current = [];
      acParticlesRef.current = null;
      avatarRef.current = null;
      avatarPartsRef.current = null;
      markerRef.current = null;
      controlsRef.current = null;
    };
  }, []);

  /* -------------------- react -> three state sync -------------------- */
  useEffect(() => {
    const applyLight = (key: string, on: boolean, intensity = 22) => {
      const item = lightsRef.current[key];
      if (!item) return;
      item.light.intensity = on ? intensity : 0;
      const mat = item.bulb.material as THREE.MeshStandardMaterial;
      mat.emissiveIntensity = on ? 2.4 : 0;
      mat.color.setHex(on ? 0xfbbf24 : 0x334155);
    };

    applyLight('kitchen_lights', devices.kitchen_lights);
    applyLight('bedroom_lights', devices.bedroom_lights);
    applyLight('washroom_lights', devices.washroom_lights);
    applyLight('bedroom_lamp', devices.bedroom_lamp, 7);

    // Cooktop burners glow when a room is being heated
    const anyHeat =
      devices.kitchen_heat > 20 || devices.bedroom_heat > 20 || devices.washroom_heat > 20;
    cooktopRingsRef.current.forEach((ring) => {
      const mat = ring.material as THREE.MeshStandardMaterial;
      mat.emissive.setHex(anyHeat ? 0xef4444 : 0x000000);
      mat.emissiveIntensity = devices.kitchen_heat > 20 ? 1.1 : anyHeat ? 0.4 : 0;
    });
  }, [devices]);

  useEffect(() => {
    const target = ROOM_TARGETS[activeLocation || 'none'] || ROOM_TARGETS.hall;
    targetPosRef.current.set(target[0], 0, target[1]);
  }, [activeLocation]);

  const resetCamera = () => {
    controlsRef.current?.reset();
  };

  return (
    <div className="home3d-shell">
      <div ref={mountRef} className="home3d-canvas" />

      <div className="hud-glass" style={{ top: 14, left: 14 }}>
        <span style={{ width: 6, height: 6, borderRadius: '50%', background: '#F59E0B', boxShadow: '0 0 8px #F59E0B' }} />
        3D ISOMETRIC • ARCHITECTURAL DIORAMA
      </div>

      <div className="hud-glass" style={{ top: 14, right: 14, pointerEvents: 'none' }}>
        <span style={{ color: '#818CF8' }}>DRAG</span> ORBIT ·{' '}
        <span style={{ color: '#818CF8' }}>SCROLL</span> ZOOM
      </div>

      <button className="hud-btn" style={{ bottom: 14, right: 14 }} onClick={resetCamera}>
        <RotateCcw size={13} />
        Reset Kamera
      </button>

      <div className="hud-glass" style={{ bottom: 14, left: 14 }}>
        {[
          { on: devices.kitchen_lights, label: 'DAPUR' },
          { on: devices.bedroom_lights, label: 'KAMAR' },
          { on: devices.washroom_lights, label: 'MANDI' },
          { on: devices.hall_music, label: 'AUDIO' },
        ].map((item) => (
          <span key={item.label} style={{ display: 'inline-flex', alignItems: 'center', gap: 4, marginRight: 6 }}>
            <span
              style={{
                width: 5,
                height: 5,
                borderRadius: '50%',
                background: item.on ? (item.label === 'AUDIO' ? '#06B6D4' : '#F59E0B') : '#334155',
                boxShadow: item.on ? `0 0 7px ${item.label === 'AUDIO' ? '#06B6D4' : '#F59E0B'}` : 'none',
              }}
            />
            {item.label}
          </span>
        ))}
      </div>
    </div>
  );
};
