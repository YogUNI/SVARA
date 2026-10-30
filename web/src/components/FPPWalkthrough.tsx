import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { DeviceState } from './FloorPlan';
import {
  createLuxuryWoodParquetTexture,
  createCalacattaMarbleTexture,
  createMicrocementWallTexture,
  createWovenFabricTexture,
  createAcousticSlatWoodTexture,
  createGoldenHourSkyTexture,
  createZenGrassTexture,
  createGraniteSlabTexture,
  createWoodRoughnessMap,
  createMarbleRoughnessMap,
} from './fpp/ProceduralTextures';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { createFPPArms } from './fpp/FPPArmsRig';
import { soundSystem } from './fpp/SoundSystem';
import { RadarMiniMap, RadarMiniMapHandle } from './fpp/RadarMiniMap';

interface FPPWalkthroughProps {
  devices: DeviceState;
  onVoiceTriggerStart: () => void;
  onVoiceTriggerEnd: () => void;
  isRecording: boolean;
  isProcessing: boolean;
  lastCommandMessage?: string;
}

/* ------------------------------------------------------------------ */
/*  Small helpers                                                      */
/* ------------------------------------------------------------------ */

function roundedRectPath(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  w: number,
  h: number,
  r: number,
) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

/** Digital thermostat wall panel with live temperature digits. */
function createThermostatCanvas() {
  const canvas = document.createElement('canvas');
  canvas.width = 256;
  canvas.height = 160;
  const ctx = canvas.getContext('2d')!;
  const draw = (temp: number) => {
    ctx.clearRect(0, 0, 256, 160);
    // Matte charcoal body
    ctx.fillStyle = '#161A1F';
    roundedRectPath(ctx, 8, 8, 240, 144, 22);
    ctx.fill();
    ctx.strokeStyle = '#3A414B';
    ctx.lineWidth = 2;
    roundedRectPath(ctx, 8, 8, 240, 144, 22);
    ctx.stroke();
    // Brand
    ctx.fillStyle = '#8B98A6';
    ctx.font = 'bold 16px monospace';
    ctx.fillText('SVARA CLIMATE', 28, 40);
    // Temperature digits
    const active = temp !== 20;
    ctx.fillStyle = active ? '#FFB454' : '#4E5863';
    ctx.font = 'bold 52px monospace';
    ctx.fillText(`${temp}°`, 28, 118);
    ctx.fillStyle = active ? '#8B98A6' : '#3A414B';
    ctx.font = '14px monospace';
    ctx.fillText(active ? (temp > 20 ? 'HEAT' : 'COOL') : 'IDLE', 22, 140);
    // Mode ring indicator
    ctx.beginPath();
    ctx.arc(208, 88, 22, 0, Math.PI * 2);
    ctx.strokeStyle = active ? '#FFB454' : '#2C333C';
    ctx.lineWidth = 5;
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(208, 88, 16, 0, Math.PI * 2);
    ctx.fillStyle = active ? (temp > 20 ? '#FF8C42' : '#5ED3E8') : '#2C333C';
    ctx.fill();
  };
  draw(20);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return { texture, draw, needsUpdate: () => { texture.needsUpdate = true; } };
}

/** HomePod-style speaker EQ ring canvas (inner halo). */
function createSpeakerRingCanvas() {
  const canvas = document.createElement('canvas');
  canvas.width = 128;
  canvas.height = 128;
  const ctx = canvas.getContext('2d')!;
  const draw = (pulse: number, active: boolean) => {
    ctx.clearRect(0, 0, 128, 128);
    ctx.globalCompositeOperation = 'lighter';
    const g = ctx.createRadialGradient(64, 64, 6, 64, 64, 62);
    g.addColorStop(0, `rgba(80, 214, 255, ${active ? 0.65 + pulse * 0.3 : 0.08})`);
    g.addColorStop(0.55, `rgba(80, 214, 255, ${active ? 0.22 + pulse * 0.2 : 0.03})`);
    g.addColorStop(1, 'rgba(80, 214, 255, 0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, 128, 128);
    ctx.globalCompositeOperation = 'source-over';
  };
  draw(0, false);
  const texture = new THREE.CanvasTexture(canvas);
  texture.colorSpace = THREE.SRGBColorSpace;
  return { texture, draw, needsUpdate: () => { texture.needsUpdate = true; } };
}

/** Printed broadsheet page — masthead, headline column, photo block. */
function createNewspaperTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 640;
  const ctx = canvas.getContext('2d')!;

  // off-white newsprint
  ctx.fillStyle = '#EDE7DA';
  ctx.fillRect(0, 0, 512, 640);
  ctx.fillStyle = 'rgba(120, 108, 90, 0.05)';
  for (let i = 0; i < 400; i++) {
    ctx.fillRect(Math.random() * 512, Math.random() * 640, 2, 2);
  }

  // masthead
  ctx.fillStyle = '#14110E';
  ctx.font = 'bold 44px Georgia, serif';
  ctx.textAlign = 'center';
  ctx.fillText('THE SVARA TIMES', 256, 62);
  ctx.font = '11px Georgia, serif';
  ctx.fillStyle = '#4A4238';
  ctx.fillText('SMART LIVING  •  ARCHITECTURE  •  DESIGN', 256, 82);
  ctx.strokeStyle = '#14110E';
  ctx.lineWidth = 3;
  ctx.beginPath(); ctx.moveTo(24, 92); ctx.lineTo(488, 92); ctx.stroke();
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(24, 97); ctx.lineTo(488, 97); ctx.stroke();

  // headline
  ctx.textAlign = 'left';
  ctx.fillStyle = '#0E0C0B';
  ctx.font = 'bold 30px Georgia, serif';
  ctx.fillText('Japandi Villas Lead', 28, 138);
  ctx.fillText('the Quiet Luxury Era', 28, 172);

  // deck
  ctx.font = 'italic 13px Georgia, serif';
  ctx.fillStyle = '#3E362C';
  ctx.fillText('Warm timber, microcement and hidden light reshape', 28, 196);
  ctx.fillText('the modern Indonesian home.', 28, 212);

  // photo block
  ctx.fillStyle = '#8A8172';
  ctx.fillRect(28, 228, 220, 150);
  ctx.fillStyle = '#6E6656';
  ctx.fillRect(28, 300, 220, 78);
  ctx.fillStyle = '#EDE7DA';
  ctx.font = '10px Georgia, serif';
  ctx.fillText('Villa SVARA, front elevation at dusk', 32, 372);

  // column text lines
  ctx.fillStyle = 'rgba(30, 26, 22, 0.72)';
  let ty = 236;
  for (let col = 0; col < 2; col++) {
    const cx = 272 + col * 118;
    for (let line = 0; line < 22; line++) {
      const wline = 96 - Math.random() * 34;
      ctx.fillRect(cx, ty + line * 10, wline, 4);
    }
  }
  // body lines under photo
  for (let line = 0; line < 16; line++) {
    ctx.fillRect(28, 400 + line * 11, 210 - Math.random() * 60, 4);
  }
  // footer rule
  ctx.strokeStyle = '#3E362C';
  ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(24, 606); ctx.lineTo(488, 606); ctx.stroke();
  ctx.font = '10px Georgia, serif';
  ctx.fillStyle = '#5A5145';
  ctx.fillText('LIFESTYLE  •  PAGE 12', 28, 626);
  ctx.textAlign = 'left';

  const tex = new THREE.CanvasTexture(canvas);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 8;
  return tex;
}

/* ------------------------------------------------------------------ */
/*  Main component                                                     */
/* ------------------------------------------------------------------ */

type Glow = { mat: THREE.MeshStandardMaterial; intensity: number };
type FixtureLight = { light: THREE.Light; intensity: number };
type Fixture = {
  lights: FixtureLight[];
  glows: Glow[];
  bulbs: THREE.Mesh[];
};

export const FPPWalkthrough: React.FC<FPPWalkthroughProps> = ({
  devices,
  onVoiceTriggerStart,
  onVoiceTriggerEnd,
  isRecording,
  isProcessing,
  lastCommandMessage,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const radarRef = useRef<RadarMiniMapHandle>(null);
  const [isLocked, setIsLocked] = useState(false);
  const [currentRoom, setCurrentRoom] = useState('🌿 Taman Rumah (Front Garden)');
  const currentRoomRef = useRef(currentRoom);

  const lightsMapRef = useRef<{ [key: string]: Fixture }>({});
  const heaterMeshRef = useRef<{ [key: string]: THREE.Mesh }>({});
  const ventParticlesRef = useRef<{ [key: string]: THREE.Points }>({});
  const thermostatRef = useRef<{ [key: string]: ReturnType<typeof createThermostatCanvas> }>({});
  const mirrorRingMatRef = useRef<THREE.MeshStandardMaterial | null>(null);
  const cooktopRingsRef = useRef<THREE.Mesh[]>([]);
  const speakerMeshRef = useRef<THREE.Mesh | null>(null);
  const speakerRingRef = useRef<ReturnType<typeof createSpeakerRingCanvas> | null>(null);

  const moveStateRef = useRef({ forward: false, backward: false, left: false, right: false, sprint: false });
  const isRecordingRef = useRef(isRecording);
  const onVoiceTriggerStartRef = useRef(onVoiceTriggerStart);
  const onVoiceTriggerEndRef = useRef(onVoiceTriggerEnd);
  const musicRef = useRef(devices.hall_music);
  const volumeRef = useRef(devices.hall_volume);

  useEffect(() => {
    currentRoomRef.current = currentRoom;
  }, [currentRoom]);

  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);

  useEffect(() => {
    onVoiceTriggerStartRef.current = onVoiceTriggerStart;
    onVoiceTriggerEndRef.current = onVoiceTriggerEnd;
  }, [onVoiceTriggerStart, onVoiceTriggerEnd]);

  useEffect(() => {
    musicRef.current = devices.hall_music;
    volumeRef.current = devices.hall_volume;
  }, [devices.hall_music, devices.hall_volume]);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const width = container.clientWidth || 800;
    const height = 520;

    /* ---------------- 1. Renderer / Scene / Camera ---------------- */
    const scene = new THREE.Scene();

    const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    container.appendChild(renderer.domElement);

    const camera = new THREE.PerspectiveCamera(76, width / height, 0.1, 220);
    camera.position.set(0, 1.65, 23); // Spawn: front garden facing the villa
    camera.rotation.order = 'YXZ';
    camera.rotation.set(0, 0, 0);

    // Golden-hour dusk atmosphere: gradient dome + layered warm fog
    const skyCanvas = document.createElement('canvas');
    skyCanvas.width = 1024;
    skyCanvas.height = 512;
    const skyCtx = skyCanvas.getContext('2d')!;
    const skyGrad = skyCtx.createLinearGradient(0, 0, 0, 512);
    skyGrad.addColorStop(0, '#1C1734');
    skyGrad.addColorStop(0.20, '#3E2949');
    skyGrad.addColorStop(0.36, '#8A4A57');
    skyGrad.addColorStop(0.46, '#D4785A');
    skyGrad.addColorStop(0.50, '#F7B972');
    skyGrad.addColorStop(0.54, '#FFD79B');
    skyGrad.addColorStop(0.60, '#E8A176');
    skyGrad.addColorStop(1, '#B97F62');
    skyCtx.fillStyle = skyGrad;
    skyCtx.fillRect(0, 0, 1024, 512);

    // Sunset glow + sun disc right on the horizon line
    const glow = skyCtx.createRadialGradient(660, 262, 6, 660, 262, 260);
    glow.addColorStop(0, 'rgba(255, 246, 219, 0.95)');
    glow.addColorStop(0.2, 'rgba(255, 213, 148, 0.7)');
    glow.addColorStop(0.55, 'rgba(255, 165, 100, 0.28)');
    glow.addColorStop(1, 'rgba(255, 140, 90, 0)');
    skyCtx.fillStyle = glow;
    skyCtx.fillRect(0, 0, 1024, 512);

    // Wispy cirrus streaks catching the last light
    for (let c = 0; c < 26; c++) {
      const cy = 60 + Math.random() * 190;
      const cx = Math.random() * 1024;
      const cw = 80 + Math.random() * 240;
      skyCtx.fillStyle = `rgba(255, ${190 + Math.random() * 45 | 0}, 150, ${0.05 + Math.random() * 0.09})`;
      skyCtx.beginPath();
      skyCtx.ellipse(cx, cy, cw, 4 + Math.random() * 7, 0, 0, Math.PI * 2);
      skyCtx.fill();
    }
    const skyTex = new THREE.CanvasTexture(skyCanvas);
    skyTex.colorSpace = THREE.SRGBColorSpace;
    const skyDome = new THREE.Mesh(
      new THREE.SphereGeometry(150, 32, 24),
      new THREE.MeshBasicMaterial({ map: skyTex, side: THREE.BackSide, fog: false, depthWrite: false }),
    );
    skyDome.position.y = 0;
    scene.add(skyDome);

    scene.background = new THREE.Color(0xD97B5F);
    scene.fog = new THREE.FogExp2(0xB96A50, 0.012);

    /* ---------------- 2. Natural golden-hour lighting ---------------- */
    const hemi = new THREE.HemisphereLight(0xFFE9C8, 0x6E5143, 0.72);
    scene.add(hemi);

    const ambient = new THREE.AmbientLight(0xFFD9A8, 0.32);
    scene.add(ambient);

    const sun = new THREE.DirectionalLight(0xFFC98F, 2.6);
    sun.position.set(22, 16, 30);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.camera.left = -32;
    sun.shadow.camera.right = 32;
    sun.shadow.camera.top = 32;
    sun.shadow.camera.bottom = -24;
    sun.shadow.camera.near = 1;
    sun.shadow.camera.far = 90;
    sun.shadow.bias = -0.0006;
    sun.shadow.normalBias = 0.03;
    scene.add(sun);

    const rim = new THREE.DirectionalLight(0xFF8C5A, 0.5);
    rim.position.set(-18, 10, -22);
    scene.add(rim);

    /* ---------------- 3. Procedural PBR texture set ---------------- */
    const woodTex = createLuxuryWoodParquetTexture();
    const marbleTex = createCalacattaMarbleTexture();
    const plasterTex = createMicrocementWallTexture();
    const linenTex = createWovenFabricTexture('#B7AC9F');
    const slateTex = createWovenFabricTexture('#8E9AA6');
    const plushTex = createWovenFabricTexture('#D9D2C4');
    const slatTex = createAcousticSlatWoodTexture();
    const sunsetTex = createGoldenHourSkyTexture();
    const grassTex = createZenGrassTexture();
    const graniteTex = createGraniteSlabTexture();
    const woodRoughTex = createWoodRoughnessMap();
    const marbleRoughTex = createMarbleRoughnessMap();
    const coveGlows: THREE.MeshStandardMaterial[] = [];

    // Shared material palette
    const matWalnut = new THREE.MeshStandardMaterial({ color: 0x4A3221, roughness: 0.55, metalness: 0.06 });
    const matCharcoal = new THREE.MeshStandardMaterial({ color: 0x23272C, roughness: 0.42, metalness: 0.45 });
    const matBlack = new THREE.MeshStandardMaterial({ color: 0x0E1114, roughness: 0.3, metalness: 0.7 });
    const matStainless = new THREE.MeshStandardMaterial({ color: 0xBEC6CE, roughness: 0.22, metalness: 0.95 });
    const matGlass = new THREE.MeshPhysicalMaterial({
      color: 0xEAF4F6,
      roughness: 0.04,
      metalness: 0,
      transparent: true,
      opacity: 0.22,
      clearcoat: 1,
      clearcoatRoughness: 0.08,
      side: THREE.DoubleSide,
    });
    const matLinen = new THREE.MeshStandardMaterial({ map: linenTex, roughness: 0.92, metalness: 0 });
    const matSlate = new THREE.MeshStandardMaterial({ map: slateTex, roughness: 0.85, metalness: 0 });
    const matPlush = new THREE.MeshStandardMaterial({ map: plushTex, roughness: 0.9, metalness: 0 });

    const addBox = (
      w: number, h: number, d: number,
      x: number, y: number, z: number,
      mat?: THREE.Material,
      cast = true,
    ) => {
      const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat ?? matWalnut);
      m.position.set(x, y, z);
      m.castShadow = cast;
      m.receiveShadow = true;
      scene.add(m);
      return m;
    };

    const addCyl = (
      rTop: number, rBot: number, h: number,
      x: number, y: number, z: number,
      mat?: THREE.Material, seg = 24,
    ) => {
      const m = new THREE.Mesh(new THREE.CylinderGeometry(rTop, rBot, h, seg), mat ?? matWalnut);
      m.position.set(x, y, z);
      m.castShadow = true;
      m.receiveShadow = true;
      scene.add(m);
      return m;
    };

    // Rounded/beveled box helper — softens every hard edge (Japandi look)
    const addRounded = (
      w: number, h: number, d: number,
      x: number, y: number, z: number,
      mat: THREE.Material, radius = 0.045, cast = true,
    ) => {
      const geo = new RoundedBoxGeometry(w, h, d, 4, Math.min(radius, Math.min(w, h, d) / 2.01));
      const m = new THREE.Mesh(geo, mat);
      m.position.set(x, y, z);
      m.castShadow = cast;
      m.receiveShadow = true;
      scene.add(m);
      return m;
    };

    /* ---------------- 4. Villa shell: floor / ceiling / walls ---------------- */
    // Wide-plank oak parquet throughout the interior
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(28, 24),
      new THREE.MeshStandardMaterial({ map: woodTex, roughnessMap: woodRoughTex, roughness: 1.0, metalness: 0.06 }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // Linen plaster ceiling
    const ceiling = new THREE.Mesh(
      new THREE.PlaneGeometry(28, 24),
      new THREE.MeshStandardMaterial({ map: plasterTex, roughness: 0.95 }),
    );
    ceiling.position.y = 3.6;
    ceiling.rotation.x = Math.PI / 2;
    scene.add(ceiling);

    const wallMat = new THREE.MeshStandardMaterial({ map: plasterTex, roughness: 0.92, metalness: 0 });
    const baseboardMat = new THREE.MeshStandardMaterial({ color: 0x3A2B1E, roughness: 0.5, metalness: 0.15 });

    const createWall = (w: number, h: number, d: number, x: number, y: number, z: number) => {
      addBox(w, h, d, x, y, z, wallMat);
      // walnut baseboard profile
      addBox(w > d ? w : 0.6, 0.16, d > w ? d : 0.6, x, 0.08, z, baseboardMat, false);
      // crown moulding shadow line
      addBox(w, 0.06, d, x, h - 0.03, z, baseboardMat, false);
    };

    // Outer perimeter
    createWall(28, 3.6, 0.5, 0, 1.8, -12);
    createWall(0.5, 3.6, 24, -14, 1.8, 0);
    createWall(0.5, 3.6, 24, 14, 1.8, 0);
    createWall(12.4, 3.6, 0.5, -7.8, 1.8, 12);
    createWall(12.4, 3.6, 0.5, 7.8, 1.8, 12);

    // Doorway lintel above the entrance
    addBox(3.4, 0.8, 0.5, 0, 3.2, 12, wallMat);

    // Glass sliding entrance door (warm dusk garden backdrop)
    addBox(3.4, 0.12, 0.18, 0, 2.94, 12.02, matCharcoal, false);
    const doorGlass = new THREE.Mesh(
      new THREE.PlaneGeometry(1.55, 2.45),
      new THREE.MeshPhysicalMaterial({
        color: 0xEAF4F6,
        roughness: 0.04,
        metalness: 0,
        transparent: true,
        opacity: 0.25,
        clearcoat: 1,
        side: THREE.DoubleSide,
      }),
    );
    doorGlass.position.set(-0.83, 1.35, 12.05);
    scene.add(doorGlass);
    const doorGlass2 = doorGlass.clone();
    doorGlass2.position.x = 0.83;
    scene.add(doorGlass2);
    addBox(0.06, 2.45, 0.1, 0, 1.35, 12.05, matCharcoal, false);
    addBox(3.3, 0.1, 0.12, 0, 0.06, 12.05, matCharcoal, false);

    // Garden-facing picture window (north wall) with sunset backdrop
    addBox(6, 2.2, 0.2, 7, 2.0, -11.9, matCharcoal, false);
    const sunsetBackdrop = new THREE.Mesh(
      new THREE.PlaneGeometry(5.7, 2.0),
      new THREE.MeshBasicMaterial({ map: sunsetTex }),
    );
    sunsetBackdrop.position.set(7, 2.05, -12.02);
    scene.add(sunsetBackdrop);
    for (let mx = -1.9; mx <= 1.9; mx += 1.9) {
      addBox(0.07, 2.0, 0.08, 7 + mx, 2.0, -11.92, matCharcoal, false);
    }
    addBox(6.1, 0.08, 0.1, 7, 3.1, -11.92, matCharcoal, false);
    addBox(6.1, 0.08, 0.1, 7, 0.92, -11.92, matCharcoal, false);

    // Interior partitions (from blueprint): center vertical + horizontal dividers
    createWall(0.5, 3.6, 8, 0, 1.8, -8);
    createWall(0.5, 3.6, 8, 0, 1.8, 8);
    createWall(8, 3.6, 0.5, -10, 1.8, 0);
    createWall(8, 3.6, 0.5, 10, 1.8, 0);

    /* ---------------- 5. KITCHEN — Japandi luxury ---------------- */
    // Monolithic charcoal base with Calacatta slab
    addBox(5.2, 0.92, 2.6, -7, 0.46, -6, matCharcoal);
    addRounded(5.35, 0.12, 2.75, -7, 0.97, -6,
      new THREE.MeshStandardMaterial({
        map: marbleTex, roughnessMap: marbleRoughTex, roughness: 0.9, metalness: 0.08,
      }), 0.05);

    // Warm indirect LED cove strip under the island overhang
    const islandCoveMat = new THREE.MeshStandardMaterial({
      color: 0xFFD9A0, emissive: 0xFFC178, emissiveIntensity: 0, roughness: 0.5,
    });
    coveGlows.push(islandCoveMat);
    addBox(5.2, 0.03, 0.04, -7, 0.9, -4.68, islandCoveMat, false);
    addBox(5.2, 0.03, 0.04, -7, 0.9, -7.32, islandCoveMat, false);

    // Induction cooktop: black glass with glowing rings
    addBox(2.0, 0.02, 1.2, -8.05, 1.09, -6, matBlack, false);
    [0, 1].forEach((i) => {
      [-5.75, -6.25].forEach((ringZ) => {
        const ring = new THREE.Mesh(
          new THREE.TorusGeometry(0.24 + i * 0.02, 0.02, 10, 30),
          new THREE.MeshStandardMaterial({ color: 0x1A1E22, emissive: 0x000000, emissiveIntensity: 0, roughness: 0.3 }),
        );
        ring.rotation.x = Math.PI / 2;
        ring.position.set(-8.05 + (i === 0 ? -0.18 : 0.38), 1.105, ringZ);
        scene.add(ring);
        cooktopRingsRef.current.push(ring);
      });
    });

    // Black sink + gooseneck faucet
    addBox(1.3, 0.1, 1.0, -6.15, 1.0, -6.5, matBlack, false);
    addBox(1.1, 0.08, 0.8, -6.15, 0.945, -6.5, new THREE.MeshStandardMaterial({ color: 0x050709, roughness: 0.2 }));
    addCyl(0.045, 0.06, 0.06, -6.15, 1.09, -6.8, matStainless, 16);
    const faucetStem = addCyl(0.022, 0.022, 0.32, -6.15, 1.68, -6.8, matStainless, 12);
    faucetStem.rotation.z = -0.25;
    const faucetArm = addCyl(0.02, 0.02, 0.22, -6.28, 1.76, -6.68, matStainless, 12);
    faucetArm.rotation.z = 0.55;

    // Stainless fridge + handle
    addBox(2.2, 2.9, 2.0, -12, 1.45, -10, matStainless);
    addBox(0.06, 1.4, 0.05, -10.98, 1.7, -10.3, matBlack, false);

    // Two floating bar stools
    [-5.8, -8.2].forEach((xx) => {
      addCyl(0.02, 0.02, 0.7, xx, 0.35, -3.55, matBlack, 8);
      addBox(0.42, 0.06, 0.42, xx, 0.76, -3.55, matWalnut, false);
      addBox(0.42, 0.3, 0.34, xx, 0.97, -3.55, matSlate, false);
    });

    // Open wall shelf with crystal glasses
    addBox(2.6, 0.06, 0.4, -12, 2.2, -5.4, matWalnut, false);
    [-12.4, -11.7, -11.0].forEach((gx) => {
      const glass = new THREE.Mesh(new THREE.CylinderGeometry(0.05, 0.035, 0.16, 18), matGlass);
      glass.position.set(gx, 2.34, -5.4);
      glass.castShadow = true;
      scene.add(glass);
    });

    /* ---------------- 6. LIVING — lounge with walnut slat TV wall ---------------- */
    // Slatted accent wall (acoustic walnut) behind the smart TV
    const slatPanel = addBox(5.6, 2.6, 0.1, 7, 1.75, 11.82,
      new THREE.MeshStandardMaterial({ map: slatTex, roughness: 0.6, metalness: 0.04 }), false);
    slatPanel.position.set(7, 1.75, 11.82);

    // Slim 75" smart TV (1.66 x 0.93 m panel) with anodized black frame
    addBox(1.74, 1.01, 0.055, 7, 2.0, 11.72, matBlack);
    const tvScreen = new THREE.Mesh(
      new THREE.PlaneGeometry(1.66, 0.93),
      new THREE.MeshStandardMaterial({ color: 0x0A0E14, roughness: 0.12, metalness: 0.85, emissive: 0x14243A, emissiveIntensity: 0.55, side: THREE.DoubleSide }),
    );
    tvScreen.position.set(7, 2.0, 11.76);
    tvScreen.rotation.y = Math.PI; // face into the lounge
    scene.add(tvScreen);
    // Bias LED strips (warm accent above & below panel)
    addBox(1.74, 0.05, 0.02, 7, 2.56, 11.6,
      new THREE.MeshStandardMaterial({ color: 0xFFB066, emissive: 0xFF9A4D, emissiveIntensity: 0.9 }), false);
    addBox(1.74, 0.05, 0.02, 7, 1.44, 11.6,
      new THREE.MeshStandardMaterial({ color: 0xFFB066, emissive: 0xFF9A4D, emissiveIntensity: 0.7 }), false);

    // L-shape modular sofa with throw pillows & folded throw
    const sofaMat = new THREE.MeshStandardMaterial({ map: plushTex, roughness: 0.88, metalness: 0 });
    addBox(6.4, 1.05, 2.6, 7, 0.55, 7.6, sofaMat);
    addBox(6.4, 1.1, 0.55, 7, 1.15, 8.9, sofaMat);
    addBox(0.6, 1.0, 2.6, 3.8, 1.0, 7.6, sofaMat);
    // Chaise extension (L leg)
    addBox(2.6, 1.05, 1.5, 10.4, 0.55, 8.6, sofaMat);
    addBox(2.6, 0.5, 0.55, 10.4, 1.15, 9.4, sofaMat);

    // Accent pillows
    const pillowPositions: Array<[number, number, number, number, number, number]> = [
      // x, y, z, rotY, width, height  (varied sizes: lumbar, square, large)
      [5.25, 1.3, 8.72, 0.15, 0.72, 0.26],
      [6.35, 1.28, 8.75, -0.12, 0.56, 0.2],
      [7.5, 1.32, 8.74, 0.1, 0.64, 0.34],
      [4.0, 1.24, 7.95, 0.5, 0.5, 0.22],
    ];
    pillowPositions.forEach(([px, py, pz, rot, pw, ph]) => {
      const p = addRounded(pw, ph, 0.42, px, py, pz,
        new THREE.MeshStandardMaterial({ map: slateTex, roughness: 0.88 }), 0.08);
      p.rotation.y = rot;
    });
    // Folded throw blanket draped on the chaise
    addBox(1.1, 0.1, 1.35, 9.9, 1.16, 8.6, new THREE.MeshStandardMaterial({ color: 0xA98B6E, roughness: 0.95 }), false);

    // Round marble coffee table with books, vase & monstera
    addCyl(0.55, 0.55, 0.1, 7, 0.42, 4.4,
      new THREE.MeshStandardMaterial({
        map: marbleTex, roughnessMap: marbleRoughTex, roughness: 0.9, metalness: 0.1,
      }), 36);
    addCyl(0.08, 0.11, 0.38, 7, 0.2, 4.4, matCharcoal, 14);
    // Design books stack
    addBox(0.5, 0.045, 0.34, 6.72, 0.5, 4.28, new THREE.MeshStandardMaterial({ color: 0x8B5E3C, roughness: 0.7 }), false);
    addBox(0.44, 0.05, 0.3, 6.74, 0.545, 4.3, new THREE.MeshStandardMaterial({ color: 0x3C4A5C, roughness: 0.7 }), false);
    addBox(0.38, 0.045, 0.26, 6.76, 0.59, 4.32, new THREE.MeshStandardMaterial({ color: 0xD9CDB0, roughness: 0.75 }), false);
    // Ceramic vase with monstera leaves
    addCyl(0.09, 0.11, 0.24, 7.35, 0.54, 4.55, new THREE.MeshStandardMaterial({ color: 0xCBB89A, roughness: 0.6 }), 18);
    const leafMat = new THREE.MeshStandardMaterial({
      color: 0x2F5D3E, roughness: 0.75, side: THREE.DoubleSide,
    });
    const leafShape = new THREE.Shape();
    leafShape.moveTo(0, 0);
    leafShape.bezierCurveTo(0.2, 0.14, 0.24, 0.42, 0, 0.6);
    leafShape.bezierCurveTo(-0.24, 0.42, -0.2, 0.14, 0, 0);
    const leafGeo = new THREE.ShapeGeometry(leafShape);
    const leafSpots: Array<[number, number, number, number]> = [
      [0.28, 0.62, 0.35, -0.9], [0.12, 0.52, -0.24, 0.8], [-0.3, 0.56, 0.2, 2.6],
      [-0.12, 0.7, -0.32, -2.2], [0.22, 0.66, -0.42, 3.2],
    ];
    leafSpots.forEach(([dx, dy, dz, ry]) => {
      const leaf = new THREE.Mesh(leafGeo, leafMat);
      leaf.position.set(7.35 + dx, 0.68 + dy, 4.55 + dz);
      leaf.rotation.y = ry;
      leaf.rotation.x = -0.4;
      leaf.scale.set(0.45, 0.5, 0.45);
      scene.add(leaf);
    });

    // --- Smart objects: printed newspaper, juice glass, sneakers ---
    const newspaperTex = createNewspaperTexture();
    const newspaper = new THREE.Mesh(
      new THREE.PlaneGeometry(0.46, 0.58),
      new THREE.MeshStandardMaterial({ map: newspaperTex, roughness: 0.95, metalness: 0 }),
    );
    newspaper.position.set(7.62, 0.478, 4.28);
    newspaper.rotation.x = -Math.PI / 2;
    newspaper.rotation.z = 0.22;
    scene.add(newspaper);
    // second folded page underneath for thickness
    const newspaperUnder = new THREE.Mesh(
      new THREE.PlaneGeometry(0.44, 0.55),
      new THREE.MeshStandardMaterial({ color: 0xE2DACA, roughness: 0.95 }),
    );
    newspaperUnder.position.set(7.6, 0.474, 4.27);
    newspaperUnder.rotation.x = -Math.PI / 2;
    newspaperUnder.rotation.z = 0.26;
    scene.add(newspaperUnder);

    // Glass of fresh juice on the marble island (specular glass + liquid)
    const juiceGlass = new THREE.Mesh(
      new THREE.CylinderGeometry(0.052, 0.044, 0.15, 24, 1, true),
      new THREE.MeshPhysicalMaterial({
        color: 0xEAF4F6, roughness: 0.03, metalness: 0,
        transparent: true, opacity: 0.28, clearcoat: 1,
        side: THREE.DoubleSide,
      }),
    );
    juiceGlass.position.set(-5.55, 1.115, -6.42);
    scene.add(juiceGlass);
    const juice = new THREE.Mesh(
      new THREE.CylinderGeometry(0.046, 0.04, 0.095, 22),
      new THREE.MeshPhysicalMaterial({
        color: 0xE8902A, roughness: 0.12, metalness: 0.05,
        transparent: true, opacity: 0.92, clearcoat: 0.7,
      }),
    );
    juice.position.set(-5.55, 1.09, -6.42);
    scene.add(juice);
    // juice glass base
    const juiceBase = new THREE.Mesh(
      new THREE.CylinderGeometry(0.046, 0.046, 0.008, 22),
      new THREE.MeshPhysicalMaterial({ color: 0xEAF4F6, roughness: 0.05, transparent: true, opacity: 0.35 }),
    );
    juiceBase.position.set(-5.55, 1.043, -6.42);
    scene.add(juiceBase);

    // Entry shoe rack with a pair of modern sneakers
    addBox(1.9, 0.05, 0.42, 2.15, 0.26, 11.35, matWalnut, false);
    addBox(1.9, 0.05, 0.42, 2.15, 0.46, 11.35, matWalnut, false);
    [1.5, 2.8].forEach((rx) => {
      addBox(0.05, 0.46, 0.05, rx, 0.23, 11.15, matBlack, false);
      addBox(0.05, 0.46, 0.05, rx, 0.23, 11.55, matBlack, false);
    });
    const sneakerUpper = new THREE.MeshStandardMaterial({ color: 0xF2F2F0, roughness: 0.75 });
    const sneakerAccent = new THREE.MeshStandardMaterial({ color: 0x2E3238, roughness: 0.6 });
    const sneakerSole = new THREE.MeshStandardMaterial({ color: 0xE8E4DC, roughness: 0.5 });
    [-0.22, 0.14].forEach((ox, si) => {
      const sx = 2.15 + ox;
      // sole
      addRounded(0.12, 0.05, 0.3, sx, 0.53, 11.33, sneakerSole, 0.02);
      // upper
      const upper = addRounded(0.11, 0.09, 0.24, sx, 0.6, 11.32, sneakerUpper, 0.035);
      upper.rotation.x = -0.04;
      // toe cap + heel accent
      addRounded(0.11, 0.06, 0.08, sx, 0.585, 11.2, si === 0 ? sneakerAccent : sneakerUpper, 0.025);
      // laces
      for (let li = 0; li < 3; li++) {
        addBox(0.07, 0.008, 0.012, sx, 0.648, 11.28 + li * 0.035, sneakerAccent, false);
      }
    });

    // Smart speaker (HomePod style) on a side credenza
    addBox(0.9, 0.55, 0.9, 3.2, 0.28, 2.6, matWalnut, true);
    const speakerBody = addCyl(0.34, 0.34, 0.72, 3.2, 0.92, 2.6, matSlate, 36);
    speakerMeshRef.current = speakerBody;
    addCyl(0.3, 0.32, 0.05, 3.2, 1.3, 2.6, matLinen, 32);
    const ringTex = createSpeakerRingCanvas();
    speakerRingRef.current = ringTex;
    const eqRing = new THREE.Mesh(
      new THREE.PlaneGeometry(0.74, 0.74),
      new THREE.MeshBasicMaterial({ map: ringTex.texture, transparent: true, depthWrite: false, side: THREE.DoubleSide }),
    );
    eqRing.position.set(3.2, 1.34, 2.6);
    eqRing.rotation.x = -Math.PI / 2;
    scene.add(eqRing);

    /* ---------------- 7. BEDROOM — floating platform bed ---------------- */
    // Woven rug
    const bedRug = new THREE.Mesh(
      new THREE.PlaneGeometry(6.5, 7.5),
      new THREE.MeshStandardMaterial({ map: slateTex, roughness: 0.94 }),
    );
    bedRug.rotation.x = -Math.PI / 2;
    bedRug.position.set(7.5, 0.02, -7.0);
    bedRug.receiveShadow = true;
    scene.add(bedRug);

    // Floating platform bed with 4 legs
    [[5.6, -9.0], [9.4, -9.0], [5.6, -5.0], [9.4, -5.0]].forEach(([lx, lz]) => {
      addCyl(0.035, 0.035, 0.34, lx as number, 0.17, lz as number, matBlack, 10);
    });
    addBox(4.8, 0.22, 6.2, 7.5, 0.36, -7.0, new THREE.MeshStandardMaterial({ color: 0x5C402A, roughness: 0.55 }));
    // Plush mattress
    addRounded(4.6, 0.42, 6.0, 7.5, 0.62, -7.0, matPlush, 0.09);
    // Upholstered headboard (padded) + walnut slat accent
    addBox(4.7, 1.35, 0.22, 7.5, 1.05, -10.15, new THREE.MeshStandardMaterial({ map: slateTex, roughness: 0.85 }));
    addBox(5.3, 2.4, 0.12, 7.5, 1.6, -10.35,
      new THREE.MeshStandardMaterial({ map: slatTex, roughness: 0.6 }), false);
    // Natural duvet with organic folds
    addRounded(4.4, 0.28, 3.6, 7.5, 0.92, -5.8, matPlush, 0.1);
    addRounded(4.3, 0.16, 3.7, 7.5, 1.1, -5.75, matPlush, 0.08);
    const duvetFold = addBox(1.2, 0.12, 3.8, 5.5, 1.08, -5.7, matPlush, false);
    duvetFold.rotation.y = 0.12;
    // Pillows
    [6.2, 8.8].forEach((px) => {
      addRounded(1.35, 0.22, 0.85, px, 1.04, -9.45,
        new THREE.MeshStandardMaterial({ map: plushTex, roughness: 0.9 }), 0.07);
    });

    // Floating nightstands + warm pendant lights (both mapped to bedroom_lamp)
    const bedsideFixtures: Fixture = { lights: [], glows: [], bulbs: [] };
    [-10.2, -3.8].forEach((bz) => {
      const stand = addBox(1.1, 0.35, 0.9, 4.1, 0.68, bz, matWalnut);
      stand.rotation.y = 0.06;
      const shadeMat = new THREE.MeshStandardMaterial({
        color: 0xF2EAD8, emissive: 0xFFB454, emissiveIntensity: 0,
        transparent: true, opacity: 0.92, roughness: 0.32,
      });
      const shade = addCyl(0.14, 0.2, 0.34, 4.1, 1.5, bz, shadeMat, 20);
      const bulb = new THREE.Mesh(
        new THREE.SphereGeometry(0.11, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0x332211 }),
      );
      bulb.position.set(4.1, 1.32, bz);
      scene.add(bulb);
      addCyl(0.014, 0.014, 0.8, 4.1, 1.85, bz, matBlack, 8);
      const light = new THREE.PointLight(0xFFB454, 0, 9, 1.8);
      light.position.set(4.1, 1.4, bz);
      scene.add(light);
      bedsideFixtures.lights.push({ light, intensity: 9 });
      bedsideFixtures.glows.push({ mat: shadeMat, intensity: 1.6 });
      bedsideFixtures.bulbs.push(bulb);
    });
    lightsMapRef.current['bedroom_lamp'] = bedsideFixtures;

    // Minimalist wardrobe
    addBox(1.2, 2.7, 4.2, 13.35, 1.35, -5.4, new THREE.MeshStandardMaterial({ color: 0x2E231B, roughness: 0.5 }));
    addBox(0.04, 2.5, 4.0, 12.92, 1.35, -5.4, matBlack, false);

    // Ceiling-mounted linear AC diffuser + falling air plume (heat/cool aura)
    const createVent = (key: string, x: number, z: number) => {
      // recessed linear slot in the ceiling
      addBox(2.4, 0.06, 0.26, x, 3.57, z, matCharcoal, false);
      const slatMat = new THREE.MeshStandardMaterial({ color: 0x0C1014, roughness: 0.42, metalness: 0.3 });
      for (let sxx = -1.0; sxx <= 1.0; sxx += 0.2) {
        addBox(0.05, 0.03, 0.22, x + sxx, 3.545, z, slatMat, false);
      }
      // slim louvre frame
      addBox(2.5, 0.02, 0.02, x, 3.53, z - 0.13, matBlack, false);
      addBox(2.5, 0.02, 0.02, x, 3.53, z + 0.13, matBlack, false);

      const count = 40;
      const positions = new Float32Array(count * 3);
      for (let i = 0; i < count; i++) {
        positions[i * 3] = x + (Math.random() - 0.5) * 2.1;
        positions[i * 3 + 1] = 2.9 + Math.random() * 0.6;
        positions[i * 3 + 2] = z + (Math.random() - 0.5) * 0.3;
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
      const mat = new THREE.PointsMaterial({
        color: 0x5ED3E8,
        size: 0.055,
        transparent: true,
        opacity: 0,
        depthWrite: false,
        blending: THREE.AdditiveBlending,
      });
      const points = new THREE.Points(geo, mat);
      scene.add(points);
      ventParticlesRef.current[key] = points;
    };
    createVent('kitchen', -7, -8.2);
    createVent('bedroom', 7, -8.2);
    createVent('washroom', -7, 3.4);

    /* ---------------- 8. WASHROOM — luxury stone spa ---------------- */
    // Black marble feature wall + round backlit mirror
    addBox(4.4, 2.6, 0.1, -8, 1.5, 11.62,
      new THREE.MeshStandardMaterial({ color: 0x14161A, roughness: 0.35, metalness: 0.4 }), false);
    const mirrorRingMat = new THREE.MeshStandardMaterial({
      color: 0x1A1E24, emissive: 0xFFD9A0, emissiveIntensity: 0, roughness: 0.3, metalness: 0.6,
    });
    mirrorRingMatRef.current = mirrorRingMat;
    const mirrorRing = new THREE.Mesh(new THREE.TorusGeometry(0.62, 0.035, 14, 40), mirrorRingMat);
    mirrorRing.position.set(-8, 2.1, 11.54);
    mirrorRing.rotation.y = Math.PI;
    scene.add(mirrorRing);
    const mirrorDisc = new THREE.Mesh(
      new THREE.CircleGeometry(0.56, 40),
      new THREE.MeshStandardMaterial({ color: 0xCFE3EA, roughness: 0.05, metalness: 0.95, side: THREE.DoubleSide }),
    );
    mirrorDisc.position.set(-8, 2.1, 11.81);
    mirrorDisc.rotation.y = Math.PI; // face the vanity area
    scene.add(mirrorDisc);

    // Floating vanity with Calacatta top & basin
    addBox(3.4, 0.55, 0.95, -8, 0.65, 11.3, matWalnut);
    addBox(3.6, 0.08, 1.05, -8, 0.96, 11.3,
      new THREE.MeshStandardMaterial({
        map: marbleTex, roughnessMap: marbleRoughTex, roughness: 0.9, metalness: 0.08,
      }));
    addCyl(0.24, 0.26, 0.16, -8, 1.06, 10.9, new THREE.MeshStandardMaterial({ color: 0xE9EEF0, roughness: 0.12, metalness: 0.05 }), 26);
    addCyl(0.14, 0.16, 0.1, -8, 1.12, 10.9, matBlack, 26);
    const tap = addCyl(0.02, 0.02, 0.4, -8, 1.42, 11.6, matStainless, 12);
    tap.rotation.z = 0.2;

    // Soaking tub
    addBox(3.4, 0.95, 5.0, -8, 0.5, 7, new THREE.MeshStandardMaterial({ color: 0xE2E7E8, roughness: 0.18, metalness: 0.12 }));
    addBox(3.0, 0.14, 4.6, -8, 0.98, 7,
      new THREE.MeshStandardMaterial({ color: 0xBFE0E6, roughness: 0.06, metalness: 0.2, transparent: true, opacity: 0.75 }));

    // Fluted glass shower screen + rain shower head
    const flutedTexCanvas = document.createElement('canvas');
    flutedTexCanvas.width = 128;
    flutedTexCanvas.height = 256;
    const fctx = flutedTexCanvas.getContext('2d')!;
    fctx.fillStyle = 'rgba(255,255,255,0.95)';
    fctx.fillRect(0, 0, 128, 256);
    for (let sx = 0; sx < 128; sx += 8) {
      fctx.fillStyle = 'rgba(255,255,255,0.35)';
      fctx.fillRect(sx, 0, 3, 256);
      fctx.fillStyle = 'rgba(0,0,0,0.12)';
      fctx.fillRect(sx + 3, 0, 2, 256);
    }
    const flutedTex = new THREE.CanvasTexture(flutedTexCanvas);
    flutedTex.colorSpace = THREE.SRGBColorSpace;
    const flutedGlass = new THREE.Mesh(
      new THREE.PlaneGeometry(0.05, 2.3),
      new THREE.MeshPhysicalMaterial({
        map: flutedTex,
        color: 0xBFE6F2,
        alphaMap: flutedTex,
        transparent: true,
        opacity: 0.32,
        roughness: 0.08,
        metalness: 0.1,
        side: THREE.DoubleSide,
      }),
    );
    flutedGlass.position.set(-4.9, 1.2, 9.5);
    scene.add(flutedGlass);

    const rainHead = addCyl(0.5, 0.56, 0.05, -4.9, 3.42, 9.5, matStainless, 30);
    rainHead.position.y = 3.42;
    addCyl(0.035, 0.035, 0.12, -4.9, 3.56, 9.5, matBlack, 10);

    /* ---------------- 9. FRONT GARDEN — modern zen landscape ---------------- */
    const gardenGrass = new THREE.Mesh(
      new THREE.PlaneGeometry(48, 24),
      new THREE.MeshStandardMaterial({ map: grassTex, roughness: 0.85, metalness: 0 }),
    );
    gardenGrass.rotation.x = -Math.PI / 2;
    gardenGrass.position.set(0, -0.012, 23.5);
    gardenGrass.receiveShadow = true;
    scene.add(gardenGrass);

    // Granite stepping-stone path from entrance to spawn
    const path = new THREE.Mesh(
      new THREE.PlaneGeometry(3.6, 15),
      new THREE.MeshStandardMaterial({ map: graniteTex, roughness: 0.8, metalness: 0.04 }),
    );
    path.rotation.x = -Math.PI / 2;
    path.position.set(0, 0.012, 19.5);
    path.receiveShadow = true;
    scene.add(path);

    // Privacy fence: horizontal dark walnut slats (both wings + gate opening)
    const fenceMat = new THREE.MeshStandardMaterial({ color: 0x3A2A1E, roughness: 0.6, metalness: 0.06 });
    const buildFence = (fromX: number, toX: number, z: number) => {
      const len = Math.abs(toX - fromX);
      const cx = (fromX + toX) / 2;
      for (let row = 0; row < 4; row++) {
        const y = 0.45 + row * 0.42;
        addBox(len, 0.12, 0.2, cx, y, z, fenceMat, false);
      }
      for (const px of [fromX, toX]) {
        addBox(0.16, 2.1, 0.2, px, 1.1, z, matCharcoal, false);
      }
    };
    buildFence(-21, -3.2, 32.4);
    buildFence(3.2, 21, 32.4);

    // Bamboo clusters in concrete planters
    const createBamboo = (x: number, z: number, count: number) => {
      addBox(0.9, 0.55, 0.9, x - 0.15, 0.28, z, new THREE.MeshStandardMaterial({ color: 0x9A9487, roughness: 0.7 }), true);
      addBox(0.8, 0.45, 0.8, x - 0.15, 0.23, z, new THREE.MeshStandardMaterial({ color: 0x4C4F46, roughness: 0.6 }), false);
      const leafM = new THREE.MeshStandardMaterial({ color: 0x5C8A50, roughness: 0.7, side: THREE.DoubleSide });
      const leafGeo2 = new THREE.PlaneGeometry(0.16, 0.4);
      for (let i = 0; i < count; i++) {
        const bx = x - 0.25 + i * 0.2;
        const h = 2.2 + Math.random() * 1.3;
        addCyl(0.028, 0.034, h, bx, 0.5 + h / 2, z + (Math.random() - 0.5) * 0.4,
          new THREE.MeshStandardMaterial({ color: 0x4E7A45, roughness: 0.6, metalness: 0.08 }), 8);
        for (let l = 0; l < 5; l++) {
          const lf = new THREE.Mesh(leafGeo2, leafM);
          lf.position.set(bx + (Math.random() - 0.5) * 0.2, 0.5 + h * 0.82 + (Math.random() - 0.5) * 0.3, z + (Math.random() - 0.5) * 0.2);
          lf.rotation.x = Math.PI / 2 + (Math.random() - 0.5);
          lf.rotation.y = Math.random() * Math.PI;
          scene.add(lf);
        }
      }
    };
    createBamboo(-12, 20, 5);
    createBamboo(12, 19, 4);
    createBamboo(-16, 26, 3);
    createBamboo(16, 27, 3);

    // Garden uplight spots along the path (warm)
    [-1.2, 0.0, 1.2].forEach((ux) => {
      const uplight = new THREE.Mesh(
        new THREE.CylinderGeometry(0.05, 0.08, 0.12, 12),
        new THREE.MeshStandardMaterial({ color: 0x23272C, emissive: 0xFFD9A0, emissiveIntensity: 0.35, roughness: 0.5 }),
      );
      uplight.position.set(ux, 0.08, 15.5);
      scene.add(uplight);
      const tiny = new THREE.PointLight(0xFFD9A0, 0.7, 5, 1.7);
      tiny.position.set(ux, 0.5, 15.5);
      scene.add(tiny);
    });

    // Modern teak garden bench
    addBox(2.2, 0.08, 0.65, 4.6, 0.44, 17, new THREE.MeshStandardMaterial({ color: 0x6E4A2F, roughness: 0.6 }), true);
    addBox(2.2, 0.5, 0.06, 4.6, 0.75, 16.68, new THREE.MeshStandardMaterial({ color: 0x6E4A2F, roughness: 0.6 }), true);
    addCyl(0.05, 0.05, 0.4, 3.9, 0.2, 17.2, matBlack, 8);
    addCyl(0.05, 0.05, 0.4, 5.3, 0.2, 17.2, matBlack, 8);

    /* ---------------- 10. Ceiling pendants & room light fixtures ---------------- */
    const fixtureMat = new THREE.MeshStandardMaterial({ color: 0x14181D, roughness: 0.35, metalness: 0.6 });
    const createPendant = (
      x: number, z: number, y = 3.4,
      shadeMat?: THREE.MeshStandardMaterial,
    ) => {
      const canopy = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.3, 0.08, 20), fixtureMat);
      canopy.position.set(x, y + 0.2, z);
      scene.add(canopy);
      const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.02, 0.02, 0.5, 8), matBlack);
      stem.position.set(x, y - 0.05, z);
      scene.add(stem);
      const shade = new THREE.Mesh(
        new THREE.CylinderGeometry(0.26, 0.34, 0.38, 24),
        shadeMat ?? new THREE.MeshStandardMaterial({
          color: 0xE8E0CF, emissive: 0xFFD9A0, emissiveIntensity: 0,
          transparent: true, opacity: 0.94, roughness: 0.32,
        }),
      );
      shade.position.set(x, y - 0.35, z);
      shade.castShadow = true;
      scene.add(shade);
      const bulb = new THREE.Mesh(
        new THREE.SphereGeometry(0.13, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0x332211 }),
      );
      bulb.position.set(x, y - 0.5, z);
      scene.add(bulb);
      return { shade, bulb };
    };

    // Kitchen: 3 pendants sharing one glow material over the island
    const kitchenShadeMat = new THREE.MeshStandardMaterial({
      color: 0xE8E0CF, emissive: 0xFFD9A0, emissiveIntensity: 0,
      transparent: true, opacity: 0.94, roughness: 0.32,
    });
    const kitchenFixture: Fixture = { lights: [], glows: [], bulbs: [] };
    [-7.6, -7.0, -6.4].forEach((dx) => {
      const p = createPendant(dx, -6, 3.4, kitchenShadeMat);
      kitchenFixture.glows.push({ mat: kitchenShadeMat, intensity: 2.1 });
      kitchenFixture.bulbs.push(p.bulb);
    });
    const kitchenLight = new THREE.PointLight(0xFFD9A0, 0, 16, 1.7);
    kitchenLight.position.set(-7, 2.9, -6);
    scene.add(kitchenLight);
    kitchenFixture.lights.push({ light: kitchenLight, intensity: 26 });
    // Under-counter LED cove follows the kitchen lights
    coveGlows.forEach((m) => kitchenFixture.glows.push({ mat: m, intensity: 1.6 }));
    lightsMapRef.current['kitchen'] = kitchenFixture;

    // Bedroom: flush ceiling pendant
    const bp = createPendant(7, -6, 3.42);
    const bedroomLight = new THREE.PointLight(0xFFD9A0, 0, 15, 1.7);
    bedroomLight.position.set(7, 2.9, -6);
    scene.add(bedroomLight);
    lightsMapRef.current['bedroom'] = {
      lights: [{ light: bedroomLight, intensity: 24 }],
      glows: [{ mat: bp.shade.material as THREE.MeshStandardMaterial, intensity: 2.0 }],
      bulbs: [bp.bulb],
    };

    // Washroom: fluted glass globe
    const washShadeMat = new THREE.MeshStandardMaterial({
      color: 0xEAF2F4, emissive: 0xFFE3B8, emissiveIntensity: 0,
      transparent: true, opacity: 0.9, roughness: 0.25,
    });
    const wp = createPendant(-7, 6, 3.42, washShadeMat);
    const washroomLight = new THREE.PointLight(0xFFE3B8, 0, 14, 1.7);
    washroomLight.position.set(-7, 2.9, 6);
    scene.add(washroomLight);
    lightsMapRef.current['washroom'] = {
      lights: [{ light: washroomLight, intensity: 22 }],
      glows: [{ mat: washShadeMat, intensity: 2.0 }],
      bulbs: [wp.bulb],
    };

    // Recessed warm 3000K ceiling spotlights (real SpotLights: penumbra .6, decay 2)
    type SpotSpec = { x: number; z: number; room: 'kitchen' | 'bedroom' | 'washroom' | 'hall' };
    const spotSpecs: SpotSpec[] = [
      { x: -8.5, z: -8.5, room: 'kitchen' },
      { x: -5.5, z: -8.5, room: 'kitchen' },
      { x: -8.5, z: -3.6, room: 'kitchen' },
      { x: 5.5, z: -8.5, room: 'bedroom' },
      { x: 9.0, z: -8.5, room: 'bedroom' },
      { x: 9.0, z: -3.6, room: 'bedroom' },
      { x: -8.5, z: 3.5, room: 'washroom' },
      { x: -5.5, z: 8.0, room: 'washroom' },
      { x: 5.0, z: 3.0, room: 'hall' },
      { x: 9.0, z: 8.5, room: 'hall' },
    ];
    const spotGlowMat = new THREE.MeshStandardMaterial({
      color: 0xFFE8C4, emissive: 0xFFE8C4, emissiveIntensity: 0.16, roughness: 0.4,
    });
    spotSpecs.forEach(({ x: sx, z: sz, room }) => {
      // recessed fixture ring + frosted lens
      addCyl(0.1, 0.13, 0.045, sx, 3.585, sz, matBlack, 18);
      addCyl(0.062, 0.062, 0.02, sx, 3.566, sz, spotGlowMat, 16);

      const spot = new THREE.SpotLight(0xFFE8C4, 0, 11, 0.72, 0.6, 2);
      spot.position.set(sx, 3.55, sz);
      spot.target.position.set(sx, 0, sz);
      spot.castShadow = false;
      scene.add(spot);
      scene.add(spot.target);

      const zone = lightsMapRef.current[room];
      if (zone) {
        zone.lights.push({ light: spot, intensity: 34 });
      } else {
        // hall spots stay as a dim always-on architectural accent
        spot.intensity = 6;
      }
    });

    /* ---------------- 11. Wall thermostats & heaters ---------------- */
    const mountThermostat = (key: string, x: number, z: number, rotY = 0) => {
      const t = createThermostatCanvas();
      thermostatRef.current[key] = t;
      const panel = new THREE.Mesh(
        new THREE.PlaneGeometry(0.85, 0.55),
        new THREE.MeshBasicMaterial({ map: t.texture }),
      );
      panel.position.set(x, 1.55, z);
      panel.rotation.y = rotY;
      scene.add(panel);
    };
    mountThermostat('kitchen', -13.7, -4, Math.PI / 2);
    mountThermostat('bedroom', 13.7, -4, -Math.PI / 2);
    mountThermostat('washroom', -13.7, 4, Math.PI / 2);

    // Slim electric radiators below thermostats
    const createHeater = (name: string, x: number, z: number, rotY = 0) => {
      const heaterMat = new THREE.MeshStandardMaterial({
        color: 0x2C2F34, emissive: 0x000000, roughness: 0.55, metalness: 0.4,
      });
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(2.4, 1.05, 0.16), heaterMat);
      mesh.position.set(x, 0.62, z);
      mesh.rotation.y = rotY;
      mesh.castShadow = true;
      scene.add(mesh);
      const finMat = new THREE.MeshStandardMaterial({ color: 0x22252A, roughness: 0.5, metalness: 0.5 });
      for (let fx = -1.0; fx <= 1.0; fx += 0.2) {
        addBox(0.04, 1.0, 0.12, x + fx, 0.6, z, finMat, false);
      }
      heaterMeshRef.current[name] = mesh;
    };
    createHeater('kitchen', -13.7, -4, Math.PI / 2);
    createHeater('bedroom', 13.7, -4, -Math.PI / 2);
    createHeater('washroom', -13.7, 4, Math.PI / 2);

    /* ---------------- 12. FPP arms on camera ---------------- */
    const armsRig = createFPPArms();
    camera.add(armsRig.armsGroup);

    // Soft camera-mounted fill light so the arms always read against the scene
    const armFill = new THREE.PointLight(0xFFE3C0, 1.15, 3.2, 2);
    armFill.position.set(0, -0.05, -0.1);
    camera.add(armFill);

    scene.add(camera);

    /* ---------------- 13. Input: WASD + Space PTT + Pointer lock ---------------- */
    const onKeyDown = (e: KeyboardEvent) => {
      switch (e.code) {
        case 'KeyW': moveStateRef.current.forward = true; break;
        case 'KeyS': moveStateRef.current.backward = true; break;
        case 'KeyA': moveStateRef.current.left = true; break;
        case 'KeyD': moveStateRef.current.right = true; break;
        case 'ShiftLeft': moveStateRef.current.sprint = true; break;
        case 'Space':
          e.preventDefault();
          if (!e.repeat) {
            soundSystem.playPTTChirp(true);
            onVoiceTriggerStartRef.current();
          }
          break;
      }
    };
    const onKeyUp = (e: KeyboardEvent) => {
      switch (e.code) {
        case 'KeyW': moveStateRef.current.forward = false; break;
        case 'KeyS': moveStateRef.current.backward = false; break;
        case 'KeyA': moveStateRef.current.left = false; break;
        case 'KeyD': moveStateRef.current.right = false; break;
        case 'ShiftLeft': moveStateRef.current.sprint = false; break;
        case 'Space':
          e.preventDefault();
          soundSystem.playPTTChirp(false);
          onVoiceTriggerEndRef.current();
          break;
      }
    };
    const onWindowBlur = () => {
      moveStateRef.current.forward = false;
      moveStateRef.current.backward = false;
      moveStateRef.current.left = false;
      moveStateRef.current.right = false;
      moveStateRef.current.sprint = false;
      if (isRecordingRef.current) {
        soundSystem.playPTTChirp(false);
        onVoiceTriggerEndRef.current();
      }
    };

    // Pointer lock + damped mouse look (no angular snapping)
    let targetYaw = 0;
    let targetPitch = 0;
    let yaw = 0;
    let pitch = 0;
    const onMouseMove = (e: MouseEvent) => {
      if (document.pointerLockElement !== renderer.domElement) return;
      const sens = 0.0022;
      targetYaw -= e.movementX * sens;
      targetPitch = Math.max(-1.55, Math.min(1.55, targetPitch - e.movementY * sens));
    };
    const onLockChange = () => {
      const locked = document.pointerLockElement === renderer.domElement;
      setIsLocked(locked);
      if (!locked) {
        yaw = camera.rotation.y;
        pitch = camera.rotation.x;
        targetYaw = yaw;
        targetPitch = pitch;
      }
    };
    renderer.domElement.addEventListener('click', () => {
      if (document.pointerLockElement !== renderer.domElement) {
        renderer.domElement.requestPointerLock();
      }
    });
    document.addEventListener('mousemove', onMouseMove);
    document.addEventListener('pointerlockchange', onLockChange);
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onWindowBlur);

    /* ---------------- 14. Animation loop — organic human locomotion ---------------- */
    let animId: number;
    let prevTime = performance.now();
    const velocity = new THREE.Vector3();
    const impulse = new THREE.Vector3();
    let walkCycle = 0;
    let lastStepPhase = 0;
    let bobSmooth = 1.65;
    let bankSmooth = 0;

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const time = performance.now();
      const rawDelta = (time - prevTime) / 1000;
      prevTime = time;
      const delta = Math.min(rawDelta, 0.05);

      const locked = document.pointerLockElement === renderer.domElement;

      if (locked) {
        // Smoothed mouse look (first-order damp — no snapping)
        const lookDamp = 1 - Math.exp(-16 * delta);
        yaw += (targetYaw - yaw) * lookDamp;
        pitch += (targetPitch - pitch) * lookDamp;
        camera.rotation.y = yaw;
        camera.rotation.x = pitch;

        // Human inertia: exponential accel + natural deceleration
        const fwd = Number(moveStateRef.current.forward) - Number(moveStateRef.current.backward);
        const strafe = Number(moveStateRef.current.right) - Number(moveStateRef.current.left);
        const sprintBoost = moveStateRef.current.sprint ? 1.6 : 1.0;
        const speed = 4.6 * sprintBoost;

        impulse.set(strafe, 0, fwd);
        if (impulse.lengthSq() > 0) impulse.normalize().multiplyScalar(speed);

        const accel = 1 - Math.exp(-9.5 * delta);
        velocity.x += (impulse.x - velocity.x) * accel;
        velocity.z += (impulse.z - velocity.z) * accel;

        // Camera-space -> world movement (yaw only, pitch-free walk)
        const sinY = Math.sin(yaw);
        const cosY = Math.cos(yaw);
        camera.position.x += (cosY * velocity.x - sinY * velocity.z) * delta;
        camera.position.z += (-sinY * velocity.x - cosY * velocity.z) * delta;

        // Continuous collision bounds (house & garden)
        const pos = camera.position;
        pos.x = Math.max(-21.0, Math.min(21.0, pos.x));
        pos.z = Math.max(-11.0, Math.min(31.0, pos.z));
        if (pos.z <= 12.0) {
          pos.x = Math.max(-13.0, Math.min(13.0, pos.x));
        }

        // Dual-frequency organic head bob (vertical stride + lateral sway)
        const moveSpeed = Math.sqrt(velocity.x * velocity.x + velocity.z * velocity.z);
        if (moveSpeed > 0.08) {
          walkCycle += delta * (moveStateRef.current.sprint ? 12.5 : 8.5);
          const bobY = Math.sin(walkCycle) * 0.042 + Math.sin(walkCycle * 2.13) * 0.014;
          const bobLateral = Math.sin(walkCycle * 0.83) * 0.016;
          bobSmooth = 1.65 + bobY;
          pos.x += cosY * bobLateral;
          pos.z += -sinY * bobLateral;

          // Footstep at lowest dip
          const currentStepPhase = Math.floor(walkCycle / Math.PI);
          if (currentStepPhase > lastStepPhase) {
            lastStepPhase = currentStepPhase;
            const surface = (pos.z > 12.0) ? 'carpet' : (pos.x < 0 && pos.z > 0 ? 'tile' : 'wood');
            soundSystem.playFootstep(surface);
          }
        } else {
          bobSmooth = THREE.MathUtils.lerp(bobSmooth, 1.65, 0.12);
        }
        pos.y = bobSmooth;

        // Dynamic camera banking on strafe (1.2–1.8° Z-roll for body weight)
        const targetBank = THREE.MathUtils.clamp(-strafe * 0.028, -0.032, 0.032);
        bankSmooth += (targetBank - bankSmooth) * (1 - Math.exp(-10 * delta));
        camera.rotation.z = bankSmooth;

        // Room tracker
        let newRoom = 'Lorong & Ruang Tamu (Hall / Living)';
        if (pos.z > 12.5) {
          newRoom = '🌿 Taman Rumah (Front Garden)';
        } else if (pos.x < 0 && pos.z < 0) {
          newRoom = 'Dapur (Kitchen)';
        } else if (pos.x > 0 && pos.z < 0) {
          newRoom = 'Kamar Tidur (Bedroom)';
        } else if (pos.x < 0 && pos.z > 0) {
          newRoom = 'Kamar Mandi (Washroom)';
        }
        if (newRoom !== currentRoomRef.current) {
          currentRoomRef.current = newRoom;
          setCurrentRoom(newRoom);
        }

        // Radar live sync
        if (radarRef.current) {
          radarRef.current.update(pos.x, pos.z, yaw);
        }

        armsRig.update(delta, moveSpeed > 0.08, moveStateRef.current.sprint, isRecordingRef.current, newRoom);
      } else {
        bobSmooth = THREE.MathUtils.lerp(bobSmooth, 1.65, 0.1);
        camera.position.y = bobSmooth;
        camera.rotation.z = THREE.MathUtils.lerp(camera.rotation.z, 0, 0.1);
        armsRig.update(delta, false, false, isRecordingRef.current, currentRoomRef.current);
      }

      // Speaker EQ ring pulse + body micro-vibration when music plays
      const ring = speakerRingRef.current;
      const body = speakerMeshRef.current;
      if (musicRef.current) {
        const pulse = 0.5 + Math.sin(time * 0.004) * 0.3;
        const vol = volumeRef.current / 100;
        ring?.draw(pulse, true);
        body?.scale.set(1 + 0.012 * vol, 1, 1 + 0.012 * vol);
      } else {
        ring?.draw(0, false);
        body?.scale.set(1, 1, 1);
      }
      ring?.needsUpdate();

      // AC vent particles drift (falling air plume)
      Object.values(ventParticlesRef.current).forEach((points) => {
        const posAttr = points.geometry.getAttribute('position') as THREE.BufferAttribute;
        for (let i = 0; i < posAttr.count; i++) {
          let yy = posAttr.getY(i) - delta * 0.55;
          if (yy < 2.25) yy = 2.85;
          posAttr.setY(i, yy);
          posAttr.setX(i, posAttr.getX(i) + Math.sin(time * 0.002 + i) * delta * 0.06);
        }
        posAttr.needsUpdate = true;
      });

      renderer.render(scene, camera);
    };
    animate();

    // Resize
    const onResize = () => {
      const w = container.clientWidth || 800;
      camera.aspect = w / height;
      camera.updateProjectionMatrix();
      renderer.setSize(w, height);
    };
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(animId);
      document.removeEventListener('mousemove', onMouseMove);
      document.removeEventListener('pointerlockchange', onLockChange);
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onWindowBlur);
      window.removeEventListener('resize', onResize);
      if (document.pointerLockElement === renderer.domElement) {
        document.exitPointerLock();
      }
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      scene.traverse((obj) => {
        const mesh = obj as THREE.Mesh;
        if (mesh.geometry) mesh.geometry.dispose();
        const mat = mesh.material as THREE.Material | THREE.Material[] | undefined;
        if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
        else if (mat) mat.dispose();
      });
      renderer.dispose();
    };
  }, []);

  /* ---------------- Device state -> visual actuation ---------------- */
  useEffect(() => {
    const setFixture = (key: string, on: boolean) => {
      const item = lightsMapRef.current[key];
      if (!item) return;
      item.lights.forEach((e) => { e.light.intensity = on ? e.intensity : 0; });
      item.glows.forEach((g) => { g.mat.emissiveIntensity = on ? g.intensity : 0; });
      item.bulbs.forEach((b) => {
        (b.material as THREE.MeshBasicMaterial).color.setHex(on ? 0xFFD9A0 : 0x332211);
      });
    };

    setFixture('kitchen', devices.kitchen_lights);
    setFixture('bedroom', devices.bedroom_lights);
    setFixture('bedroom_lamp', devices.bedroom_lamp);
    setFixture('washroom', devices.washroom_lights);

    // Backlit mirror ring follows washroom light
    const mirrorMat = mirrorRingMatRef.current;
    if (mirrorMat) {
      mirrorMat.emissiveIntensity = devices.washroom_lights ? 1.5 : 0;
    }

    soundSystem.playSwitchClick(
      devices.kitchen_lights || devices.bedroom_lights || devices.washroom_lights || devices.bedroom_lamp,
    );
    soundSystem.setAmbientMusic(devices.hall_music);

    // Heater thermal glow + thermostat digits + AC vent plume
    const updateClimate = (key: string, temp: number) => {
      const heater = heaterMeshRef.current[key];
      if (heater) {
        const mat = heater.material as THREE.MeshStandardMaterial;
        if (temp > 20) {
          mat.emissive.setHex(0xC4492F);
          mat.emissiveIntensity = Math.min(1.4, 0.35 + (temp - 20) * 0.18);
        } else {
          mat.emissive.setHex(0x000000);
          mat.emissiveIntensity = 0;
        }
      }
      const thermo = thermostatRef.current[key];
      if (thermo) {
        thermo.draw(temp);
        thermo.needsUpdate();
      }
      const plume = ventParticlesRef.current[key];
      if (plume) {
        const pm = plume.material as THREE.PointsMaterial;
        if (temp !== 20) {
          pm.opacity = 0.28 + Math.min(0.45, Math.abs(temp - 20) * 0.05);
          pm.color.setHex(temp > 20 ? 0xFF9A4D : 0x5ED3E8);
        } else {
          pm.opacity = 0;
        }
      }
    };
    updateClimate('kitchen', devices.kitchen_heat);
    updateClimate('bedroom', devices.bedroom_heat);
    updateClimate('washroom', devices.washroom_heat);

    // Induction cooktop rings
    const cookOn = devices.kitchen_heat !== 20;
    cooktopRingsRef.current.forEach((ring) => {
      const m = ring.material as THREE.MeshStandardMaterial;
      m.emissiveIntensity = cookOn ? (devices.kitchen_heat > 20 ? 1.3 : 0.6) : 0;
      m.emissive.setHex(cookOn ? 0xFF4A2A : 0x000000);
    });
  }, [devices]);

  return (
    <div className="fpp-shell">
      {/* 3D WebGL Canvas */}
      <div
        ref={containerRef}
        style={{ width: '100%', height: '100%', cursor: isLocked ? 'none' : 'pointer' }}
      />

      {/* Minimal smart-glass crosshair with live audio ring */}
      <div className={`fpp-crosshair ${isRecording ? 'live' : ''}`}>
        <div className="fpp-ticks">
          <span />
          <span />
          <span />
          <span />
        </div>
        <div className="fpp-dot" />
        {isRecording && (
          <>
            <span className="fpp-listen-ring" />
            <span className="fpp-listen-ring d1" />
          </>
        )}
      </div>

      {/* Top-center room notification card */}
      <div className="fpp-room-card" key={currentRoom}>
        <span className="room-dot" />
        <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.18 }}>
          <span className="room-kicker">ZONA AKTIF</span>
          <span className="room-name">{currentRoom}</span>
        </div>
      </div>

      {/* Minimal control legend */}
      <div className="fpp-help">
        <span className="fpp-key-chip">
          <kbd>WASD</kbd> Jalan
        </span>
        <span className="fpp-key-chip">
          <kbd>SHIFT</kbd> Lari
        </span>
        <span className="fpp-key-chip">
          <kbd>SPASI</kbd> Tahan bicara
        </span>
      </div>

      {/* Architectural blueprint mini-map */}
      <RadarMiniMap
        ref={radarRef}
        initialX={0}
        initialZ={23}
        devices={devices}
      />

      {isRecording && (
        <div className="fpp-recording">
          <span className="rec-dot" />
          MENDENGARKAN • LEPAS SPASI UNTUK KIRIM
        </div>
      )}

      {isProcessing && (
        <div
          className="fpp-recording"
          style={{
            borderColor: 'rgba(245, 158, 11, 0.5)',
            background: 'rgba(245, 158, 11, 0.14)',
            color: '#FCD34D',
          }}
        >
          <span className="rec-dot" style={{ background: '#F59E0B', boxShadow: '0 0 10px #F59E0B' }} />
          MODEL ONNX MEMPROSES…
        </div>
      )}

      {lastCommandMessage && (
        <div className="fpp-subtitle" key={lastCommandMessage}>
          <span className="who">SVARA</span>
          <span className="msg">“{lastCommandMessage}”</span>
        </div>
      )}

      {!isLocked && (
        <div
          className="fpp-overlay"
          onClick={() => {
            const canvas = containerRef.current?.querySelector('canvas');
            if (canvas && document.pointerLockElement !== canvas) {
              canvas.requestPointerLock();
            }
          }}
        >
          <span className="hud-glass" style={{ position: 'static', marginBottom: 10 }}>
            FIRST-PERSON WALKTHROUGH
          </span>
          <div className="enter-title">Masuk ke Rumah SVARA</div>
          <div className="enter-sub">
            Jelajahi rumah dengan pandangan 360°, berjalan dengan WASD, dan tahan SPASI untuk memberi
            perintah suara langsung ke model SLU.
          </div>
          <div className="enter-cta">▶ Mulai Walkthrough</div>
        </div>
      )}
    </div>
  );
};