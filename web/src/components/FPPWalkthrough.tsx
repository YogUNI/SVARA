import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { PointerLockControls } from 'three/examples/jsm/controls/PointerLockControls.js';
import { DeviceState } from './FloorPlan';
import {
  createWoodFloorTexture,
  createMarbleTexture,
  createWallPlasterTexture,
  createFabricRugTexture,
  createSkyboxTexture,
} from './fpp/ProceduralTextures';
import { createFPPArms } from './fpp/FPPArmsRig';
import { soundSystem } from './fpp/SoundSystem';
import { RadarMiniMap } from './fpp/RadarMiniMap';

interface FPPWalkthroughProps {
  devices: DeviceState;
  onVoiceTriggerStart: () => void;
  onVoiceTriggerEnd: () => void;
  isRecording: boolean;
  isProcessing: boolean;
  lastCommandMessage?: string;
}

export const FPPWalkthrough: React.FC<FPPWalkthroughProps> = ({
  devices,
  onVoiceTriggerStart,
  onVoiceTriggerEnd,
  isRecording,
  isProcessing,
  lastCommandMessage,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [isLocked, setIsLocked] = useState(false);
  const [currentRoom, setCurrentRoom] = useState('Lorong (Living / Hall)');
  const [playerCoord, setPlayerCoord] = useState({ x: 0, z: 4, rotY: 0 });
  
  // Three.js references
  const controlsRef = useRef<PointerLockControls | null>(null);
  const lightsMapRef = useRef<{ [key: string]: { pointLight: THREE.PointLight; mesh: THREE.Mesh } }>({});
  const heaterMeshRef = useRef<{ [key: string]: THREE.Mesh }>({});
  const moveStateRef = useRef({ forward: false, backward: false, left: false, right: false, sprint: false });
  const isRecordingRef = useRef(isRecording);
  const onVoiceTriggerStartRef = useRef(onVoiceTriggerStart);
  const onVoiceTriggerEndRef = useRef(onVoiceTriggerEnd);

  useEffect(() => {
    isRecordingRef.current = isRecording;
  }, [isRecording]);

  useEffect(() => {
    onVoiceTriggerStartRef.current = onVoiceTriggerStart;
    onVoiceTriggerEndRef.current = onVoiceTriggerEnd;
  }, [onVoiceTriggerStart, onVoiceTriggerEnd]);

  useEffect(() => {
    if (!containerRef.current) return;
    const container = containerRef.current;
    const width = container.clientWidth || 800;
    const height = 520;

    // 1. Scene, Camera (Perspective Human Eye 75 FOV), Renderer
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0E191D); // Deep modern dark room atmosphere
    scene.fog = new THREE.FogExp2(0x0E191D, 0.035);

    const camera = new THREE.PerspectiveCamera(75, width / height, 0.1, 100);
    camera.position.set(0, 1.65, 4); // Eye level 1.65 meters

    // 2. Attach FPP Arms & Smartwatch to Camera
    const armsRig = createFPPArms();
    camera.add(armsRig.armsGroup);
    scene.add(camera);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    container.appendChild(renderer.domElement);

    // 2. PointerLockControls setup
    const controls = new PointerLockControls(camera, renderer.domElement);
    controlsRef.current = controls;

    controls.addEventListener('lock', () => setIsLocked(true));
    controls.addEventListener('unlock', () => setIsLocked(false));

    // 3. Ambient & Directional Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.35);
    scene.add(ambientLight);

    // 4. Generate Procedural PBR Textures
    const woodFloorTex = createWoodFloorTexture();
    const marbleTex = createMarbleTexture();
    const wallPlasterTex = createWallPlasterTexture();
    const rugTex = createFabricRugTexture();
    const skyTex = createSkyboxTexture();

    // Floor with Authentic Wood Plank Texture & Specular Glaze
    const floorGeo = new THREE.PlaneGeometry(28, 24);
    const floorMat = new THREE.MeshStandardMaterial({
      map: woodFloorTex,
      roughness: 0.35,
      metalness: 0.08,
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // Ceiling with Smooth Slate
    const ceilingGeo = new THREE.PlaneGeometry(28, 24);
    const ceilingMat = new THREE.MeshStandardMaterial({ color: 0x1A252A, roughness: 0.85 });
    const ceiling = new THREE.Mesh(ceilingGeo, ceilingMat);
    ceiling.position.y = 3.6;
    ceiling.rotation.x = Math.PI / 2;
    scene.add(ceiling);

    // Outer & Partition Walls with Textured Plaster & Wooden Baseboards
    const wallMat = new THREE.MeshStandardMaterial({
      map: wallPlasterTex,
      roughness: 0.75,
      metalness: 0.02,
    });
    const baseboardMat = new THREE.MeshStandardMaterial({ color: 0x12171A, roughness: 0.4 });

    const createWallWithBaseboard = (w: number, h: number, d: number, x: number, y: number, z: number) => {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), wallMat);
      mesh.position.set(x, y, z);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      scene.add(mesh);

      // Add baseboard trimming along the bottom
      const bbHeight = 0.18;
      const bb = new THREE.Mesh(
        new THREE.BoxGeometry(w > d ? w : 0.54, bbHeight, d > w ? d : 0.54),
        baseboardMat
      );
      bb.position.set(x, bbHeight / 2, z);
      scene.add(bb);
    };

    // Outer perimeter walls (height 3.6m)
    createWallWithBaseboard(28, 3.6, 0.5, 0, 1.8, -12); // North
    createWallWithBaseboard(28, 3.6, 0.5, 0, 1.8, 12);  // South
    createWallWithBaseboard(0.5, 3.6, 24, -14, 1.8, 0); // West
    createWallWithBaseboard(0.5, 3.6, 24, 14, 1.8, 0);  // East

    // Partition walls with doorways
    createWallWithBaseboard(0.5, 3.6, 8, 0, 1.8, -8);  // North center divider
    createWallWithBaseboard(0.5, 3.6, 8, 0, 1.8, 8);   // South center divider
    createWallWithBaseboard(8, 3.6, 0.5, -10, 1.8, 0); // West horizontal divider
    createWallWithBaseboard(8, 3.6, 0.5, 10, 1.8, 0);  // East horizontal divider

    // Windows with Glass & Outdoor Night Sky View (East Wall Window)
    const windowFrame = new THREE.Mesh(
      new THREE.BoxGeometry(6, 2.2, 0.2),
      new THREE.MeshStandardMaterial({ color: 0x0E1417, roughness: 0.3 })
    );
    windowFrame.position.set(7, 2.0, -11.9);
    scene.add(windowFrame);

    const outdoorBackdrop = new THREE.Mesh(
      new THREE.PlaneGeometry(8, 4),
      new THREE.MeshBasicMaterial({ map: skyTex })
    );
    outdoorBackdrop.position.set(7, 2.0, -12.1);
    scene.add(outdoorBackdrop);

    // 5. Furniture & Objects per Zone with High-End Materials
    // === ZONE 1: DAPUR / KITCHEN (X: -7, Z: -6) ===
    // Island with Glossy Veined Black Marble Countertop
    const islandBase = new THREE.Mesh(
      new THREE.BoxGeometry(5.2, 1.0, 2.6),
      new THREE.MeshStandardMaterial({ color: 0x1A2226, roughness: 0.5 })
    );
    islandBase.position.set(-7, 0.5, -6);
    islandBase.castShadow = true;
    islandBase.receiveShadow = true;
    scene.add(islandBase);

    const marbleTop = new THREE.Mesh(
      new THREE.BoxGeometry(5.4, 0.14, 2.8),
      new THREE.MeshStandardMaterial({
        map: marbleTex,
        roughness: 0.15,
        metalness: 0.2,
      })
    );
    marbleTop.position.set(-7, 1.07, -6);
    marbleTop.castShadow = true;
    marbleTop.receiveShadow = true;
    scene.add(marbleTop);

    // Modern Induction Cooktop (Glass with glowing red rings)
    const stoveTop = new THREE.Mesh(
      new THREE.BoxGeometry(2.0, 0.02, 1.2),
      new THREE.MeshStandardMaterial({ color: 0x050505, roughness: 0.05, metalness: 0.8 })
    );
    stoveTop.position.set(-8.0, 1.15, -6);
    scene.add(stoveTop);

    // Refrigerator Stainless Steel
    const fridge = new THREE.Mesh(
      new THREE.BoxGeometry(2.2, 2.9, 2.0),
      new THREE.MeshStandardMaterial({ color: 0x8A969E, metalness: 0.75, roughness: 0.25 })
    );
    fridge.position.set(-12, 1.45, -10);
    fridge.castShadow = true;
    scene.add(fridge);

    // === ZONE 2: KAMAR TIDUR / BEDROOM (X: 7, Z: -6) ===
    // Bedroom Rug with Woven Fabric Texture
    const bedRug = new THREE.Mesh(
      new THREE.PlaneGeometry(6.5, 7.5),
      new THREE.MeshStandardMaterial({ map: rugTex, roughness: 0.9 })
    );
    bedRug.rotation.x = -Math.PI / 2;
    bedRug.position.set(7.5, 0.02, -7.0);
    bedRug.receiveShadow = true;
    scene.add(bedRug);

    // Bed Frame & Mattress
    const bedFrame = new THREE.Mesh(
      new THREE.BoxGeometry(4.8, 0.5, 6.2),
      new THREE.MeshStandardMaterial({ color: 0x2A1C14, roughness: 0.6 })
    );
    bedFrame.position.set(7.5, 0.25, -7.0);
    bedFrame.castShadow = true;
    scene.add(bedFrame);

    const mattress = new THREE.Mesh(
      new THREE.BoxGeometry(4.4, 0.55, 5.8),
      new THREE.MeshStandardMaterial({ color: 0xE8EDE9, roughness: 0.85 })
    );
    mattress.position.set(7.5, 0.7, -7.0);
    mattress.castShadow = true;
    scene.add(mattress);

    // Pillows
    const pillowGeo = new THREE.BoxGeometry(1.6, 0.25, 1.0);
    const pillowMat = new THREE.MeshStandardMaterial({ color: 0x2F6B5E, roughness: 0.7 });
    const pillow1 = new THREE.Mesh(pillowGeo, pillowMat);
    pillow1.position.set(6.2, 1.05, -9.2);
    const pillow2 = new THREE.Mesh(pillowGeo, pillowMat);
    pillow2.position.set(8.8, 1.05, -9.2);
    scene.add(pillow1);
    scene.add(pillow2);

    // Nightstand & Bedside Lamp
    const nightstand = new THREE.Mesh(
      new THREE.BoxGeometry(1.2, 0.9, 1.2),
      new THREE.MeshStandardMaterial({ color: 0x2A1C14, roughness: 0.5 })
    );
    nightstand.position.set(4.3, 0.45, -9.2);
    nightstand.castShadow = true;
    scene.add(nightstand);

    const deskLamp = new THREE.Mesh(
      new THREE.CylinderGeometry(0.2, 0.45, 0.65),
      new THREE.MeshStandardMaterial({ color: 0xF2EAD8, roughness: 0.3 })
    );
    deskLamp.position.set(4.3, 1.25, -9.2);
    scene.add(deskLamp);

    // === ZONE 3: KAMAR MANDI / WASHROOM (X: -7, Z: 6) ===
    const bathTub = new THREE.Mesh(
      new THREE.BoxGeometry(3.6, 1.0, 5.2),
      new THREE.MeshStandardMaterial({ color: 0xEAEFEF, roughness: 0.1, metalness: 0.05 })
    );
    bathTub.position.set(-8, 0.5, 7);
    bathTub.castShadow = true;
    scene.add(bathTub);

    // Bathroom Mirror Vanity
    const vanityMirror = new THREE.Mesh(
      new THREE.BoxGeometry(3.0, 1.8, 0.08),
      new THREE.MeshStandardMaterial({ color: 0xC0D6DF, metalness: 0.95, roughness: 0.05 })
    );
    vanityMirror.position.set(-8, 2.2, 11.9);
    scene.add(vanityMirror);

    // === ZONE 4: LORONG & RUANG KELUARGA (X: 7, Z: 6) ===
    // Living Room Woven Geometric Rug
    const livingRug = new THREE.Mesh(
      new THREE.PlaneGeometry(8.5, 6.5),
      new THREE.MeshStandardMaterial({ map: rugTex, roughness: 0.9 })
    );
    livingRug.rotation.x = -Math.PI / 2;
    livingRug.position.set(7.0, 0.02, 6.5);
    livingRug.receiveShadow = true;
    scene.add(livingRug);

    // Modern Emerald Sectional Sofa
    const sofa = new THREE.Mesh(
      new THREE.BoxGeometry(6.8, 1.2, 2.8),
      new THREE.MeshStandardMaterial({ color: 0x1C4A40, roughness: 0.75 })
    );
    sofa.position.set(7, 0.6, 7.5);
    sofa.castShadow = true;
    scene.add(sofa);

    // Coffee Table with Glass Top
    const tableBase = new THREE.Mesh(
      new THREE.BoxGeometry(3.6, 0.45, 1.8),
      new THREE.MeshStandardMaterial({ color: 0x151B1E })
    );
    tableBase.position.set(7, 0.25, 4.5);
    scene.add(tableBase);

    // Flat Wall Smart TV 65-Inch
    const tvScreen = new THREE.Mesh(
      new THREE.BoxGeometry(4.8, 2.5, 0.1),
      new THREE.MeshStandardMaterial({ color: 0x080A0C, roughness: 0.1, metalness: 0.9 })
    );
    tvScreen.position.set(7, 2.2, 11.88);
    scene.add(tvScreen);

    // Smart Speaker Tower
    const speaker = new THREE.Mesh(
      new THREE.CylinderGeometry(0.38, 0.38, 1.1, 24),
      new THREE.MeshStandardMaterial({ color: 0x111618, metalness: 0.85, roughness: 0.25 })
    );
    speaker.position.set(2.5, 0.55, 3.2);
    speaker.castShadow = true;
    scene.add(speaker);

    // 6. Dynamic Room Light Fixtures (PointLights with Real Inverse-Square Attenuation)
    const setupRoomLight = (name: string, x: number, z: number) => {
      const pLight = new THREE.PointLight(0xF2B33D, 0, 14, 1.5);
      pLight.position.set(x, 3.2, z);
      pLight.castShadow = true;
      pLight.shadow.bias = -0.002;
      scene.add(pLight);

      // Visual bulb mesh
      const bulb = new THREE.Mesh(
        new THREE.SphereGeometry(0.3, 16, 16),
        new THREE.MeshBasicMaterial({ color: 0x333333 })
      );
      bulb.position.set(x, 3.4, z);
      scene.add(bulb);

      lightsMapRef.current[name] = { pointLight: pLight, mesh: bulb };
    };

    setupRoomLight('kitchen', -7, -6);
    setupRoomLight('bedroom', 7, -6);
    setupRoomLight('washroom', -7, 6);

    // Bedside Lamp spot
    const bedsideLight = new THREE.PointLight(0xF2B33D, 0, 8, 2);
    bedsideLight.position.set(4.5, 1.6, -9);
    scene.add(bedsideLight);
    lightsMapRef.current['bedroom_lamp'] = { pointLight: bedsideLight, mesh: deskLamp };

    // 7. Wall Radiator Heaters with Thermal Emissive Glow
    const createHeater = (name: string, x: number, z: number, rotY: number = 0) => {
      const heaterGeo = new THREE.BoxGeometry(2.4, 1.2, 0.2);
      const heaterMat = new THREE.MeshStandardMaterial({
        color: 0x333333,
        emissive: 0x000000,
        roughness: 0.6,
      });
      const mesh = new THREE.Mesh(heaterGeo, heaterMat);
      mesh.position.set(x, 1.0, z);
      mesh.rotation.y = rotY;
      scene.add(mesh);
      heaterMeshRef.current[name] = mesh;
    };

    createHeater('kitchen', -13.7, -4, Math.PI / 2);
    createHeater('bedroom', 13.7, -4, -Math.PI / 2);
    createHeater('washroom', -13.7, 4, Math.PI / 2);

    // 8. Key Listeners for WASD + Spacebar Voice
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

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', onWindowBlur);

    // 9. Animation Loop (60 FPS Physics & Head Bobbing)
    let animId: number;
    let prevTime = performance.now();
    const velocity = new THREE.Vector3();
    const direction = new THREE.Vector3();
    let walkCycle = 0;
    let lastStepPhase = 0;

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const time = performance.now();
      const delta = (time - prevTime) / 1000;
      prevTime = time;

      if (controls.isLocked) {
        // Friction damping
        velocity.x -= velocity.x * 10.0 * delta;
        velocity.z -= velocity.z * 10.0 * delta;

        direction.z = Number(moveStateRef.current.forward) - Number(moveStateRef.current.backward);
        direction.x = Number(moveStateRef.current.right) - Number(moveStateRef.current.left);
        direction.normalize();

        const speed = moveStateRef.current.sprint ? 14.0 : 7.5;
        if (moveStateRef.current.forward || moveStateRef.current.backward) {
          velocity.z -= direction.z * speed * delta;
        }
        if (moveStateRef.current.left || moveStateRef.current.right) {
          velocity.x -= direction.x * speed * delta;
        }

        controls.moveRight(-velocity.x * delta);
        controls.moveForward(-velocity.z * delta);

        // Collision Bounds (Stay inside home boundary)
        const pos = camera.position;
        pos.x = Math.max(-13.0, Math.min(13.0, pos.x));
        pos.z = Math.max(-11.0, Math.min(11.0, pos.z));

        // Sinusoidal Head Bobbing (Real Human Walking Rhythm) & Footstep Triggers
        const moveSpeed = Math.sqrt(velocity.x * velocity.x + velocity.z * velocity.z);
        if (moveSpeed > 0.05) {
          walkCycle += delta * (moveStateRef.current.sprint ? 14 : 9);
          pos.y = 1.65 + Math.sin(walkCycle) * 0.05;

          // Footstep audio triggered at lowest dip of head bob (step contact)
          const currentStepPhase = Math.floor(walkCycle / Math.PI);
          if (currentStepPhase > lastStepPhase) {
            lastStepPhase = currentStepPhase;
            const surface = (pos.x < 0 && pos.z > 0) ? 'tile' : (pos.x > 0 && pos.z < 0 ? 'carpet' : 'wood');
            soundSystem.playFootstep(surface);
          }
        } else {
          pos.y = THREE.MathUtils.lerp(pos.y, 1.65, 0.1);
        }

        // Room Location Tracker based on player coordinates
        if (pos.x < 0 && pos.z < 0) {
          setCurrentRoom('Dapur (Kitchen)');
        } else if (pos.x > 0 && pos.z < 0) {
          setCurrentRoom('Kamar Tidur (Bedroom)');
        } else if (pos.x < 0 && pos.z > 0) {
          setCurrentRoom('Kamar Mandi (Washroom)');
        } else {
          setCurrentRoom('Lorong & Ruang Tamu (Hall / Living)');
        }

        // Sync Player Coordinates with In-Game Radar MiniMap
        setPlayerCoord({
          x: pos.x,
          z: pos.z,
          rotY: camera.rotation.y,
        });

        // Animate FPP Arms & Smartwatch Rig
        armsRig.update(
          delta,
          moveSpeed > 0.05,
          moveStateRef.current.sprint,
          isRecordingRef.current
        );
      } else {
        // Idle breathing and wristwatch animation when pointer lock is not engaged
        armsRig.update(delta, false, false, isRecordingRef.current);
      }

      // Speaker vibration animation if music playing
      if (devices.hall_music) {
        const pulse = 1.0 + Math.sin(time * 0.02) * 0.06;
        speaker.scale.set(pulse, 1.0, pulse);
      } else {
        speaker.scale.set(1.0, 1.0, 1.0);
      }

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animId);
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', onWindowBlur);
      if (container.contains(renderer.domElement)) {
        container.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  // Sync Lights with real-time DeviceState
  useEffect(() => {
    const updateLightState = (key: string, isOn: boolean, intensity: number = 3.5) => {
      const item = lightsMapRef.current[key];
      if (!item) return;
      if (isOn) {
        item.pointLight.intensity = intensity;
        (item.mesh.material as THREE.MeshBasicMaterial).color.setHex(0xF2B33D);
      } else {
        item.pointLight.intensity = 0.0;
        (item.mesh.material as THREE.MeshBasicMaterial).color.setHex(0x333333);
      }
    };

    updateLightState('kitchen', devices.kitchen_lights);
    updateLightState('bedroom', devices.bedroom_lights);
    updateLightState('bedroom_lamp', devices.bedroom_lamp, 2.5);
    updateLightState('washroom', devices.washroom_lights);

    // Sync Thermal Heaters
    const updateHeaterGlow = (key: string, temp: number) => {
      const mesh = heaterMeshRef.current[key];
      if (!mesh) return;
      const mat = mesh.material as THREE.MeshStandardMaterial;
      if (temp > 20) {
        mat.emissive.setHex(0xC4492F);
        mat.emissiveIntensity = Math.min(1.5, 0.4 + (temp - 20) * 0.2);
      } else {
        mat.emissive.setHex(0x000000);
        mat.emissiveIntensity = 0.0;
      }
    };

    // Play tactile switch click audio when light toggles
    soundSystem.playSwitchClick(
      devices.kitchen_lights || devices.bedroom_lights || devices.washroom_lights || devices.bedroom_lamp
    );

    // Sync Ambient Lounge Music
    soundSystem.setAmbientMusic(devices.hall_music);

    updateHeaterGlow('kitchen', devices.kitchen_heat);
    updateHeaterGlow('bedroom', devices.bedroom_heat);
    updateHeaterGlow('washroom', devices.washroom_heat);
  }, [devices]);

  return (
    <div style={{ position: 'relative', width: '100%', height: '520px', borderRadius: '12px', overflow: 'hidden' }}>
      {/* 3D WebGL Canvas */}
      <div
        ref={containerRef}
        onClick={() => controlsRef.current?.lock()}
        style={{ width: '100%', height: '100%', cursor: isLocked ? 'none' : 'pointer' }}
      />

      {/* Crosshair (Reticle) in Center Screen */}
      <div style={{
        position: 'absolute',
        top: '50%',
        left: '50%',
        transform: 'translate(-50%, -50%)',
        pointerEvents: 'none',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
      }}>
        <div style={{
          width: isRecording ? 16 : 6,
          height: isRecording ? 16 : 6,
          borderRadius: '50%',
          backgroundColor: isRecording ? '#C4492F' : 'rgba(255, 255, 255, 0.75)',
          border: isRecording ? '2px solid #FFF' : '1px solid rgba(0,0,0,0.4)',
          transition: 'all 120ms ease',
        }} />
      </div>

      {/* Top HUD: Current Room & Guide */}
      <div style={{
        position: 'absolute',
        top: 14,
        left: 16,
        display: 'flex',
        flexDirection: 'column',
        gap: 6,
        pointerEvents: 'none',
      }}>
        <div style={{
          background: 'rgba(22, 48, 58, 0.88)',
          color: '#F6F8F4',
          padding: '6px 14px',
          borderRadius: '6px',
          fontSize: '0.85rem',
          fontWeight: 700,
          borderLeft: '4px solid #2F6B5E',
          backdropFilter: 'blur(4px)',
        }}>
          📍 Lokasi: {currentRoom}
        </div>
        <div style={{
          background: 'rgba(22, 48, 58, 0.7)',
          color: '#D5DDD9',
          padding: '4px 10px',
          borderRadius: '4px',
          fontSize: '0.75rem',
        }}>
          🎮 Kontrol: <strong>WASD</strong> (Jalan) | <strong>Shift</strong> (Lari) | <strong>Spasi</strong> (Tahan Bicara)
        </div>
      </div>

      {/* Radar Mini-Map (Top Right) */}
      <RadarMiniMap
        playerX={playerCoord.x}
        playerZ={playerCoord.z}
        playerRotationY={playerCoord.rotY}
        currentRoom={currentRoom}
        devices={devices}
      />

      {/* Voice Status Indicator (When Spacebar pressed) */}
      {isRecording && (
        <div style={{
          position: 'absolute',
          top: 176,
          right: 16,
          background: '#C4492F',
          color: '#FFF',
          padding: '6px 16px',
          borderRadius: '999px',
          fontSize: '0.85rem',
          fontWeight: 700,
          animation: 'pulse 1s infinite',
          boxShadow: '0 0 16px rgba(196, 73, 47, 0.7)',
        }}>
          🎙️ Mendengarkan Suara Anda... (Lepas Spasi untuk Kirim)
        </div>
      )}

      {isProcessing && (
        <div style={{
          position: 'absolute',
          top: 176,
          right: 16,
          background: '#2F6B5E',
          color: '#FFF',
          padding: '6px 16px',
          borderRadius: '999px',
          fontSize: '0.85rem',
          fontWeight: 700,
        }}>
          ⚡ Model AI Sedang Memproses...
        </div>
      )}

      {/* Subtitle Message Bar / AI Response Feedback */}
      {lastCommandMessage && (
        <div style={{
          position: 'absolute',
          bottom: 24,
          left: '50%',
          transform: 'translateX(-50%)',
          background: 'rgba(14, 25, 29, 0.88)',
          color: '#3EE08F',
          border: '1px solid #2F6B5E',
          padding: '8px 24px',
          borderRadius: '999px',
          fontSize: '0.92rem',
          fontWeight: 700,
          backdropFilter: 'blur(8px)',
          boxShadow: '0 4px 20px rgba(0,0,0,0.5)',
          pointerEvents: 'none',
          display: 'flex',
          alignItems: 'center',
          gap: 8,
        }}>
          <span>🤖 SVARA:</span>
          <span style={{ color: '#F6F8F4' }}>"{lastCommandMessage}"</span>
        </div>
      )}

      {/* Click to Play Overlay when not locked */}
      {!isLocked && (
        <div
          onClick={() => controlsRef.current?.lock()}
          style={{
            position: 'absolute',
            inset: 0,
            background: 'rgba(14, 25, 29, 0.75)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#F6F8F4',
            cursor: 'pointer',
            backdropFilter: 'blur(3px)',
          }}
        >
          <div style={{
            fontSize: '1.6rem',
            fontWeight: 800,
            marginBottom: '0.5rem',
            letterSpacing: '-0.02em',
          }}>
            KLIK UNTUK MASUK KE MODE FPP
          </div>
          <div style={{ fontSize: '0.95rem', color: '#A9B8B2', marginBottom: '1.25rem' }}>
            Nengok 360° dengan Mouse, Jalan dengan WASD, dan Tahan SPASI untuk Bicara
          </div>
          <div style={{
            backgroundColor: '#2F6B5E',
            color: '#FFF',
            padding: '0.6rem 1.4rem',
            borderRadius: '999px',
            fontWeight: 700,
            fontSize: '0.9rem',
          }}>
            ▶ Masuk ke Rumah (FPP Walkthrough)
          </div>
        </div>
      )}
    </div>
  );
};
