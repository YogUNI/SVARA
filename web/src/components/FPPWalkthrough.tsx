import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { PointerLockControls } from 'three/examples/jsm/controls/PointerLockControls.js';
import { DeviceState } from './FloorPlan';

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
  
  // Three.js references
  const controlsRef = useRef<PointerLockControls | null>(null);
  const lightsMapRef = useRef<{ [key: string]: { pointLight: THREE.PointLight; mesh: THREE.Mesh } }>({});
  const heaterMeshRef = useRef<{ [key: string]: THREE.Mesh }>({});
  const moveStateRef = useRef({ forward: false, backward: false, left: false, right: false, sprint: false });

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
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.22);
    scene.add(ambientLight);

    // 4. House Architecture (Rooms, Floor, Ceiling, Walls)
    const floorGeo = new THREE.PlaneGeometry(28, 24);
    const floorMat = new THREE.MeshStandardMaterial({
      color: 0x2A3733,
      roughness: 0.5,
      metalness: 0.1,
    });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    scene.add(floor);

    // Ceiling
    const ceilingGeo = new THREE.PlaneGeometry(28, 24);
    const ceilingMat = new THREE.MeshStandardMaterial({ color: 0x162227, roughness: 0.9 });
    const ceiling = new THREE.Mesh(ceilingGeo, ceilingMat);
    ceiling.position.y = 3.6;
    ceiling.rotation.x = Math.PI / 2;
    scene.add(ceiling);

    // Outer & Partition Walls
    const wallMat = new THREE.MeshStandardMaterial({ color: 0x1B2C33, roughness: 0.7 });
    const createWall = (w: number, h: number, d: number, x: number, y: number, z: number) => {
      const mesh = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), wallMat);
      mesh.position.set(x, y, z);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      scene.add(mesh);
    };

    // Outer perimeter walls (height 3.6m)
    createWall(28, 3.6, 0.5, 0, 1.8, -12); // North
    createWall(28, 3.6, 0.5, 0, 1.8, 12);  // South
    createWall(0.5, 3.6, 24, -14, 1.8, 0); // West
    createWall(0.5, 3.6, 24, 14, 1.8, 0);  // East

    // Partition walls with doorways
    // Center divider (X = 0) with door opening at center
    createWall(0.5, 3.6, 8, 0, 1.8, -8);  // North segment
    createWall(0.5, 3.6, 8, 0, 1.8, 8);   // South segment
    // Horizontal divider (Z = 0) with doorway
    createWall(8, 3.6, 0.5, -10, 1.8, 0); // West segment
    createWall(8, 3.6, 0.5, 10, 1.8, 0);  // East segment

    // 5. Furniture & Objects per Zone
    // === ZONE 1: DAPUR / KITCHEN (X: -7, Z: -6) ===
    const kitchenIsland = new THREE.Mesh(
      new THREE.BoxGeometry(5, 1.1, 2.5),
      new THREE.MeshStandardMaterial({ color: 0x6E7F80, roughness: 0.2 })
    );
    kitchenIsland.position.set(-7, 0.55, -6);
    kitchenIsland.castShadow = true;
    kitchenIsland.receiveShadow = true;
    scene.add(kitchenIsland);

    // Refrigerator
    const fridge = new THREE.Mesh(
      new THREE.BoxGeometry(2, 2.8, 2),
      new THREE.MeshStandardMaterial({ color: 0xA9B8B2, metalness: 0.4 })
    );
    fridge.position.set(-12, 1.4, -10);
    fridge.castShadow = true;
    scene.add(fridge);

    // === ZONE 2: KAMAR TIDUR / BEDROOM (X: 7, Z: -6) ===
    const bed = new THREE.Mesh(
      new THREE.BoxGeometry(4.5, 0.8, 6),
      new THREE.MeshStandardMaterial({ color: 0x2F6B5E, roughness: 0.8 })
    );
    bed.position.set(8, 0.4, -7);
    bed.castShadow = true;
    scene.add(bed);

    // Nightstand & Lamp
    const nightstand = new THREE.Mesh(
      new THREE.BoxGeometry(1.2, 0.9, 1.2),
      new THREE.MeshStandardMaterial({ color: 0x483C32 })
    );
    nightstand.position.set(4.5, 0.45, -9);
    nightstand.castShadow = true;
    scene.add(nightstand);

    // Bedside Table Lamp
    const deskLamp = new THREE.Mesh(
      new THREE.CylinderGeometry(0.2, 0.4, 0.6),
      new THREE.MeshStandardMaterial({ color: 0xE8ECE6 })
    );
    deskLamp.position.set(4.5, 1.2, -9);
    scene.add(deskLamp);

    // === ZONE 3: KAMAR MANDI / WASHROOM (X: -7, Z: 6) ===
    const bathTub = new THREE.Mesh(
      new THREE.BoxGeometry(3.5, 0.9, 5),
      new THREE.MeshStandardMaterial({ color: 0xE8ECE6, roughness: 0.1 })
    );
    bathTub.position.set(-8, 0.45, 7);
    bathTub.castShadow = true;
    scene.add(bathTub);

    // === ZONE 4: LORONG / HALL & LIVING ROOM (X: 7, Z: 6) ===
    const sofa = new THREE.Mesh(
      new THREE.BoxGeometry(6.5, 1.1, 2.8),
      new THREE.MeshStandardMaterial({ color: 0x3D5A50, roughness: 0.7 })
    );
    sofa.position.set(7, 0.55, 7);
    sofa.castShadow = true;
    scene.add(sofa);

    // Smart Speaker
    const speaker = new THREE.Mesh(
      new THREE.CylinderGeometry(0.35, 0.35, 0.8, 24),
      new THREE.MeshStandardMaterial({ color: 0x111111, metalness: 0.8, roughness: 0.3 })
    );
    speaker.position.set(2.5, 0.9, 3);
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
          if (!e.repeat) onVoiceTriggerStart();
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
          onVoiceTriggerEnd();
          break;
      }
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);

    // 9. Animation Loop (60 FPS Physics & Head Bobbing)
    let animId: number;
    let prevTime = performance.now();
    const velocity = new THREE.Vector3();
    const direction = new THREE.Vector3();
    let walkCycle = 0;

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

        // Sinusoidal Head Bobbing (Real Human Walking Rhythm)
        const moveSpeed = Math.sqrt(velocity.x * velocity.x + velocity.z * velocity.z);
        if (moveSpeed > 0.05) {
          walkCycle += delta * (moveStateRef.current.sprint ? 14 : 9);
          pos.y = 1.65 + Math.sin(walkCycle) * 0.05;
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

      {/* Voice Status Indicator (When Spacebar pressed) */}
      {isRecording && (
        <div style={{
          position: 'absolute',
          top: 14,
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
          top: 14,
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
