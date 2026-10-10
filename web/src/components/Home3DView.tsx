import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { DeviceState } from './FloorPlan';

interface Home3DViewProps {
  devices: DeviceState;
  activeLocation: string | null;
  onToggleDevice: (key: keyof DeviceState) => void;
}

export const Home3DView: React.FC<Home3DViewProps> = ({ devices, activeLocation, onToggleDevice }) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const characterRef = useRef<THREE.Group | null>(null);
  const targetPosRef = useRef<THREE.Vector3>(new THREE.Vector3(0, 0, 0));
  const lightsMeshRef = useRef<{ [key: string]: { mesh: THREE.Mesh; pointLight: THREE.PointLight } }>({});

  useEffect(() => {
    if (!mountRef.current) return;
    const width = mountRef.current.clientWidth || 700;
    const height = 480;

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0xE8ECE6); // var(--plaster)

    // Isometric-like camera
    const aspect = width / height;
    const d = 16;
    const camera = new THREE.OrthographicCamera(-d * aspect, d * aspect, d, -d, 1, 1000);
    camera.position.set(20, 24, 20);
    camera.lookAt(0, 0, 0);

    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(width, height);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    mountRef.current.appendChild(renderer.domElement);

    // Ambient light
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.65);
    scene.add(ambientLight);

    // Directional sun light
    const dirLight = new THREE.DirectionalLight(0xfffaed, 0.6);
    dirLight.position.set(15, 30, 10);
    dirLight.castShadow = true;
    dirLight.shadow.mapSize.width = 1024;
    dirLight.shadow.mapSize.height = 1024;
    scene.add(dirLight);

    // Floor Base (Home Foundation)
    const floorGeo = new THREE.BoxGeometry(26, 0.6, 22);
    const floorMat = new THREE.MeshLambertMaterial({ color: 0xF6F8F4 });
    const floor = new THREE.Mesh(floorGeo, floorMat);
    floor.position.y = -0.3;
    floor.receiveShadow = true;
    scene.add(floor);

    // Walls function
    const createWall = (w: number, h: number, depth: number, x: number, y: number, z: number) => {
      const wallGeo = new THREE.BoxGeometry(w, h, depth);
      const wallMat = new THREE.MeshLambertMaterial({ color: 0x16303A });
      const wall = new THREE.Mesh(wallGeo, wallMat);
      wall.position.set(x, y, z);
      wall.castShadow = true;
      wall.receiveShadow = true;
      scene.add(wall);
    };

    // Outer & Partition Walls
    createWall(0.5, 3.5, 22, -13, 1.75, 0); // West wall
    createWall(26, 3.5, 0.5, 0, 1.75, -11); // North wall
    createWall(0.5, 3.5, 22, 0, 1.75, 0);   // Center dividing wall
    createWall(26, 3.5, 0.5, 0, 1.75, 0);   // Horizontal dividing wall

    // Room labels helper
    const addRoomFurniture = () => {
      // Dapur (Kitchen) Furniture (-6.5, 0, -5.5)
      const counterGeo = new THREE.BoxGeometry(5, 1.5, 2);
      const counterMat = new THREE.MeshLambertMaterial({ color: 0xA9B8B2 });
      const counter = new THREE.Mesh(counterGeo, counterMat);
      counter.position.set(-8, 0.75, -9);
      counter.castShadow = true;
      scene.add(counter);

      // Kamar Tidur (Bedroom) Bed (6.5, 0, -5.5)
      const bedGeo = new THREE.BoxGeometry(4.5, 1.2, 6);
      const bedMat = new THREE.MeshLambertMaterial({ color: 0x2F6B5E });
      const bed = new THREE.Mesh(bedGeo, bedMat);
      bed.position.set(7, 0.6, -7);
      bed.castShadow = true;
      scene.add(bed);

      // Kamar Mandi (Washroom) (-6.5, 0, 5.5)
      const bathGeo = new THREE.BoxGeometry(3.5, 1.2, 3.5);
      const bathMat = new THREE.MeshLambertMaterial({ color: 0xD5DDD9 });
      const bath = new THREE.Mesh(bathGeo, bathMat);
      bath.position.set(-8, 0.6, 6);
      bath.castShadow = true;
      scene.add(bath);

      // Lorong / Hall (Living & Sound) (6.5, 0, 5.5)
      const sofaGeo = new THREE.BoxGeometry(6, 1.4, 3);
      const sofaMat = new THREE.MeshLambertMaterial({ color: 0x597268 });
      const sofa = new THREE.Mesh(sofaGeo, sofaMat);
      sofa.position.set(6.5, 0.7, 6);
      sofa.castShadow = true;
      scene.add(sofa);
    };
    addRoomFurniture();

    // Ceiling Light Fixtures
    const createCeilingLight = (x: number, z: number, key: string) => {
      const bulbGeo = new THREE.SphereGeometry(0.5, 16, 16);
      const bulbMat = new THREE.MeshBasicMaterial({ color: 0x777777 });
      const bulb = new THREE.Mesh(bulbGeo, bulbMat);
      bulb.position.set(x, 4.2, z);

      const pLight = new THREE.PointLight(0xF2B33D, 0, 16);
      pLight.position.set(x, 3.8, z);
      scene.add(bulb);
      scene.add(pLight);
      lightsMeshRef.current[key] = { mesh: bulb, pointLight: pLight };
    };

    createCeilingLight(-6.5, -5.5, 'kitchen');
    createCeilingLight(6.5, -5.5, 'bedroom');
    createCeilingLight(-6.5, 5.5, 'washroom');

    // 3D Human Avatar Character (Sims-style Character)
    const charGroup = new THREE.Group();

    // Body
    const bodyGeo = new THREE.CylinderGeometry(0.45, 0.45, 1.4, 16);
    const bodyMat = new THREE.MeshLambertMaterial({ color: 0x2F6B5E });
    const body = new THREE.Mesh(bodyGeo, bodyMat);
    body.position.y = 1.3;
    body.castShadow = true;
    charGroup.add(body);

    // Head
    const headGeo = new THREE.SphereGeometry(0.5, 16, 16);
    const headMat = new THREE.MeshLambertMaterial({ color: 0xE4B08A });
    const head = new THREE.Mesh(headGeo, headMat);
    head.position.y = 2.4;
    head.castShadow = true;
    charGroup.add(head);

    // Hair
    const hairGeo = new THREE.SphereGeometry(0.52, 16, 16, 0, Math.PI * 2, 0, Math.PI * 0.5);
    const hairMat = new THREE.MeshLambertMaterial({ color: 0x16303A });
    const hair = new THREE.Mesh(hairGeo, hairMat);
    hair.position.y = 2.42;
    charGroup.add(hair);

    // Interactive Floating Pointer / Diamond over Avatar
    const diamondGeo = new THREE.OctahedronGeometry(0.35);
    const diamondMat = new THREE.MeshBasicMaterial({ color: 0x2F6B5E });
    const diamond = new THREE.Mesh(diamondGeo, diamondMat);
    diamond.position.y = 3.4;
    charGroup.add(diamond);

    charGroup.position.set(0, 0, 2);
    scene.add(charGroup);
    characterRef.current = charGroup;

    // Animation Loop
    let animId: number;
    const clock = new THREE.Clock();

    const animate = () => {
      animId = requestAnimationFrame(animate);
      const delta = clock.getDelta();

      // Smooth walk towards target room position
      if (characterRef.current && targetPosRef.current) {
        const curPos = characterRef.current.position;
        const target = targetPosRef.current;
        const dist = curPos.distanceTo(target);

        if (dist > 0.08) {
          curPos.lerp(target, 4.0 * delta);
          // Look towards walking direction
          characterRef.current.lookAt(target.x, curPos.y, target.z);
          // Walking bounce
          body.position.y = 1.3 + Math.sin(clock.getElapsedTime() * 12) * 0.08;
        } else {
          body.position.y = 1.3;
        }
      }

      // Rotate floating pointer
      diamond.rotation.y += 2.0 * delta;

      renderer.render(scene, camera);
    };
    animate();

    // Clean up
    return () => {
      cancelAnimationFrame(animId);
      if (mountRef.current && renderer.domElement) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
    };
  }, []);

  // Update Character position based on Voice Target Room
  useEffect(() => {
    if (!activeLocation) return;
    if (activeLocation === 'kitchen') {
      targetPosRef.current.set(-6.5, 0, -5.5);
    } else if (activeLocation === 'bedroom') {
      targetPosRef.current.set(6.5, 0, -5.5);
    } else if (activeLocation === 'washroom') {
      targetPosRef.current.set(-6.5, 0, 5.5);
    } else {
      // Hall / Living
      targetPosRef.current.set(3.5, 0, 3.5);
    }
  }, [activeLocation]);

  // Update Lights state in 3D Scene
  useEffect(() => {
    const updateLight = (key: string, isOn: boolean) => {
      const item = lightsMeshRef.current[key];
      if (!item) return;
      if (isOn) {
        (item.mesh.material as THREE.MeshBasicMaterial).color.setHex(0xF2B33D);
        item.pointLight.intensity = 2.2;
      } else {
        (item.mesh.material as THREE.MeshBasicMaterial).color.setHex(0x555555);
        item.pointLight.intensity = 0.0;
      }
    };

    updateLight('kitchen', devices.kitchen_lights);
    updateLight('bedroom', devices.bedroom_lights);
    updateLight('washroom', devices.washroom_lights);
  }, [devices]);

  return (
    <div style={{ position: 'relative', width: '100%', borderRadius: '12px', overflow: 'hidden' }}>
      <div ref={mountRef} style={{ width: '100%', height: '480px' }} />
      <div style={{
        position: 'absolute',
        top: 12,
        left: 14,
        background: 'rgba(22, 48, 58, 0.85)',
        color: '#F6F8F4',
        padding: '6px 12px',
        borderRadius: '6px',
        fontSize: '0.82rem',
        fontWeight: 600,
        pointerEvents: 'none',
      }}>
        🎮 Mode 3D Interaktif (Sims Game Style) — Karakter berjalan otomatis ke ruangan perintah suara!
      </div>
    </div>
  );
};
