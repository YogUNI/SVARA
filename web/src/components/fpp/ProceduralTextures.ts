import * as THREE from 'three';

/**
 * Procedural PBR Texture Generator for SVARA Smart Home
 * Generates dynamic high-res textures via HTML5 Canvas (Zero external asset download overhead)
 */

// 1. Realistic Oak Wood Flooring Texture (Plank & Grain with Roughness)
export function createWoodFloorTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 1024;
  const ctx = canvas.getContext('2d')!;

  // Base warm wood tone
  ctx.fillStyle = '#654B38';
  ctx.fillRect(0, 0, 1024, 1024);

  // Planks configuration
  const plankHeight = 64;
  const plankWidth = 256;

  for (let y = 0; y < 1024; y += plankHeight) {
    const offsetX = (y / plankHeight) % 2 === 0 ? 0 : plankWidth / 2;
    for (let x = -plankWidth; x < 1024 + plankWidth; x += plankWidth) {
      // Wood plank subtle shade variations
      const toneVariance = Math.floor((Math.random() - 0.5) * 36);
      const r = Math.min(255, Math.max(0, 115 + toneVariance));
      const g = Math.min(255, Math.max(0, 85 + toneVariance));
      const b = Math.min(255, Math.max(0, 60 + toneVariance));

      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fillRect(x + offsetX, y, plankWidth, plankHeight);

      // Wood grain lines
      ctx.strokeStyle = `rgba(50, 35, 20, 0.22)`;
      ctx.lineWidth = 1;
      for (let i = 0; i < 8; i++) {
        const grainY = y + Math.random() * plankHeight;
        ctx.beginPath();
        ctx.moveTo(x + offsetX, grainY);
        ctx.bezierCurveTo(
          x + offsetX + plankWidth * 0.3, grainY + (Math.random() - 0.5) * 6,
          x + offsetX + plankWidth * 0.7, grainY + (Math.random() - 0.5) * 6,
          x + offsetX + plankWidth, grainY
        );
        ctx.stroke();
      }

      // Plank bevel borders
      ctx.strokeStyle = 'rgba(25, 15, 10, 0.75)';
      ctx.lineWidth = 2;
      ctx.strokeRect(x + offsetX, y, plankWidth, plankHeight);
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(4, 4);
  return texture;
}

// 2. High-End Black Veined Marble Texture (Kitchen Counter & Bathroom Vanity)
export function createMarbleTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 1024;
  const ctx = canvas.getContext('2d')!;

  // Dark charcoal base
  ctx.fillStyle = '#1A1D20';
  ctx.fillRect(0, 0, 1024, 1024);

  // Marble Veins
  const drawVein = (color: string, width: number, count: number) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    for (let c = 0; c < count; c++) {
      ctx.beginPath();
      let cx = Math.random() * 1024;
      let cy = Math.random() * 1024;
      ctx.moveTo(cx, cy);
      for (let s = 0; s < 6; s++) {
        cx += (Math.random() - 0.45) * 260;
        cy += (Math.random() - 0.45) * 260;
        ctx.lineTo(cx, cy);
      }
      ctx.stroke();
    }
  };

  // Subtle smoky background veins
  drawVein('rgba(255, 255, 255, 0.08)', 8, 12);
  // Crisp distinct white veins
  drawVein('rgba(240, 245, 250, 0.55)', 2.5, 9);
  // Gold/Amber accent micro-veins
  drawVein('rgba(235, 185, 95, 0.35)', 1.2, 5);

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(2, 2);
  return texture;
}

// 3. Contemporary Textured Wall Plaster
export function createWallPlasterTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  // Modern off-white / light slate plaster base
  ctx.fillStyle = '#2A3B42';
  ctx.fillRect(0, 0, 512, 512);

  // Micro-noise texture for authentic plaster roughness
  const imgData = ctx.getImageData(0, 0, 512, 512);
  const data = imgData.data;
  for (let i = 0; i < data.length; i += 4) {
    const noise = (Math.random() - 0.5) * 22;
    data[i] = Math.min(255, Math.max(0, data[i] + noise));
    data[i + 1] = Math.min(255, Math.max(0, data[i + 1] + noise));
    data[i + 2] = Math.min(255, Math.max(0, data[i + 2] + noise));
  }
  ctx.putImageData(imgData, 0, 0);

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(6, 3);
  return texture;
}

// 4. Woven Fabric / Carpet Texture (Cozy Bedroom & Living Area)
export function createFabricRugTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  // Base rug color (Pine slate)
  ctx.fillStyle = '#264841';
  ctx.fillRect(0, 0, 512, 512);

  // Woven thread cross-hatch pattern
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
  ctx.lineWidth = 1;
  for (let i = 0; i < 512; i += 4) {
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, 512);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(512, i);
    ctx.stroke();
  }

  // Geometric border
  ctx.strokeStyle = '#D5DDD9';
  ctx.lineWidth = 12;
  ctx.strokeRect(16, 16, 480, 480);

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  return texture;
}

// 5. Starry Night Outdoor Scenery for Windows
export function createSkyboxTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 1024;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  // Deep night gradient
  const grad = ctx.createLinearGradient(0, 0, 0, 512);
  grad.addColorStop(0, '#04090F');
  grad.addColorStop(0.7, '#0B1924');
  grad.addColorStop(1, '#152C38');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, 1024, 512);

  // Distant stars & garden silhouette
  ctx.fillStyle = '#FFFFFF';
  for (let s = 0; s < 180; s++) {
    const sx = Math.random() * 1024;
    const sy = Math.random() * 320;
    const sz = Math.random() * 1.8;
    ctx.beginPath();
    ctx.arc(sx, sy, sz, 0, Math.PI * 2);
    ctx.fill();
  }

  // Soft distant trees silhouette
  ctx.fillStyle = '#050D12';
  for (let t = 0; t < 1024; t += 32) {
    const th = 60 + Math.random() * 50;
    ctx.beginPath();
    ctx.arc(t, 512, th, Math.PI, 0);
    ctx.fill();
  }

  return new THREE.CanvasTexture(canvas);
}

// 6. Lush Cartoon Stylized Grass Texture
export function createCartoonGrassTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  // Fresh vibrant cartoon green base
  ctx.fillStyle = '#489A3E';
  ctx.fillRect(0, 0, 512, 512);

  // Stylized grass blades patches
  const bladeColors = ['#5CB84D', '#6BCB58', '#388031', '#78D663'];
  for (let i = 0; i < 400; i++) {
    const gx = Math.random() * 512;
    const gy = Math.random() * 512;
    const color = bladeColors[Math.floor(Math.random() * bladeColors.length)];
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(gx, gy);
    ctx.lineTo(gx + 4 + Math.random() * 4, gy - 8 - Math.random() * 8);
    ctx.lineTo(gx + 8, gy);
    ctx.closePath();
    ctx.fill();
  }

  // Cute tiny cartoon wildflowers (yellow, white, pink)
  const flowerColors = ['#FFD152', '#FFFFFF', '#FF85A1'];
  for (let f = 0; f < 35; f++) {
    const fx = Math.random() * 512;
    const fy = Math.random() * 512;
    ctx.fillStyle = flowerColors[Math.floor(Math.random() * flowerColors.length)];
    ctx.beginPath();
    ctx.arc(fx, fy, 3.5, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#FFAA00';
    ctx.beginPath();
    ctx.arc(fx, fy, 1.2, 0, Math.PI * 2);
    ctx.fill();
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(6, 6);
  return texture;
}

// 7. Stylized Stone Pavers / Garden Walkway Texture
export function createStonePathTexture(): THREE.CanvasTexture {
  const canvas = document.createElement('canvas');
  canvas.width = 512;
  canvas.height = 512;
  const ctx = canvas.getContext('2d')!;

  ctx.fillStyle = '#3F4E46'; // Dirt/moss mortar base
  ctx.fillRect(0, 0, 512, 512);

  // Rounded cartoon stepping stones
  const stoneColors = ['#8A9A90', '#A3B4AA', '#76877D', '#B5C4BB'];
  for (let y = 16; y < 512; y += 64) {
    for (let x = 16; x < 512; x += 64) {
      const sx = x + (Math.random() - 0.5) * 12;
      const sy = y + (Math.random() - 0.5) * 12;
      const color = stoneColors[Math.floor(Math.random() * stoneColors.length)];
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.roundRect(sx, sy, 52, 52, [14, 14, 14, 14]);
      ctx.fill();
      ctx.strokeStyle = 'rgba(0, 0, 0, 0.25)';
      ctx.lineWidth = 3;
      ctx.stroke();
    }
  }

  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(2, 6);
  return texture;
}
