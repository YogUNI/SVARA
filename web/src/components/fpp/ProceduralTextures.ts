import * as THREE from 'three';

/**
 * Procedural PBR Texture Engine — SVARA "Japandi Luxury Villa" overhaul
 * All textures 1024px+ generated via HTML5 Canvas (zero external assets).
 */

type RGB = [number, number, number];

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

function makeCanvas(w: number, h: number): [HTMLCanvasElement, CanvasRenderingContext2D] {
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  return [canvas, ctx];
}

function finalizeTexture(
  canvas: HTMLCanvasElement,
  repeatX = 1,
  repeatY = 1,
): THREE.CanvasTexture {
  const texture = new THREE.CanvasTexture(canvas);
  texture.wrapS = THREE.RepeatWrapping;
  texture.wrapT = THREE.RepeatWrapping;
  texture.repeat.set(repeatX, repeatY);
  texture.anisotropy = 8;
  texture.colorSpace = THREE.SRGBColorSpace;
  return texture;
}

/** In-memory singleton cache to prevent re-generating multi-megapixel canvases on re-mounts. */
const textureCache = new Map<string, THREE.CanvasTexture>();

function getOrCreateTexture(key: string, factory: () => THREE.CanvasTexture): THREE.CanvasTexture {
  let cached = textureCache.get(key);
  if (!cached) {
    cached = factory();
    textureCache.set(key, cached);
  }
  return cached.clone();
}


/** Per-pixel micro-roughness noise overlay. */
function grain(ctx: CanvasRenderingContext2D, w: number, h: number, amount = 16) {
  const img = ctx.getImageData(0, 0, w, h);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const n = (Math.random() - 0.5) * amount;
    d[i] = clamp(d[i] + n, 0, 255);
    d[i + 1] = clamp(d[i + 1] + n, 0, 255);
    d[i + 2] = clamp(d[i + 2] + n, 0, 255);
  }
  ctx.putImageData(img, 0, 0);
}

/** Soft large-scale mottling for plaster/marble depth. */
function blotches(ctx: CanvasRenderingContext2D, w: number, h: number, count = 24) {
  for (let i = 0; i < count; i++) {
    const x = Math.random() * w;
    const y = Math.random() * h;
    const r = 60 + Math.random() * 220;
    const dark = Math.random() > 0.5;
    const g = ctx.createRadialGradient(x, y, 0, x, y, r);
    g.addColorStop(0, dark ? 'rgba(60, 50, 40, 0.045)' : 'rgba(255, 250, 240, 0.05)');
    g.addColorStop(1, 'rgba(0,0,0,0)');
    ctx.fillStyle = g;
    ctx.fillRect(0, 0, w, h);
  }
}

/* =====================================================================
 * 1. LUXURY WOOD — wide-plank natural oak parquet
 *    organic grain, per-plank tone variance, chamfered seams, knots
 * ===================================================================== */
function _generateLuxuryWoodParquetTexture(): THREE.CanvasTexture {
  const SIZE = 1024;
  const [canvas, ctx] = makeCanvas(SIZE, SIZE);

  const base: RGB = [168, 132, 96];
  ctx.fillStyle = `rgb(${base[0]},${base[1]},${base[2]})`;
  ctx.fillRect(0, 0, SIZE, SIZE);

  const plankW = 146;
  const plankH = SIZE * 2.2; // rows staggered

  const drawPlank = (x: number, y: number) => {
    // per-plank tone variation
    const tone = Math.floor((Math.random() - 0.5) * 42);
    const r = clamp(base[0] + tone, 0, 255);
    const g = clamp(base[1] + tone * 0.8, 0, 255);
    const b = clamp(base[2] + tone * 0.6, 0, 255);

    // soft vertical gradient across plank
    const grad = ctx.createLinearGradient(x, y, x + plankW, y + plankH);
    grad.addColorStop(0, `rgb(${r * 0.92},${g * 0.92},${b * 0.92})`);
    grad.addColorStop(0.5, `rgb(${r},${g},${b})`);
    grad.addColorStop(1, `rgb(${r * 0.9},${g * 0.9},${b * 0.9})`);
    ctx.fillStyle = grad;
    ctx.fillRect(x, y, plankW + 2, plankH + 2);

    // natural grain: wavy longitudinal streaks
    ctx.lineCap = 'round';
    for (let s = 0; s < 14; s++) {
      const gx = x + 6 + Math.random() * (plankW - 12);
      ctx.strokeStyle = `rgba(96, 66, 40, ${0.10 + Math.random() * 0.16})`;
      ctx.lineWidth = 0.8 + Math.random() * 1.5;
      ctx.beginPath();
      ctx.moveTo(gx, y - 4);
      ctx.bezierCurveTo(
        gx + (Math.random() - 0.5) * 14, y + plankH * 0.3,
        gx + (Math.random() - 0.5) * 14, y + plankH * 0.6,
        gx + (Math.random() - 0.5) * 10, y + plankH + 4,
      );
      ctx.stroke();
    }

    // occasional knot
    if (Math.random() < 0.42) {
      const kx = x + 26 + Math.random() * (plankW - 52);
      const ky = y + 60 + Math.random() * (plankH - 120);
      for (let ring = 0; ring < 3; ring++) {
        ctx.strokeStyle = `rgba(74, 46, 24, ${0.26 - ring * 0.06})`;
        ctx.lineWidth = 1.3;
        ctx.beginPath();
        ctx.ellipse(kx, ky, 4 + ring * 3.4, 6 + ring * 4.2, 0, 0, Math.PI * 2);
        ctx.stroke();
      }
      ctx.fillStyle = 'rgba(68, 44, 24, 0.85)';
      ctx.beginPath();
      ctx.ellipse(kx, ky, 2.6, 3.4, 0, 0, Math.PI * 2);
      ctx.fill();
    }

    // chamfer bevel: dark seam + light edge
    ctx.strokeStyle = 'rgba(24, 14, 8, 0.9)';
    ctx.lineWidth = 2.4;
    ctx.strokeRect(x, y, plankW, plankH);
    ctx.strokeStyle = 'rgba(255, 240, 214, 0.22)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 1.5, y + 1.5, plankW - 3, plankH - 3);
  };

  // staggered rows (brick pattern wide-plank)
  const rowCount = 4;
  const rowH = SIZE / rowCount;
  for (let row = 0; row < rowCount; row++) {
    const y = row * rowH;
    const offset = row % 2 === 0 ? 0 : -plankW / 2;
    for (let x = offset - plankW; x < SIZE + plankW; x += plankW) {
      drawPlank(x, y);
    }
  }

  grain(ctx, SIZE, SIZE, 7);

  return finalizeTexture(canvas, 4, 2);
}

/* =====================================================================
 * 2. CALACATTA GOLD MARBLE — Italian white marble, organic grey-gold veins
 * ===================================================================== */
function _generateCalacattaMarbleTexture(): THREE.CanvasTexture {
  const SIZE = 1024;
  const [canvas, ctx] = makeCanvas(SIZE, SIZE);

  // warm ivory base
  const grad = ctx.createLinearGradient(0, 0, SIZE, SIZE);
  grad.addColorStop(0, '#F2EDE3');
  grad.addColorStop(0.5, '#FAF6EE');
  grad.addColorStop(1, '#EFE9DD');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, SIZE, SIZE);

  blotches(ctx, SIZE, SIZE, 20);

  // organic vein walker
  const drawVein = (color: string, width: number, count: number, chaos = 0.24) => {
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    for (let c = 0; c < count; c++) {
      let vx = Math.random() * SIZE;
      let vy = Math.random() * SIZE;
      const dir = Math.random() * Math.PI * 2;
      ctx.strokeStyle = color;
      ctx.lineWidth = width + Math.random() * width;
      ctx.beginPath();
      ctx.moveTo(vx, vy);
      for (let s = 0; s < 10; s++) {
        const a = dir + (Math.random() - 0.5) * chaos * 2;
        vx += Math.cos(a) * (90 + Math.random() * 70);
        vy += Math.sin(a) * (90 + Math.random() * 70);
        ctx.lineTo(clamp(vx, -20, SIZE + 20), clamp(vy, -20, SIZE + 20));
      }
      ctx.stroke();
    }
  };

  // broad smoky grey veins, fine gold veins, hairline champagne
  drawVein('rgba(146, 143, 138, 0.42)', 5, 7, 0.3);
  drawVein('rgba(198, 170, 112, 0.34)', 1.6, 6, 0.42);
  drawVein('rgba(210, 205, 196, 0.5)', 1.1, 9, 0.5);

  grain(ctx, SIZE, SIZE, 5);
  return finalizeTexture(canvas, 2, 2);
}

/* =====================================================================
 * 3. MICROCEMENT WALL — warm linen-taupe plaster, micro sand texture
 * ===================================================================== */
function _generateMicrocementWallTexture(): THREE.CanvasTexture {
  const SIZE = 1024;
  const [canvas, ctx] = makeCanvas(SIZE, SIZE);

  const base: RGB = [205, 196, 182];
  ctx.fillStyle = `rgb(${base[0]},${base[1]},${base[2]})`;
  ctx.fillRect(0, 0, SIZE, SIZE);

  blotches(ctx, SIZE, SIZE, 30);
  grain(ctx, SIZE, SIZE, 9);

  // sand speckle: thousands of tiny tonal dots
  for (let i = 0; i < 2200; i++) {
    const sx = Math.random() * SIZE;
    const sy = Math.random() * SIZE;
    const t = Math.random();
    ctx.fillStyle = t < 0.4
      ? 'rgba(150, 138, 120, 0.14)'
      : t < 0.75
        ? 'rgba(255, 250, 240, 0.12)'
        : 'rgba(110, 98, 82, 0.14)';
    ctx.beginPath();
    ctx.arc(sx, sy, 0.7 + Math.random() * 1.2, 0, Math.PI * 2);
    ctx.fill();
  }

  // sparse hairline micro cracks
  ctx.strokeStyle = 'rgba(150, 138, 120, 0.22)';
  ctx.lineWidth = 0.8;
  for (let i = 0; i < 7; i++) {
    let cx = Math.random() * SIZE;
    let cy = Math.random() * SIZE;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    for (let s = 0; s < 6; s++) {
      cx += (Math.random() - 0.5) * 150;
      cy += (Math.random() - 0.5) * 150;
      ctx.lineTo(clamp(cx, 0, SIZE), clamp(cy, 0, SIZE));
    }
    ctx.stroke();
  }

  return finalizeTexture(canvas, 3, 3);
}

/* =====================================================================
 * 4. WOVEN FABRIC — linen slub weave for sofa, rugs, duvets
 * ===================================================================== */
function _generateWovenFabricTexture(baseColor = '#B7AC9F'): THREE.CanvasTexture {
  const SIZE = 1024;
  const [canvas, ctx] = makeCanvas(SIZE, SIZE);

  ctx.fillStyle = baseColor;
  ctx.fillRect(0, 0, SIZE, SIZE);

  // fine woven thread pattern (warp + weft)
  const step = 8;
  for (let i = 0; i < SIZE; i += step) {
    ctx.strokeStyle = 'rgba(30, 24, 18, 0.13)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(i, 0);
    ctx.lineTo(i, SIZE);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(0, i);
    ctx.lineTo(SIZE, i);
    ctx.stroke();
  }
  // bright thread highlights
  for (let i = 0; i < SIZE; i += step * 2) {
    ctx.strokeStyle = 'rgba(255, 250, 240, 0.1)';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.moveTo(i + 1, 0);
    ctx.lineTo(i + 1, SIZE);
    ctx.stroke();
  }

  // slub nubs (irregular thread thickenings)
  for (let i = 0; i < 900; i++) {
    const x = Math.random() * SIZE;
    const y = Math.random() * SIZE;
    const len = 4 + Math.random() * 8;
    ctx.strokeStyle = `rgba(${Math.random() > 0.5 ? '255,250,240' : '40,32,24'}, ${0.08 + Math.random() * 0.08})`;
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + len, y + (Math.random() - 0.5) * 2);
    ctx.stroke();
  }

  grain(ctx, SIZE, SIZE, 4);
  return finalizeTexture(canvas, 2, 2);
}

/* =====================================================================
 * 5. ACOUSTIC WALNUT SLATS — vertical wood slat wall panels
 * ===================================================================== */
function _generateAcousticSlatWoodTexture(): THREE.CanvasTexture {
  const SIZE = 1024;
  const [canvas, ctx] = makeCanvas(SIZE, SIZE);

  // dark shadow gap background
  ctx.fillStyle = '#191009';
  ctx.fillRect(0, 0, SIZE, SIZE);

  const slatW = 76;
  const gapW = 14;

  let x = 0;
  while (x < SIZE) {
    const tone = (Math.random() - 0.5) * 30;
    const r = clamp(104 + tone, 0, 255);
    const g = clamp(70 + tone * 0.8, 0, 255);
    const b = clamp(48 + tone * 0.6, 0, 255);

    const grad = ctx.createLinearGradient(x, 0, x + slatW, 0);
    grad.addColorStop(0, `rgb(${r * 0.82},${g * 0.82},${b * 0.82})`);
    grad.addColorStop(0.18, `rgb(${r},${g},${b})`);
    grad.addColorStop(0.86, `rgb(${r * 1.04},${g * 1.04},${b * 1.04})`);
    grad.addColorStop(1, `rgb(${r * 0.78},${g * 0.78},${b * 0.78})`);
    ctx.fillStyle = grad;
    ctx.fillRect(x, 0, slatW, SIZE);

    // vertical walnut grain
    for (let s = 0; s < 10; s++) {
      const gx = x + 6 + Math.random() * (slatW - 12);
      ctx.strokeStyle = `rgba(34, 20, 10, ${0.1 + Math.random() * 0.12})`;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.bezierCurveTo(
        gx + (Math.random() - 0.5) * 6, SIZE * 0.33,
        gx + (Math.random() - 0.5) * 6, SIZE * 0.66,
        gx + (Math.random() - 0.5) * 5, SIZE,
      );
      ctx.stroke();
    }

    // rounded slat top & bottom shadow
    ctx.fillStyle = 'rgba(0, 0, 0, 0.28)';
    ctx.fillRect(x, 0, slatW, 3);
    ctx.fillRect(x, SIZE - 3, slatW, 3);

    x += slatW + gapW;
  }

  return finalizeTexture(canvas, 2, 1);
}

/* =====================================================================
 * 6. GOLDEN HOUR / WARM DUSK SKY — grand dusk backdrop for windows/garden
 * ===================================================================== */
function _generateGoldenHourSkyTexture(): THREE.CanvasTexture {
  const W = 1024;
  const H = 512;
  const [canvas, ctx] = makeCanvas(W, H);

  const grad = ctx.createLinearGradient(0, 0, 0, H);
  grad.addColorStop(0, '#2E2645');
  grad.addColorStop(0.32, '#7D4A5E');
  grad.addColorStop(0.55, '#C86F54');
  grad.addColorStop(0.78, '#EFA368');
  grad.addColorStop(0.92, '#F8CF9A');
  grad.addColorStop(1, '#FBE6C2');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, W, H);

  // sun glow near horizon
  const sunX = W * 0.68;
  const sunY = H * 0.78;
  const sun = ctx.createRadialGradient(sunX, sunY, 4, sunX, sunY, 220);
  sun.addColorStop(0, 'rgba(255, 244, 214, 0.95)');
  sun.addColorStop(0.25, 'rgba(255, 222, 160, 0.55)');
  sun.addColorStop(1, 'rgba(255, 200, 130, 0)');
  ctx.fillStyle = sun;
  ctx.fillRect(0, 0, W, H);

  // wispy sunset clouds
  for (let c = 0; c < 16; c++) {
    const cy = Math.random() * H * 0.5;
    const cx = Math.random() * W;
    const cw = 120 + Math.random() * 240;
    ctx.fillStyle = `rgba(255, 190, 140, ${0.05 + Math.random() * 0.1})`;
    ctx.beginPath();
    ctx.ellipse(cx, cy, cw, 6 + Math.random() * 10, 0, 0, Math.PI * 2);
    ctx.fill();
  }

  // distant hills silhouette
  ctx.fillStyle = '#4A3847';
  ctx.beginPath();
  ctx.moveTo(0, H);
  for (let x = 0; x <= W; x += 24) {
    ctx.lineTo(x, H - 34 - Math.sin(x * 0.012 + 2) * 18 - Math.cos(x * 0.031) * 10);
  }
  ctx.lineTo(W, H);
  ctx.closePath();
  ctx.fill();

  // stars in upper sky
  ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
  for (let s = 0; s < 60; s++) {
    ctx.beginPath();
    ctx.arc(Math.random() * W, Math.random() * H * 0.3, 0.6 + Math.random(), 0, Math.PI * 2);
    ctx.fill();
  }

  return finalizeTexture(canvas, 1, 1);
}

/* =====================================================================
 * 7. ZEN LAWN — HD manicured grass for the modern front garden
 * ===================================================================== */
function _generateZenGrassTexture(): THREE.CanvasTexture {
  const SIZE = 1024;
  const [canvas, ctx] = makeCanvas(SIZE, SIZE);

  // deep natural green base with subtle tonal drift
  const base: RGB = [66, 114, 66];
  const grad = ctx.createLinearGradient(0, 0, SIZE, SIZE);
  grad.addColorStop(0, `rgb(${base[0]},${base[1]},${base[2]})`);
  grad.addColorStop(1, `rgb(${base[0] * 0.9},${base[1] * 0.92},${base[2] * 0.9})`);
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, SIZE, SIZE);

  // thousands of fine grass blades
  for (let i = 0; i < 3600; i++) {
    const gx = Math.random() * SIZE;
    const gy = Math.random() * SIZE;
    const len = 3 + Math.random() * 7;
    const dir = (Math.random() - 0.5) * 0.8;
    const bright = Math.random() > 0.5;
    ctx.strokeStyle = bright
      ? `rgba(120, ${160 + Math.random() * 30}, 96, ${0.25 + Math.random() * 0.2})`
      : `rgba(28, ${58 + Math.random() * 20}, 30, ${0.25 + Math.random() * 0.2})`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(gx, gy);
    ctx.lineTo(gx + len * dir, gy - len);
    ctx.stroke();
  }

  grain(ctx, SIZE, SIZE, 6);
  return finalizeTexture(canvas, 6, 3);
}

/* =====================================================================
 * 8. GRANITE STEPPING STONES — zen garden walkway pavers
 * ===================================================================== */
function _generateGraniteSlabTexture(): THREE.CanvasTexture {
  const SIZE = 512;
  const [canvas, ctx] = makeCanvas(SIZE, SIZE);

  // pebble/gravel filler base
  ctx.fillStyle = '#59524C';
  ctx.fillRect(0, 0, SIZE, SIZE);
  for (let i = 0; i < 500; i++) {
    ctx.fillStyle = `rgba(${105 + Math.random() * 40}, ${97 + Math.random() * 40}, ${85 + Math.random() * 36}, 0.5)`;
    ctx.beginPath();
    ctx.arc(Math.random() * SIZE, Math.random() * SIZE, 1 + Math.random() * 2, 0, Math.PI * 2);
    ctx.fill();
  }

  // irregular granite slabs
  const slabColors: RGB[] = [[158, 152, 144], [170, 164, 156], [148, 142, 134]];
  const slabCount = 5;
  for (let s = 0; s < slabCount; s++) {
    const sx = 60 + Math.random() * (SIZE - 120);
    const sy = 60 + Math.random() * (SIZE - 120);
    const sw = 70 + Math.random() * 40;
    // rounded organic slab
    ctx.fillStyle = `rgb(${slabColors[s % 3].join(',')})`;
    ctx.beginPath();
    const radius = 18;
    ctx.roundRect(sx, sy, sw, sw * 0.72, radius);
    ctx.fill();

    // granite speckle
    for (let p = 0; p < 130; p++) {
      const dark = Math.random() > 0.55;
      ctx.fillStyle = dark ? 'rgba(40, 38, 36, 0.35)' : 'rgba(255, 255, 255, 0.3)';
      ctx.beginPath();
      ctx.arc(sx + Math.random() * sw, sy + Math.random() * sw * 0.72, 0.6 + Math.random() * 1.4, 0, Math.PI * 2);
      ctx.fill();
    }

    // bevel edge
    ctx.strokeStyle = 'rgba(30, 26, 22, 0.7)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.roundRect(sx, sy, sw, sw * 0.72, radius);
    ctx.stroke();
    ctx.strokeStyle = 'rgba(255, 250, 240, 0.18)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(sx + 2, sy + 2, sw - 4, sw * 0.72 - 4, radius);
    ctx.stroke();
  }

  return finalizeTexture(canvas, 2, 8);
}

/* =====================================================================
 * PBR ROUGHNESS MAPS — grayscale micro-variation so highlights break up
 * realistically instead of reading as a single flat gloss value.
 * ===================================================================== */
function _generateWoodRoughnessMap(): THREE.CanvasTexture {
  const SIZE = 512;
  const [canvas, ctx] = makeCanvas(SIZE, SIZE);
  // Base: satin-varnished oak (mid grey ≈ 0.45 roughness)
  ctx.fillStyle = '#7A7A7A';
  ctx.fillRect(0, 0, SIZE, SIZE);

  // Plank seams are duller (dirt collects) → lighter grey
  for (let y = 0; y < SIZE; y += SIZE / 4) {
    ctx.fillStyle = 'rgba(214, 214, 214, 0.55)';
    ctx.fillRect(0, y, SIZE, 3);
  }
  // Grain lines are slightly glossier (burnished) → darker
  for (let i = 0; i < 900; i++) {
    const gx = Math.random() * SIZE;
    const gy = Math.random() * SIZE;
    ctx.strokeStyle = Math.random() > 0.5 ? 'rgba(40, 40, 40, 0.25)' : 'rgba(220, 220, 220, 0.2)';
    ctx.lineWidth = 0.8 + Math.random() * 1.4;
    ctx.beginPath();
    ctx.moveTo(gx, gy);
    ctx.lineTo(gx + (Math.random() - 0.5) * 8, gy + 20 + Math.random() * 40);
    ctx.stroke();
  }
  grain(ctx, SIZE, SIZE, 18);

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(4, 2);
  return tex;
}

function _generateMarbleRoughnessMap(): THREE.CanvasTexture {
  const SIZE = 512;
  const [canvas, ctx] = makeCanvas(SIZE, SIZE);
  // Polished Calacatta: mostly very smooth
  ctx.fillStyle = '#2E2E2E';
  ctx.fillRect(0, 0, SIZE, SIZE);

  // Veins are marginally duller than the field
  for (let c = 0; c < 26; c++) {
    let vx = Math.random() * SIZE;
    let vy = Math.random() * SIZE;
    const dir = Math.random() * Math.PI * 2;
    ctx.strokeStyle = 'rgba(150, 150, 150, 0.35)';
    ctx.lineWidth = 1 + Math.random() * 3;
    ctx.beginPath();
    ctx.moveTo(vx, vy);
    for (let s2 = 0; s2 < 8; s2++) {
      vx += Math.cos(dir + (Math.random() - 0.5) * 0.7) * 60;
      vy += Math.sin(dir + (Math.random() - 0.5) * 0.7) * 60;
      ctx.lineTo(clamp(vx, 0, SIZE), clamp(vy, 0, SIZE));
    }
    ctx.stroke();
  }
  grain(ctx, SIZE, SIZE, 10);

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  tex.repeat.set(2, 2);
  return tex;
}


/* =====================================================================
 * Cached Public Texture Factories (Fast singleton + cloned descriptors)
 * ===================================================================== */
export function createLuxuryWoodParquetTexture(): THREE.CanvasTexture {
  return getOrCreateTexture('luxuryWoodParquet', _generateLuxuryWoodParquetTexture);
}

export function createCalacattaMarbleTexture(): THREE.CanvasTexture {
  return getOrCreateTexture('calacattaMarble', _generateCalacattaMarbleTexture);
}

export function createMicrocementWallTexture(): THREE.CanvasTexture {
  return getOrCreateTexture('microcementWall', _generateMicrocementWallTexture);
}

export function createWovenFabricTexture(baseColor = '#B7AC9F'): THREE.CanvasTexture {
  return getOrCreateTexture('wovenFabric_' + baseColor, () => _generateWovenFabricTexture(baseColor));
}

export function createAcousticSlatWoodTexture(): THREE.CanvasTexture {
  return getOrCreateTexture('acousticSlatWood', _generateAcousticSlatWoodTexture);
}

export function createGoldenHourSkyTexture(): THREE.CanvasTexture {
  return getOrCreateTexture('goldenHourSky', _generateGoldenHourSkyTexture);
}

export function createZenGrassTexture(): THREE.CanvasTexture {
  return getOrCreateTexture('zenGrass', _generateZenGrassTexture);
}

export function createGraniteSlabTexture(): THREE.CanvasTexture {
  return getOrCreateTexture('graniteSlab', _generateGraniteSlabTexture);
}

export function createWoodRoughnessMap(): THREE.CanvasTexture {
  return getOrCreateTexture('woodRoughness', _generateWoodRoughnessMap);
}

export function createMarbleRoughnessMap(): THREE.CanvasTexture {
  return getOrCreateTexture('marbleRoughness', _generateMarbleRoughnessMap);
}

/* =====================================================================
 * Backwards-compatible aliases (Home3DView + older imports)
 * ===================================================================== */
export const createWoodFloorTexture = createLuxuryWoodParquetTexture;
export const createMarbleTexture = createCalacattaMarbleTexture;
export const createWallPlasterTexture = createMicrocementWallTexture;
export const createFabricRugTexture = createWovenFabricTexture;
export const createSkyboxTexture = createGoldenHourSkyTexture;
export const createCartoonGrassTexture = createZenGrassTexture;
export const createStonePathTexture = createGraniteSlabTexture;