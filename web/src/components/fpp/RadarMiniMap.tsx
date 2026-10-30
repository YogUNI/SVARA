import React, { useImperativeHandle, forwardRef, useRef, useEffect } from 'react';
import { DeviceState } from '../FloorPlan';

export interface RadarMiniMapHandle {
  update: (playerX: number, playerZ: number, playerYaw: number) => void;
}

interface RadarMiniMapProps {
  initialX: number;
  initialZ: number;
  devices: DeviceState;
}

/**
 * Architectural Blueprint Mini-Map — luxury floor-plan style radar HUD.
 * Warm blueprint paper + ink lines, glowing device markers and a compass.
 */
export const RadarMiniMap = forwardRef<RadarMiniMapHandle, RadarMiniMapProps>(({
  initialX,
  initialZ,
  devices,
}, ref) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const stateRef = useRef({
    x: initialX,
    z: initialZ,
    yaw: 0,
  });

  const drawRadar = (playerX: number, playerZ: number, playerYaw: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const size = 160;
    const center = size / 2;
    const radius = size / 2 - 4;

    ctx.clearRect(0, 0, size, size);

    // 1. Circular clipping area
    ctx.save();
    ctx.beginPath();
    ctx.arc(center, center, radius, 0, Math.PI * 2);
    ctx.clip();

    // Blueprint paper background (aged warm architectural paper)
    const bg = ctx.createRadialGradient(center, center, 4, center, center, radius);
    bg.addColorStop(0, '#1A2332');
    bg.addColorStop(0.75, '#141C29');
    bg.addColorStop(1, '#0D141F');
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, size, size);

    // Fine blueprint grid
    ctx.strokeStyle = 'rgba(148, 184, 224, 0.10)';
    ctx.lineWidth = 0.6;
    for (let gx = 0; gx < size; gx += 10) {
      ctx.beginPath();
      ctx.moveTo(gx, 0);
      ctx.lineTo(gx, size);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(0, gx);
      ctx.lineTo(size, gx);
      ctx.stroke();
    }

    // 2. World -> map transform.
    // World bounds: House X [-14, 14], Z [-12, 12] + Front Garden Z [12, 31]
    const mapScale = 4.3;
    const toMapX = (wx: number) => center + (wx - playerX) * mapScale;
    const toMapY = (wz: number) => center + (wz - playerZ) * mapScale;

    // House perimeter (drawn in blueprint ink)
    ctx.strokeStyle = 'rgba(190, 210, 235, 0.55)';
    ctx.lineWidth = 1.8;
    const left = toMapX(-14);
    const top = toMapY(-12);
    ctx.strokeRect(left, top, 28 * mapScale, 24 * mapScale);

    // Room partition walls (thin ink)
    ctx.strokeStyle = 'rgba(160, 190, 225, 0.35)';
    ctx.lineWidth = 1.1;
    ctx.beginPath();
    ctx.moveTo(toMapX(0), toMapY(-12));
    ctx.lineTo(toMapX(0), toMapY(-4));
    ctx.moveTo(toMapX(0), toMapY(4));
    ctx.lineTo(toMapX(0), toMapY(12));
    ctx.moveTo(toMapX(-14), toMapY(0));
    ctx.lineTo(toMapX(-4), toMapY(0));
    ctx.moveTo(toMapX(4), toMapY(0));
    ctx.lineTo(toMapX(14), toMapY(0));
    ctx.stroke();

    // Room labels (architectural notation)
    ctx.font = '8px monospace';
    ctx.fillStyle = 'rgba(200, 220, 245, 0.75)';
    ctx.fillText('DAPUR', toMapX(-7) - 17, toMapY(-5));
    ctx.fillText('KAMAR', toMapX(7) - 17, toMapY(-5));
    ctx.fillText('MANDI', toMapX(-7) - 17, toMapY(7));
    ctx.fillText('LOUNGE', toMapX(7) - 19, toMapY(7));

    // Front garden lawn + walkway zone
    ctx.strokeStyle = 'rgba(120, 200, 150, 0.4)';
    ctx.lineWidth = 1.4;
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(toMapX(-21), toMapY(12), 42 * mapScale, 19 * mapScale);
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(120, 200, 150, 0.14)';
    ctx.fillRect(toMapX(-1.5), toMapY(12), 3 * mapScale, 14 * mapScale);

    // 3. Device markers
    const drawDeviceDot = (wx: number, wz: number, isOn: boolean, label: string) => {
      const mx = toMapX(wx);
      const my = toMapY(wz);
      ctx.beginPath();
      ctx.arc(mx, my, isOn ? 5 : 3.6, 0, Math.PI * 2);
      ctx.fillStyle = isOn ? '#F2B24A' : '#5A6B7E';
      ctx.fill();
      if (isOn) {
        const halo = ctx.createRadialGradient(mx, my, 1, mx, my, 12);
        halo.addColorStop(0, 'rgba(242, 178, 74, 0.5)');
        halo.addColorStop(1, 'rgba(242, 178, 74, 0)');
        ctx.fillStyle = halo;
        ctx.beginPath();
        ctx.arc(mx, my, 12, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.strokeStyle = 'rgba(235, 245, 255, 0.85)';
      ctx.lineWidth = 0.8;
      ctx.beginPath();
      ctx.arc(mx, my, isOn ? 5 : 3.6, 0, Math.PI * 2);
      ctx.stroke();
      ctx.fillStyle = 'rgba(210, 228, 248, 0.9)';
      ctx.font = '7px monospace';
      ctx.fillText(label, mx + 6, my + 3);
    };

    drawDeviceDot(-7, -6, devices.kitchen_lights, 'Dapur');
    drawDeviceDot(7, -6, devices.bedroom_lights, 'Kamar');
    drawDeviceDot(-7, 6, devices.washroom_lights, 'Toilet');
    drawDeviceDot(0, 0, devices.hall_music, 'Audio');

    // 4. Compass rings (fine blueprint circles)
    ctx.strokeStyle = 'rgba(150, 185, 225, 0.28)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(center, center, radius * 0.42, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(center, center, radius * 0.7, 0, Math.PI * 2);
    ctx.stroke();

    // Compass cardinal marks
    ctx.fillStyle = 'rgba(190, 214, 240, 0.9)';
    ctx.font = 'bold 9px monospace';
    ctx.fillText('N', center - 3, 14);
    ctx.fillText('S', center - 3.5, size - 6);
    ctx.fillText('E', size - 12, center + 3);
    ctx.fillText('W', 9, center + 3);

    // 5. Player position + heading cone (compass rose style)
    ctx.save();
    ctx.translate(center, center);
    ctx.rotate(-playerYaw);

    const cone = ctx.createRadialGradient(0, 0, 0, 0, 0, 34);
    cone.addColorStop(0, 'rgba(105, 200, 255, 0.55)');
    cone.addColorStop(1, 'rgba(105, 200, 255, 0)');
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, 34, -Math.PI / 2 - 0.42, -Math.PI / 2 + 0.42);
    ctx.closePath();
    ctx.fillStyle = cone;
    ctx.fill();

    // Heading arrow
    ctx.beginPath();
    ctx.moveTo(0, -10);
    ctx.lineTo(5.5, 6);
    ctx.lineTo(0, 3);
    ctx.lineTo(-5.5, 6);
    ctx.closePath();
    ctx.fillStyle = '#69C8FF';
    ctx.fill();

    ctx.restore();

    // 6. Exterior bezel (brushed metal ring)
    ctx.strokeStyle = 'rgba(120, 155, 195, 0.8)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(center, center, radius, 0, Math.PI * 2);
    ctx.stroke();
    ctx.strokeStyle = 'rgba(210, 228, 248, 0.4)';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.arc(center, center, radius - 2, 0, Math.PI * 2);
    ctx.stroke();

    ctx.restore();
  };

  useImperativeHandle(ref, () => ({
    update: (x: number, z: number, yaw: number) => {
      stateRef.current.x = x;
      stateRef.current.z = z;
      stateRef.current.yaw = yaw;
      drawRadar(x, z, yaw);
    },
  }));

  useEffect(() => {
    drawRadar(stateRef.current.x, stateRef.current.z, stateRef.current.yaw);
  }, [devices]);

  return (
    <div style={{
      position: 'absolute',
      top: 14,
      right: 14,
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      pointerEvents: 'none',
    }}>
      <canvas
        ref={canvasRef}
        width={160}
        height={160}
        style={{
          borderRadius: '50%',
          boxShadow: '0 6px 20px rgba(0, 0, 0, 0.7), inset 0 0 0 1px rgba(150, 185, 225, 0.35)',
        }}
      />
      <div style={{
        marginTop: 4,
        background: 'rgba(16, 24, 36, 0.88)',
        color: '#B8D4F2',
        border: '1px solid rgba(120, 160, 210, 0.5)',
        padding: '2px 8px',
        borderRadius: '999px',
        fontSize: '0.66rem',
        fontWeight: 700,
        fontFamily: 'monospace',
        letterSpacing: '0.06em',
      }}>
        FLOOR PLAN • LIVE
      </div>
    </div>
  );
});

RadarMiniMap.displayName = 'RadarMiniMap';