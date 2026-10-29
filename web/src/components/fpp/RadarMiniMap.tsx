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

    const size = 150;
    const center = size / 2;
    const radius = size / 2 - 4;

    ctx.clearRect(0, 0, size, size);

    // 1. Circular Radar Clipping Area
    ctx.save();
    ctx.beginPath();
    ctx.arc(center, center, radius, 0, Math.PI * 2);
    ctx.clip();

    // Radar Dark Glass Background
    ctx.fillStyle = 'rgba(10, 18, 22, 0.92)';
    ctx.fillRect(0, 0, size, size);

    // 2. Map coordinates transformation
    // World bounds: House X [-14, 14], Z [-12, 12] + Front Garden Z [12, 31]
    const mapScale = 4.4; // pixels per meter
    const toMapX = (wx: number) => center + (wx - playerX) * mapScale;
    const toMapY = (wz: number) => center + (wz - playerZ) * mapScale;

    // Draw Interior Floor Plan Boundaries relative to player
    ctx.strokeStyle = 'rgba(148, 163, 184, 0.42)';
    ctx.lineWidth = 1.6;

    // Outer Perimeter House Walls
    const left = toMapX(-14);
    const top = toMapY(-12);
    const width = 28 * mapScale;
    const height = 24 * mapScale;
    ctx.strokeRect(left, top, width, height);

    // Dividing Walls
    ctx.beginPath();
    // Center vertical wall (with hallway gap)
    ctx.moveTo(toMapX(0), toMapY(-12));
    ctx.lineTo(toMapX(0), toMapY(-4));
    ctx.moveTo(toMapX(0), toMapY(4));
    ctx.lineTo(toMapX(0), toMapY(12));

    // Horizontal walls
    ctx.moveTo(toMapX(-14), toMapY(0));
    ctx.lineTo(toMapX(-4), toMapY(0));
    ctx.moveTo(toMapX(4), toMapY(0));
    ctx.lineTo(toMapX(14), toMapY(0));
    ctx.stroke();

    // Front Garden Lawn Outline & Walkway
    ctx.strokeStyle = 'rgba(16, 185, 129, 0.38)';
    ctx.strokeRect(toMapX(-21), toMapY(12), 42 * mapScale, 19 * mapScale);
    ctx.fillStyle = 'rgba(16, 185, 129, 0.16)';
    ctx.fillRect(toMapX(-1.5), toMapY(12), 3 * mapScale, 14 * mapScale);

    // 3. Smart Device Icons / Dots
    const drawDeviceDot = (wx: number, wz: number, isOn: boolean, label: string) => {
      const mx = toMapX(wx);
      const my = toMapY(wz);
      ctx.beginPath();
      ctx.arc(mx, my, 4, 0, Math.PI * 2);
      ctx.fillStyle = isOn ? '#FBBF24' : '#475569';
      ctx.fill();
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = '#94A3B8';
      ctx.font = '7px sans-serif';
      ctx.fillText(label, mx + 5, my + 3);
    };

    drawDeviceDot(-7, -6, devices.kitchen_lights, 'Dapur');
    drawDeviceDot(7, -6, devices.bedroom_lights, 'Kamar');
    drawDeviceDot(-7, 6, devices.washroom_lights, 'Toilet');
    drawDeviceDot(0, 0, devices.hall_music, 'Audio');

    // 4. Concentric Radar Rings
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.22)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(center, center, radius * 0.4, 0, Math.PI * 2);
    ctx.arc(center, center, radius * 0.75, 0, Math.PI * 2);
    ctx.stroke();

    // 5. Player Dot & Directional Cone (PUBG Mobile Style)
    // Three.js PointerLockControls yaw angle points along negative Z when yaw=0 (North)
    ctx.save();
    ctx.translate(center, center);
    ctx.rotate(-playerYaw); // Exact 1:1 camera yaw sync

    const coneGradient = ctx.createRadialGradient(0, 0, 0, 0, 0, 36);
    coneGradient.addColorStop(0, 'rgba(56, 189, 248, 0.6)');
    coneGradient.addColorStop(1, 'rgba(56, 189, 248, 0.0)');

    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, 36, -Math.PI / 2 - 0.45, -Math.PI / 2 + 0.45);
    ctx.closePath();
    ctx.fillStyle = coneGradient;
    ctx.fill();

    // Player Direction Arrow
    ctx.beginPath();
    ctx.moveTo(0, -10);
    ctx.lineTo(6, 6);
    ctx.lineTo(0, 3);
    ctx.lineTo(-6, 6);
    ctx.closePath();
    ctx.fillStyle = '#38BDF8';
    ctx.fill();
    ctx.restore();

    ctx.restore(); // End clipping

    // 6. Exterior Tactical Radar Bezel
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.75)';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(center, center, radius, 0, Math.PI * 2);
    ctx.stroke();

    // Cardinal direction ticks (N, S, E, W)
    ctx.fillStyle = '#A5B4FC';
    ctx.font = 'bold 9px monospace';
    ctx.fillText('N', center - 3, 12);
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
        width={150}
        height={150}
        style={{
          borderRadius: '50%',
          boxShadow: '0 4px 16px rgba(0, 0, 0, 0.65)',
        }}
      />
      <div style={{
        marginTop: 4,
        background: 'rgba(11, 15, 23, 0.85)',
        color: '#A5B4FC',
        border: '1px solid rgba(99, 102, 241, 0.45)',
        padding: '2px 8px',
        borderRadius: '999px',
        fontSize: '0.68rem',
        fontWeight: 700,
        fontFamily: 'monospace',
        letterSpacing: '0.04em',
      }}>
        RADAR • LIVE SYNC
      </div>
    </div>
  );
});

RadarMiniMap.displayName = 'RadarMiniMap';
