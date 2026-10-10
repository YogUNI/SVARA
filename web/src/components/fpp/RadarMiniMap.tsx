import React, { useEffect, useRef } from 'react';
import { DeviceState } from '../FloorPlan';

interface RadarMiniMapProps {
  playerX: number; // Three.js X (-14 to 14)
  playerZ: number; // Three.js Z (-12 to 12)
  playerRotationY: number; // Camera yaw rotation angle in radians
  currentRoom: string;
  devices: DeviceState;
}

export const RadarMiniMap: React.FC<RadarMiniMapProps> = ({
  playerX,
  playerZ,
  playerRotationY,
  devices,
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
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
    ctx.fillStyle = 'rgba(10, 18, 22, 0.90)';
    ctx.fillRect(0, 0, size, size);

    // 2. Map coordinates transformation
    // World bounds: X [-14, 14] (width 28), Z [-12, 12] (height 24)
    const mapScale = 4.6; // pixels per meter
    const toMapX = (wx: number) => center + (wx - playerX) * mapScale;
    const toMapY = (wz: number) => center + (wz - playerZ) * mapScale;

    // Draw Interior Floor Plan Boundaries relative to player
    ctx.strokeStyle = 'rgba(47, 107, 94, 0.45)';
    ctx.lineWidth = 1.5;

    // Outer Perimeter
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

    // 3. Smart Device Icons / Dots
    const drawDeviceDot = (wx: number, wz: number, isOn: boolean, label: string) => {
      const mx = toMapX(wx);
      const my = toMapY(wz);
      ctx.beginPath();
      ctx.arc(mx, my, 4, 0, Math.PI * 2);
      ctx.fillStyle = isOn ? '#F2B33D' : '#4E6166';
      ctx.fill();
      ctx.strokeStyle = '#FFFFFF';
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.fillStyle = '#C2D1CC';
      ctx.font = '7px sans-serif';
      ctx.fillText(label, mx + 5, my + 3);
    };

    drawDeviceDot(-7, -6, devices.kitchen_lights, 'Dapur');
    drawDeviceDot(7, -6, devices.bedroom_lights, 'Kamar');
    drawDeviceDot(-7, 6, devices.washroom_lights, 'Toilet');
    drawDeviceDot(0, 0, devices.hall_music, 'Audio');

    // 4. Concentric Radar Rings
    ctx.strokeStyle = 'rgba(78, 161, 142, 0.25)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.arc(center, center, radius * 0.4, 0, Math.PI * 2);
    ctx.arc(center, center, radius * 0.75, 0, Math.PI * 2);
    ctx.stroke();

    // 5. Player Dot & Directional Cone (PUBG Mobile Style)
    // Cone of Vision
    ctx.save();
    ctx.translate(center, center);
    ctx.rotate(playerRotationY);

    const coneGradient = ctx.createRadialGradient(0, 0, 0, 0, 0, 36);
    coneGradient.addColorStop(0, 'rgba(62, 224, 143, 0.55)');
    coneGradient.addColorStop(1, 'rgba(62, 224, 143, 0.0)');

    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.arc(0, 0, 36, -Math.PI / 2 - 0.45, -Math.PI / 2 + 0.45);
    ctx.closePath();
    ctx.fillStyle = coneGradient;
    ctx.fill();

    // Player Direction Arrow
    ctx.beginPath();
    ctx.moveTo(0, -9);
    ctx.lineTo(6, 6);
    ctx.lineTo(0, 3);
    ctx.lineTo(-6, 6);
    ctx.closePath();
    ctx.fillStyle = '#3EE08F';
    ctx.fill();
    ctx.restore();

    ctx.restore(); // End clipping

    // 6. Exterior Tactical Radar Bezel
    ctx.strokeStyle = '#2F6B5E';
    ctx.lineWidth = 2.5;
    ctx.beginPath();
    ctx.arc(center, center, radius, 0, Math.PI * 2);
    ctx.stroke();

    // Cardinal direction ticks (N, S, E, W)
    ctx.fillStyle = '#3EE08F';
    ctx.font = 'bold 9px monospace';
    ctx.fillText('N', center - 3, 12);
  }, [playerX, playerZ, playerRotationY, devices]);

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
        background: 'rgba(10, 18, 22, 0.85)',
        color: '#3EE08F',
        border: '1px solid rgba(47, 107, 94, 0.6)',
        padding: '2px 8px',
        borderRadius: '999px',
        fontSize: '0.68rem',
        fontWeight: 700,
        fontFamily: 'monospace',
        letterSpacing: '0.04em',
      }}>
        RADAR GPS: ACTIVE
      </div>
    </div>
  );
};
