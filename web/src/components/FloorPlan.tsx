import React from 'react';
import { Lightbulb, Flame, Volume2, ShieldAlert } from 'lucide-react';

export interface DeviceState {
  kitchen_lights: boolean;
  kitchen_heat: number;
  bedroom_lights: boolean;
  bedroom_lamp: boolean;
  bedroom_heat: number;
  washroom_lights: boolean;
  washroom_heat: number;
  hall_music: boolean;
  hall_volume: number;
}

interface FloorPlanProps {
  devices: DeviceState;
  activeLocation: string | null;
  onToggleDevice: (deviceKey: keyof DeviceState) => void;
}

export const FloorPlan: React.FC<FloorPlanProps> = ({ devices, activeLocation, onToggleDevice }) => {
  return (
    <div className="floorplan-wrapper" style={{ minHeight: '440px', padding: '16px' }}>
      <svg
        viewBox="0 0 800 500"
        style={{ width: '100%', height: '100%', display: 'block' }}
      >
        <defs>
          {/* Radial light glow filter */}
          <radialGradient id="lightGlow" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#F2B33D" stopOpacity="0.5" />
            <stop offset="70%" stopColor="#F2B33D" stopOpacity="0.15" />
            <stop offset="100%" stopColor="#F2B33D" stopOpacity="0" />
          </radialGradient>
        </defs>

        {/* Outer boundary */}
        <rect x="20" y="20" width="760" height="460" fill="#F6F8F4" stroke="#16303A" strokeWidth="6" rx="12" />

        {/* ================= ZONE 1: DAPUR (KITCHEN) ================= */}
        <g id="zone-kitchen">
          <rect
            x="20"
            y="20"
            width="380"
            height="230"
            fill={activeLocation === 'kitchen' ? '#EDF3F0' : '#F6F8F4'}
            stroke="#16303A"
            strokeWidth="4"
          />
          {devices.kitchen_lights && (
            <circle cx="210" cy="135" r="140" fill="url(#lightGlow)" pointerEvents="none" />
          )}
          <text x="45" y="55" fill="#16303A" fontSize="18" fontWeight="700">Dapur (Kitchen)</text>
          
          {/* Kitchen Light Toggle */}
          <g
            transform="translate(140, 100)"
            cursor="pointer"
            onClick={() => onToggleDevice('kitchen_lights')}
          >
            <circle cx="20" cy="20" r="24" fill={devices.kitchen_lights ? "#F2B33D" : "#E8ECE6"} stroke="#16303A" strokeWidth="2" />
            <text x="20" y="26" textAnchor="middle" fill="#16303A" fontSize="16">💡</text>
            <text x="20" y="58" textAnchor="middle" fill="#16303A" fontSize="12" fontWeight="600">
              {devices.kitchen_lights ? "ON" : "OFF"}
            </text>
          </g>

          {/* Kitchen Heat */}
          <g
            transform="translate(260, 100)"
            cursor="pointer"
            onClick={() => onToggleDevice('kitchen_heat')}
          >
            <rect x="0" y="0" width="60" height="42" rx="6" fill={devices.kitchen_heat > 20 ? "#C4492F" : "#E8ECE6"} stroke="#16303A" strokeWidth="2" />
            <text x="30" y="25" textAnchor="middle" fill={devices.kitchen_heat > 20 ? "#FFF" : "#16303A"} fontSize="13" fontWeight="700">
              {devices.kitchen_heat}°C
            </text>
            <text x="30" y="58" textAnchor="middle" fill="#16303A" fontSize="12" fontWeight="600">Pemanas</text>
          </g>
        </g>

        {/* ================= ZONE 2: KAMAR TIDUR (BEDROOM) ================= */}
        <g id="zone-bedroom">
          <rect
            x="400"
            y="20"
            width="380"
            height="230"
            fill={activeLocation === 'bedroom' ? '#EDF3F0' : '#F6F8F4'}
            stroke="#16303A"
            strokeWidth="4"
          />
          {devices.bedroom_lights && (
            <circle cx="590" cy="135" r="140" fill="url(#lightGlow)" pointerEvents="none" />
          )}
          <text x="425" y="55" fill="#16303A" fontSize="18" fontWeight="700">Kamar Tidur (Bedroom)</text>
          
          {/* Bedroom Light */}
          <g
            transform="translate(500, 100)"
            cursor="pointer"
            onClick={() => onToggleDevice('bedroom_lights')}
          >
            <circle cx="20" cy="20" r="24" fill={devices.bedroom_lights ? "#F2B33D" : "#E8ECE6"} stroke="#16303A" strokeWidth="2" />
            <text x="20" y="26" textAnchor="middle" fill="#16303A" fontSize="16">💡</text>
            <text x="20" y="58" textAnchor="middle" fill="#16303A" fontSize="12" fontWeight="600">
              {devices.bedroom_lights ? "ON" : "OFF"}
            </text>
          </g>

          {/* Bedroom Lamp */}
          <g
            transform="translate(600, 100)"
            cursor="pointer"
            onClick={() => onToggleDevice('bedroom_lamp')}
          >
            <circle cx="20" cy="20" r="24" fill={devices.bedroom_lamp ? "#F2B33D" : "#E8ECE6"} stroke="#16303A" strokeWidth="2" />
            <text x="20" y="26" textAnchor="middle" fill="#16303A" fontSize="16">🛋️</text>
            <text x="20" y="58" textAnchor="middle" fill="#16303A" fontSize="12" fontWeight="600">Lamp</text>
          </g>

          {/* Bedroom Heat */}
          <g
            transform="translate(700, 100)"
            cursor="pointer"
            onClick={() => onToggleDevice('bedroom_heat')}
          >
            <rect x="0" y="0" width="55" height="42" rx="6" fill={devices.bedroom_heat > 20 ? "#C4492F" : "#E8ECE6"} stroke="#16303A" strokeWidth="2" />
            <text x="27" y="25" textAnchor="middle" fill={devices.bedroom_heat > 20 ? "#FFF" : "#16303A"} fontSize="13" fontWeight="700">
              {devices.bedroom_heat}°C
            </text>
            <text x="27" y="58" textAnchor="middle" fill="#16303A" fontSize="12" fontWeight="600">Suhu</text>
          </g>
        </g>

        {/* ================= ZONE 3: KAMAR MANDI (WASHROOM) ================= */}
        <g id="zone-washroom">
          <rect
            x="20"
            y="250"
            width="380"
            height="230"
            fill={activeLocation === 'washroom' ? '#EDF3F0' : '#F6F8F4'}
            stroke="#16303A"
            strokeWidth="4"
          />
          {devices.washroom_lights && (
            <circle cx="210" cy="365" r="140" fill="url(#lightGlow)" pointerEvents="none" />
          )}
          <text x="45" y="285" fill="#16303A" fontSize="18" fontWeight="700">Kamar Mandi (Washroom)</text>
          
          {/* Washroom Lights */}
          <g
            transform="translate(160, 330)"
            cursor="pointer"
            onClick={() => onToggleDevice('washroom_lights')}
          >
            <circle cx="20" cy="20" r="24" fill={devices.washroom_lights ? "#F2B33D" : "#E8ECE6"} stroke="#16303A" strokeWidth="2" />
            <text x="20" y="26" textAnchor="middle" fill="#16303A" fontSize="16">💡</text>
            <text x="20" y="58" textAnchor="middle" fill="#16303A" fontSize="12" fontWeight="600">
              {devices.washroom_lights ? "ON" : "OFF"}
            </text>
          </g>

          {/* Washroom Heat */}
          <g
            transform="translate(260, 330)"
            cursor="pointer"
            onClick={() => onToggleDevice('washroom_heat')}
          >
            <rect x="0" y="0" width="60" height="42" rx="6" fill={devices.washroom_heat > 20 ? "#C4492F" : "#E8ECE6"} stroke="#16303A" strokeWidth="2" />
            <text x="30" y="25" textAnchor="middle" fill={devices.washroom_heat > 20 ? "#FFF" : "#16303A"} fontSize="13" fontWeight="700">
              {devices.washroom_heat}°C
            </text>
            <text x="30" y="58" textAnchor="middle" fill="#16303A" fontSize="12" fontWeight="600">Pemanas</text>
          </g>
        </g>

        {/* ================= ZONE 4: LORONG & MUSIK (HALL) ================= */}
        <g id="zone-hall">
          <rect
            x="400"
            y="250"
            width="380"
            height="230"
            fill={activeLocation === 'none' || activeLocation === null ? '#F6F8F4' : '#F6F8F4'}
            stroke="#16303A"
            strokeWidth="4"
          />
          <text x="425" y="285" fill="#16303A" fontSize="18" fontWeight="700">Lorong & Suara (Hall)</text>
          
          {/* Music System */}
          <g
            transform="translate(520, 330)"
            cursor="pointer"
            onClick={() => onToggleDevice('hall_music')}
          >
            <rect x="0" y="0" width="80" height="48" rx="8" fill={devices.hall_music ? "#2F6B5E" : "#E8ECE6"} stroke="#16303A" strokeWidth="2" />
            <text x="40" y="28" textAnchor="middle" fill={devices.hall_music ? "#FFF" : "#16303A"} fontSize="14" fontWeight="700">
              {devices.hall_music ? "🎵 PLAY" : "⏸ PAUSE"}
            </text>
            <text x="40" y="66" textAnchor="middle" fill="#16303A" fontSize="12" fontWeight="600">Musik</text>
          </g>

          {/* Volume Level */}
          <g transform="translate(640, 330)">
            <rect x="0" y="0" width="90" height="48" rx="8" fill="#E8ECE6" stroke="#16303A" strokeWidth="2" />
            <text x="45" y="28" textAnchor="middle" fill="#16303A" fontSize="14" fontWeight="700">
              🔊 {devices.hall_volume * 10}%
            </text>
            <text x="45" y="66" textAnchor="middle" fill="#16303A" fontSize="12" fontWeight="600">Volume</text>
          </g>
        </g>
      </svg>
    </div>
  );
};
