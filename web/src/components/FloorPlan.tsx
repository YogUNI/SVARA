import React from 'react';

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
  onAdjustDevice?: (deviceKey: keyof DeviceState, delta: number) => void;
}

/* -------------------------------------------------------------------------
   Small SVG controls reused in every room
   ------------------------------------------------------------------------- */

interface ToggleControlProps {
  x: number;
  y: number;
  width?: number;
  label: string;
  on: boolean;
  accent?: string;
  onToggle: () => void;
}

const ToggleControl: React.FC<ToggleControlProps> = ({
  x,
  y,
  width = 132,
  label,
  on,
  accent = '#F59E0B',
  onToggle,
}) => {
  const trackX = width - 31;
  const knobX = on ? width - 14 : width - 24;
  return (
    <g
      transform={`translate(${x} ${y})`}
      className="fp-control"
      role="button"
      tabIndex={0}
      aria-label={`${label} ${on ? 'aktif' : 'mati'}`}
      onClick={(e) => {
        e.stopPropagation();
        onToggle();
      }}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onToggle();
        }
      }}
    >
      <rect
        className="fp-control-bg"
        width={width}
        height={44}
        rx={12}
        fill="rgba(9,13,21,0.86)"
        stroke={on ? `${accent}88` : 'rgba(255,255,255,0.10)'}
        strokeWidth={1}
      />
      <circle
        cx={20}
        cy={22}
        r={11}
        fill={on ? `${accent}22` : 'rgba(255,255,255,0.05)'}
        stroke={on ? `${accent}66` : 'rgba(255,255,255,0.10)'}
        strokeWidth={1}
      />
      {/* bulb glyph */}
      <path
        d="M20 15.5a4.6 4.6 0 0 0-2.6 8.4c.5.4.8 1 .8 1.6h3.6c0-.6.3-1.2.8-1.6a4.6 4.6 0 0 0-2.6-8.4Z"
        fill="none"
        stroke={on ? accent : '#64748B'}
        strokeWidth={1.3}
        strokeLinejoin="round"
      />
      <path d="M18.4 27.4h3.2" stroke={on ? accent : '#64748B'} strokeWidth={1.2} strokeLinecap="round" />

      <text x={38} y={17} className="fp-control-label">
        {label}
      </text>
      <text x={38} y={33} className={`fp-control-value ${on ? '' : 'off'}`}>
        {on ? 'ON' : 'OFF'}
      </text>

      <rect
        x={trackX}
        y={13}
        width={24}
        height={18}
        rx={9}
        fill={on ? `${accent}33` : 'rgba(255,255,255,0.06)'}
        stroke={on ? `${accent}88` : 'rgba(255,255,255,0.12)'}
        strokeWidth={1}
      />
      <circle cx={knobX} cy={22} r={6} fill={on ? accent : '#64748B'} />
    </g>
  );
};

interface StepperControlProps {
  x: number;
  y: number;
  width?: number;
  label: string;
  value: string;
  accent?: string;
  hot?: boolean;
  onDecrease: () => void;
  onIncrease: () => void;
}

const StepperControl: React.FC<StepperControlProps> = ({
  x,
  y,
  width = 150,
  label,
  value,
  accent = '#F59E0B',
  hot = false,
  onDecrease,
  onIncrease,
}) => {
  const minusX = width - 44;
  const plusX = width - 22;
  return (
    <g transform={`translate(${x} ${y})`}>
      <rect
        width={width}
        height={44}
        rx={12}
        fill="rgba(9,13,21,0.86)"
        stroke={hot ? `${accent}88` : 'rgba(255,255,255,0.10)'}
        strokeWidth={1}
      />
      <text x={13} y={16} className="fp-control-label">
        {label}
      </text>
      <text x={13} y={33} className="fp-readout" style={{ fill: hot ? accent : '#E2E8F0' }}>
        {value}
      </text>

      <g
        className="fp-control"
        role="button"
        tabIndex={0}
        aria-label={`${label} turunkan`}
        onClick={(e) => {
          e.stopPropagation();
          onDecrease();
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onDecrease();
          }
        }}
      >
        <rect
          x={minusX}
          y={12}
          width={18}
          height={20}
          rx={5}
          fill="rgba(255,255,255,0.06)"
          stroke="rgba(255,255,255,0.14)"
        />
        <path d={`M${minusX + 5} 22h8`} stroke="#CBD5E1" strokeWidth={1.4} strokeLinecap="round" />
      </g>

      <g
        className="fp-control"
        role="button"
        tabIndex={0}
        aria-label={`${label} naikkan`}
        onClick={(e) => {
          e.stopPropagation();
          onIncrease();
        }}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            onIncrease();
          }
        }}
      >
        <rect
          x={plusX}
          y={12}
          width={18}
          height={20}
          rx={5}
          fill="rgba(255,255,255,0.06)"
          stroke="rgba(255,255,255,0.14)"
        />
        <path
          d={`M${plusX + 5} 22h8M${plusX + 9} 18v8`}
          stroke="#CBD5E1"
          strokeWidth={1.4}
          strokeLinecap="round"
        />
      </g>
    </g>
  );
};

/* -------------------------------------------------------------------------
   Floor plan
   ------------------------------------------------------------------------- */

export const FloorPlan: React.FC<FloorPlanProps> = ({
  devices,
  activeLocation,
  onToggleDevice,
  onAdjustDevice,
}) => {
  const adjust = (key: keyof DeviceState, delta: number) => {
    if (onAdjustDevice) onAdjustDevice(key, delta);
    else onToggleDevice(key);
  };

  return (
    <div className="floorplan-wrapper">
      <svg viewBox="0 0 920 640" className="floorplan-svg" role="img" aria-label="Denah rumah SVARA mode 2D CAD">
        <defs>
          <radialGradient id="warmBloom" cx="50%" cy="42%" r="68%">
            <stop offset="0%" stopColor="#F59E0B" stopOpacity="0.42" />
            <stop offset="42%" stopColor="#F59E0B" stopOpacity="0.18" />
            <stop offset="100%" stopColor="#F59E0B" stopOpacity="0" />
          </radialGradient>

          <radialGradient id="lampBloom" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#FBBF24" stopOpacity="0.55" />
            <stop offset="55%" stopColor="#F59E0B" stopOpacity="0.22" />
            <stop offset="100%" stopColor="#F59E0B" stopOpacity="0" />
          </radialGradient>

          <radialGradient id="heatAura" cx="50%" cy="50%" r="50%">
            <stop offset="0%" stopColor="#EF4444" stopOpacity="0.35" />
            <stop offset="100%" stopColor="#EF4444" stopOpacity="0" />
          </radialGradient>

          <linearGradient id="roomGrad" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#111A28" />
            <stop offset="100%" stopColor="#0D1521" />
          </linearGradient>

          <pattern id="blueprintGrid" width="20" height="20" patternUnits="userSpaceOnUse">
            <path d="M20 0H0V20" fill="none" stroke="rgba(148,163,184,0.055)" strokeWidth="0.5" />
          </pattern>

          <filter id="zoneGlow" x="-30%" y="-30%" width="160%" height="160%">
            <feGaussianBlur stdDeviation="5" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>

          <filter id="softGlow" x="-60%" y="-60%" width="220%" height="220%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>

        {/* CAD title block */}
        <text x="40" y="26" className="fp-dimension-text" style={{ letterSpacing: '0.18em' }}>
          SVARA • DENAH LANTAI 01
        </text>
        <text x="880" y="26" textAnchor="end" className="fp-dimension-text" style={{ letterSpacing: '0.16em' }}>
          ARSITEKTUR CAD • SKALA 1:100
        </text>

        {/* Foundation plate */}
        <rect x="40" y="44" width="840" height="552" fill="#0D1521" />
        <rect x="40" y="44" width="840" height="552" fill="url(#blueprintGrid)" />

        {/* Room surfaces */}
        <rect x="47" y="51" width="406" height="262" fill="url(#roomGrad)" />
        <rect x="467" y="51" width="406" height="262" fill="url(#roomGrad)" />
        <rect x="47" y="327" width="406" height="262" fill="url(#roomGrad)" />
        <rect x="467" y="327" width="406" height="262" fill="url(#roomGrad)" />

        {/* Light blooms (under furniture) */}
        <rect
          x="47"
          y="51"
          width="406"
          height="262"
          fill="url(#warmBloom)"
          className="fp-bloom"
          opacity={devices.kitchen_lights ? 1 : 0}
          pointerEvents="none"
        />
        <rect
          x="467"
          y="51"
          width="406"
          height="262"
          fill="url(#warmBloom)"
          className="fp-bloom"
          opacity={devices.bedroom_lights ? 1 : 0}
          pointerEvents="none"
        />
        <circle
          cx="564"
          cy="124"
          r="96"
          fill="url(#lampBloom)"
          className="fp-bloom"
          opacity={devices.bedroom_lamp ? 1 : 0}
          pointerEvents="none"
        />
        <rect
          x="47"
          y="327"
          width="406"
          height="262"
          fill="url(#warmBloom)"
          className="fp-bloom"
          opacity={devices.washroom_lights ? 1 : 0}
          pointerEvents="none"
        />

        {/* Ambient heat auras */}
        <circle
          cx="150"
          cy="220"
          r="130"
          fill="url(#heatAura)"
          className="fp-bloom"
          opacity={devices.kitchen_heat > 20 ? 0.9 : 0}
          pointerEvents="none"
        />
        <circle
          cx="790"
          cy="180"
          r="130"
          fill="url(#heatAura)"
          className="fp-bloom"
          opacity={devices.bedroom_heat > 20 ? 0.9 : 0}
          pointerEvents="none"
        />
        <circle
          cx="300"
          cy="430"
          r="130"
          fill="url(#heatAura)"
          className="fp-bloom"
          opacity={devices.washroom_heat > 20 ? 0.9 : 0}
          pointerEvents="none"
        />

        {/* ======================= KITCHEN FURNITURE ======================= */}
        <g id="furniture-kitchen">
          {/* Counter run along the west wall */}
          <rect x="58" y="120" width="54" height="128" rx="4" className="fp-furniture" />
          <line x1="58" y1="162" x2="112" y2="162" className="fp-furniture-line" />
          <line x1="58" y1="204" x2="112" y2="204" className="fp-furniture-line" />
          {/* Sink */}
          <rect x="65" y="128" width="40" height="26" rx="5" className="fp-furniture-soft" />
          <circle cx="85" cy="141" r="8" className="fp-muted-line" />
          <circle cx="85" cy="124" r="2.4" fill="#64748B" />
          {/* Cooktop */}
          <rect x="64" y="170" width="42" height="30" rx="4" className="fp-furniture-soft" />
          {[74, 96].map((cx) =>
            [180, 191].map((cy) => <circle key={`${cx}-${cy}`} cx={cx} cy={cy} r="3.4" className="fp-muted-line" />),
          )}

          {/* Island with prep top */}
          <rect x="170" y="150" width="180" height="76" rx="6" className="fp-furniture" />
          <rect x="177" y="157" width="102" height="62" rx="4" className="fp-furniture-soft" />
          <line x1="228" y1="157" x2="228" y2="219" className="fp-furniture-line" />
          <circle cx="200" cy="188" r="9" className="fp-muted-line" />

          {/* Stools */}
          {[200, 250, 300].map((cx) => (
            <circle key={`stool-${cx}`} cx={cx} cy={238} r="9" className="fp-furniture" />
          ))}

          {/* Fridge */}
          <rect x="370" y="120" width="62" height="76" rx="5" className="fp-furniture" />
          <line x1="370" y1="146" x2="432" y2="146" className="fp-furniture-line" />
          <rect x="424" y="132" width="2.5" height="10" rx="1" fill="#64748B" />
          <rect x="424" y="158" width="2.5" height="14" rx="1" fill="#64748B" />

          {/* Breakfast table */}
          <circle cx="392" cy="232" r="19" className="fp-furniture" />
          <circle cx="392" cy="232" r="13" className="fp-muted-line" />
          <rect x="376" y="208" width="14" height="9" rx="3" className="fp-furniture-soft" />
          <rect x="394" y="208" width="14" height="9" rx="3" className="fp-furniture-soft" />

          {/* Plant */}
          <circle cx="66" cy="284" r="10" className="fp-furniture" />
          <circle cx="66" cy="284" r="5" className="fp-muted-line" />
        </g>

        {/* ======================= BEDROOM FURNITURE ======================= */}
        <g id="furniture-bedroom">
          {/* Rug */}
          <ellipse cx="670" cy="212" rx="150" ry="84" className="fp-furniture-soft" opacity="0.75" />
          <ellipse cx="670" cy="212" rx="136" ry="72" className="fp-muted-line" />
          <ellipse cx="670" cy="212" rx="120" ry="58" className="fp-muted-line" />

          {/* Platform bed */}
          <rect x="590" y="110" width="180" height="140" rx="6" className="fp-furniture" />
          {/* Headboard */}
          <rect x="584" y="100" width="192" height="12" rx="4" className="fp-furniture-soft" />
          {/* Pillows */}
          <rect x="604" y="122" width="72" height="32" rx="9" className="fp-furniture-soft" />
          <rect x="684" y="122" width="72" height="32" rx="9" className="fp-furniture-soft" />
          {/* Blanket */}
          <rect x="600" y="186" width="160" height="58" rx="7" className="fp-furniture-soft" />
          <line x1="600" y1="198" x2="760" y2="198" className="fp-furniture-line" />
          {/* Throw stripe */}
          <rect x="600" y="214" width="160" height="8" fill="#334155" opacity="0.55" />

          {/* Nightstands + lamps */}
          <rect x="548" y="110" width="32" height="32" rx="4" className="fp-furniture" />
          <circle cx="564" cy="126" r="6" fill={devices.bedroom_lamp ? '#FBBF24' : '#475569'} />
          <rect x="780" y="110" width="32" height="32" rx="4" className="fp-furniture" />
          <circle cx="796" cy="126" r="6" fill="#475569" />

          {/* Wardrobe */}
          <rect x="818" y="152" width="40" height="110" rx="4" className="fp-furniture" />
          <line x1="838" y1="152" x2="838" y2="262" className="fp-furniture-line" />
          <line x1="818" y1="206" x2="858" y2="206" className="fp-furniture-line" />
        </g>

        {/* ======================= WASHROOM FURNITURE ======================= */}
        <g id="furniture-washroom">
          {/* Bathtub */}
          <rect x="90" y="380" width="140" height="110" rx="16" className="fp-furniture" />
          <rect x="100" y="390" width="120" height="90" rx="12" className="fp-muted-line" />
          <circle cx="160" cy="435" r="3" fill="#64748B" />

          {/* Shower */}
          <rect x="260" y="380" width="90" height="90" rx="5" className="fp-furniture" />
          <line x1="260" y1="425" x2="350" y2="425" className="fp-furniture-line" />
          <line x1="306" y1="380" x2="306" y2="425" className="fp-furniture-line" />
          <circle cx="286" cy="404" r="8" className="fp-muted-line" />
          <circle cx="342" cy="440" r="2.5" fill="#64748B" />

          {/* Vanity + basin */}
          <rect x="378" y="380" width="66" height="70" rx="5" className="fp-furniture" />
          <ellipse cx="411" cy="410" rx="20" ry="14" className="fp-furniture-soft" />
          <circle cx="411" cy="410" r="4" className="fp-muted-line" />

          {/* Toilet */}
          <rect x="390" y="478" width="42" height="18" rx="4" className="fp-furniture" />
          <ellipse cx="411" cy="510" rx="18" ry="22" className="fp-furniture" />
          <ellipse cx="411" cy="510" rx="10" ry="14" className="fp-muted-line" />

          {/* Towel rail */}
          <line x1="66" y1="382" x2="66" y2="470" className="fp-furniture-line" />
          <line x1="60" y1="392" x2="72" y2="392" className="fp-furniture-line" />
          <line x1="60" y1="458" x2="72" y2="458" className="fp-furniture-line" />

          {/* Plant */}
          <circle cx="430" cy="566" r="9" className="fp-furniture" />
        </g>

        {/* ======================= HALL FURNITURE ======================= */}
        <g id="furniture-hall">
          {/* Rug */}
          <rect x="500" y="404" width="220" height="146" rx="10" className="fp-furniture-soft" opacity="0.7" />
          <rect x="512" y="416" width="196" height="122" rx="6" className="fp-muted-line" />

          {/* TV console + wall screen */}
          <rect x="560" y="390" width="180" height="16" rx="4" className="fp-furniture" />
          <rect x="600" y="384" width="100" height="4" rx="2" fill="#38BDF8" opacity="0.5" />

          {/* Sofa */}
          <rect x="520" y="420" width="160" height="54" rx="10" className="fp-furniture" />
          <rect x="528" y="428" width="144" height="38" rx="7" className="fp-furniture-soft" />
          <line x1="600" y1="428" x2="600" y2="466" className="fp-furniture-line" />
          <rect x="510" y="426" width="12" height="42" rx="5" className="fp-furniture" />
          <rect x="678" y="426" width="12" height="42" rx="5" className="fp-furniture" />

          {/* Coffee table */}
          <rect x="560" y="490" width="100" height="34" rx="6" className="fp-furniture" />
          <rect x="570" y="498" width="80" height="18" rx="4" className="fp-muted-line" />

          {/* Speaker with live equalizer */}
          <rect x="790" y="390" width="56" height="84" rx="8" className="fp-furniture" />
          <rect x="798" y="398" width="40" height="68" rx="5" className="fp-furniture-soft" />
          {devices.hall_music ? (
            <g>
              {[0, 1, 2, 3].map((i) => (
                <rect
                  key={`eq-${i}`}
                  x={804 + i * 9}
                  y={430}
                  width={5}
                  height={28}
                  rx={2}
                  fill="#06B6D4"
                  className="eq-bar"
                  style={{ animationDelay: `${i * 0.11}s` }}
                />
              ))}
            </g>
          ) : (
            <g>
              {[0, 1, 2, 3].map((i) => (
                <rect key={`eq-off-${i}`} x={804 + i * 9} y={448} width={5} height={10} rx={2} fill="#334155" />
              ))}
            </g>
          )}
          {/* Volume ticks */}
          <line x1="798" y1="474" x2="838" y2="474" stroke="#475569" strokeWidth="1.5" />
        </g>

        {/* ======================= WALLS & OPENINGS ======================= */}
        {/* Outer shell */}
        <rect x="40" y="44" width="840" height="14" fill="#334155" />
        <rect x="40" y="582" width="600" height="14" fill="#334155" />
        <rect x="700" y="582" width="180" height="14" fill="#334155" />
        <rect x="40" y="44" width="14" height="552" fill="#334155" />
        <rect x="866" y="44" width="14" height="552" fill="#334155" />

        {/* Central partitions with door openings */}
        <rect x="453" y="44" width="14" height="96" fill="#334155" />
        <rect x="453" y="200" width="14" height="230" fill="#334155" />
        <rect x="453" y="490" width="14" height="106" fill="#334155" />

        <rect x="40" y="313" width="60" height="14" fill="#334155" />
        <rect x="160" y="313" width="560" height="14" fill="#334155" />
        <rect x="780" y="313" width="100" height="14" fill="#334155" />

        {/* Wall highlights */}
        {[
          [47, 51, 406, 262],
          [467, 51, 406, 262],
          [47, 327, 406, 262],
          [467, 327, 406, 262],
        ].map(([x, y, w, h]) => (
          <rect key={`edge-${x}-${y}`} x={x} y={y} width={w} height={h} className="fp-wall-edge" fill="none" />
        ))}

        {/* Windows */}
        <g>
          <rect x="180" y="44" width="120" height="14" fill="#0B1220" stroke="#38BDF8" strokeOpacity="0.35" />
          <line x1="180" y1="51" x2="300" y2="51" stroke="#38BDF8" strokeOpacity="0.55" strokeWidth="1.4" />
          <rect x="600" y="44" width="160" height="14" fill="#0B1220" stroke="#38BDF8" strokeOpacity="0.35" />
          <line x1="600" y1="51" x2="760" y2="51" stroke="#38BDF8" strokeOpacity="0.55" strokeWidth="1.4" />
          <rect x="866" y="400" width="14" height="120" fill="#0B1220" stroke="#38BDF8" strokeOpacity="0.35" />
          <line x1="873" y1="400" x2="873" y2="520" stroke="#38BDF8" strokeOpacity="0.55" strokeWidth="1.4" />
          <rect x="40" y="380" width="14" height="120" fill="#0B1220" stroke="#38BDF8" strokeOpacity="0.35" />
        </g>

        {/* Door swings */}
        <g>
          <path d="M460 140 L520 140" className="fp-door" />
          <path d="M520 140 A60 60 0 0 1 460 200" className="fp-door" />
          <path d="M460 490 L400 490" className="fp-door" />
          <path d="M400 490 A60 60 0 0 1 460 430" className="fp-door" />
          <path d="M160 320 L160 260" className="fp-door" />
          <path d="M160 260 A60 60 0 0 1 100 320" className="fp-door" />
          <path d="M720 320 L720 380" className="fp-door" />
          <path d="M720 380 A60 60 0 0 1 780 320" className="fp-door" />
          <path d="M700 589 L700 529" className="fp-door" />
          <path d="M700 529 A60 60 0 0 1 640 589" className="fp-door" />
        </g>

        {/* ======================= ROOM LABELS ======================= */}
        <g pointerEvents="none">
          <text x="62" y="82" className="fp-room-label">
            Dapur
          </text>
          <text x="62" y="98" className="fp-room-sub">
            KITCHEN • 14 m²
          </text>

          <text x="482" y="82" className="fp-room-label">
            Kamar Tidur
          </text>
          <text x="482" y="98" className="fp-room-sub">
            BEDROOM • 18 m²
          </text>

          <text x="62" y="358" className="fp-room-label">
            Kamar Mandi
          </text>
          <text x="62" y="374" className="fp-room-sub">
            WASHROOM • 8 m²
          </text>

          <text x="482" y="358" className="fp-room-label">
            Lorong
          </text>
          <text x="482" y="374" className="fp-room-sub">
            HALL &amp; AUDIO • 12 m²
          </text>
        </g>

        {/* ======================= ACTIVE ZONE HALOS ======================= */}
        <rect
          x="48.5"
          y="52.5"
          width="403"
          height="259"
          rx="2"
          fill="none"
          stroke="#6366F1"
          strokeWidth="2.4"
          filter="url(#zoneGlow)"
          className={`zone-halo ${activeLocation === 'kitchen' ? 'is-active' : ''}`}
          pointerEvents="none"
        />
        <rect
          x="468.5"
          y="52.5"
          width="403"
          height="259"
          rx="2"
          fill="none"
          stroke="#6366F1"
          strokeWidth="2.4"
          filter="url(#zoneGlow)"
          className={`zone-halo ${activeLocation === 'bedroom' ? 'is-active' : ''}`}
          pointerEvents="none"
        />
        <rect
          x="48.5"
          y="328.5"
          width="403"
          height="259"
          rx="2"
          fill="none"
          stroke="#6366F1"
          strokeWidth="2.4"
          filter="url(#zoneGlow)"
          className={`zone-halo ${activeLocation === 'washroom' ? 'is-active' : ''}`}
          pointerEvents="none"
        />
        <rect
          x="468.5"
          y="328.5"
          width="403"
          height="259"
          rx="2"
          fill="none"
          stroke="#6366F1"
          strokeWidth="2.4"
          filter="url(#zoneGlow)"
          className={`zone-halo ${activeLocation === 'none' ? 'is-active' : ''}`}
          pointerEvents="none"
        />

        {/* ======================= INTERACTIVE CONTROLS ======================= */}
        <g>
          {/* Kitchen */}
          <StepperControl
            x={140}
            y={252}
            label="SUHU DAPUR"
            value={`${devices.kitchen_heat}°C`}
            hot={devices.kitchen_heat > 20}
            accent="#EF4444"
            onDecrease={() => adjust('kitchen_heat', -2)}
            onIncrease={() => adjust('kitchen_heat', 2)}
          />
          <ToggleControl
            x={300}
            y={252}
            label="LAMPU DAPUR"
            on={devices.kitchen_lights}
            onToggle={() => onToggleDevice('kitchen_lights')}
          />

          {/* Bedroom */}
          <ToggleControl
            x={486}
            y={258}
            width={116}
            label="LAMPU"
            on={devices.bedroom_lights}
            onToggle={() => onToggleDevice('bedroom_lights')}
          />
          <ToggleControl
            x={610}
            y={258}
            width={116}
            label="LAMPU TIDUR"
            on={devices.bedroom_lamp}
            accent="#FBBF24"
            onToggle={() => onToggleDevice('bedroom_lamp')}
          />
          <StepperControl
            x={734}
            y={258}
            width={126}
            label="SUHU"
            value={`${devices.bedroom_heat}°C`}
            hot={devices.bedroom_heat > 20}
            accent="#EF4444"
            onDecrease={() => adjust('bedroom_heat', -2)}
            onIncrease={() => adjust('bedroom_heat', 2)}
          />

          {/* Washroom */}
          <ToggleControl
            x={66}
            y={530}
            label="LAMPU MANDI"
            on={devices.washroom_lights}
            onToggle={() => onToggleDevice('washroom_lights')}
          />
          <StepperControl
            x={210}
            y={530}
            label="SUHU MANDI"
            value={`${devices.washroom_heat}°C`}
            hot={devices.washroom_heat > 20}
            accent="#EF4444"
            onDecrease={() => adjust('washroom_heat', -2)}
            onIncrease={() => adjust('washroom_heat', 2)}
          />

          {/* Hall */}
          <ToggleControl
            x={486}
            y={532}
            label="MUSIK LORONG"
            on={devices.hall_music}
            accent="#06B6D4"
            onToggle={() => onToggleDevice('hall_music')}
          />
          <StepperControl
            x={648}
            y={532}
            label="VOLUME"
            value={`${devices.hall_volume * 10}%`}
            hot={devices.hall_music}
            accent="#06B6D4"
            onDecrease={() => adjust('hall_volume', -1)}
            onIncrease={() => adjust('hall_volume', 1)}
          />
        </g>

        {/* ======================= DIMENSION LINE ======================= */}
        <g pointerEvents="none">
          <line x1="40" y1="620" x2="880" y2="620" className="fp-dimension" />
          <line x1="40" y1="615" x2="40" y2="625" className="fp-dimension" />
          <line x1="880" y1="615" x2="880" y2="625" className="fp-dimension" />
          <line x1="460" y1="615" x2="460" y2="625" className="fp-dimension" />
          <rect x="418" y="612" width="84" height="16" rx="8" fill="#0B0F17" />
          <text x="460" y="623" textAnchor="middle" className="fp-dimension-text">
            18.64 m
          </text>
        </g>
      </svg>
    </div>
  );
};
