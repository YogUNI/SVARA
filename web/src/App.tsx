import React, { useEffect, useRef, useState } from 'react';
import {
  Mic,
  Upload,
  RotateCcw,
  Box,
  Map,
  Gamepad2,
  Home,
  FlaskConical,
  Info,
  Zap,
  Package,
  MapPin,
  Activity,
  Check,
  Minus,
  Plus,
  Lightbulb,
  Music,
  Volume2,
  Thermometer,
  Loader2,
  Clock,
  Sparkles,
  Flame,
} from 'lucide-react';
import { FloorPlan, DeviceState } from './components/FloorPlan';
import { Home3DView } from './components/Home3DView';
import { FPPWalkthrough } from './components/FPPWalkthrough';

interface PredictionResult {
  accepted: boolean;
  intent: {
    action: string;
    object: string;
    location: string;
  };
  confidence: {
    action: number;
    object: number;
    location: number;
    overall: number;
  };
  threshold: number;
  effect: {
    type: string;
    message: string;
  };
  timing_ms: {
    decode_ms: number;
    preprocess_ms: number;
    inference_ms: number;
    total_ms: number;
  };
}

interface EventLogItem {
  id: string;
  time: string;
  command: string;
  accepted: boolean;
  message: string;
  confidence: number;
}

const INITIAL_DEVICES: DeviceState = {
  kitchen_lights: false,
  kitchen_heat: 20,
  bedroom_lights: false,
  bedroom_lamp: false,
  bedroom_heat: 20,
  washroom_lights: false,
  washroom_heat: 20,
  hall_music: false,
  hall_volume: 5,
};

const SAMPLE_COMMANDS = [
  'Nyalakan lampu di dapur',
  'Tolong matikan lampu kamar mandi',
  'Nyalakan musiknya dong',
  'Gedein volumenya tolong ya',
  'Matikan lampu tidur',
  'Turn on the lights in the kitchen',
];

const SLOT_LIST = [
  { key: 'action', label: 'Action', Icon: Zap, color: '#818CF8' },
  { key: 'object', label: 'Object', Icon: Package, color: '#FBBF24' },
  { key: 'location', label: 'Location', Icon: MapPin, color: '#22D3EE' },
] as const;

const TOGGLE_TILES = [
  { key: 'kitchen_lights', label: 'Lampu Dapur', Icon: Lightbulb, audio: false },
  { key: 'bedroom_lights', label: 'Lampu Kamar', Icon: Lightbulb, audio: false },
  { key: 'bedroom_lamp', label: 'Lampu Tidur', Icon: Flame, audio: false },
  { key: 'washroom_lights', label: 'Lampu Mandi', Icon: Lightbulb, audio: false },
  { key: 'hall_music', label: 'Musik Lorong', Icon: Music, audio: true },
] as const;

const CLIMATE_ROWS = [
  { key: 'kitchen_heat', label: 'Suhu Dapur' },
  { key: 'bedroom_heat', label: 'Suhu Kamar' },
  { key: 'washroom_heat', label: 'Suhu Mandi' },
] as const;

// Helper: Resample any input sample rate (e.g. 48kHz / 44.1kHz) to exact 16kHz
function resampleTo16k(audioBuffer: Float32Array, origSampleRate: number): Float32Array {
  if (origSampleRate === 16000) return audioBuffer;
  const ratio = 16000 / origSampleRate;
  const newLength = Math.round(audioBuffer.length * ratio);
  const result = new Float32Array(newLength);
  for (let i = 0; i < newLength; i++) {
    const origIndex = i / ratio;
    const indexFloor = Math.floor(origIndex);
    const indexCeil = Math.min(audioBuffer.length - 1, indexFloor + 1);
    const weight = origIndex - indexFloor;
    result[i] = (1 - weight) * audioBuffer[indexFloor] + weight * audioBuffer[indexCeil];
  }
  return result;
}

// Helper to encode Float32Array into clean 16kHz Mono 16-bit PCM WAV
function encodeWAV(samples: Float32Array, inputSampleRate: number): Blob {
  const targetSamples = resampleTo16k(samples, inputSampleRate);
  const targetRate = 16000;
  const buffer = new ArrayBuffer(44 + targetSamples.length * 2);
  const view = new DataView(buffer);

  /* RIFF identifier */
  writeString(view, 0, 'RIFF');
  /* file length */
  view.setUint32(4, 36 + targetSamples.length * 2, true);
  /* RIFF type */
  writeString(view, 8, 'WAVE');
  /* format chunk identifier */
  writeString(view, 12, 'fmt ');
  /* format chunk length */
  view.setUint32(16, 16, true);
  /* sample format (raw) */
  view.setUint16(20, 1, true);
  /* channel count (mono) */
  view.setUint16(22, 1, true);
  /* sample rate */
  view.setUint32(24, targetRate, true);
  /* byte rate (sample rate * block align) */
  view.setUint32(28, targetRate * 2, true);
  /* block align (channel count * bytes per sample) */
  view.setUint16(32, 2, true);
  /* bits per sample */
  view.setUint16(34, 16, true);
  /* data chunk identifier */
  writeString(view, 36, 'data');
  /* data chunk length */
  view.setUint32(40, targetSamples.length * 2, true);

  // float to 16-bit PCM
  let offset = 44;
  for (let i = 0; i < targetSamples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, targetSamples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
  }

  return new Blob([view], { type: 'audio/wav' });
}

function writeString(view: DataView, offset: number, string: string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

/* -------------------------------------------------------------------------
   Live microphone spectrum (Web Audio AnalyserNode -> canvas)
   ------------------------------------------------------------------------- */

const AudioSpectrum: React.FC<{
  analyserRef: { current: AnalyserNode | null };
  active: boolean;
  processing: boolean;
}> = ({ analyserRef, active, processing }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    let raf = 0;
    const binCount = 56;
    const data = new Uint8Array(128);

    let cachedW = canvas.clientWidth || 300;
    let cachedH = canvas.clientHeight || 50;
    let cachedDpr = Math.min(window.devicePixelRatio || 1, 2);

    const updateSize = () => {
      cachedW = canvas.clientWidth || 300;
      cachedH = canvas.clientHeight || 50;
      cachedDpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(cachedW * cachedDpr);
      canvas.height = Math.floor(cachedH * cachedDpr);
    };
    updateSize();

    const resizeObserver = new ResizeObserver(() => {
      updateSize();
    });
    resizeObserver.observe(canvas);

    const draw = () => {
      raf = requestAnimationFrame(draw);
      const w = cachedW;
      const h = cachedH;
      const dpr = cachedDpr;
      if (!w || !h) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);

      // center reference line
      ctx.strokeStyle = 'rgba(148, 163, 184, 0.13)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, h / 2);
      ctx.lineTo(w, h / 2);
      ctx.stroke();

      const analyser = analyserRef.current;

      if (analyser && active) {
        analyser.getByteFrequencyData(data);
        const barW = w / binCount;
        for (let i = 0; i < binCount; i++) {
          const v = data[Math.floor((i * data.length) / binCount)] / 255;
          const bh = Math.max(2, v * h * 0.86);
          const x = i * barW + barW * 0.18;
          const y = h / 2 - bh / 2;
          const grad = ctx.createLinearGradient(0, y, 0, y + bh);
          grad.addColorStop(0, '#22D3EE');
          grad.addColorStop(1, '#3B82F6');
          ctx.fillStyle = grad;
          ctx.globalAlpha = 0.5 + v * 0.5;
          ctx.fillRect(x, y, barW * 0.64, bh);
        }
        ctx.globalAlpha = 1;
      } else {
        const t = performance.now() / 1000;
        ctx.beginPath();
        for (let x = 0; x <= w; x += 3) {
          const y =
            h / 2 +
            Math.sin(x * 0.024 + t * 1.3) * 2.6 * Math.sin(x * 0.007 + t * 0.7);
          if (x === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.strokeStyle = processing ? '#FBBF24' : '#3B82F6';
        ctx.globalAlpha = processing ? 0.8 : 0.4;
        ctx.lineWidth = 1.4;
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    };

    draw();
    return () => {
      cancelAnimationFrame(raf);
      resizeObserver.disconnect();
    };
  }, [active, processing, analyserRef]);

  return <canvas ref={canvasRef} className="vis-canvas" />;
};

/* -------------------------------------------------------------------------
   App
   ------------------------------------------------------------------------- */

export default function App() {
  const [activeTab, setActiveTab] = useState<'home' | 'lab' | 'about'>('home');
  // Deep-linkable view mode via ?view=2d|3d|fpp (useful for demo videos and QA)
  const [viewMode, setViewMode] = useState<'fpp' | '3d' | '2d'>(() => {
    const param = new URLSearchParams(window.location.search).get('view');
    return param === '2d' || param === 'fpp' ? param : '3d';
  });
  const [devices, setDevices] = useState<DeviceState>(INITIAL_DEVICES);
  const [activeLocation, setActiveLocation] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastResult, setLastResult] = useState<PredictionResult | null>(null);
  const [events, setEvents] = useState<EventLogItem[]>([]);
  const [serverOnline, setServerOnline] = useState(true);
  const [copied, setCopied] = useState<string | null>(null);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Float32Array[]>([]);
  const processorRef = useRef<ScriptProcessorNode | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const isRecordingRef = useRef(false);
  const shouldStopRef = useRef(false);
  const activeLocationTimerRef = useRef<number | null>(null);

  // Check health on mount
  useEffect(() => {
    fetch('/api/health')
      .then((res) => res.json())
      .then((data) => setServerOnline(data.status === 'healthy' || data.status === 'ok'))
      .catch(() => setServerOnline(false));
  }, []);

  useEffect(() => {
    return () => {
      if (activeLocationTimerRef.current) window.clearTimeout(activeLocationTimerRef.current);
    };
  }, []);

  const handleToggleDevice = (key: keyof DeviceState) => {
    setDevices((prev) => {
      if (typeof prev[key] === 'boolean') {
        return { ...prev, [key]: !prev[key] };
      } else if (key.includes('heat')) {
        return { ...prev, [key]: prev[key] >= 24 ? 18 : (prev[key] as number) + 2 };
      } else if (key.includes('volume')) {
        return { ...prev, [key]: prev[key] >= 10 ? 1 : (prev[key] as number) + 2 };
      }
      return prev;
    });
  };

  const adjustDevice = (key: keyof DeviceState, delta: number) => {
    setDevices((prev) => {
      const current = prev[key];
      if (typeof current !== 'number') return prev;
      const min = key.includes('heat') ? 16 : 0;
      const max = key.includes('heat') ? 30 : 10;
      const next = Math.max(min, Math.min(max, current + delta));
      return { ...prev, [key]: next };
    });
  };

  const applyIntentToDevices = (intent: { action: string; object: string; location: string }) => {
    setActiveLocation(intent.location);
    if (activeLocationTimerRef.current) window.clearTimeout(activeLocationTimerRef.current);
    activeLocationTimerRef.current = window.setTimeout(() => setActiveLocation(null), 1800);

    setDevices((prev) => {
      const next = { ...prev };
      const { action, object, location } = intent;

      // Lights / lamp
      if (object === 'lights') {
        const val = action === 'activate';
        if (location === 'kitchen') next.kitchen_lights = val;
        else if (location === 'bedroom') next.bedroom_lights = val;
        else if (location === 'washroom') next.washroom_lights = val;
        else if (location === 'none') {
          next.kitchen_lights = val;
          next.bedroom_lights = val;
          next.washroom_lights = val;
        }
      } else if (object === 'lamp') {
        next.bedroom_lamp = action === 'activate';
      }

      // Heat
      if (object === 'heat') {
        const delta = action === 'increase' ? 2 : action === 'decrease' ? -2 : 0;
        if (location === 'kitchen')
          next.kitchen_heat = Math.max(16, Math.min(30, next.kitchen_heat + delta));
        else if (location === 'bedroom')
          next.bedroom_heat = Math.max(16, Math.min(30, next.bedroom_heat + delta));
        else if (location === 'washroom')
          next.washroom_heat = Math.max(16, Math.min(30, next.washroom_heat + delta));
      }

      // Music / volume
      if (object === 'music') {
        next.hall_music = action === 'activate';
      } else if (object === 'volume') {
        const vDelta = action === 'increase' ? 2 : -2;
        next.hall_volume = Math.max(0, Math.min(10, next.hall_volume + vDelta));
      }

      return next;
    });
  };

  const sendAudioBlob = async (blob: Blob) => {
    setIsProcessing(true);
    const formData = new FormData();
    formData.append('audio', blob, 'command.wav');

    try {
      const res = await fetch('/api/predict', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({ detail: 'Format error' }));
        alert(`Gagal: ${errJson.detail || res.statusText}`);
        return;
      }

      const data: PredictionResult = await res.json();
      if (!data || !data.intent) return;

      setLastResult(data);

      if (data.accepted) {
        applyIntentToDevices(data.intent);
      }

      const newLog: EventLogItem = {
        id: Math.random().toString(),
        time: new Date().toLocaleTimeString('id-ID', {
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit',
        }),
        command: `${data.intent.action} ${data.intent.object} ${data.intent.location}`,
        accepted: data.accepted,
        message: data.effect?.message || 'Selesai',
        confidence: data.confidence?.overall || 0,
      };
      setEvents((prev) => [newLog, ...prev.slice(0, 9)]);
    } catch (err: any) {
      console.error('Error during prediction API call:', err);
      alert(`Terjadi kesalahan jaringan atau server: ${err?.message || err}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const startRecording = async () => {
    // Avoid double start
    if (isRecordingRef.current) return;
    shouldStopRef.current = false;
    isRecordingRef.current = true;
    setIsRecording(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: false,
          noiseSuppression: false,
          autoGainControl: true,
        },
      });

      // If user released spacebar while mic permission dialog was resolving
      if (shouldStopRef.current) {
        stream.getTracks().forEach((t) => t.stop());
        isRecordingRef.current = false;
        setIsRecording(false);
        return;
      }

      mediaStreamRef.current = stream;
      audioChunksRef.current = [];

      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      const audioCtx = new AudioCtx();
      audioCtxRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);

      // AnalyserNode drives the live waveform visualiser
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyser.smoothingTimeConstant = 0.75;
      source.connect(analyser);
      analyserRef.current = analyser;

      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = (e) => {
        const inputData = e.inputBuffer.getChannelData(0);
        audioChunksRef.current.push(new Float32Array(inputData));
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);
    } catch (err) {
      isRecordingRef.current = false;
      setIsRecording(false);
      analyserRef.current = null;
      alert('Microphone permission diperlukan atau gunakan tombol Unggah Audio!');
    }
  };

  const stopRecording = () => {
    shouldStopRef.current = true;
    if (!isRecordingRef.current) return;
    isRecordingRef.current = false;
    setIsRecording(false);

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    analyserRef.current = null;
    if (audioCtxRef.current) {
      const sr = audioCtxRef.current.sampleRate;
      try {
        audioCtxRef.current.close();
      } catch {
        // ignore already closed
      }
      audioCtxRef.current = null;

      // Concatenate Float32Arrays
      let totalLength = 0;
      for (const chunk of audioChunksRef.current) totalLength += chunk.length;
      const merged = new Float32Array(totalLength);
      let offset = 0;
      for (const chunk of audioChunksRef.current) {
        merged.set(chunk, offset);
        offset += chunk.length;
      }

      if (merged.length > 3200) {
        // Voice Activity Detection / Silence Trimming:
        // Find first and last sample above threshold to avoid sending dead silence
        let startIdx = 0;
        let endIdx = merged.length - 1;
        const silenceThreshold = 0.015;

        for (let i = 0; i < merged.length; i++) {
          if (Math.abs(merged[i]) > silenceThreshold) {
            startIdx = Math.max(0, i - Math.floor(sr * 0.15)); // 150ms buffer
            break;
          }
        }
        for (let i = merged.length - 1; i >= 0; i--) {
          if (Math.abs(merged[i]) > silenceThreshold) {
            endIdx = Math.min(merged.length - 1, i + Math.floor(sr * 0.2)); // 200ms buffer
            break;
          }
        }

        // Cap maximum duration to 3.5s to guarantee a responsive inference path
        const maxSamples = Math.floor(sr * 3.5);
        if (endIdx - startIdx > maxSamples) {
          endIdx = startIdx + maxSamples;
        }

        const trimmed = endIdx > startIdx + 3200 ? merged.subarray(startIdx, endIdx) : merged;
        const wavBlob = encodeWAV(trimmed, sr);
        sendAudioBlob(wavBlob);
      }
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop());
      mediaStreamRef.current = null;
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      sendAudioBlob(file);
    }
    e.target.value = '';
  };

  // Global Spacebar push-to-talk (all modes except FPP, which handles it in-scene)
  const spaceHandlersRef = useRef({ startRecording, stopRecording });
  spaceHandlersRef.current = { startRecording, stopRecording };

  useEffect(() => {
    if (activeTab !== 'home' || viewMode === 'fpp') return;

    const onKeyDown = (e: KeyboardEvent) => {
      if (e.code !== 'Space' || e.repeat) return;
      const target = e.target as HTMLElement | null;
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) {
        return;
      }
      e.preventDefault();
      spaceHandlersRef.current.startRecording();
    };
    const onKeyUp = (e: KeyboardEvent) => {
      if (e.code !== 'Space') return;
      e.preventDefault();
      spaceHandlersRef.current.stopRecording();
    };

    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, [activeTab, viewMode]);

  const copyPhrase = async (phrase: string) => {
    try {
      await navigator.clipboard.writeText(phrase);
      setCopied(phrase);
      window.setTimeout(() => setCopied(null), 1800);
    } catch {
      // Clipboard API can be unavailable outside a secure context; ignore silently.
    }
  };

  const latencyLabel = lastResult ? `${Math.round(lastResult.timing_ms.total_ms)} ms` : '— ms';

  const viewTitle =
    viewMode === 'fpp'
      ? 'Simulasi POV Orang Pertama'
      : viewMode === '3d'
        ? 'Diorama Rumah Cerdas 3D'
        : 'Denah Arsitektur 2D';

  const viewSub =
    viewMode === 'fpp'
      ? 'Walkthrough first-person • WASD untuk berjalan, SPASI untuk bicara'
      : viewMode === '3d'
        ? 'Isometrik interaktif • drag untuk orbit, scroll untuk zoom'
        : 'Blueprint CAD interaktif • klik kontrol perangkat di setiap ruangan';

  return (
    <div className="app-container">
      {/* ============================== HEADER ============================== */}
      <header className="navbar">
        <div className="brand">
          <div className="brand-mark">S</div>
          <div className="brand-text">
            <span className="brand-name">SVARA</span>
            <span className="brand-badge">
              <span className="dot" />
              E2E Transformer SLU • Wav2Vec 2.0 INT8
            </span>
          </div>
        </div>

        <nav className="nav-tabs" aria-label="Navigasi utama">
          <button
            className={`nav-btn ${activeTab === 'home' ? 'active' : ''}`}
            onClick={() => setActiveTab('home')}
          >
            <Home size={14} />
            Live Demo
          </button>
          <button
            className={`nav-btn ${activeTab === 'lab' ? 'active' : ''}`}
            onClick={() => setActiveTab('lab')}
          >
            <FlaskConical size={14} />
            Lab Model
          </button>
          <button
            className={`nav-btn ${activeTab === 'about' ? 'active' : ''}`}
            onClick={() => setActiveTab('about')}
          >
            <Info size={14} />
            Tentang
          </button>
        </nav>

        <div className="top-actions">
          <div className="latency-pill" title="Total latency inferensi terakhir">
            <Zap size={12} />
            {latencyLabel}
          </div>
          <div className="status-chip">
            <span className={`status-dot ${serverOnline ? 'ready' : 'error'}`} />
            {serverOnline ? 'ONNX Ready' : 'Server Offline'}
          </div>
        </div>
      </header>

      {/* ============================ HOME VIEW ============================ */}
      {activeTab === 'home' && (
        <main className="main-content">
          <section className="stage-grid">
            {/* -------------------------- VIEWPORT ------------------------- */}
            <div className="panel viewer-panel">
              <div className="viewer-toolbar">
                <div>
                  <div className="panel-kicker">Live Digital Twin</div>
                  <h2 className="panel-title">{viewTitle}</h2>
                  <p className="panel-sub">{viewSub}</p>
                </div>

                <div className="viewer-actions">
                  <div className="mode-tabs" role="tablist" aria-label="Mode visualisasi">
                    <button
                      role="tab"
                      aria-selected={viewMode === '2d'}
                      data-mode="2d"
                      className={`mode-tab ${viewMode === '2d' ? 'active' : ''}`}
                      onClick={() => setViewMode('2d')}
                    >
                      <Map size={13} />
                      <span>2D Denah CAD</span>
                    </button>
                    <button
                      role="tab"
                      aria-selected={viewMode === '3d'}
                      data-mode="3d"
                      className={`mode-tab ${viewMode === '3d' ? 'active' : ''}`}
                      onClick={() => setViewMode('3d')}
                    >
                      <Box size={13} />
                      <span>3D Isometrik</span>
                    </button>
                    <button
                      role="tab"
                      aria-selected={viewMode === 'fpp'}
                      data-mode="fpp"
                      className={`mode-tab ${viewMode === 'fpp' ? 'active' : ''}`}
                      onClick={() => setViewMode('fpp')}
                    >
                      <Gamepad2 size={13} />
                      <span>3D First-Person</span>
                    </button>
                  </div>

                  <button
                    className="ghost-btn"
                    onClick={() => setDevices(INITIAL_DEVICES)}
                    title="Kembalikan seluruh perangkat ke kondisi awal"
                  >
                    <RotateCcw size={13} />
                    Reset
                  </button>
                </div>
              </div>

              <div className="view-stage">
                {viewMode === 'fpp' ? (
                  <FPPWalkthrough
                    devices={devices}
                    onVoiceTriggerStart={startRecording}
                    onVoiceTriggerEnd={stopRecording}
                    isRecording={isRecording}
                    isProcessing={isProcessing}
                    lastCommandMessage={lastResult?.effect.message}
                  />
                ) : viewMode === '3d' ? (
                  <Home3DView
                    devices={devices}
                    activeLocation={activeLocation}
                    onToggleDevice={handleToggleDevice}
                  />
                ) : (
                  <FloorPlan
                    devices={devices}
                    activeLocation={activeLocation}
                    onToggleDevice={handleToggleDevice}
                    onAdjustDevice={adjustDevice}
                  />
                )}
              </div>
            </div>

            {/* -------------------------- INSPECTOR ------------------------ */}
            <aside className="inspector">
              {/* SLU result */}
              <div className="panel result-panel">
                <div className="panel-header">
                  <div>
                    <div className="panel-kicker">SLU Inspector</div>
                    <h3 className="panel-title">Hasil Prediksi Suara</h3>
                  </div>
                  {lastResult && (
                    <span className={`status-pill ${lastResult.accepted ? 'ok' : 'warn'}`}>
                      {lastResult.accepted ? <Check size={11} /> : <Activity size={11} />}
                      {lastResult.accepted ? 'ACCEPTED' : 'LOW CONF'}
                    </span>
                  )}
                </div>

                {lastResult ? (
                  <div>
                    <div className={`status-strip ${lastResult.accepted ? 'ok' : 'warn'}`}>
                      <span className="status-label">
                        <span className="status-glow" />
                        {lastResult.accepted ? 'DITERIMA' : 'DITOLAK (RAGU)'}
                      </span>
                      <span className="threshold-chip">Threshold {lastResult.threshold.toFixed(2)}</span>
                    </div>

                    <p className="effect-quote">“{lastResult.effect.message}”</p>

                    <div className="slot-grid">
                      {SLOT_LIST.map((slot) => {
                        const Icon = slot.Icon;
                        const value = lastResult.intent[slot.key];
                        const pct = Math.round(lastResult.confidence[slot.key] * 100);
                        return (
                          <div
                            key={slot.key}
                            className="slot-card"
                            style={
                              {
                                '--slot-color': slot.color,
                                borderColor: `${slot.color}2E`,
                              } as React.CSSProperties
                            }
                          >
                            <div className="slot-card-head">
                              <span className="slot-label">
                                <Icon size={12} color={slot.color} />
                                {slot.label}
                              </span>
                              <span className="slot-pct">{pct}%</span>
                            </div>
                            <div className="slot-value">{value}</div>
                            <div className="slot-meter">
                              <div className="slot-meter-fill" style={{ width: `${pct}%` }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>

                    <div className="timing-row">
                      <span className="timing-pill">
                        Decode <strong>{Math.round(lastResult.timing_ms.decode_ms)}ms</strong>
                      </span>
                      <span className="timing-pill">
                        Preprocess <strong>{Math.round(lastResult.timing_ms.preprocess_ms)}ms</strong>
                      </span>
                      <span className="timing-pill">
                        ONNX INT8 <strong>{Math.round(lastResult.timing_ms.inference_ms)}ms</strong>
                      </span>
                      <span className="timing-pill">
                        Total <strong>{Math.round(lastResult.timing_ms.total_ms)}ms</strong>
                      </span>
                    </div>
                  </div>
                ) : (
                  <div className="empty-state">
                    <Sparkles size={20} />
                    <span>
                      Tahan tombol mikrofon atau tekan <strong>Spacebar</strong>, lalu ucapkan perintah
                      dalam Bahasa Indonesia atau Inggris.
                    </span>
                    <span>Slot action / object / location dan skor keyakinan akan tampil di sini.</span>
                  </div>
                )}
              </div>

              {/* Quick device controls */}
              <div className="panel">
                <div className="panel-header tight">
                  <div>
                    <div className="panel-kicker">Quick Controls</div>
                    <h3 className="panel-title">Kontrol Perangkat</h3>
                  </div>
                  <span className="threshold-chip">MANUAL</span>
                </div>

                <div className="quick-grid">
                  {TOGGLE_TILES.map((tile) => {
                    const Icon = tile.Icon;
                    const on = devices[tile.key];
                    return (
                      <button
                        key={tile.key}
                        className={`device-tile ${on ? 'on' : ''} ${tile.audio ? 'audio' : ''}`}
                        onClick={() => handleToggleDevice(tile.key)}
                        aria-pressed={on}
                      >
                        <span className="tile-icon">
                          <Icon size={14} />
                        </span>
                        <span className="tile-copy">
                          <span className="tile-name">{tile.label}</span>
                          <span className="tile-state">{on ? 'ON' : 'OFF'}</span>
                        </span>
                      </button>
                    );
                  })}
                </div>

                <div className="numeric-list">
                  {CLIMATE_ROWS.map((row) => (
                    <div className="numeric-row" key={row.key}>
                      <span className="numeric-label">
                        <Thermometer size={12} />
                        {row.label}
                      </span>
                      <div className="stepper">
                        <button
                          onClick={() => adjustDevice(row.key, -2)}
                          aria-label={`${row.label} turunkan`}
                        >
                          <Minus size={10} />
                        </button>
                        <span className={`step-value ${devices[row.key] > 20 ? 'hot' : ''}`}>
                          {devices[row.key]}°C
                        </span>
                        <button
                          onClick={() => adjustDevice(row.key, 2)}
                          aria-label={`${row.label} naikkan`}
                        >
                          <Plus size={10} />
                        </button>
                      </div>
                    </div>
                  ))}

                  <div className="numeric-row">
                    <span className="numeric-label">
                      <Volume2 size={12} />
                      Volume Lorong
                    </span>
                    <div className="stepper">
                      <button onClick={() => adjustDevice('hall_volume', -1)} aria-label="Volume turunkan">
                        <Minus size={10} />
                      </button>
                      <span className="step-value">{devices.hall_volume * 10}%</span>
                      <button onClick={() => adjustDevice('hall_volume', 1)} aria-label="Volume naikkan">
                        <Plus size={10} />
                      </button>
                    </div>
                  </div>
                </div>
              </div>

              {/* Event timeline */}
              <div className="panel">
                <div className="panel-header tight">
                  <div>
                    <div className="panel-kicker">Event Timeline</div>
                    <h3 className="panel-title">Riwayat Perintah</h3>
                  </div>
                  <span className="threshold-chip">{events.length} / 10</span>
                </div>

                {events.length > 0 ? (
                  <div className="timeline">
                    {events.map((ev) => (
                      <div key={ev.id} className={`timeline-item ${ev.accepted ? 'ok' : 'reject'}`}>
                        <span className="timeline-dot" />
                        <div className="timeline-main">
                          <div className="timeline-command">{ev.command}</div>
                          <div className="timeline-message">{ev.message}</div>
                        </div>
                        <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 1 }}>
                          <span className={ev.accepted ? 'badge-accepted' : 'badge-rejected'}>
                            {Math.round(ev.confidence * 100)}%
                          </span>
                          <span className="timeline-time">{ev.time}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="empty-state">
                    <Clock size={18} />
                    <span>Belum ada riwayat aktivitas.</span>
                  </div>
                )}
              </div>
            </aside>
          </section>

          {/* ============================ HERO PTT ============================ */}
          <section className="panel hero-ptt">
            <div className="mic-cluster">
              <div className="mic-orb-wrap">
                {isRecording && (
                  <>
                    <span className="mic-ring" />
                    <span className="mic-ring d1" />
                    <span className="mic-ring d2" />
                  </>
                )}
                <button
                  className={`mic-orb ${isRecording ? 'recording' : ''}`}
                  onMouseDown={startRecording}
                  onMouseUp={stopRecording}
                  onTouchStart={startRecording}
                  onTouchEnd={stopRecording}
                  onContextMenu={(e) => e.preventDefault()}
                  disabled={isProcessing}
                  aria-label="Tahan untuk merekam perintah suara"
                >
                  {isProcessing ? <Loader2 size={26} className="spin" /> : <Mic size={26} />}
                </button>
              </div>

              <div className="mic-meta">
                <span
                  className={`mic-state ${isRecording ? 'live' : isProcessing ? 'busy' : ''}`}
                >
                  {isProcessing
                    ? 'Memproses model…'
                    : isRecording
                      ? 'Merekam — lepas untuk kirim'
                      : 'Tahan untuk bicara'}
                </span>
                <span className="mic-hint">SPACEBAR ATAU TAHAN · 16 kHz MONO WAV</span>
              </div>
            </div>

            <div className="wavezone">
              <AudioSpectrum analyserRef={analyserRef} active={isRecording} processing={isProcessing} />
              <div className="chips">
                <span className="chips-label">Coba ucapkan</span>
                {SAMPLE_COMMANDS.map((phrase) => (
                  <button
                    key={phrase}
                    className={`phrase-chip ${copied === phrase ? 'copied' : ''}`}
                    onClick={() => copyPhrase(phrase)}
                    title="Klik untuk menyalin perintah"
                  >
                    {copied === phrase && <Check size={10} style={{ marginRight: 4 }} />}“{phrase}”
                  </button>
                ))}
              </div>
            </div>

            <label className="upload-btn">
              <Upload size={16} />
              <span>Unggah Audio</span>
              <small>.WAV · 16 kHz</small>
              <input type="file" accept="audio/*" onChange={handleFileUpload} style={{ display: 'none' }} />
            </label>
          </section>
        </main>
      )}

      {/* ============================= LAB VIEW ============================= */}
      {activeTab === 'lab' && (
        <main className="main-content">
          <div className="panel">
            <div className="panel-header">
              <div>
                <div className="panel-kicker">Evaluation Lab</div>
                <h2 className="panel-title" style={{ fontSize: '1.15rem' }}>
                  Laboratorium Evaluasi Model SVARA
                </h2>
              </div>
              <span className="status-pill ok">
                <Check size={11} />
                VERIFIED RUNS
              </span>
            </div>

            <p className="panel-sub" style={{ marginBottom: '1.1rem' }}>
              Data terverifikasi dari hasil eksperimen GPU NVIDIA RTX 4060 dan tersimpan di{' '}
              <code>reports/tables/table_ablation.csv</code>.
            </p>

            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Run ID &amp; Model</th>
                    <th>Split</th>
                    <th>Exact-Match</th>
                    <th>95% CI</th>
                    <th>Action</th>
                    <th>Object</th>
                    <th>Location</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td className="strong">M0 (wav2vec2-base)</td>
                    <td>Split A (Standard)</td>
                    <td className="metric">99.60%</td>
                    <td>[99.39%, 99.79%]</td>
                    <td>99.76%</td>
                    <td>99.89%</td>
                    <td>99.89%</td>
                  </tr>
                  <tr>
                    <td className="strong">M0 (wav2vec2-base)</td>
                    <td>Split B1 (Unseen Speaker)</td>
                    <td className="metric">95.87%</td>
                    <td>[95.33%, 96.38%]</td>
                    <td>98.84%</td>
                    <td>96.84%</td>
                    <td>99.61%</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        </main>
      )}

      {/* ============================ ABOUT VIEW ============================ */}
      {activeTab === 'about' && (
        <main className="main-content">
          <div className="panel" style={{ maxWidth: 860, margin: '0 auto' }}>
            <div className="panel-header">
              <div>
                <div className="panel-kicker">About</div>
                <h2 className="panel-title" style={{ fontSize: '1.15rem' }}>
                  Tentang Proyek SVARA
                </h2>
              </div>
              <span className="status-pill ok">
                <Zap size={11} />
                E2E SLU
              </span>
            </div>

            <div className="prose">
              <p>
                <strong>SVARA (Smart Voice Assistant for Residential Automation)</strong> adalah proyek tugas
                akhir mata kuliah Deep Learning di Program Studi Teknik Informatika, Universitas Mercu Buana.
              </p>
              <p>
                Model ini mengimplementasikan pendekatan <strong>End-to-End Spoken Language Understanding
                (SLU)</strong> berbasis Transformer (<code>facebook/wav2vec2-base</code>) tanpa perantara
                Automatic Speech Recognition (ASR). Audio gelombang mentah (16 kHz) langsung dipetakan menjadi
                kombinasi tiga slot semantik: <code>action</code>, <code>object</code>, dan{' '}
                <code>location</code>.
              </p>
              <p>
                <strong>Tim Pengembang:</strong>
              </p>
              <ul className="team-list">
                <li>• Yoga — ML Pipeline, Training, Model Export, Backend &amp; Web</li>
                <li>• Haikal — Dataset Validation, QA, Dokumentasi &amp; Demo Video</li>
              </ul>
            </div>
          </div>
        </main>
      )}
    </div>
  );
}
