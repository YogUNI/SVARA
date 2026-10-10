import React, { useState, useRef, useEffect } from 'react';
import { Mic, Upload, RotateCcw, Box, Map, Gamepad2 } from 'lucide-react';
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
  "turn on the lights in the kitchen",
  "switch off the lights in the washroom",
  "turn up the heat in the bedroom",
  "resume the music",
  "turn up the volume",
  "turn off the lamp",
];

// Helper to encode AudioBuffer to real 16kHz Mono WAV
function encodeWAV(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  /* RIFF identifier */
  writeString(view, 0, 'RIFF');
  /* file length */
  view.setUint32(4, 36 + samples.length * 2, true);
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
  view.setUint32(24, sampleRate, true);
  /* byte rate (sample rate * block align) */
  view.setUint32(28, sampleRate * 2, true);
  /* block align (channel count * bytes per sample) */
  view.setUint16(32, 2, true);
  /* bits per sample */
  view.setUint16(34, 16, true);
  /* data chunk identifier */
  writeString(view, 36, 'data');
  /* data chunk length */
  view.setUint32(40, samples.length * 2, true);

  // float to 16-bit PCM
  let offset = 44;
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([view], { type: 'audio/wav' });
}

function writeString(view: DataView, offset: number, string: string) {
  for (let i = 0; i < string.length; i++) {
    view.setUint8(offset + i, string.charCodeAt(i));
  }
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'home' | 'lab' | 'about'>('home');
  const [viewMode, setViewMode] = useState<'fpp' | '3d' | '2d'>('fpp');
  const [devices, setDevices] = useState<DeviceState>(INITIAL_DEVICES);
  const [activeLocation, setActiveLocation] = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [lastResult, setLastResult] = useState<PredictionResult | null>(null);
  const [events, setEvents] = useState<EventLogItem[]>([]);
  const [serverOnline, setServerOnline] = useState(true);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const audioChunksRef = useRef<Float32Array[]>([]);
  const processorRef = useRef<ScriptProcessorNode | null>(null);

  // Check health on mount
  useEffect(() => {
    fetch('/api/health')
      .then(res => res.json())
      .then(data => setServerOnline(data.status === 'healthy' || data.status === 'ok'))
      .catch(() => setServerOnline(false));
  }, []);

  const handleToggleDevice = (key: keyof DeviceState) => {
    setDevices(prev => {
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

  const applyIntentToDevices = (intent: { action: string; object: string; location: string }) => {
    setActiveLocation(intent.location);
    setTimeout(() => setActiveLocation(null), 1800);

    setDevices(prev => {
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
        if (location === 'kitchen') next.kitchen_heat = Math.max(16, Math.min(30, next.kitchen_heat + delta));
        else if (location === 'bedroom') next.bedroom_heat = Math.max(16, Math.min(30, next.bedroom_heat + delta));
        else if (location === 'washroom') next.washroom_heat = Math.max(16, Math.min(30, next.washroom_heat + delta));
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
        time: new Date().toLocaleTimeString('id-ID', { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
        command: `${data.intent.action} ${data.intent.object} ${data.intent.location}`,
        accepted: data.accepted,
        message: data.effect?.message || 'Selesai',
        confidence: data.confidence?.overall || 0,
      };
      setEvents(prev => [newLog, ...prev.slice(0, 9)]);
    } catch (err: any) {
      console.error('Error during prediction API call:', err);
      alert(`Terjadi kesalahan jaringan atau server: ${err?.message || err}`);
    } finally {
      setIsProcessing(false);
    }
  };

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      audioChunksRef.current = [];

      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      const audioCtx = new AudioCtx({ sampleRate: 16000 });
      audioCtxRef.current = audioCtx;

      const source = audioCtx.createMediaStreamSource(stream);
      const processor = audioCtx.createScriptProcessor(4096, 1, 1);
      processorRef.current = processor;

      processor.onaudioprocess = e => {
        const inputData = e.inputBuffer.getChannelData(0);
        audioChunksRef.current.push(new Float32Array(inputData));
      };

      source.connect(processor);
      processor.connect(audioCtx.destination);
      setIsRecording(true);
    } catch (err) {
      alert("Microphone permission diperlukan atau gunakan tombol Unggah Audio!");
    }
  };

  const stopRecording = () => {
    if (!isRecording) return;
    setIsRecording(false);

    if (processorRef.current) {
      processorRef.current.disconnect();
      processorRef.current = null;
    }
    if (audioCtxRef.current) {
      const sr = audioCtxRef.current.sampleRate;
      audioCtxRef.current.close();
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
        const wavBlob = encodeWAV(merged, sr);
        sendAudioBlob(wavBlob);
      }
    }
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach(t => t.stop());
      mediaStreamRef.current = null;
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      sendAudioBlob(file);
    }
  };

  return (
    <div className="app-container">
      {/* Top Navbar */}
      <header className="navbar">
        <div className="brand">
          <span className="brand-title">SVARA</span>
          <span className="brand-badge">Transformer SLU (wav2vec2)</span>
        </div>

        <nav className="nav-links">
          <button
            className={`nav-btn ${activeTab === 'home' ? 'active' : ''}`}
            onClick={() => setActiveTab('home')}
          >
            Rumah (Live Demo)
          </button>
          <button
            className={`nav-btn ${activeTab === 'lab' ? 'active' : ''}`}
            onClick={() => setActiveTab('lab')}
          >
            Lab (Evaluasi Model)
          </button>
          <button
            className={`nav-btn ${activeTab === 'about' ? 'active' : ''}`}
            onClick={() => setActiveTab('about')}
          >
            Tentang
          </button>
        </nav>

        <div className="status-chip">
          <span className={`status-dot ${serverOnline ? 'ready' : 'error'}`} />
          <span>{serverOnline ? 'Model ONNX Siap' : 'Server Offline'}</span>
        </div>
      </header>

      {/* Main Content View */}
      {activeTab === 'home' && (
        <main className="main-content">
          <div className="stage-grid">
            {/* Left Column: Interactive Floor Plan / 3D Home View */}
            <div className="panel">
              <div className="panel-header">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                  <h2 className="panel-title">
                    {viewMode === 'fpp'
                      ? '🎮 Simulasi POV Orang Pertama (FPP Game Mode)'
                      : viewMode === '3d'
                      ? 'Simulasi 3D Rumah Cerdas (Isometric)'
                      : 'Denah Rumah 2D (Floor Plan)'}
                  </h2>
                  <div style={{ display: 'flex', background: 'var(--mist-light)', borderRadius: '6px', padding: '2px' }}>
                    <button
                      className="nav-btn"
                      style={{
                        padding: '3px 10px',
                        fontSize: '0.8rem',
                        backgroundColor: viewMode === 'fpp' ? 'var(--brick)' : 'transparent',
                        color: viewMode === 'fpp' ? '#FFF' : 'var(--ink)',
                        fontWeight: viewMode === 'fpp' ? 700 : 500,
                      }}
                      onClick={() => setViewMode('fpp')}
                    >
                      <Gamepad2 size={13} style={{ display: 'inline', marginRight: 4 }} />
                      Mode FPP (Game)
                    </button>
                    <button
                      className="nav-btn"
                      style={{
                        padding: '3px 8px',
                        fontSize: '0.8rem',
                        backgroundColor: viewMode === '3d' ? 'var(--ink)' : 'transparent',
                        color: viewMode === '3d' ? 'var(--paper)' : 'var(--ink)',
                      }}
                      onClick={() => setViewMode('3d')}
                    >
                      <Box size={13} style={{ display: 'inline', marginRight: 4 }} />
                      3D Isometrik
                    </button>
                    <button
                      className="nav-btn"
                      style={{
                        padding: '3px 8px',
                        fontSize: '0.8rem',
                        backgroundColor: viewMode === '2d' ? 'var(--ink)' : 'transparent',
                        color: viewMode === '2d' ? 'var(--paper)' : 'var(--ink)',
                      }}
                      onClick={() => setViewMode('2d')}
                    >
                      <Map size={13} style={{ display: 'inline', marginRight: 4 }} />
                      2D Blueprint
                    </button>
                  </div>
                </div>

                <button
                  className="nav-btn"
                  style={{ fontSize: '0.85rem' }}
                  onClick={() => setDevices(INITIAL_DEVICES)}
                >
                  <RotateCcw size={14} style={{ display: 'inline', marginRight: 4 }} />
                  Reset Perangkat
                </button>
              </div>

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
                />
              )}
            </div>

            {/* Right Column: Prediction & Slot Cards */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
              <div className="panel">
                <div className="panel-header">
                  <h3 className="panel-title">Hasil Prediksi Suara</h3>
                  {lastResult && (
                    <span className={lastResult.accepted ? 'badge-accepted' : 'badge-rejected'}>
                      {lastResult.accepted ? 'Diterima' : 'Ditolak (Ragu)'}
                    </span>
                  )}
                </div>

                {lastResult ? (
                  <div>
                    <div style={{ marginBottom: '1rem', fontStyle: 'italic', color: '#16303A' }}>
                      "{lastResult.effect.message}"
                    </div>

                    <div className="slot-row">
                      <span className="slot-name">Action</span>
                      <span className="slot-val">{lastResult.intent.action}</span>
                      <div className="slot-bar-wrap">
                        <div className="slot-bar-bg">
                          <div
                            className="slot-bar-fill"
                            style={{ width: `${Math.round(lastResult.confidence.action * 100)}%` }}
                          />
                        </div>
                        <span className="slot-pct">
                          {Math.round(lastResult.confidence.action * 100)}%
                        </span>
                      </div>
                    </div>

                    <div className="slot-row">
                      <span className="slot-name">Object</span>
                      <span className="slot-val">{lastResult.intent.object}</span>
                      <div className="slot-bar-wrap">
                        <div className="slot-bar-bg">
                          <div
                            className="slot-bar-fill"
                            style={{ width: `${Math.round(lastResult.confidence.object * 100)}%` }}
                          />
                        </div>
                        <span className="slot-pct">
                          {Math.round(lastResult.confidence.object * 100)}%
                        </span>
                      </div>
                    </div>

                    <div className="slot-row">
                      <span className="slot-name">Location</span>
                      <span className="slot-val">{lastResult.intent.location}</span>
                      <div className="slot-bar-wrap">
                        <div className="slot-bar-bg">
                          <div
                            className="slot-bar-fill"
                            style={{ width: `${Math.round(lastResult.confidence.location * 100)}%` }}
                          />
                        </div>
                        <span className="slot-pct">
                          {Math.round(lastResult.confidence.location * 100)}%
                        </span>
                      </div>
                    </div>

                    <div style={{ marginTop: '1rem', fontSize: '0.85rem', color: '#68827B', display: 'flex', justifyContent: 'space-between' }}>
                      <span>Inference: {Math.round(lastResult.timing_ms.inference_ms)} ms</span>
                      <span>Total Latency: {Math.round(lastResult.timing_ms.total_ms)} ms</span>
                    </div>
                  </div>
                ) : (
                  <div style={{ textAlign: 'center', padding: '2rem 1rem', color: '#A9B8B2' }}>
                    Tahan tombol mikrofon di bawah atau unggah audio untuk melihat reaksi model.
                  </div>
                )}
              </div>

              {/* Event Log */}
              <div className="panel" style={{ flex: 1 }}>
                <div className="panel-header">
                  <h3 className="panel-title">Riwayat Perintah (Log)</h3>
                  <span style={{ fontSize: '0.8rem', color: '#A9B8B2' }}>10 Terakhir</span>
                </div>

                {events.length > 0 ? (
                  events.map(ev => (
                    <div key={ev.id} className="event-item">
                      <div>
                        <span style={{ fontWeight: 600 }}>{ev.command}</span>
                        <div style={{ fontSize: '0.75rem', color: '#68827B' }}>{ev.time}</div>
                      </div>
                      <span className={ev.accepted ? 'badge-accepted' : 'badge-rejected'}>
                        {Math.round(ev.confidence * 100)}%
                      </span>
                    </div>
                  ))
                ) : (
                  <div style={{ color: '#A9B8B2', fontSize: '0.85rem', textAlign: 'center', padding: '1rem' }}>
                    Belum ada riwayat aktivitas.
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Bottom Push-to-Talk & Controls */}
          <div className="control-bar">
            <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
              <button
                className={`mic-btn ${isRecording ? 'recording' : ''}`}
                onMouseDown={startRecording}
                onMouseUp={stopRecording}
                onTouchStart={startRecording}
                onTouchEnd={stopRecording}
                disabled={isProcessing}
              >
                <Mic size={22} />
                <span>
                  {isProcessing
                    ? 'Memproses Model...'
                    : isRecording
                    ? 'Lepas untuk Mengirim...'
                    : 'Tahan untuk Bicara'}
                </span>
              </button>

              <label className="nav-btn" style={{ cursor: 'pointer', padding: '0.9rem 1.25rem', border: '1px solid var(--mist)', borderRadius: '999px', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Upload size={18} />
                <span>Unggah Audio</span>
                <input
                  type="file"
                  accept="audio/*"
                  onChange={handleFileUpload}
                  style={{ display: 'none' }}
                />
              </label>
            </div>

            {/* Suggested Commands Pills */}
            <div className="suggested-chips">
              <span style={{ fontSize: '0.8rem', color: '#A9B8B2', marginRight: '0.5rem', alignSelf: 'center' }}>
                Coba ucapkan:
              </span>
              {SAMPLE_COMMANDS.map((phrase, idx) => (
                <span key={idx} className="phrase-chip">
                  "{phrase}"
                </span>
              ))}
            </div>
          </div>
        </main>
      )}

      {/* Lab View */}
      {activeTab === 'lab' && (
        <main className="main-content">
          <div className="panel">
            <h2 className="panel-title" style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>
              Laboratorium Evaluasi Model SVARA
            </h2>
            <p style={{ marginBottom: '1.5rem', color: '#68827B' }}>
              Data terverifikasi langsung dari hasil eksperimen GPU NVIDIA RTX 4060 dan tersimpan di <code>reports/tables/table_ablation.csv</code>.
            </p>

            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
              <thead>
                <tr style={{ borderBottom: '2px solid var(--mist)', padding: '0.75rem 0' }}>
                  <th style={{ padding: '0.75rem' }}>Run ID & Model</th>
                  <th style={{ padding: '0.75rem' }}>Split</th>
                  <th style={{ padding: '0.75rem' }}>Exact-Match</th>
                  <th style={{ padding: '0.75rem' }}>95% CI</th>
                  <th style={{ padding: '0.75rem' }}>Action</th>
                  <th style={{ padding: '0.75rem' }}>Object</th>
                  <th style={{ padding: '0.75rem' }}>Location</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: '1px solid var(--mist-light)' }}>
                  <td style={{ padding: '0.75rem', fontWeight: 600 }}>M0 (wav2vec2-base)</td>
                  <td style={{ padding: '0.75rem' }}>Split A (Standard)</td>
                  <td style={{ padding: '0.75rem', fontWeight: 700, color: 'var(--pine)' }}>99.60%</td>
                  <td style={{ padding: '0.75rem' }}>[99.39%, 99.79%]</td>
                  <td style={{ padding: '0.75rem' }}>99.76%</td>
                  <td style={{ padding: '0.75rem' }}>99.89%</td>
                  <td style={{ padding: '0.75rem' }}>99.89%</td>
                </tr>
                <tr style={{ borderBottom: '1px solid var(--mist-light)' }}>
                  <td style={{ padding: '0.75rem', fontWeight: 600 }}>M0 (wav2vec2-base)</td>
                  <td style={{ padding: '0.75rem' }}>Split B1 (Unseen Speaker)</td>
                  <td style={{ padding: '0.75rem', fontWeight: 700, color: 'var(--pine)' }}>95.87%</td>
                  <td style={{ padding: '0.75rem' }}>[95.33%, 96.38%]</td>
                  <td style={{ padding: '0.75rem' }}>98.84%</td>
                  <td style={{ padding: '0.75rem' }}>96.84%</td>
                  <td style={{ padding: '0.75rem' }}>99.61%</td>
                </tr>
              </tbody>
            </table>
          </div>
        </main>
      )}

      {/* About View */}
      {activeTab === 'about' && (
        <main className="main-content">
          <div className="panel" style={{ maxWidth: '800px', margin: '0 auto' }}>
            <h2 className="panel-title" style={{ fontSize: '1.5rem', marginBottom: '1rem' }}>
              Tentang Proyek SVARA
            </h2>
            <p style={{ marginBottom: '1rem', lineHeight: '1.7' }}>
              <strong>SVARA (Smart Voice Assistant for Residential Automation)</strong> adalah proyek tugas akhir mata kuliah Deep Learning di Program Studi Teknik Informatika, Universitas Mercu Buana.
            </p>
            <p style={{ marginBottom: '1rem', lineHeight: '1.7' }}>
              Model ini mengimplementasikan pendekatan <strong>End-to-End Spoken Language Understanding (SLU)</strong> berbasis Transformer (<code>facebook/wav2vec2-base</code>) tanpa perantara Automatic Speech Recognition (ASR). Audio gelombang mentah (16 kHz) langsung dipetakan menjadi kombinasi tiga slot semantik: <code>action</code>, <code>object</code>, dan <code>location</code>.
            </p>
            <p style={{ marginBottom: '1rem', lineHeight: '1.7' }}>
              <strong>Tim Pengembang:</strong><br />
              • Yoga (ML Pipeline, Training, Model Export, Backend & Web)<br />
              • Haikal (Dataset Validation, QA, Dokumentasi & Demo Video)
            </p>
          </div>
        </main>
      )}
    </div>
  );
}
