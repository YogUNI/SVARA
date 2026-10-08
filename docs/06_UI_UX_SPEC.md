# 06 — UI / UX Specification

## 1. Product framing
- **Subject:** a voice-controlled home you can talk to, plus a research view that proves the model works.
- **Audience:** lecturer and classmates during a demo (first-time users), and report readers who want evidence.
- **Primary job of the UI:** let a stranger press a button, say one command, and *see the house react* within a
  second or two. Secondary job: let a reader inspect results honestly (the "Lab").
- **Memorable element (spend the boldness here, keep everything else quiet):** a top-down **floor plan** of a small
  home where devices physically respond: lights glow, the heater shows its temperature, music bars pulse.
  Nothing else on the page competes with it.
- Language of UI copy: Indonesian by default, with an EN toggle only if time remains. Keep labels plain and literal.

## 2. Information architecture
| View | Route | Purpose |
|---|---|---|
| Rumah (Live) | `/` | microphone, floor plan, prediction, event log |
| Lab | `/lab` | evidence: protocols, ablation, confusion matrices, noise curve, reject curve, fairness, own set, systems metrics |
| Tentang | `/about` | dataset, method in 5 sentences, limitations, license/consent note, team |
Top bar: product name, three links, a small status chip (`Model siap` / `Memuat model` / `Server tidak terjangkau`).

## 3. Visual system (tokens)
Avoid the common generated-page defaults (cream + terracotta, near-black + acid accent, identical rounded cards,
tracked ALL-CAPS eyebrows, "→" on every link, monospace micro-labels). Use these deliberate choices instead:

| Token | Name | Hex | Use |
|---|---|---|---|
| `--plaster` | Plaster | `#E8ECE6` | page background |
| `--ink` | Ink | `#16303A` | text, walls of the floor plan |
| `--pine` | Pine | `#2F6B5E` | primary actions, selected states |
| `--lamp` | Lamp | `#F2B33D` | **only** for "device is on / active" glow |
| `--brick` | Brick | `#C4492F` | rejected / error / recording |
| `--mist` | Mist | `#A9B8B2` | borders, off-state devices, dividers |
| `--paper` | Paper | `#F6F8F4` | raised surfaces (sparingly) |
Dark theme (optional): swap `--plaster` -> `#0F2128`, `--ink` -> `#E8ECE6`; keep `--lamp` and `--brick`.
Contrast: body text >= 4.5:1; check `--lamp` text on `--plaster` (do not put amber text on light backgrounds).

Typography (Google Fonts, provide system fallbacks):
- UI and headings: **Familjen Grotesk** (weights 400/500/700).
- Long text in Lab/About (captions, interpretations): **Newsreader** (400/500), slightly larger line-height than sans.
- Scale: 14 / 16 / 20 / 28 / 44 px; line length < 72 characters; sentence case everywhere; no all-caps labels.
- Numbers in tables use tabular figures (`font-variant-numeric: tabular-nums`).

Shape and space: 8-px spacing grid. Two radii only: 4 px (inputs, chips) and 16 px (floor plan frame, main panels).
Vary hierarchy by size and spacing rather than by wrapping everything in identical cards. No gradient washes.
Motion: one orchestrated moment only (page load: floor plan walls draw in, ~600 ms). Everything else is
action-driven and short (120–200 ms): a device lighting up, a chip appearing. Respect `prefers-reduced-motion`.

## 4. View: Rumah (Live)
Desktop wireframe (>= 1024 px):
```
+----------------------------------------------------------------------------+
| SVARA            Rumah   Lab   Tentang                  [Model siap]   |
+------------------------------------------+---------------------------------+
|                                          |  Perintah terakhir              |
|   +----------------+----------------+    |  action   [ activate  ] ####    |
|   |  Dapur         |  Kamar tidur   |    |  object   [ lights    ] ####    |
|   |  (lampu, pemanas)| (lampu, lamp,|    |  location [ kitchen   ] ###     |
|   |                |   pemanas)     |    |  Diterima | 95 ms               |
|   +----------------+----------------+    |  "Lampu dapur menyala"          |
|   |  Kamar mandi   |  Lorong        |    +---------------------------------+
|   |  (lampu, pemanas)| musik/volume |    |  Riwayat                        |
|   +----------------+----------------+    |  12:01 activate lights kitchen  |
|        FLOOR PLAN (SVG)                  |  12:00 ditolak (ragu)           |
+------------------------------------------+---------------------------------+
|  [  Tahan untuk bicara  (atau tekan Spasi)  ]   ~~~ waveform ~~~   [Unggah] |
+----------------------------------------------------------------------------+
```
Mobile (< 700 px): floor plan on top, result below, mic button fixed at the bottom (thumb reachable), history in a collapsible sheet.

### Floor plan (SVG, the hero)
- Four zones: Dapur (kitchen), Kamar tidur (bedroom), Kamar mandi (washroom), Lorong (hall, holds music/volume).
  Map zones to `location` values; `none` highlights the whole plan.
- Walls in `--ink`, room floors in `--paper`. Devices:
  - Lights: ceiling circle; on = `--lamp` fill + soft radial glow across the room floor; brightness controls glow opacity.
  - Lamp: floor-lamp icon with a small cone of light when on.
  - Heat: radiator glyph with the target temperature as text (e.g., `22 °C`), tinted `--brick` only while on.
  - Music/volume: speaker in the hall; 10 volume ticks; animated bars only while music is playing.
- When a command is accepted: the affected device animates (≤ 200 ms) and its room gets a brief outline in `--pine`.
- Clicking a device toggles it manually (useful for resetting the demo and for judges to explore).
- Rejected command: no device change; the result panel shows the muted best guess, labelled as uncertain.
- Unsupported-but-recognized (`bring`, `change language`): show an "assistant task" ribbon above the plan
  (e.g., "Permintaan antar: koran") with no device effect; language chip updates for `change language`.

### Microphone control
- Large push-to-talk button: hold to record (mouse, touch) or hold Space; release to send. Max recording length = model max seconds;
  show a thin progress ring that fills toward the limit.
- States and labels: `Tahan untuk bicara` (idle), `Merekam…` (recording, button turns `--brick`), `Memproses…` (spinner, button disabled), `Selesai`.
- Waveform: live `AnalyserNode` bars while recording; after recording, a static mini waveform of what was sent.
- Permission denied: explain how to allow the microphone and offer **file upload** as the fallback (WAV/MP3/WebM accepted client-side, converted to 16 kHz WAV).
- Suggested commands: a quiet row of 5–6 example phrases (text only, no audio files) that rotate through different intents.
  They are suggestions of what to say, not buttons.

### Result panel
- Three slot rows (action, object, location). Each row: value, confidence bar (width only, no gradient), numeric %.
- Status line: `Diterima` (Pine) or `Ditolak: kurang yakin` (Brick) + the threshold used + total latency and model latency.
- One plain sentence describing the effect (from the API `effect.message`).
- "Detail" disclosure: top-k intents, valid-intent flag, audio duration. Collapsed by default.
- No transcript is shown: the model is end-to-end and does not produce text. Say so in the About page, not in the main UI.

### Event log
Latest first, max 50, each line: time, slots, accepted/rejected, latency. "Reset rumah" button (confirm not needed; undo toast).

## 5. View: Lab (evidence)
Purpose: make the evaluation readable in under 5 minutes. All data from `/api/results`. No hardcoded numbers.
Sections (anchored, sticky side index on desktop):
1. **Ringkasan**: what was trained, on which split, the headline exact-match accuracy with CI and the model size/latency chosen. Large type, no decoration.
2. **Split A vs B**: grouped bars for exact-match and slot accuracies; caption explains phrase leakage in 2 sentences, with the measured overlap percentage.
3. **Ablation**: sortable table (variant, split, exact-match mean ± std, params, size, latency) + accuracy-vs-latency scatter with labelled points.
4. **Confusion matrices**: slot tabs (action/object/location/joint); joint 31×31 heatmap with hover tooltip; sequential single-hue colour scale (Pine); toggle row-normalised/raw.
5. **Noise**: line chart accuracy vs SNR per model (clean at the right end).
6. **Reject curve**: risk–coverage chart with the chosen threshold marked; OOD false-accept rate beside it.
7. **Fairness**: horizontal bars per group with CIs; groups with n < 30 hatched and labelled "sampel kecil".
8. **Uji lapangan (own set)**: quiet vs noisy, per device; show n.
9. **Sistem**: size/latency/RTF table for fp32, int8, truncated.
Each section: short human-written caption (Newsreader) stating what to notice, plus a "Sumber" line with the run id(s).
Empty state per section: "Hasil belum tersedia. Jalankan `python scripts/make_report_assets.py`." (plain, no apology).
Charts: Recharts; every chart has a text alternative (table toggle) and colour-blind-safe encodings (not colour alone).

## 6. View: Tentang
Short, plain sections: What this is (3 sentences), How it works (diagram: audio -> wav2vec2 -> three heads -> intent -> home),
Dataset and license notice, Limitations (English-only commands, clean-ish training data, small field test, simulated devices),
Privacy (audio is processed in memory and not stored, if true — keep this sentence accurate to the deployed config), Team.

## 7. Components (suggested)
`TopBar`, `StatusChip`, `FloorPlan` (+ `Room`, `DeviceLights`, `DeviceLamp`, `DeviceHeat`, `DeviceMusic`), `PushToTalk`,
`Waveform`, `ResultPanel`, `SlotRow`, `EventLog`, `LabSection`, `Charts/*`, `EmptyState`, `Toast`.
State: React state + small context for session/home state; server is the source of truth for device state per session.
Data layer: typed API client (`web/src/api.ts`) mirroring the schemas in docs/05.

## 8. Audio capture details (client)
- `getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: false}})`; keep noise suppression off by default
  so the model sees audio similar to training (make this a documented setting; test both).
- Record with MediaRecorder, decode via `AudioContext.decodeAudioData`, resample to 16 kHz mono with `OfflineAudioContext`, encode PCM16 WAV in JS, POST as multipart.
- Trim leading/trailing silence conservatively (energy threshold) and cap to the model max length; never send > limit.
- Handle iOS Safari quirks (user gesture to start AudioContext; MediaRecorder MIME differences).

## 9. Accessibility and quality floor
Keyboard: Space/Enter operate push-to-talk, Tab order follows visual order, visible focus ring (2 px Pine offset).
Screen readers: live region announces results ("Perintah diterima: aktifkan lampu dapur"). Colour is never the only
signal (icons/text for on/off and accepted/rejected). Touch targets ≥ 44 px. Responsive 360–1440 px. Reduced motion respected.
Lighthouse accessibility ≥ 90 as a target, measured and reported.

## 10. Copy rules
Plain verbs, sentence case, no filler. A button names its action ("Reset rumah", "Unggah rekaman"). Errors state what happened and what to do
("Mikrofon diblokir. Izinkan akses mikrofon di pengaturan browser, atau unggah file rekaman.") and never apologise. The same action keeps the same name everywhere.

## 11. Not in v1
Authentication, accounts, wake-word, multi-user rooms, real IoT control, saving user audio, i18n beyond ID/EN.
