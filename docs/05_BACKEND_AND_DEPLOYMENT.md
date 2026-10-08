# 05 — Backend, Export and Deployment

## 1. Export pipeline
1. Choose the deployment run (best trade-off from docs/02 §5–6, justified by measured numbers).
2. `scripts/export_onnx.py --run <run_dir> [--quantize int8]`
   - Wrapper module: input `waveform: float32 [batch, samples]` (already normalized), output three logit tensors
     (plus joint logits if the joint head is used). Dynamic axes for batch and samples.
   - Export with a recent opset supported by onnxruntime; test with different lengths (1 s, 2 s, 4 s).
   - Avoid attention-mask inputs (see docs/02 gotchas). Pooling must be inside the graph or replicated exactly in numpy.
3. Parity test: PyTorch vs ONNX logits on 20+ clips (max abs diff within tolerance, same argmax).
4. Quantization: `onnxruntime.quantization.quantize_dynamic` (QInt8 weights). Measure accuracy delta on the test
   split and the own set; accept only if the drop is small and documented. Keep fp32 as fallback.
5. Write `models/svara.onnx` and `models/svara.meta.json`:
```json
{
  "schema": 1, "sample_rate": 16000, "max_audio_seconds": "<from base.yaml>",
  "normalize": "zero_mean_unit_var",
  "slots": {"action": ["..."], "object": ["..."], "location": ["..."]},
  "intent_map": "<embedded copy of configs/intent_map.json>",
  "confidence": {"method": "min_head_softmax", "threshold": "<from validation>"},
  "model": {"run_id": "...", "precision": "fp32|int8", "layers": 12},
  "license_note": "Trained on Fluent Speech Commands; academic use."
}
```
Never hardcode label lists in the server: always load from meta JSON.

## 2. Backend (FastAPI + onnxruntime)
Layout: `src/svara/serve/{app.py, inference.py, device_state.py, schemas.py}`.

### Endpoints
| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | liveness + model loaded flag |
| GET | `/api/model-info` | contents of meta JSON (without secrets), model size, precision |
| POST | `/api/predict` | multipart `audio` (WAV 16 kHz mono preferred) -> prediction JSON |
| GET | `/api/results` | serves `reports/metrics/summary.json` (empty-state JSON if absent) |
| GET | `/api/devices` | current simulated home state (per session) |
| POST | `/api/devices/reset` | reset simulated state |
| GET | `/` and assets | serves built frontend (`web/dist`) |

### `/api/predict` response (schema v1)
```json
{
  "accepted": true,
  "intent": {"action":"activate","object":"lights","location":"kitchen"},
  "confidence": {"action":0.99,"object":0.98,"location":0.97,"overall":0.97},
  "threshold": 0.9,
  "top_k": [{"intent_id": 12, "slots": {"...":"..."}, "score": 0.97}],
  "valid_intent": true,
  "effect": {"type":"device","changes":[{"device":"lights","room":"kitchen","state":"on"}], "message":"Kitchen lights on"},
  "timing_ms": {"decode":4,"preprocess":2,"inference":85,"total":95},
  "audio": {"duration_s": 1.8, "sample_rate": 16000}
}
```
- `accepted=false` when overall confidence < threshold or the slot tuple is not a valid intent: then `effect` is
  `{"type":"none","message":"Not sure what you meant"}` and the state is not changed. Numbers above are illustrative
  shapes only, not real results.
- Validation: reject files > ~1 MB, duration > max + margin, non-audio; return 422 with a clear error code
  (`AUDIO_TOO_LONG`, `AUDIO_UNREADABLE`, `AUDIO_TOO_SHORT`, `SAMPLE_RATE_UNSUPPORTED`).
- Resample on the server if the client sends another rate (soundfile + a simple resampler); the web client already sends 16 kHz.
- Concurrency: one onnxruntime session loaded at startup; set intra-op threads from env; a semaphore limits parallel inference.
- CORS: allow the dev origin only (`localhost:5173`); in production the app is same-origin.
- Privacy: do not store uploaded audio by default. Optional env `SAVE_DEMO_AUDIO=false`.
- Logging: structured log line per request (id, durations, accepted flag), no audio content.

## 3. Simulated home state (`device_state.py`, `configs/device_map.yaml`)
Rooms: `kitchen`, `bedroom`, `washroom`, and `global` (for location `none`). State per session (cookie/session id) in memory.

| Object | State model | Effects |
|---|---|---|
| lights | per room: on/off, brightness 0–100 (step 20) | activate/deactivate; increase/decrease = brightness step |
| lamp | per room: on/off | activate/deactivate |
| heat | per room: on/off, target temp °C (step 1, 16–30) | activate/deactivate; increase/decrease = ±1 °C |
| music | global: playing/stopped | activate/deactivate |
| volume | global: 0–10 | increase/decrease = ±1 step |
| newspaper / juice / socks / shoes with `bring` | no device | `effect.type = "assistant_task"`, message "Fetch request: <object>" |
| `change language` + language object | UI badge only | `effect.type = "ui_language"`; shows chosen language chip; no real translation |

Rules:
- `location = none` means **all rooms** for lights/lamp/heat, and global scope for music/volume. This is a design
  decision; make it a config flag.
- Generate the list of valid (action, object, location) combos from `intent_map.json`; for each combo, look up its
  handler in `device_map.yaml`. A combo without a handler returns `effect.type="none"` with message "Recognized, no simulated effect" (log it, then add a handler).
- State transitions must be pure functions (state, intent) -> (new_state, effect) to be unit-tested.

## 4. Docker
Single multi-stage image:
1. Node stage builds `web/` -> `web/dist`.
2. Python slim stage installs `requirements-serve.txt` (onnxruntime, fastapi, uvicorn, numpy, soundfile, pyyaml), copies
   `src/`, `models/svara.onnx`, `models/svara.meta.json`, `reports/metrics/summary.json`, `web/dist`.
3. `CMD uvicorn svara.serve.app:app --host 0.0.0.0 --port ${PORT:-7860}`.
Image must run without GPU and without internet. Keep image size small; do not include PyTorch in the serve image.
(`soundfile` needs libsndfile; install it in the image.)

## 5. Hosting options (pick one, verify current free-tier limits before committing)
| Option | Pros | Cons / checks |
|---|---|---|
| Hugging Face Spaces (Docker SDK) | free tier, natural fit for HF-based project, public URL | verify CPU/RAM limits and cold start; port conventions |
| Render / Railway / Fly.io | simple Docker deploys | free tiers change; check RAM (ONNX model + runtime) and sleep behavior |
| Run locally on a laptop for the live presentation | zero risk of hosting limits | needs a recorded backup video |
Always prepare a **local fallback** (`docker run -p 7860:7860 ...`) and a recorded demo video for the presentation.
Browser microphone requires HTTPS (or `localhost`): make sure the hosted URL is HTTPS.

## 6. Performance targets (to be measured, then reported; do not assume)
Measure on the target CPU: model size, p50/p95 inference latency for 1–3 s clips, end-to-end latency in the browser
(record stop -> result visible). Include network latency separately from model latency in the report.

## 7. Optional extensions
- Gradio fallback demo (`app_gradio.py`) for quick sharing if the custom web app slips.
- Raspberry Pi or ESP32-relay prototype (stretch): same ONNX model on a Pi, relay toggled by GPIO.
- Streaming/VAD-based endpointing (stretch): current design is push-to-talk with a fixed max length.

## 8. Tests
- `tests/test_device_state.py`: every intent in `intent_map.json` maps to a handler or a documented fallback; transitions pure.
- `tests/test_api.py`: FastAPI TestClient with synthetic WAV (sine/silence) checks schema, errors, reject path.
- `tests/test_inference_parity.py`: server preprocess equals training preprocess (docs/02 §10).
