"""FastAPI backend server for SVARA Smart Voice Assistant.

Endpoints:
- GET  /api/health: liveness + model status
- GET  /api/model-info: model metadata, slots, precision
- POST /api/predict: audio upload -> prediction, confidence, and simulated smart home effects
- GET  /api/results: summary metrics JSON
- GET  /api/devices: simulated home state
- POST /api/devices/reset: reset simulated home state
- GET  /: static frontend hosting (if web/dist exists)

Task: P8-01, P8-04
Reference: docs/05 §2
"""

import json
import os
from typing import Dict, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from svara.serve.device_state import get_initial_home_state, transition_device_state
from svara.serve.inference import SLUInferenceEngine
from svara.serve.schemas import (
    ActionEffect,
    AudioMetadata,
    ConfidenceScores,
    HealthResponse,
    IntentSlots,
    ModelInfoResponse,
    PredictionResponse,
    TimingBreakdown,
)

app = FastAPI(
    title="SVARA API",
    description="Smart Voice Assistant for Residential Automation Backend",
    version="1.0.0",
)

# CORS: Allow local frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:7860"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singletons for inference engine and in-memory simulated home state
engine: Optional[SLUInferenceEngine] = None
simulated_home_state = get_initial_home_state()


@app.on_event("startup")
def startup_event():
    global engine
    onnx_path = os.getenv("SVARA_ONNX_PATH", "models/svara.onnx")
    meta_path = os.getenv("SVARA_META_PATH", "models/svara.meta.json")

    # Automatic fallback to svara_int8.onnx if svara.onnx is not present
    if not os.path.exists(onnx_path) and os.path.exists("models/svara_int8.onnx"):
        onnx_path = "models/svara_int8.onnx"
        if os.path.exists("models/svara_int8.meta.json"):
            meta_path = "models/svara_int8.meta.json"

    engine = SLUInferenceEngine(onnx_path=onnx_path, meta_path=meta_path)


@app.get("/api/health", response_model=HealthResponse)
def get_health():
    """Health and model readiness check."""
    return HealthResponse(
        status="healthy",
        model_loaded=engine.is_loaded if engine else False,
        version="1.0.0",
    )


@app.get("/api/model-info", response_model=ModelInfoResponse)
def get_model_info():
    """Return model metadata, vocabularies, and confidence threshold."""
    if not engine or not engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded or configured.",
        )
    meta = engine.meta
    return ModelInfoResponse(
        model_name=meta.get("model", {}).get("run_id", "svara_w2v2"),
        precision=meta.get("model", {}).get("precision", "fp32"),
        sample_rate=meta.get("sample_rate", 16000),
        max_audio_seconds=meta.get("max_audio_seconds", 5.0),
        slots=meta.get("slots", {}),
        threshold=meta.get("confidence", {}).get("threshold", 0.85),
        license_note=meta.get("license_note", "Fluent Speech Commands"),
    )


@app.post("/api/predict", response_model=PredictionResponse)
async def predict_command(audio: UploadFile = File(...)):
    """Predict slots, check confidence threshold, and update simulated smart home devices."""
    global simulated_home_state

    if not engine or not engine.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine not ready. Model files not found.",
        )

    # 1. Validate file size (< 2 MB)
    audio_bytes = await audio.read()
    if len(audio_bytes) > 2 * 1024 * 1024:
        raise HTTPException(status_code=422, detail="AUDIO_TOO_LONG: File size exceeds 2 MB.")
    if len(audio_bytes) < 100:
        raise HTTPException(status_code=422, detail="AUDIO_TOO_SHORT: Audio data empty or corrupted.")

    try:
        pred_res = engine.predict_audio(audio_bytes)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"AUDIO_UNREADABLE: {str(e)}")

    # 2. Determine state transition effect
    slots = pred_res["intent"]
    if pred_res["accepted"]:
        simulated_home_state, effect_dict = transition_device_state(
            simulated_home_state,
            action=slots["action"],
            obj=slots["object"],
            location=slots["location"],
        )
    else:
        effect_dict = {
            "type": "none",
            "changes": [],
            "message": "Perintah tidak dapat dipastikan (confidence di bawah ambang batas).",
        }

    return PredictionResponse(
        accepted=pred_res["accepted"],
        intent=IntentSlots(**slots),
        confidence=ConfidenceScores(**pred_res["confidence"]),
        threshold=pred_res["threshold"],
        valid_intent=pred_res["valid_intent"],
        effect=ActionEffect(**effect_dict),
        timing_ms=TimingBreakdown(**pred_res["timing_ms"]),
        audio=AudioMetadata(**pred_res["audio"]),
    )


@app.get("/api/devices")
def get_devices():
    """Return current simulated smart home state."""
    return simulated_home_state


@app.post("/api/devices/reset")
def reset_devices():
    """Reset simulated smart home state to initial conditions."""
    global simulated_home_state
    simulated_home_state = get_initial_home_state()
    return {"status": "reset", "state": simulated_home_state}


@app.get("/api/results")
def get_results():
    """Serve summary metrics JSON from reports."""
    summary_path = "reports/metrics/summary.json"
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"status": "empty", "message": "No evaluated run results available yet."}


# Mount built static frontend if present
dist_path = os.path.join("web", "dist")
if os.path.exists(dist_path):
    app.mount("/", StaticFiles(directory=dist_path, html=True), name="static")
