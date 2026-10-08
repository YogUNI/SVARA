"""Pydantic data schemas for SVARA FastAPI backend.

Task: P8-01
Reference: docs/05 §2
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class IntentSlots(BaseModel):
    action: str
    object: str
    location: str


class ConfidenceScores(BaseModel):
    action: float
    object: float
    location: float
    overall: float


class TopKIntent(BaseModel):
    intent_id: int
    slots: IntentSlots
    score: float


class DeviceChange(BaseModel):
    device: str
    room: str
    state: Any


class ActionEffect(BaseModel):
    type: str  # 'device', 'assistant_task', 'ui_language', or 'none'
    changes: List[DeviceChange] = Field(default_factory=list)
    message: str


class TimingBreakdown(BaseModel):
    decode_ms: float
    preprocess_ms: float
    inference_ms: float
    total_ms: float


class AudioMetadata(BaseModel):
    duration_s: float
    sample_rate: int


class PredictionResponse(BaseModel):
    accepted: bool
    intent: Optional[IntentSlots] = None
    confidence: Optional[ConfidenceScores] = None
    threshold: float
    top_k: List[TopKIntent] = Field(default_factory=list)
    valid_intent: bool
    effect: ActionEffect
    timing_ms: TimingBreakdown
    audio: AudioMetadata


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str = "1.0.0"


class ModelInfoResponse(BaseModel):
    model_name: str
    precision: str
    sample_rate: int
    max_audio_seconds: float
    slots: Dict[str, List[str]]
    threshold: float
    license_note: str
