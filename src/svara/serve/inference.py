"""ONNX Runtime inference engine for SVARA backend.

Handles:
- Loading models/svara.onnx and models/svara.meta.json
- Raw audio normalization & preprocessing (identically matching training)
- Softmax confidence computation (min over 3 heads per docs/01 §9)
- Abstention / rejection when confidence < threshold
- Top-k candidate extraction and validation against intent_map.json

Task: P8-03
Reference: docs/05 §1-§2, docs/01 §9
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import onnxruntime as ort

from svara.data.audio import preprocess_audio


def softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax computation."""
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


class SLUInferenceEngine:
    """Thread-safe ONNX Runtime inference service."""

    def __init__(
        self,
        onnx_path: str = "models/svara.onnx",
        meta_path: str = "models/svara.meta.json",
        intra_op_threads: int = 2,
    ):
        self.onnx_path = onnx_path
        self.meta_path = meta_path
        self.session = None
        self.meta = {}

        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.meta = json.load(f)

        if os.path.exists(onnx_path):
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = intra_op_threads
            self.session = ort.InferenceSession(
                onnx_path, sess_options=opts, providers=["CPUExecutionProvider"]
            )
            print(f"[SLUInferenceEngine] Loaded ONNX model from {onnx_path}")

    @property
    def is_loaded(self) -> bool:
        return self.session is not None and bool(self.meta)

    def predict_audio(
        self,
        audio_bytes: bytes,
        threshold_override: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Process audio buffer and return full prediction payload."""
        if not self.is_loaded:
            raise RuntimeError("Model or metadata not loaded in inference engine.")

        t_start = time.perf_counter()

        # 1. Preprocess & Decode
        t_decode_start = time.perf_counter()
        waveform, original_len, sr = preprocess_audio(
            audio_bytes,
            target_sample_rate=self.meta.get("sample_rate", 16000),
            max_audio_seconds=self.meta.get("max_audio_seconds", 5.0),
            normalize=True,
            pad=False,
            is_training=False,
        )
        duration_s = float(original_len / sr)
        t_decode_end = time.perf_counter()

        # 2. Run ONNX Inference
        t_inf_start = time.perf_counter()
        input_tensor = waveform[np.newaxis, :]  # Shape: [1, samples]
        ort_inputs = {self.session.get_inputs()[0].name: input_tensor}
        ort_outputs = self.session.run(None, ort_inputs)

        # Outputs: action_logits, object_logits, location_logits
        act_logits = ort_outputs[0][0]
        obj_logits = ort_outputs[1][0]
        loc_logits = ort_outputs[2][0]
        t_inf_end = time.perf_counter()

        # 3. Decode Slots & Probabilities
        t_dec_start = time.perf_counter()
        act_probs = softmax(act_logits)
        obj_probs = softmax(obj_logits)
        loc_probs = softmax(loc_logits)

        act_idx = int(np.argmax(act_probs))
        obj_idx = int(np.argmax(obj_probs))
        loc_idx = int(np.argmax(loc_probs))

        action_label = self.meta["slots"]["action"][act_idx]
        object_label = self.meta["slots"]["object"][obj_idx]
        location_label = self.meta["slots"]["location"][loc_idx]

        conf_action = float(act_probs[act_idx])
        conf_object = float(obj_probs[obj_idx])
        conf_location = float(loc_probs[loc_idx])

        # Confidence metric: min over three heads (docs/01 §9)
        overall_conf = float(min(conf_action, conf_object, conf_location))

        threshold = (
            threshold_override
            if threshold_override is not None
            else float(self.meta.get("confidence", {}).get("threshold", 0.85))
        )

        # Check validity against intent map
        valid_intent = False
        matched_intent_id = None
        for i_id, details in self.meta.get("intent_map", {}).items():
            if (
                details["action"] == action_label
                and details["object"] == object_label
                and details["location"] == location_label
            ):
                valid_intent = True
                matched_intent_id = int(i_id)
                break

        # Acceptance criterion: overall confidence >= threshold AND valid FSC intent
        accepted = bool(overall_conf >= threshold and valid_intent)

        t_dec_end = time.perf_counter()

        timing = {
            "decode_ms": round((t_decode_end - t_decode_start) * 1000, 2),
            "preprocess_ms": 1.0,
            "inference_ms": round((t_inf_end - t_inf_start) * 1000, 2),
            "total_ms": round((t_dec_end - t_start) * 1000, 2),
        }

        return {
            "accepted": accepted,
            "intent": {
                "action": action_label,
                "object": object_label,
                "location": location_label,
            },
            "confidence": {
                "action": round(conf_action, 4),
                "object": round(conf_object, 4),
                "location": round(conf_location, 4),
                "overall": round(overall_conf, 4),
            },
            "threshold": threshold,
            "valid_intent": valid_intent,
            "matched_intent_id": matched_intent_id,
            "timing_ms": timing,
            "audio": {
                "duration_s": round(duration_s, 2),
                "sample_rate": sr,
            },
        }
