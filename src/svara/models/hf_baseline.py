"""HuggingFace-native Wav2Vec2ForSequenceClassification cross-check baseline.

Purpose:
- Sanity check our custom multi-head model (P3-01 / M0) against HF's official implementation.
- Joint 31-intent classification directly using HF SequenceClassification head.
- Provides an independent benchmark implementation within the HF ecosystem.

Task: P3-17
Reference: docs/12 §5, docs/10 P3-17
"""

import json
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoConfig, Wav2Vec2Config, Wav2Vec2ForSequenceClassification


class HFWav2Vec2ClassificationBaseline(nn.Module):
    """HF-native SequenceClassification wrapper mapping 31 joint intents to slot outputs."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = "facebook/wav2vec2-base",
        num_labels: int = 31,
        intent_map_path: str = "configs/intent_map.json",
        freeze_feature_encoder: bool = True,
        config_only: bool = False,
    ):
        super().__init__()
        self.num_labels = num_labels

        # Load slot mappings for decoding joint intent to action, object, location
        with open(intent_map_path, "r", encoding="utf-8") as f:
            imap_data = json.load(f)

        self.intent_map = imap_data["intent_map"]
        self.slot_vocab = imap_data["slot_vocab"]

        # Build tensor lookups from joint intent index -> slot indices
        action_lookup = []
        object_lookup = []
        location_lookup = []

        act_to_id = {name: i for i, name in enumerate(self.slot_vocab["action"])}
        obj_to_id = {name: i for i, name in enumerate(self.slot_vocab["object"])}
        loc_to_id = {name: i for i, name in enumerate(self.slot_vocab["location"])}

        for i in range(num_labels):
            item = self.intent_map[str(i)]
            action_lookup.append(act_to_id[item["action"]])
            object_lookup.append(obj_to_id[item["object"]])
            location_lookup.append(loc_to_id[item["location"]])

        self.register_buffer("action_lookup", torch.tensor(action_lookup, dtype=torch.long))
        self.register_buffer("object_lookup", torch.tensor(object_lookup, dtype=torch.long))
        self.register_buffer("location_lookup", torch.tensor(location_lookup, dtype=torch.long))

        if config_only:
            config = Wav2Vec2Config(
                num_labels=num_labels,
                hidden_size=768,
                num_hidden_layers=2,
                num_attention_heads=12,
                intermediate_size=3072,
            )
            self.model = Wav2Vec2ForSequenceClassification(config)
        else:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                num_labels=num_labels,
            )
            self.model = Wav2Vec2ForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path,
                config=config,
            )

        if freeze_feature_encoder:
            self.model.freeze_feature_encoder()

    def forward(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Returns:
            Dictionary with 'joint' logits (shape [B, 31]), as well as mapped
            slot logits for 'action', 'object', 'location' for unified evaluation.
        """
        # HF Wav2Vec2ForSequenceClassification expects input_values: [B, T]
        outputs = self.model(waveform)
        joint_logits = outputs.logits  # [B, 31]

        # Decode joint logits to slot pseudo-logits
        # Log-probabilities over 31 intents
        log_probs = torch.log_softmax(joint_logits, dim=-1)

        b = waveform.size(0)
        n_act = len(self.slot_vocab["action"])
        n_obj = len(self.slot_vocab["object"])
        n_loc = len(self.slot_vocab["location"])

        act_logits = torch.full((b, n_act), -1e4, device=waveform.device)
        obj_logits = torch.full((b, n_obj), -1e4, device=waveform.device)
        loc_logits = torch.full((b, n_loc), -1e4, device=waveform.device)

        # Marginalize/aggregate log-probabilities per slot
        for i in range(self.num_labels):
            a_idx = self.action_lookup[i].item()
            o_idx = self.object_lookup[i].item()
            l_idx = self.location_lookup[i].item()

            p = log_probs[:, i]
            act_logits[:, a_idx] = torch.logaddexp(act_logits[:, a_idx], p)
            obj_logits[:, o_idx] = torch.logaddexp(obj_logits[:, o_idx], p)
            loc_logits[:, l_idx] = torch.logaddexp(loc_logits[:, l_idx], p)

        return {
            "joint": joint_logits,
            "action": act_logits,
            "object": obj_logits,
            "location": loc_logits,
        }
