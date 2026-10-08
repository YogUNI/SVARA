"""Transformer SLU model based on wav2vec 2.0 (facebook/wav2vec2-base).

Tasks: P3-01, P4-02 (layer truncation), E2/E3 support
Reference: docs/12 §2, §3, §4, docs/02 §1, docs/11 V-11..V-18

Architecture:
- Input: [B, T] 16 kHz raw audio waveform (zero-padded per batch, length-bucketed).
- Feature Encoder: 7-layer temporal CNN (frozen by default, group norm in first layer).
- Transformer Encoder: 12 layers (hidden_size 768), relative positional embeddings.
- Length-aware mean pooling over valid acoustic frames.
- 3 classification heads (action, object, location) + optional joint head (31 intents).
"""

from typing import Dict, Optional, Union

import torch
import torch.nn as nn
from transformers import AutoConfig, Wav2Vec2Config, Wav2Vec2Model


class Wav2VecSLU(nn.Module):
    """wav2vec 2.0 based Spoken Language Understanding model."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = "facebook/wav2vec2-base",
        n_action: int = 6,
        n_object: int = 14,
        n_location: int = 4,
        n_joint: Optional[int] = 31,
        keep_layers: Optional[int] = None,
        freeze_feature_encoder: bool = True,
        freeze_encoder: bool = False,
        dropout: float = 0.1,
        mask_time_prob: float = 0.05,
        layerdrop: float = 0.0,
        config_only: bool = False,
    ):
        """Initialize Wav2VecSLU.

        Args:
            pretrained_model_name_or_path: HuggingFace model identifier or local directory.
            n_action: Number of action classes.
            n_object: Number of object classes.
            n_location: Number of location classes.
            n_joint: Optional number of joint intent classes (e.g. 31).
            keep_layers: Number of transformer encoder layers to keep (1..12). None keeps all.
            freeze_feature_encoder: If True, freeze temporal CNN feature encoder weights.
            freeze_encoder: If True, freeze the entire Transformer encoder (only heads trainable).
            dropout: Dropout probability before classification heads.
            mask_time_prob: Probability of masking time steps during training (SpecAugment).
            layerdrop: LayerDrop probability.
            config_only: If True, initialize with random weights from config (for fast offline tests).
        """
        super().__init__()
        self.pretrained_name = pretrained_model_name_or_path
        self.freeze_feature_encoder_flag = freeze_feature_encoder
        self.freeze_encoder = freeze_encoder
        self.keep_layers = keep_layers

        # 1. Load config or pretrained model
        if config_only:
            config = Wav2Vec2Config(
                mask_time_prob=mask_time_prob,
                layerdrop=layerdrop,
                hidden_size=768,
                num_hidden_layers=keep_layers if keep_layers else 12,
                num_attention_heads=12,
                intermediate_size=3072,
            )
            self.enc = Wav2Vec2Model(config)
        else:
            config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path,
                mask_time_prob=mask_time_prob,
                layerdrop=layerdrop,
            )
            self.enc = Wav2Vec2Model.from_pretrained(
                pretrained_model_name_or_path,
                config=config,
            )

        # 2. Freeze CNN feature encoder (always per docs/12 §3)
        if freeze_feature_encoder:
            self.enc.freeze_feature_encoder()

        # 3. Layer truncation (E5 / P4-02 support)
        if keep_layers is not None and keep_layers < len(self.enc.encoder.layers):
            self.enc.encoder.layers = nn.ModuleList(self.enc.encoder.layers[:keep_layers])
            self.enc.config.num_hidden_layers = keep_layers

        # 4. Freeze full Transformer encoder if requested (E2)
        if freeze_encoder:
            for p in self.enc.parameters():
                p.requires_grad = False

        d = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)

        # 5. Classification heads
        self.heads = nn.ModuleDict(
            {
                "action": nn.Linear(d, n_action),
                "object": nn.Linear(d, n_object),
                "location": nn.Linear(d, n_location),
            }
        )
        self.joint = nn.Linear(d, n_joint) if n_joint is not None else None

    def train(self, mode: bool = True):
        """Override train to ensure frozen encoder stays in eval mode (docs/12 §3, §4)."""
        super().train(mode)
        if self.freeze_encoder and mode:
            # Frozen encoder must not apply dropout or SpecAugment time-masking
            self.enc.eval()
        return self

    def forward(
        self,
        waveform: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            waveform: [B, T] raw audio tensor normalized to zero-mean, unit-variance.
            lengths: [B] valid lengths in audio samples. If None, assumes all samples valid.

        Returns:
            Dictionary with logits for 'action', 'object', 'location' (and 'joint' if configured).
        """
        # Note: Do NOT pass attention_mask to Wav2Vec2Model base (docs/12 §4 gotcha #1).
        # Group norm in conv layer does not support mask and causes mismatch.
        h = self.enc(waveform).last_hidden_state  # [B, T_frames, hidden_dim]

        if lengths is not None:
            # Compute frame-level lengths using the wav2vec2 feature extractor formula
            f_len = self.enc._get_feat_extract_output_lengths(lengths)
            # Create boolean mask: [B, T_frames, 1]
            t_frames = h.size(1)
            seq_range = torch.arange(t_frames, device=h.device)[None, :]
            mask = (seq_range < f_len[:, None]).unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = h.mean(dim=1)

        pooled = self.drop(pooled)

        logits = {slot: head(pooled) for slot, head in self.heads.items()}
        if self.joint is not None:
            logits["joint"] = self.joint(pooled)

        return logits
