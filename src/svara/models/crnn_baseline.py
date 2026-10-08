"""CRNN baseline model for Spoken Language Understanding (B1 / E1).

Architecture per docs/02 §2:
- 64-bin log-mel filterbank (25ms window, 10ms hop)
- 2-3 Conv2d blocks with BatchNorm, ReLU, MaxPool
- 2-layer Bidirectional GRU
- Mean pooling over time
- Three classification heads (action, object, location) + optional joint head

Task: P2-07
Reference: docs/02 §2, docs/11 V-11..V-18
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torchaudio.transforms as T


class LogMelSpectrogram(nn.Module):
    """Log-mel filterbank extractor from raw waveform."""

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,  # 25 ms
        hop_length: int = 160,  # 10 ms
        n_mels: int = 64,
    ):
        super().__init__()
        self.mel_transform = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]
        mel = self.mel_transform(x)  # [B, n_mels, time_frames]
        # Log compression with stabilization
        log_mel = torch.log(torch.clamp(mel, min=1e-5))
        return log_mel


class CRNNBaseline(nn.Module):
    """Convolutional Recurrent Neural Network baseline for SLU."""

    def __init__(
        self,
        n_action: int = 6,
        n_object: int = 14,
        n_location: int = 4,
        n_joint: Optional[int] = 31,
        n_mels: int = 64,
        conv_channels: int = 64,
        rnn_hidden: int = 128,
        rnn_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.feature_extractor = LogMelSpectrogram(n_mels=n_mels)

        # 3 Conv blocks
        self.conv = nn.Sequential(
            nn.Conv2d(1, conv_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # n_mels/2, time/2
            nn.Dropout2d(dropout),
            nn.Conv2d(conv_channels, conv_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),  # n_mels/4, time/4
            nn.Dropout2d(dropout),
            nn.Conv2d(conv_channels * 2, conv_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(conv_channels * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),  # n_mels/8, time/4
            nn.Dropout2d(dropout),
        )

        rnn_input_dim = (n_mels // 8) * (conv_channels * 2)

        self.rnn = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=rnn_hidden,
            num_layers=rnn_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if rnn_layers > 1 else 0.0,
        )

        d = rnn_hidden * 2  # Bidirectional
        self.drop = nn.Dropout(dropout)

        self.heads = nn.ModuleDict(
            {
                "action": nn.Linear(d, n_action),
                "object": nn.Linear(d, n_object),
                "location": nn.Linear(d, n_location),
            }
        )
        self.joint = nn.Linear(d, n_joint) if n_joint is not None else None

    def forward(
        self, waveform: torch.Tensor, lengths: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            waveform: [B, T] raw audio tensor.
            lengths: [B] lengths in raw audio samples.
        """
        # 1. Feature extraction -> [B, 1, n_mels, time]
        x = self.feature_extractor(waveform).unsqueeze(1)

        # 2. Convolutional blocks -> [B, C, F, T']
        x = self.conv(x)
        b, c, f, t = x.shape

        # 3. Reshape for RNN: [B, T', C*F]
        x = x.permute(0, 3, 1, 2).contiguous().view(b, t, c * f)

        # 4. Bidirectional GRU
        rnn_out, _ = self.rnn(x)  # [B, T', 2*H]

        # 5. Length-aware pooling over time
        if lengths is not None:
            # Conv downsamples time by factor of 4
            downsampled_lens = torch.clamp((lengths / 160 / 4).ceil().long(), min=1, max=t)
            time_mask = (
                torch.arange(t, device=waveform.device)[None, :]
                < downsampled_lens[:, None]
            ).unsqueeze(-1)
            pooled = (rnn_out * time_mask).sum(dim=1) / time_mask.sum(dim=1).clamp(min=1)
        else:
            pooled = rnn_out.mean(dim=1)

        pooled = self.drop(pooled)

        logits = {slot: head(pooled) for slot, head in self.heads.items()}
        if self.joint is not None:
            logits["joint"] = self.joint(pooled)

        return logits
