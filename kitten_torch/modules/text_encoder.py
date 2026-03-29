"""
Text Encoder (CNN + BiLSTM).

Architecture (from ONNX weight analysis):
  Embedding(178, 128)
  → 6 × [Conv1d(128, 128, k=5, pad=2) + LayerNorm(128) + ReLU]
  → BiLSTM(input=128, hidden=64) → output=128
  → Linear(128, 512)  [text_proj]

Output: (batch, T, 512) — passed to decoder
Intermediate (before text_proj): (batch, T, 128) — fed to predictor
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLayerNorm(nn.Module):
    """Conv1d + LayerNorm (over channel dim) + ReLU."""

    def __init__(self, channels: int, kernel: int = 5):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel, padding=kernel // 2)
        # ONNX stores LayerNorm on the channel dim as (gamma, beta) with shape (C,)
        # standard LayerNorm over last dim → we permute
        self.norm = nn.LayerNorm(channels)  # normalized_shape = C

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T)
        x = self.conv(x)
        # LayerNorm over channel dim: permute → (B, T, C)
        x = self.norm(x.transpose(1, 2)).transpose(1, 2)
        return F.relu(x)


class TextEncoder(nn.Module):
    """
    Maps token ids → (B, T, 128) for decoder input.

    v0.8 architecture: 2 CNN blocks (down from 6), no separate text_proj.
    text_proj = h_lstm = 128-dim BiLSTM output.
    """

    def __init__(self, vocab_size: int = 178, d_model: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cnn = nn.ModuleList([CNNLayerNorm(d_model) for _ in range(2)])
        # BiLSTM: hidden_size=64 per direction → output=128
        self.lstm = nn.LSTM(
            d_model, 64, num_layers=1, batch_first=True, bidirectional=True
        )
        # No text_proj in v0.8: decoder takes 128-dim features directly

    def forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (B, T) int64

        Returns:
            h_lstm: (B, T, 128)  — for decoder (text_proj) and predictor
            h_lstm: (B, T, 128)  — same tensor returned twice for API compat
        """
        x = self.embedding(input_ids)   # (B, T, 128)
        x = x.transpose(1, 2)           # (B, 128, T) for Conv1d
        for layer in self.cnn:
            x = layer(x)
        x = x.transpose(1, 2)           # (B, T, 128) for LSTM
        h_lstm, _ = self.lstm(x)        # (B, T, 128)
        return h_lstm, h_lstm
