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
    Maps token ids → (B, T, 512) for decoder input.
    Also exposes intermediate 128-dim output for the predictor.
    """

    def __init__(self, vocab_size: int = 178, d_model: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.cnn = nn.ModuleList([CNNLayerNorm(d_model) for _ in range(6)])
        # BiLSTM: hidden_size=64 per direction → output=128
        self.lstm = nn.LSTM(
            d_model, 64, num_layers=1, batch_first=True, bidirectional=True
        )
        self.text_proj = nn.Linear(128, 512)

    def forward(
        self, input_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: (B, T) int64

        Returns:
            h_proj: (B, T, 512)  — for decoder
            h_lstm: (B, T, 128)  — for predictor
        """
        x = self.embedding(input_ids)   # (B, T, 128)
        x = x.transpose(1, 2)           # (B, 128, T) for Conv1d
        for layer in self.cnn:
            x = layer(x)
        x = x.transpose(1, 2)           # (B, T, 128) for LSTM
        h_lstm, _ = self.lstm(x)        # (B, T, 128)
        h_proj = self.text_proj(h_lstm) # (B, T, 512)
        return h_proj, h_lstm
