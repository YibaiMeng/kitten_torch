"""
Predictor module.

Architecture (from ONNX weight analysis):
  Input: concat(text_cnn_lstm: 128, bert_encoder: 128) = 256-dim

  text_encoder (2 stacked BiLSTMs):
    lstms.0: BiLSTM(256 → H=64) → 128
    lstms.1.fc: Linear(128, 256)
    lstms.2: BiLSTM(256 → H=64) → 128
    lstms.3.fc: Linear(128, 256)
    Output: 256-dim

  Duration branch:
    /lstm: BiLSTM(256 → H=64) → 128
    duration_proj.linear_layer: Linear(128, 50) → log-duration bins

  After length regulation (LR expand with predicted durations):

  Shared branch:
    /shared: BiLSTM(256 → H=64) → 128

  F0 predictor (3 AdaIN ResBlocks):
    F0.0: AdaINResBlock(128, 128, style_half=128)
    F0.1: AdaINUpsampleBlock(128→64, T→2T, style_half=128)
    F0.2: AdaINResBlock(64, 64, style_half=128)
    F0_proj: Conv1d(64, 1, k=1) → scalar F0 at 2T frames

  N predictor (same structure as F0):
    N.0, N.1, N.2, N_proj: same shapes as F0
"""
from __future__ import annotations

import math as _math
import torch
import torch.nn as nn
import torch.nn.functional as F

_INV_SQRT2 = 1.0 / _math.sqrt(2.0)


# ------------------------------------------------------------------ #
#  AdaIN Normalization                                                 #
# ------------------------------------------------------------------ #

class AdaIN1d(nn.Module):
    """
    Adaptive Instance Normalization for 1D sequences.
    Applies instance norm, then style-conditioned scale + shift.

    fc: Linear(style_half, 2*channels) → [scale, shift]
    norm: InstanceNorm1d(channels, affine=True) — affine params
          serve as "base" scale/shift before style modulation
    """

    def __init__(self, channels: int, style_half: int = 128):
        super().__init__()
        self.norm = nn.InstanceNorm1d(channels, affine=True)
        self.fc = nn.Linear(style_half, 2 * channels)

    def forward(self, x: torch.Tensor, style_half: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, T)
        style_half: (B, style_half)  — half the style vector
        """
        # Instance norm (with learned affine scale/shift)
        x = self.norm(x)
        # Style modulation: (B, 2C) → split into scale, shift
        style = self.fc(style_half)          # (B, 2C)
        gamma, beta = style.chunk(2, dim=1)  # each (B, C)
        gamma = gamma.unsqueeze(2)           # (B, C, 1)
        beta = beta.unsqueeze(2)
        return (gamma + 1.0) * x + beta    # gamma is residual: +1 = identity at zero


# ------------------------------------------------------------------ #
#  Predictor ResBlocks                                                 #
# ------------------------------------------------------------------ #

class PredResBlock(nn.Module):
    """
    AdaIN residual block (no channel change, no spatial change).
    Used in F0.0, F0.2, N.0, N.2.

    conv1, conv2: Conv1d(C, C, k=3, pad=1)
    """

    def __init__(self, channels: int, style_half: int = 128):
        super().__init__()
        self.norm1 = AdaIN1d(channels, style_half)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.norm2 = AdaIN1d(channels, style_half)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor, style_half: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x, style_half)
        x = F.leaky_relu(x, 0.2)
        x = self.conv1(x)
        x = self.norm2(x, style_half)
        x = F.leaky_relu(x, 0.2)
        x = self.conv2(x)
        return (x + residual) * _INV_SQRT2


class PredUpsampleBlock(nn.Module):
    """
    AdaIN block with 2× temporal upsample and channel halving.
    Used in F0.1 and N.1.

    pool: depthwise ConvTranspose1d(in_ch, in_ch, k=3, stride=2)
      → temporal 2× upsample applied FIRST to input
    conv1x1: Conv1d(in_ch, out_ch, k=1) for residual shortcut
    norm1: AdaIN(in_ch)
    conv1: Conv1d(in_ch, out_ch, k=3, pad=1)
    norm2: AdaIN(out_ch)
    conv2: Conv1d(out_ch, out_ch, k=3, pad=1)
    """

    def __init__(self, in_ch: int, out_ch: int, style_half: int = 128):
        super().__init__()
        self.pool = nn.ConvTranspose1d(in_ch, in_ch, 3, stride=2, groups=in_ch)
        self.conv1x1 = nn.Conv1d(in_ch, out_ch, 1)
        self.norm1 = AdaIN1d(in_ch, style_half)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm2 = AdaIN1d(out_ch, style_half)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)

    def _pool_upsample(self, x: torch.Tensor) -> torch.Tensor:
        """2× upsample via depthwise ConvTranspose1d, trimmed to exact 2T."""
        target_T = x.shape[-1] * 2
        x = self.pool(x)
        if x.shape[-1] > target_T:
            x = x[..., :target_T]
        elif x.shape[-1] < target_T:
            x = F.pad(x, (0, target_T - x.shape[-1]))
        return x

    def forward(self, x: torch.Tensor, style_half: torch.Tensor) -> torch.Tensor:
        # ONNX F0.1/N.1 architecture (verified):
        # residual = conv1x1(Resize(x, 2×, nearest))
        # h = norm1(x_original) → LeakyReLU → pool(ConvTranspose 2×) → conv1 → norm2 → LeakyReLU → conv2
        # output = (h + residual) * 1/√2
        residual = self.conv1x1(
            F.interpolate(x, size=x.shape[-1] * 2, mode='nearest')
        )
        h = self.norm1(x, style_half)
        h = F.leaky_relu(h, 0.2)
        h = self._pool_upsample(h)
        h = self.conv1(h)
        h = self.norm2(h, style_half)
        h = F.leaky_relu(h, 0.2)
        h = self.conv2(h)
        return (h + residual) * _INV_SQRT2


# ------------------------------------------------------------------ #
#  LSTM-based text encoder (6 BiLSTM layers)                          #
# ------------------------------------------------------------------ #

class PredTextEncoder(nn.Module):
    """
    N BiLSTM layers with style-conditioned AdaIN after each (v0.8: n_layers=2).

    Architecture (verified from ONNX):
      Initial input: cat([bert_out (128), style_half (128)]) = 256-dim
      Each step:
        BiLSTM(256→64×2=128)
        LayerNorm(128)
        FC(style_half → 256) → gamma(128) + beta(128)
        adain = (gamma + 1) * LN_out + beta   [AdaIN]
        h = adain + lstm_out                   [residual]
        output = cat([h, style_half.expand]) = 256-dim → next BiLSTM

    lstms[even] = BiLSTM, lstms[odd] = FC(style_half→256)
    """

    def __init__(self, hidden: int = 64, n_layers: int = 2, style_half: int = 128):
        super().__init__()
        lstm_in = style_half * 2  # 256 = 128 bert + 128 style
        lstm_out = hidden * 2     # 128

        combined = []
        lns = []
        for _ in range(n_layers):
            combined.append(nn.LSTM(lstm_in, hidden, batch_first=True, bidirectional=True))
            combined.append(nn.Linear(style_half, 2 * lstm_out))  # style_half → 256 (gamma+beta)
            lns.append(nn.LayerNorm(lstm_out, elementwise_affine=False))
        self.lstms = nn.ModuleList(combined)
        self.lns = nn.ModuleList(lns)

    def forward(self, bert_out: torch.Tensor, style_half: torch.Tensor) -> torch.Tensor:
        """
        bert_out: (B, T, 128)
        style_half: (B, 128)
        Returns: (B, T, 256) with left=adain_out, right=style_half.expand
        """
        B, T, _ = bert_out.shape
        s = style_half.unsqueeze(1).expand(-1, T, -1)  # (B, T, 128)
        x = torch.cat([bert_out, s], dim=-1)           # (B, T, 256) initial input

        for i in range(0, len(self.lstms), 2):
            lstm = self.lstms[i]
            fc   = self.lstms[i + 1]
            ln   = self.lns[i // 2]

            lstm_out, _ = lstm(x)                      # (B, T, 128)
            ln_out = ln(lstm_out)                      # (B, T, 128)

            style_cond = fc(style_half)                # (B, 256)
            gamma, beta = style_cond.chunk(2, dim=-1)  # each (B, 128)
            adain = (gamma.unsqueeze(1) + 1.0) * ln_out + beta.unsqueeze(1)

            x = torch.cat([adain, s], dim=-1)          # (B, T, 256) — no lstm_out residual

        return x  # (B, T, 256)


# ------------------------------------------------------------------ #
#  Duration predictor                                                  #
# ------------------------------------------------------------------ #

class DurationPredictor(nn.Module):
    """
    BiLSTM(256→128) → Linear(128, 50) → 50 log-duration bins.
    Uses softmax + weighted average to get continuous duration per phoneme.
    """

    def __init__(self, input_dim: int = 256, hidden: int = 64, n_bins: int = 50):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden * 2, n_bins)
        self.n_bins = n_bins
        # Log-duration bin centers: 0..n_bins-1 mapped to log-durations
        # In StyleTTS2 the duration predictor outputs log durations via softmax
        # bins: typically range from log(0.1) to log(max_duration)

    def forward(
        self, x: torch.Tensor, speed: float = 1.0
    ) -> torch.Tensor:
        """
        x: (B, T, 256)
        Returns: durations (B, T) as integer frame counts
        """
        h, _ = self.lstm(x)  # (B, T, 128)
        logits = self.linear(h)  # (B, T, 50)
        # Soft-max weighted duration
        probs = logits.softmax(dim=-1)  # (B, T, 50)
        # Bin values: 0..49 as float
        bins = torch.arange(self.n_bins, device=x.device, dtype=x.dtype)
        duration_log = (probs * bins).sum(dim=-1)  # (B, T)
        # Convert log to actual duration in frames (exp), adjust for speed
        duration = duration_log.exp() / speed  # (B, T)
        return duration


# ------------------------------------------------------------------ #
#  Full Predictor                                                      #
# ------------------------------------------------------------------ #

class Predictor(nn.Module):
    """
    Full predictor module:
    1. concat(text_lstm_128, bert_128) → 256
    2. text_encoder: 6×BiLSTM+FC → 256
    3. duration branch: BiLSTM + Linear(128, 50) → durations
    4. (After LR expand:) shared BiLSTM(256→128)
    5. F0/N ResBlocks → F0/N predictions at 2T frames
    """

    def __init__(self, style_half: int = 128):
        super().__init__()
        self.text_encoder = PredTextEncoder(hidden=64, n_layers=2, style_half=style_half)

        # Duration branch
        self.lstm = nn.LSTM(256, 64, batch_first=True, bidirectional=True)
        self.duration_proj = nn.Linear(128, 50)  # linear_layer in ONNX

        # Shared frame-level branch
        self.shared = nn.LSTM(256, 64, batch_first=True, bidirectional=True)

        # F0 predictor
        self.F0 = nn.ModuleList([
            PredResBlock(128, style_half),            # F0.0
            PredUpsampleBlock(128, 64, style_half),   # F0.1
            PredResBlock(64, style_half),             # F0.2
        ])
        self.F0_proj = nn.Conv1d(64, 1, 1)

        # N (energy/noise) predictor — same structure as F0
        self.N = nn.ModuleList([
            PredResBlock(128, style_half),
            PredUpsampleBlock(128, 64, style_half),
            PredResBlock(64, style_half),
        ])
        self.N_proj = nn.Conv1d(64, 1, 1)

    def forward(
        self,
        bert_h: torch.Tensor,     # (B, T, 128) from ALBERT
        style: torch.Tensor,      # (B, 256) style embedding
        speed: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            text_enc_out: (B, T, 256) text encoder output (for LR expand)
            durations: (B, T) int64 frame counts per phoneme
        """
        s2 = style[:, 128:]  # (B, 128) — used in text encoder AdaIN

        # Text encoder: bert_out + style_half → 6×BiLSTM+AdaIN → 256
        x = self.text_encoder(bert_h, s2)  # (B, T, 256)

        # Duration branch: BiLSTM → Linear(128, 50) → sigmoid sum / speed → round
        dur_h, _ = self.lstm(x)                    # (B, T, 128)
        dur_logits = self.duration_proj(dur_h)      # (B, T, 50)
        # Duration = sum of sigmoid activations (soft count of active bins)
        dur_cont = dur_logits.sigmoid().sum(dim=-1) / speed  # (B, T)
        durations = dur_cont.round().long().clamp(min=1)     # (B, T) integer frames

        return x, durations

    def forward_frame(
        self,
        lr_features: torch.Tensor,  # (B, T_frames, 256) after length regulation
        style: torch.Tensor,        # (B, 256)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Frame-level prediction after LR expansion.

        Returns:
            shared_h: (B, T_frames, 128)
            F0_pred: (B, 1, 2*T_frames)
            N_pred: (B, 1, 2*T_frames)
        """
        s1 = style[:, :128]
        s2 = style[:, 128:]

        h, _ = self.shared(lr_features)  # (B, T, 128)

        # F0 branch: (B, C, T) format
        f0 = h.transpose(1, 2)           # (B, 128, T)
        for i, blk in enumerate(self.F0):
            f0 = blk(f0, s2)
        f0 = self.F0_proj(f0)             # (B, 1, 2T)

        # N branch
        n = h.transpose(1, 2)
        for i, blk in enumerate(self.N):
            n = blk(n, s2)
        n = self.N_proj(n)                # (B, 1, 2T)

        return h, f0, n
