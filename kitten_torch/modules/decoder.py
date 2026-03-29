"""
Acoustic Decoder (AdaIN encode + decode blocks).

Architecture (from ONNX weight analysis):

encode block (single):
  Input: cat(text_proj=512, F0=1, N=1) = 514 channels
  conv1x1: Conv1d(514, 256, k=1)  — residual shortcut
  norm1: AdaIN(514, style_half=128) → fc: Linear(128, 1028=2*514)
  conv1: Conv1d(514, 256, k=3, pad=1)
  norm2: AdaIN(256, style_half=128) → fc: Linear(128, 512=2*256)
  conv2: Conv1d(256, 256, k=3, pad=1)
  out = conv2 + residual

decode blocks (×4):
  Input: cat(encode_out=256, asr_res=64, F0=1, N=1) = 322 channels
  conv1x1: Conv1d(322, 256, k=1)  — residual shortcut
  norm1: AdaIN(322, style_half=128) → fc: Linear(128, 644=2*322)
  conv1: Conv1d(322, 256, k=3, pad=1)
  norm2: AdaIN(256, style_half=128) → fc: Linear(128, 512=2*256)
  conv2: Conv1d(256, 256, k=3, pad=1)
  out = conv2 + residual

  decode.3 additionally has:
    pool: depthwise ConvTranspose1d(322, 322, k=3, stride=2)
    applied BEFORE main computation to get 2× temporal resolution

asr_res: Conv1d(512, 64, k=1)  — maps text_proj (512) to 64

F0_conv: Conv1d(1, 1, k=3) — smooth F0 before concat
N_conv: Conv1d(1, 1, k=3)  — smooth N before concat

Output from decode.3: (B, 256, 2T)  → goes to generator
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaIN1d(nn.Module):
    """Instance Norm + style-conditioned scale/shift."""

    def __init__(self, channels: int, style_half: int = 128):
        super().__init__()
        self.norm = nn.InstanceNorm1d(channels, affine=True)
        self.fc = nn.Linear(style_half, 2 * channels)

    def forward(self, x: torch.Tensor, style_half: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        style = self.fc(style_half)           # (B, 2C)
        gamma, beta = style.chunk(2, dim=1)   # (B, C)
        return (gamma.unsqueeze(2) + 1.0) * x + beta.unsqueeze(2)


import math as _math
_INV_SQRT2 = 1.0 / _math.sqrt(2.0)  # 0.7071...


class AdaINResBlock(nn.Module):
    """
    AdaIN residual block with optional temporal upsample (for decode.3).

    ONNX architecture (verified):
      No upsample:
        norm1(x) → LeakyReLU → conv1 → norm2 → LeakyReLU → conv2
        residual = conv1x1(x)
        output = (conv2 + residual) * (1/√2)

      upsample=True (decode.3 only):
        norm1(x_orig) → LeakyReLU → pool(ConvTranspose, 2×) → conv1 → norm2 → conv2
        residual = conv1x1(F.interpolate(x_orig, 2×, nearest))
        output = (conv2 + residual) * (1/√2)

    The 1/√2 scaling is applied to ALL blocks (encode + decode 0-3).
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        style_half: int = 128,
        upsample: bool = False,
    ):
        super().__init__()
        self.upsample = upsample
        if upsample:
            self.pool = nn.ConvTranspose1d(in_ch, in_ch, 3, stride=2, groups=in_ch)

        self.conv1x1 = nn.Conv1d(in_ch, out_ch, 1)
        self.norm1 = AdaIN1d(in_ch, style_half)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.norm2 = AdaIN1d(out_ch, style_half)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)

    def _pool_upsample(self, x: torch.Tensor) -> torch.Tensor:
        """2× upsample using depthwise ConvTranspose1d (trim to exact 2T)."""
        target_T = x.shape[-1] * 2
        x = self.pool(x)
        if x.shape[-1] > target_T:
            x = x[..., :target_T]
        elif x.shape[-1] < target_T:
            x = F.pad(x, (0, target_T - x.shape[-1]))
        return x

    def forward(self, x: torch.Tensor, style_half: torch.Tensor) -> torch.Tensor:
        if self.upsample:
            # ONNX decode.3: norm1 on original x, pool after activation
            # Residual: nearest-neighbor resize of original x
            residual = self.conv1x1(
                F.interpolate(x, size=x.shape[-1] * 2, mode='nearest')
            )
            h = self.norm1(x, style_half)
            h = F.leaky_relu(h, 0.2)
            h = self._pool_upsample(h)   # upsample AFTER activation
            h = self.conv1(h)
        else:
            residual = self.conv1x1(x)
            h = self.norm1(x, style_half)
            h = F.leaky_relu(h, 0.2)
            h = self.conv1(h)

        h = self.norm2(h, style_half)
        h = F.leaky_relu(h, 0.2)
        h = self.conv2(h)
        return (h + residual) * _INV_SQRT2


class AcousticDecoder(nn.Module):
    """
    Acoustic decoder: text features + F0/N → acoustic features.

    Input:
        text_enc: (B, T, 512) — from text_encoder.text_proj
        f0: (B, 1, T)         — at T frames (first half of 2T F0 predictions)
        n: (B, 1, T)          — energy at T frames
        style: (B, 256)       — voice style embedding

    Output:
        features: (B, 256, 2T) — feature maps for generator
        f0_2T: (B, 1, 2T)      — F0 at 2T resolution for generator
    """

    def __init__(self, style_half: int = 128):
        super().__init__()
        # Map text_proj (512) → 64-dim for decoder concat
        self.asr_res = nn.Conv1d(512, 64, 1)

        # Stride-2 convs: downsample F0/N from 2T → T (ONNX verified: stride=2)
        self.F0_conv = nn.Conv1d(1, 1, 3, stride=2, padding=1)
        self.N_conv = nn.Conv1d(1, 1, 3, stride=2, padding=1)

        # Encode block: 514-ch input (512 text + 1 F0 + 1 N)
        self.encode = AdaINResBlock(514, 256, style_half, upsample=False)

        # Decode blocks: 322-ch input (256 encode + 64 asr_res + 1 F0 + 1 N)
        # decode.3 has 2× temporal upsample
        self.decode = nn.ModuleList([
            AdaINResBlock(322, 256, style_half, upsample=False),  # 0
            AdaINResBlock(322, 256, style_half, upsample=False),  # 1
            AdaINResBlock(322, 256, style_half, upsample=False),  # 2
            AdaINResBlock(322, 256, style_half, upsample=True),   # 3: 2×
        ])

    def forward(
        self,
        text_enc: torch.Tensor,  # (B, T, 512)
        f0: torch.Tensor,        # (B, 1, 2T) full-res F0 from predictor
        n: torch.Tensor,         # (B, 1, 2T) full-res N
        style: torch.Tensor,     # (B, 256)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (features (B, 256, 2T), f0_2T (B, 1, 2T))."""
        # ONNX verified: ALL blocks (encode + decode 0-3) use style[:128]
        s1 = style[:, :128]   # (B, 128) — used for ALL blocks

        # text_enc: (B, T, 512) → (B, 512, T)
        te = text_enc.transpose(1, 2)  # (B, 512, T)

        # f0/n are at 2T resolution from predictor; stride-2 conv downsamples to T
        f0_2T = f0                      # (B, 1, 2T) — kept at 2T for generator
        f0_T = self.F0_conv(f0)         # (B, 1, T) via stride=2
        n_T = self.N_conv(n)            # (B, 1, T) via stride=2

        # Encode: concat text (512) + F0 (1) + N (1) = 514
        enc_input = torch.cat([te, f0_T, n_T], dim=1)  # (B, 514, T)
        enc_out = self.encode(enc_input, s1)             # (B, 256, T)

        # asr_res: project text_enc (512) to 64
        asr = self.asr_res(te)  # (B, 64, T)

        # Decode: cat(encode=256, asr=64, F0=1, N=1) = 322 for each block
        # All decode blocks use s1 = style[:128] (verified from ONNX)
        x = enc_out
        for i, block in enumerate(self.decode):
            dec_input = torch.cat([x, asr, f0_T, n_T], dim=1)  # (B, 322, T)
            x = block(dec_input, s1)                              # decode.3 upsamples to 2T

        return x, f0_2T
