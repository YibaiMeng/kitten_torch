"""
HiFi-GAN + iSTFT-Net Generator (reverse-engineered from ONNX).

Architecture (verified via ONNX intermediate tensor probing):

1. SineGenerator: 9 harmonics from F0 → (B, 1, N_samples)
   - sines scaled by 0.1 before harmonic mixing
   - voiced mask: f0 > 10 Hz; unvoiced frames get noise (amplitude 1/3) instead of sines
   - Tanh on output (≈identity at these small scales)

2. Forward learned STFT: (B, 1, N) → (B, 22, N//5) — stride=5, n_fft=20
   - Edge (replicate) padding of n_fft//2=10 on each side
   - Output: [magnitude=sqrt(re²+im²), phase=atan2(im,re)] (NOT raw real/imag)
   - channels [0:11] = magnitude (≥0), channels [11:22] = phase (-π to π)

3. Inverse STFT: (B, 22, frames) → (B, 1, N)
   - conv_post outputs [log_magnitude (11 ch), phase_angle (11 ch)]
   - mag = exp(log_mag), phase_sin = sin(phase)
   - audio = ConvTranspose(mag*cos(phase_sin), w_real) - ConvTranspose(mag*sin(phase_sin), w_imag)

Generator stages (for T=generator input frames, 2× predictor frames):
  Input: features (B, 256, T), sine_stft (B, 22, N//5)

  Stage 1:
    noise_0 = noise_convs.0(sine_stft)      # Conv1d(22,128,k=12,stride=6) → (B,128,T10)
    ups_0 = ups.0(features)                 # ConvTranspose1d(256,128,k=20,stride=10) → (B,128,T10)
    noise_rb0 = noise_res.0(noise_0, s2)    # AdaINResBlock(128, k=7, 3 dilations)
    x = ups_0 + noise_rb0
    r0 = resblocks.0(x, s2)                 # AdaINResBlock(128, k=3, 3 dilations)
    r1 = resblocks.1(x, s2)                 # AdaINResBlock(128, k=3, 3 dilations)
    x = (r0 + r1) * 0.5                     # MRF average

  Stage 2:
    noise_1 = noise_convs.1(sine_stft)      # Conv1d(22,64,k=1) → (B,64,T60)
    ups_1 = ups.1(x)                        # ConvTranspose1d(128,64,k=12,stride=6) → (B,64,T60)
    ups_1 = reflection_pad(ups_1)           # pad by 1 to match noise_1 size
    noise_rb1 = noise_res.1(noise_1, s2)    # AdaINResBlock(64, k=11, 3 dilations)
    x = ups_1 + noise_rb1
    r2 = resblocks.2(x, s2)                 # AdaINResBlock(64, k=3, 3 dilations)
    r3 = resblocks.3(x, s2)                 # AdaINResBlock(64, k=3, 3 dilations)
    x = (r2 + r3) * 0.5                     # MRF average

  Post:
    x = conv_post(x)                        # Conv1d(64, 22, k=7)
    audio = inverse_stft(x)                 # learned iSTFT, stride=5

LeakyReLU slopes (verified from ONNX attributes):
  - before ups.0: alpha=0.1
  - before ups.1: alpha=0.1
  - before conv_post: alpha=0.01

ONNX quantization note:
  All convs and AdaIN linear layers in resblocks and noise_res use ConvInteger/MatMulInteger
  (int8 weights + DynamicQuantizeLinear activations). The PT model uses float32 throughout
  (matching training-time behavior). This causes inherent numerical differences vs ONNX.

Notes:
- Both resblocks within each stage use the SAME input (MRF pattern), then averaged
- noise_res processes only the noise signal (not the main features)
- Style first half (s1 = style[:,:128]) conditions all AdaIN blocks in the generator (verified from ONNX)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------------ #
#  AdaIN Normalization                                                 #
# ------------------------------------------------------------------ #

class AdaIN1d(nn.Module):
    def __init__(self, channels: int, style_half: int = 128):
        super().__init__()
        self.norm = nn.InstanceNorm1d(channels, affine=True)
        self.fc = nn.Linear(style_half, 2 * channels)

    def forward(self, x: torch.Tensor, style_half: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        g, b = self.fc(style_half).chunk(2, dim=1)
        return (g.unsqueeze(2) + 1.0) * x + b.unsqueeze(2)


# ------------------------------------------------------------------ #
#  AdaIN ResBlock (for generator)                                      #
# ------------------------------------------------------------------ #

class GenResBlock(nn.Module):
    """
    3-dilation AdaIN ResBlock with Snake activation.

    ONNX architecture (verified):
      convs1: dilations [1, 3, 5] — causal dilation pattern
      convs2: dilations [1, 1, 1]
      activation: Snake(x; α) = x + sin²(α·x)/α  (NOT LeakyReLU)
      alpha1[i], alpha2[i]: per-channel learned frequency params, shape (1, C, 1)
    """

    def __init__(self, channels: int, style_half: int = 128, kernel: int = 3):
        super().__init__()
        n = 3  # 3 dilation steps
        dilations = [1, 3, 5]
        self.adain1 = nn.ModuleList([AdaIN1d(channels, style_half) for _ in range(n)])
        self.adain2 = nn.ModuleList([AdaIN1d(channels, style_half) for _ in range(n)])
        self.alpha1 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for _ in range(n)])
        self.alpha2 = nn.ParameterList([nn.Parameter(torch.ones(1, channels, 1)) for _ in range(n)])
        # convs1 use dilations [1,3,5]; convs2 use dilation=1
        self.convs1 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel,
                      dilation=d, padding=d * (kernel - 1) // 2)
            for d in dilations
        ])
        self.convs2 = nn.ModuleList([
            nn.Conv1d(channels, channels, kernel, padding=(kernel - 1) // 2)
            for _ in range(n)
        ])

    @staticmethod
    def snake(x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        """Snake activation: f(x) = x + sin²(α·x)/α"""
        return x + torch.sin(alpha * x).pow(2) / alpha

    def forward(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        for i in range(3):
            h = self.adain1[i](x, s)
            h = self.snake(h, self.alpha1[i])
            h = self.convs1[i](h)
            h = self.adain2[i](h, s)
            h = self.snake(h, self.alpha2[i])
            h = self.convs2[i](h)
            x = x + h
        return x


# ------------------------------------------------------------------ #
#  Sine Generator                                                      #
# ------------------------------------------------------------------ #

class SineGenerator(nn.Module):
    """
    Generates multi-harmonic sinusoidal excitation from F0.
    Linear(9→1) weights the harmonics.
    """

    def __init__(self, sample_rate: int = 24000, n_harmonics: int = 9):
        super().__init__()
        self.sr = sample_rate
        self.n_h = n_harmonics
        self.l_linear = nn.Linear(n_harmonics, 1)

    def forward(self, f0: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """f0: (B, 1, N_samples) in Hz → (B, 1, N_samples)

        ONNX-verified architecture:
          - voiced = f0 > 10 Hz (threshold from ONNX)
          - sines scaled by 0.1 (Mul_10 in ONNX, Constant_34=0.1)
          - voiced sines: 0.1 * sin * voiced_mask
          - noise: 0.003 * N(0,1) for voiced, 0.333 * N(0,1) for unvoiced
          - l_linear mixes harmonics, then Tanh (≈identity for small values)
        """
        B, _, N = f0.shape
        device = f0.device

        # Voiced mask: f0 > 10 Hz (matches ONNX Constant_31=10.0)
        voiced = (f0 > 10.0).float()  # (B, 1, N)

        # Multi-harmonic frequencies
        h = torch.arange(1, self.n_h + 1, device=device, dtype=f0.dtype).view(1, self.n_h, 1)
        f0_h = f0 * h  # (B, n_h, N)

        # Phase accumulation
        phase_inc = 2.0 * math.pi * f0_h / self.sr
        phase = torch.cumsum(phase_inc, dim=-1)

        if not deterministic:
            init = torch.rand(B, self.n_h, 1, device=device) * 2 * math.pi
        else:
            init = torch.zeros(B, self.n_h, 1, device=device)
        sines = torch.sin(phase + init)  # (B, n_h, N)

        # ONNX: scale sines by 0.1, apply voiced mask (Mul_14 = 0.1 * sin * voiced)
        voiced_sines = 0.1 * sines * voiced  # (B, n_h, N)

        if not deterministic:
            # noise_amp = 0.003 for voiced, 1/3 for unvoiced (ONNX Add_4)
            noise_amp = voiced * 0.003 + (1.0 - voiced) * (1.0 / 3.0)
            voiced_sines = voiced_sines + noise_amp * torch.randn_like(sines)

        # Harmonic mixing: (B, N, n_h) × Linear(n_h, 1) → (B, N, 1) → (B, 1, N)
        out = self.l_linear(voiced_sines.transpose(1, 2)).transpose(1, 2)
        # Tanh (≈identity for small values, included for completeness)
        out = torch.tanh(out)
        return out


# ------------------------------------------------------------------ #
#  Learned iSTFT (forward + backward STFT)                            #
# ------------------------------------------------------------------ #

class LearnedISTFT(nn.Module):
    """
    Learned forward and inverse STFT filters.
    n_fft=20 → 11 bins. stride (hop)=5.

    Forward STFT: Conv1d(1, 22, k=20, stride=5) → (B, 22, frames)
    Inverse STFT: per-bin ConvTranspose1d × 11 → sum → (B, 1, N)
    """

    def __init__(self, n_fft: int = 20, hop: int = 5):
        super().__init__()
        self.n_bins = n_fft // 2 + 1  # 11
        self.n_fft = n_fft
        self.hop = hop
        # Forward filters
        self.weight_forward_real = nn.Parameter(torch.randn(self.n_bins, 1, n_fft))
        self.weight_forward_imag = nn.Parameter(torch.randn(self.n_bins, 1, n_fft))
        # Backward (synthesis) filters
        self.weight_backward_real = nn.Parameter(torch.randn(self.n_bins, 1, n_fft))
        self.weight_backward_imag = nn.Parameter(torch.randn(self.n_bins, 1, n_fft))

    def forward_stft(self, x: torch.Tensor) -> torch.Tensor:
        """(B, 1, N) → (B, 22, N//hop + 1)
        ONNX: edge (replicate) padding of n_fft//2=10, then Conv with stride=hop.
        Output: [magnitude (sqrt(re²+im²)), atan2_phase] — NOT raw real/imag.
        channels [0:11] = magnitude (≥0), channels [11:22] = phase (-π to π).
        """
        edge_pad = self.n_fft // 2  # 10
        x = F.pad(x, (edge_pad, edge_pad), mode='replicate')
        real = F.conv1d(x, self.weight_forward_real, stride=self.hop)
        imag = F.conv1d(x, self.weight_forward_imag, stride=self.hop)
        magnitude = torch.sqrt(real ** 2 + imag ** 2)
        phase = torch.atan2(imag, real)
        return torch.cat([magnitude, phase], dim=1)

    def inverse_stft(self, spec: torch.Tensor) -> torch.Tensor:
        """
        (B, 22, frames) → (B, 1, N_approx)

        conv_post output encodes [log_magnitude (11), phase_angle (11)].
        ONNX converts polar → Cartesian before synthesis:
          mag       = exp(log_mag)          channels [0:11]
          phase_sin = sin(phase_angle)      channels [11:22], wraps to [-1,1]
          real      = mag * cos(phase_sin)
          imag      = mag * sin(phase_sin)
          audio     = ConvTranspose(real, w_real) - ConvTranspose(imag, w_imag)
        """
        log_mag   = spec[:, :self.n_bins, :]   # (B, 11, frames)
        phase     = spec[:, self.n_bins:, :]   # (B, 11, frames)

        mag        = torch.exp(log_mag)
        phase_sin  = torch.sin(phase)
        real       = mag * torch.cos(phase_sin)
        imag       = mag * torch.sin(phase_sin)

        real_out = F.conv_transpose1d(real, self.weight_backward_real, stride=self.hop)
        imag_out = F.conv_transpose1d(imag, self.weight_backward_imag, stride=self.hop)
        return real_out - imag_out

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        return self.inverse_stft(spec)


# ------------------------------------------------------------------ #
#  Full Generator                                                      #
# ------------------------------------------------------------------ #

class Generator(nn.Module):
    """Full HiFi-GAN + iSTFT generator."""

    def __init__(self, style_half: int = 128, n_harmonics: int = 9, sr: int = 24000):
        super().__init__()
        self.sine_gen = SineGenerator(sr, n_harmonics)

        # Upsamplers (ONNX: pads=[5,5] and pads=[3,3] respectively)
        self.ups = nn.ModuleList([
            nn.ConvTranspose1d(256, 128, 20, stride=10, padding=5),
            nn.ConvTranspose1d(128, 64, 12, stride=6, padding=3),
        ])

        # Noise injection convs (ONNX: noise_convs.0 pads=[3,3], noise_convs.1 pads=[0,0])
        self.noise_convs = nn.ModuleList([
            nn.Conv1d(22, 128, 12, stride=6, padding=3),
            nn.Conv1d(22, 64, 1),
        ])

        # Noise resblocks (process only the noise signal)
        # noise_res.0: k=7, 128-ch
        # noise_res.1: k=11, 64-ch
        self.noise_res = nn.ModuleList([
            GenResBlock(128, style_half, kernel=7),
            GenResBlock(64, style_half, kernel=11),
        ])

        # Main resblocks (MRF: 2 per stage, same input, sum outputs)
        # All use k=3 (verified from ONNX weight shapes)
        self.resblocks = nn.ModuleList([
            GenResBlock(128, style_half, kernel=3),  # stage 1
            GenResBlock(128, style_half, kernel=3),  # stage 1
            GenResBlock(64, style_half, kernel=3),   # stage 2
            GenResBlock(64, style_half, kernel=3),   # stage 2
        ])

        # Post-conv
        self.conv_post = nn.Conv1d(64, 22, 7, padding=3)

        # Learned iSTFT
        self.stft = LearnedISTFT(n_fft=20, hop=5)

    def forward(
        self,
        features: torch.Tensor,    # (B, 256, T) from decoder
        f0_samples: torch.Tensor,  # (B, 1, N_samples)
        style: torch.Tensor,       # (B, 256)
        deterministic: bool = False,
    ) -> torch.Tensor:
        """Returns audio (B, 1, N_samples)."""
        s = style[:, :128]  # first half for generator (verified from ONNX: style[:, :128])

        # Sinusoidal source
        sine = self.sine_gen(f0_samples, deterministic)  # (B, 1, N)

        # Forward STFT of sine → noise features
        sine_stft = self.stft.forward_stft(sine)       # (B, 22, ~N//5)
        noise_0 = self.noise_convs[0](sine_stft)       # (B, 128, ~T×10)
        noise_1 = self.noise_convs[1](sine_stft)       # (B, 64, ~T×60)

        # Noise resblocks
        n0 = self.noise_res[0](noise_0, s)             # (B, 128, ~T×10)
        n1 = self.noise_res[1](noise_1, s)             # (B, 64, ~T×60)

        # Stage 1
        x = F.leaky_relu(features, 0.1)
        x = self.ups[0](x)                             # (B, 128, T×10)
        # Align n0 to x
        n0 = _match_length(n0, x.shape[-1])
        x = x + n0

        # MRF: both resblocks from same x, then average (ONNX: Add + Div by 2.0)
        r0 = self.resblocks[0](x, s)
        r1 = self.resblocks[1](x, s)
        x = (r0 + r1) * 0.5                            # (B, 128, T×10)

        # Stage 2
        x = F.leaky_relu(x, 0.1)
        x = self.ups[1](x)                             # (B, 64, T×60)
        # Reflection-pad to match noise_1 length
        x = _match_length_pad(x, n1.shape[-1])
        x = x + n1

        # MRF: both resblocks from same x, then average (ONNX: Add + Div by 2.0)
        r2 = self.resblocks[2](x, s)
        r3 = self.resblocks[3](x, s)
        x = (r2 + r3) * 0.5                            # (B, 64, T×60)

        # Post-processing + iSTFT
        x = F.leaky_relu(x, 0.01)
        x = self.conv_post(x)                          # (B, 22, T×60)
        audio = self.stft.inverse_stft(x)              # (B, 1, N)
        return audio


def _match_length(x: torch.Tensor, target: int) -> torch.Tensor:
    """Trim or zero-pad x to match target temporal length."""
    if x.shape[-1] > target:
        return x[..., :target]
    elif x.shape[-1] < target:
        return F.pad(x, (0, target - x.shape[-1]))
    return x


def _match_length_pad(x: torch.Tensor, target: int) -> torch.Tensor:
    """Reflect-pad x to reach target length (or trim if longer)."""
    if x.shape[-1] >= target:
        return x[..., :target]
    pad = target - x.shape[-1]
    return F.pad(x, (0, pad), mode='reflect')
