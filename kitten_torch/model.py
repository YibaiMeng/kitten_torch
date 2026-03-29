"""
KittenTTS nano — full PyTorch model.

Inference pipeline:
  1. Phonemize text → token IDs (same as original KittenTTS)
  2. TextEncoder: token IDs → text_lstm=128 (returned twice for API compat; no text_proj in v0.8)
  3. ALBERT: token IDs → bert_out=128
  4. Predictor (phoneme-level):
     a. Concat(bert_out, style_half) → 2×BiLSTM+FC → 256
     b. Duration LSTM → log-duration bins → frame counts
  5. Length Regulation (LR): expand phoneme features to frame level
  6. Predictor (frame-level):
     a. Shared LSTM → 128
     b. F0/N ResBlocks → F0 at 2T frames, N at 2T frames
  7. Acoustic Decoder:
     a. Concat(text_lstm_128, F0_T, N_T) → encode block → 256
     b. 4× decode blocks with F0/N → 256 at 2T frames
  8. Generator:
     a. SineGenerator from upsampled F0
     b. 10×6× upsample + iSTFT → audio at 24kHz
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.text_encoder import TextEncoder
from .modules.bert import Albert
from .modules.predictor import Predictor
from .modules.decoder import AcousticDecoder
from .modules.generator import Generator


class KittenTTSTorch(nn.Module):
    """
    PyTorch reconstruction of KittenTTS nano.

    Args:
        onnx_path: path to .onnx file (used for weight loading)
        voices_path: path to voices.npz
    """

    def __init__(self):
        super().__init__()
        # Sub-modules
        self.text_encoder = TextEncoder(vocab_size=178, d_model=128)
        self.bert = Albert(
            vocab_size=178,
            embedding_dim=128,
            hidden=768,
            heads=12,
            intermediate=2048,
            num_layers=12,
            max_pos=512,
            output_dim=128,
        )
        self.predictor = Predictor(style_half=128)
        self.decoder = AcousticDecoder(style_half=128)
        self.decoder.generator = Generator(style_half=128, n_harmonics=9, sr=24000)

    @torch.inference_mode()
    def forward(
        self,
        input_ids: torch.Tensor,  # (B, T) int64 — phoneme token IDs
        style: torch.Tensor,       # (B, 256) voice embedding
        speed: float = 1.0,
        deterministic: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) phoneme IDs with BOS/EOS = 0
            style: (B, 256) voice embedding from voices.npz
            speed: speaking rate multiplier (>1 = faster)
            deterministic: if True, suppress random noise for reproducibility

        Returns:
            audio: (B, N_samples) float32 audio at 24kHz
        """
        B, T = input_ids.shape
        device = input_ids.device

        # 1. Text encoding (v0.8: both outputs are 128-dim h_lstm)
        text_proj, text_lstm = self.text_encoder(input_ids)  # (B,T,128), (B,T,128)

        # 2. ALBERT
        bert_out = self.bert(input_ids)                       # (B, T, 128)

        # 3. Predictor — phoneme level
        text_enc_out, durations = self.predictor.forward(
            bert_out, style, speed
        )
        # text_enc_out: (B, T, 256), durations: (B, T) float frame counts

        # 4. Length Regulation: expand phoneme features to frame level
        # Convert durations to integer frame counts, then repeat features
        dur_int = durations.round().long().clamp(min=1)  # (B, T)

        # LR expand: for each batch item, repeat each phoneme feature dur[i] times
        lr_features = self._length_regulate(text_enc_out, dur_int)  # (B, T_frames, 256)

        # 5. Predictor — frame level
        shared_h, f0_pred, n_pred = self.predictor.forward_frame(lr_features, style)
        # shared_h: (B, T_frames, 128)
        # f0_pred: (B, 1, 2*T_frames)  — F0 in log/linear scale at 2T resolution
        # n_pred: (B, 1, 2*T_frames)   — energy at 2T resolution

        # 6. Acoustic Decoder
        # text_proj at T_frames (LR-expanded, 128-dim in v0.8)
        text_proj_lr = self._length_regulate_proj(text_proj, dur_int)  # (B, T_frames, 128)

        features, f0_2T = self.decoder(text_proj_lr, f0_pred, n_pred, style)
        # features: (B, 256, 2*T_frames)
        # f0_2T: (B, 1, 2*T_frames)

        # 7. Upsample F0 to sample level for SineGenerator
        T2 = features.shape[-1]  # 2*T_frames
        # Expected audio length: T2 * 60 * 5 = T2 * 300 (approx)
        N_approx = T2 * 300  # rough estimate; actual may differ slightly
        f0_samples = F.interpolate(
            f0_2T.float(),
            size=N_approx,
            mode='linear',
            align_corners=False,
        )  # (B, 1, N_approx)
        # F0 is in log scale in some models, but here it's direct Hz (clamped ≥ 0)
        f0_hz = F.relu(f0_samples)  # ensure non-negative

        # 8. Generator
        audio = self.decoder.generator(features, f0_hz, style, deterministic)
        # audio: (B, 1, N_samples)
        return audio.squeeze(1)  # (B, N_samples)

    @staticmethod
    def _length_regulate(
        features: torch.Tensor,  # (B, T, D)
        durations: torch.Tensor, # (B, T) int64
    ) -> torch.Tensor:
        """Expand phoneme features to frame level using integer durations."""
        B, T, D = features.shape
        # Build expanded features per batch item
        outputs = []
        for b in range(B):
            chunks = []
            for t in range(T):
                d = int(durations[b, t].item())
                if d > 0:
                    chunks.append(features[b, t:t+1].expand(d, -1))
            if chunks:
                outputs.append(torch.cat(chunks, dim=0))
            else:
                outputs.append(features.new_zeros(1, D))

        # Pad to same length
        max_len = max(o.shape[0] for o in outputs)
        padded = features.new_zeros(B, max_len, D)
        for b, o in enumerate(outputs):
            padded[b, :o.shape[0]] = o
        return padded

    @staticmethod
    def _length_regulate_proj(
        features: torch.Tensor,  # (B, T, D)
        durations: torch.Tensor, # (B, T) int64
    ) -> torch.Tensor:
        """Same as _length_regulate but for a different feature tensor."""
        return KittenTTSTorch._length_regulate(features, durations)


def build_model(onnx_path: str | Path) -> KittenTTSTorch:
    """Construct and load weights from ONNX file."""
    from .load_weights import load_weights
    model = KittenTTSTorch()
    model.eval()
    load_weights(model, onnx_path)
    return model
