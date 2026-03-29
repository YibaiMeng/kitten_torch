"""
kitten_torch — PyTorch reconstruction of KittenTTS nano.

Public API mirrors the original kittentts package:
    from kitten_torch import KittenTTS
    tts = KittenTTS()
    audio = tts.generate("Hello world")
    tts.generate_to_file("Hello world", "out.wav")
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch

from .model import KittenTTSTorch, build_model
from .tokenizer import Tokenizer


SAMPLE_RATE = 24000


def _try_import_phonemizer():
    try:
        from phonemizer.backend import EspeakBackend
        return EspeakBackend(
            language="en-us",
            preserve_punctuation=True,
            with_stress=True,
        )
    except Exception as e:
        raise RuntimeError(f"phonemizer/espeak-ng not available: {e}")


class KittenTTS:
    """
    PyTorch-based KittenTTS. Same API as the original.

    Args:
        model_path: path to .onnx model file (used for weights)
        voices_path: path to voices.npz
        device: torch device (default cpu)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        voices_path: Optional[str] = None,
        device: str = "cpu",
    ):
        # Download model if not specified
        if model_path is None or voices_path is None:
            from huggingface_hub import hf_hub_download
            repo = "KittenML/kitten-tts-nano-0.1"
            if model_path is None:
                model_path = hf_hub_download(repo, "kitten_tts_nano_v0_1.onnx")
            if voices_path is None:
                voices_path = hf_hub_download(repo, "voices.npz")

        self._device = torch.device(device)
        self._model = build_model(model_path)
        self._model.to(self._device)
        self._model.eval()

        self._voices = np.load(voices_path)
        self._phonemizer = _try_import_phonemizer()
        self._tokenizer = Tokenizer()

    def _phonemize(self, text: str) -> list[int]:
        """Convert text to phoneme token IDs."""
        return self._tokenizer.encode(self._phonemizer, text)

    def generate(
        self,
        text: str,
        voice: str = "expr-voice-2-m",
        speed: float = 1.0,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Generate speech audio.

        Args:
            text: input text (English)
            voice: voice name from voices.npz
            speed: speaking rate (1.0 = normal)
            deterministic: suppress random noise for reproducibility

        Returns:
            audio: (N_samples,) float32 numpy array at 24kHz
        """
        ids = self._phonemize(text)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
        style_np = self._voices[voice]  # (1, 256)
        style = torch.from_numpy(style_np).to(self._device)

        with torch.inference_mode():
            audio = self._model(input_ids, style, speed=speed, deterministic=deterministic)

        return audio.squeeze(0).cpu().numpy()

    def generate_to_file(
        self,
        text: str,
        path: str,
        voice: str = "expr-voice-2-m",
        speed: float = 1.0,
    ) -> None:
        """Generate speech and save to a WAV file."""
        audio = self.generate(text, voice=voice, speed=speed)
        sf.write(path, audio, SAMPLE_RATE)
