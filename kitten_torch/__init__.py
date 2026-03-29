"""
kitten_torch — PyTorch reconstruction of KittenTTS nano.

Public API mirrors the latest kittentts package (v0.8+):
    from kitten_torch import KittenTTS
    tts = KittenTTS()
    audio = tts.generate("Hello world")
    tts.generate_to_file("Hello world", "out.wav")
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import soundfile as sf
import torch

from .model import KittenTTSTorch, build_model
from .tokenizer import Tokenizer


SAMPLE_RATE = 24000
_DEFAULT_REPO = "KittenML/kitten-tts-nano-0.8-int8"


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


def _try_import_preprocessor():
    try:
        from kittentts.preprocess import TextPreprocessor
        return TextPreprocessor(remove_punctuation=False)
    except Exception:
        return None


def ensure_punctuation(text: str) -> str:
    """Ensure text ends with punctuation; add comma if not."""
    text = text.strip()
    if text and text[-1] not in '.!?,;:':
        text = text + ','
    return text


def chunk_text(text: str, max_len: int = 400) -> list[str]:
    """Split text into ≤max_len chunks at sentence boundaries."""
    sentences = re.split(r'[.!?]+', text)
    chunks = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        if len(sentence) <= max_len:
            chunks.append(ensure_punctuation(sentence))
        else:
            words = sentence.split()
            temp = ""
            for word in words:
                if len(temp) + len(word) + 1 <= max_len:
                    temp += (" " + word) if temp else word
                else:
                    if temp:
                        chunks.append(ensure_punctuation(temp.strip()))
                    temp = word
            if temp:
                chunks.append(ensure_punctuation(temp.strip()))
    return chunks or [ensure_punctuation(text)]


class KittenTTS:
    """
    PyTorch-based KittenTTS. Same API as the latest kittentts package.

    Args:
        model_path: path to .onnx model file (used for weights); downloads from HF if None
        voices_path: path to voices.npz; downloads from HF if None
        device: torch device (default cpu)
        repo: HuggingFace repo ID (default: kitten-tts-nano-0.8)
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        voices_path: Optional[str] = None,
        device: str = "cpu",
        repo: str = _DEFAULT_REPO,
    ):
        from huggingface_hub import hf_hub_download

        # Config-based download (mirrors kittentts get_model.py)
        config_path = hf_hub_download(repo_id=repo, filename="config.json")
        with open(config_path) as f:
            config = json.load(f)

        if model_path is None:
            model_path = hf_hub_download(repo_id=repo, filename=config["model_file"])
        if voices_path is None:
            voices_path = hf_hub_download(repo_id=repo, filename=config["voices"])

        self._speed_priors: dict[str, float] = config.get("speed_priors", {})
        self._voice_aliases: dict[str, str] = config.get("voice_aliases", {})

        self._device = torch.device(device)
        self._model = build_model(model_path)
        self._model.to(self._device)
        self._model.eval()

        self._voices = np.load(voices_path)
        self._phonemizer = _try_import_phonemizer()
        self._tokenizer = Tokenizer()
        self._preprocessor = _try_import_preprocessor()

        # Voice lists
        self.available_voices: list[str] = [
            'expr-voice-2-m', 'expr-voice-2-f', 'expr-voice-3-m', 'expr-voice-3-f',
            'expr-voice-4-m', 'expr-voice-4-f', 'expr-voice-5-m', 'expr-voice-5-f',
        ]
        self.all_voice_names: list[str] = [
            'Bella', 'Jasper', 'Luna', 'Bruno', 'Rosie', 'Hugo', 'Kiki', 'Leo'
        ]

    def _resolve_voice(self, voice: str) -> str:
        return self._voice_aliases.get(voice, voice)

    def _phonemize(self, text: str) -> list[int]:
        return self._tokenizer.encode(self._phonemizer, text)

    def _generate_chunk(
        self,
        text: str,
        voice: str,
        speed: float = 1.0,
        deterministic: bool = False,
    ) -> np.ndarray:
        """Generate audio for a single text chunk."""
        ids = self._phonemize(text)
        input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)

        voice_data = self._voices[voice]          # (N, 256)
        ref_id = min(len(text), voice_data.shape[0] - 1)
        style_np = voice_data[ref_id:ref_id + 1]  # (1, 256)
        style = torch.from_numpy(style_np).to(self._device)

        with torch.inference_mode():
            audio = self._model(input_ids, style, speed=speed, deterministic=deterministic)

        return audio.squeeze(0).cpu().numpy()

    def generate(
        self,
        text: str,
        voice: str = "expr-voice-5-m",
        speed: float = 1.0,
        deterministic: bool = False,
        clean_text: bool = False,
    ) -> np.ndarray:
        """
        Generate speech audio, handling long text via chunking.

        Returns:
            audio: (N_samples,) float32 numpy array at 24kHz
        """
        voice = self._resolve_voice(voice)
        if voice in self._speed_priors:
            speed = speed * self._speed_priors[voice]
        if clean_text and self._preprocessor is not None:
            text = self._preprocessor(text)

        chunks = chunk_text(text)
        return np.concatenate([
            self._generate_chunk(c, voice, speed, deterministic) for c in chunks
        ])

    def generate_stream(
        self,
        text: str,
        voice: str = "expr-voice-5-m",
        speed: float = 1.0,
        deterministic: bool = False,
        clean_text: bool = False,
    ) -> Generator[np.ndarray, None, None]:
        """Generate audio chunk-by-chunk as a generator."""
        voice = self._resolve_voice(voice)
        if voice in self._speed_priors:
            speed = speed * self._speed_priors[voice]
        if clean_text and self._preprocessor is not None:
            text = self._preprocessor(text)

        for chunk in chunk_text(text):
            yield self._generate_chunk(chunk, voice, speed, deterministic)

    def generate_to_file(
        self,
        text: str,
        path: str,
        voice: str = "expr-voice-5-m",
        speed: float = 1.0,
        sample_rate: int = SAMPLE_RATE,
        clean_text: bool = False,
    ) -> None:
        """Generate speech and save to a WAV file."""
        audio = self.generate(text, voice=voice, speed=speed, clean_text=clean_text)
        sf.write(path, audio, sample_rate)
