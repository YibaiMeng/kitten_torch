"""
Phoneme tokenizer for KittenTTS.

Loads the character vocabulary from phoneme_vocab.json (extracted once from
kittentts source so we don't need to inspect it at runtime).

Usage:
    from kitten_torch.tokenizer import Tokenizer
    tok = Tokenizer()
    ids = tok.encode(phonemizer, "Hello, how are you today?")
"""
from __future__ import annotations

import json
import os

_VOCAB_PATH = os.path.join(os.path.dirname(__file__), "phoneme_vocab.json")
_EMBEDDING_SIZE = 178  # text_encoder embedding has 178 entries (0–177)


class Tokenizer:
    """
    Character-level phoneme tokenizer.

    The vocab is extracted from the kittentts reference implementation and
    saved to phoneme_vocab.json.  Index 0 is used as both BOS/EOS and pad.

    Characters with assigned index >= _EMBEDDING_SIZE (e.g. ᵻ at 179) are
    clamped to _EMBEDDING_SIZE - 1 so they don't crash the embedding lookup.
    """

    def __init__(self, vocab_path: str = _VOCAB_PATH):
        with open(vocab_path, encoding="utf-8") as f:
            data = json.load(f)
        chars = list(data["vocab"])
        # Use first-occurrence index so duplicates in the vocab string
        # (backslash at 174/177 and apostrophe at 175/178) resolve to the
        # lower index, which is within the valid embedding range.
        self._word_index: dict[str, int] = {}
        for i, c in enumerate(chars):
            if c not in self._word_index:
                self._word_index[c] = i

    def encode(self, phonemizer, text: str) -> list[int]:
        """
        Phonemize text and return a list of integer token IDs.

        phonemizer: object with a .phonemize([str]) -> [str] method
                    (e.g. tts._phonemizer from KittenTTS)
        """
        import re
        phonemized = phonemizer.phonemize([text])[0]
        normalized = " ".join(re.findall(r"\w+|[^\w\s]", phonemized))
        ids = (
            [0]
            + [
                min(self._word_index[c], _EMBEDDING_SIZE - 1)
                for c in normalized
                if c in self._word_index
            ]
            + [0]
        )
        return ids
