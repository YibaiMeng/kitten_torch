# kitten-torch

PyTorch reconstruction of [KittenTTS nano](https://huggingface.co/KittenML/kitten-tts-nano-0.8-int8) — a StyleTTS2-based TTS model originally distributed as a quantized ONNX file.

The goal is a clean, readable PyTorch implementation that produces audio numerically close to the original ONNX, suitable for fine-tuning or further research.

> **Based on kitten-tts-nano-0.8-int8.**
> The architecture, weights, and tokenizer were all reverse-engineered from
> `kitten_tts_nano_v0_8.onnx` as distributed by [KittenML](https://huggingface.co/KittenML)
> and the `kittentts` pip package (version used during development: 0.8.x).
> If the upstream ONNX or package changes, this implementation may need updating.

## Architecture

KittenTTS nano is a StyleTTS2-inspired model with a HiFi-GAN + iSTFT vocoder:

```
Text → Phonemizer (espeak-ng) → token IDs
                                     │
                    ┌────────────────┤
                    │                │
               TextEncoder       ALBERT
              (CNN+BiLSTM)    (12-layer, shared)
                    │                │
                    └────────┬───────┘
                             │ 256-dim
                         Predictor
                    ┌────────┴────────┐
               Duration          Text Encoder
               (BiLSTM)         (2×BiLSTM+AdaIN)
                    │                │
               Length Regulation (LR expand)
                                     │
                              Frame Predictor
                         ┌───────────┴───────────┐
                       F0 (3×AdaIN ResBlocks)   N (energy)
                                     │
                          Acoustic Decoder
                         (4× AdaIN decode blocks)
                                     │
                               Generator
                    ┌────────────────┴───────────────┐
               SineGenerator               features (256-dim)
               (9 harmonics)                         │
                    │              ┌──────────────────┤
               Forward STFT     ups.0 (×10)       ups.1 (×6)
                    │           + noise_res       + noise_res
                    └────────── + MRF (2 resblocks averaged)
                                     │
                               conv_post → iSTFT
                                     │
                               audio @ 24kHz
```

**Style conditioning**: a 256-dim voice embedding is split into two 128-dim halves:
- `style[:, :128]` → all generator and acoustic decoder AdaIN blocks
- `style[:, 128:]` → predictor text encoder, F0/N branch AdaIN blocks

**ONNX quantization**: the original model uses ConvInteger/MatMulInteger (int8 weights + DynamicQuantizeLinear activations) for all convolutions and linear layers in the resblocks and AdaIN FC layers. This implementation simulates DQL via forward pre-hooks to stay numerically close to ONNX.

## Installation

```bash
pip install -e .
# requires: torch, onnx, onnxruntime, soundfile, phonemizer, espeakng-loader
# phonemizer also needs espeak-ng system library:
#   apt install espeak-ng   # Debian/Ubuntu
#   brew install espeak     # macOS
```

## Usage

```python
from kitten_torch import KittenTTS

# Downloads model from HuggingFace on first run
tts = KittenTTS()

# Generate to numpy array (float32, 24kHz)
audio = tts.generate("Hello, how are you today?", voice="expr-voice-2-m")

# Available voices: expr-voice-{2,3,4,5}-{m,f}

# Save to file
tts.generate_to_file("Hello world", "out.wav", voice="expr-voice-3-f", speed=1.0)
```

Or with explicit model paths (no HuggingFace download):

```python
tts = KittenTTS(
    model_path="/path/to/kitten_tts_nano_v0_8.onnx",
    voices_path="/path/to/voices.npz",
)
```

## Tokenizer

The phoneme character vocabulary is saved in `kitten_torch/phoneme_vocab.json` so it can be loaded without inspecting the `kittentts` package source at runtime:

```python
from kitten_torch.tokenizer import Tokenizer
tok = Tokenizer()
ids = tok.encode(phonemizer, "Hello world")
```

The vocab has 178 valid entries (indices 0–177). Two characters in the raw vocab string (`'` and `\`) appear as duplicates at indices 178–179; these are clamped to 177. The IPA character `ᵻ` (index 179 in the source) is similarly clamped.

## Quality

PT and ONNX produce very close output across tested voices and sentences:

| Metric | Value |
|--------|-------|
| Duration ratio (PT/ONNX) | 0.97–1.08 |
| Amplitude ratio (PT/ONNX) | 0.92–1.33 (most near 1.0) |

The residual differences come from quantization noise in the int8 LSTM and resblock operations, plus random phase offsets in the SineGenerator — not audible as a quality difference.

## Blind Test Scripts

Generate a paired A/B listening test:

```bash
python gen_ab_test.py          # 20 pairs → ab_test/pair_001_A.wav, pair_001_B.wav, ...
python gen_ab_test.py --n 30   # more pairs for more statistical power
```

Score your guesses:

```bash
python score_ab_test.py                     # interactive
python score_ab_test.py --guesses AABBA...  # one A/B per pair
```

Needs **15+/20** correct for p < 0.05 (models sound statistically different).

## File Structure

```
kitten_torch/
  __init__.py          KittenTTS class + public API
  model.py             KittenTTSTorch nn.Module (full pipeline)
  load_weights.py      ONNX → PyTorch weight mapping + DQL hooks
  weight_loader.py     ONNX initializer extraction utilities
  tokenizer.py         Phoneme tokenizer (uses phoneme_vocab.json)
  phoneme_vocab.json   Saved character vocabulary (178 entries)
  config.py            Architecture hyperparameters
  modules/
    text_encoder.py    CNN + BiLSTM text encoder
    bert.py            ALBERT encoder (12 shared-weight layers)
    predictor.py       Duration + F0/N predictor (AdaIN ResBlocks)
    decoder.py         Acoustic decoder (AdaIN encode/decode)
    generator.py       HiFi-GAN generator + SineGen + iSTFT

generate_audio.py      Demo script: generate samples for all voices
gen_ab_test.py         Generate paired A/B blind test clips
score_ab_test.py       Score A/B test guesses + binomial p-value

research/              Investigation scripts used during reverse-engineering
  debug_*.py           Step-by-step ONNX vs PT intermediate comparisons
  probe_*.py           Injection probes to isolate divergence sources
```
