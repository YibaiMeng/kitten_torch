"""KittenTTS nano model configuration (reverse-engineered from ONNX)."""
from dataclasses import dataclass


@dataclass
class KittenConfig:
    # Vocabulary / text
    vocab_size: int = 178         # embedding table rows (0..177)
    text_dim: int = 128           # CNN text encoder hidden dim

    # CNN text encoder
    text_cnn_layers: int = 6
    text_cnn_kernel: int = 5

    # ALBERT text encoder
    bert_hidden: int = 768
    bert_proj_in: int = 128       # embedding_hidden_mapping_in input dim
    bert_heads: int = 12
    bert_intermediate: int = 2048
    bert_layers: int = 12         # shared weight layers (ALBERT)
    bert_max_pos: int = 512

    # Style
    style_dim: int = 256          # voice embedding dim (split into 128+128 halves)

    # Predictor
    predictor_dim: int = 512      # concatenated input: 128(cnn) + 768(bert) ... projected
    predictor_lstm_hidden: int = 256   # BiLSTM hidden (256 per direction = 512 total)
    predictor_lstm_layers: int = 6     # stacked BiLSTMs
    duration_lstm_hidden: int = 256    # /lstm single BiLSTM

    # Duration (frame length prediction)
    duration_proj_out: int = 1    # scalar log-duration per phoneme

    # Shared (frame-level) predictor LSTM
    shared_lstm_hidden: int = 256

    # F0 / energy predictor
    f0_resblock_dim: int = 256
    f0_resblock_layers: int = 3
    n_mels: int = 80              # not used in inference path but consistent

    # Decoder (AdaIN)
    decoder_dim: int = 128        # AdaIN encode/decode channel base
    # encode: 512→128, 128→256 w/ stride-2 downsample  (but actual is different - see analysis)
    # decode.3: depthwise ConvTranspose stride-2 (2× upsample)

    # Generator (HiFi-GAN style + iSTFT)
    upsample_initial_channel: int = 256
    upsample_rates: tuple = (10, 6)        # ConvTranspose upsampling (10× then 6×)
    upsample_kernel_sizes: tuple = (20, 12)
    resblock_kernel_sizes: tuple = (3, 7, 11)
    resblock_dilation_sizes: tuple = ((1, 3, 5), (1, 3, 5), (1, 3, 5))
    n_harmonics: int = 9          # SineGenerator harmonics

    # iSTFT vocoder
    istft_n_fft: int = 20         # → 11 bins
    istft_hop_length: int = 5     # stride
    istft_win_length: int = 20

    # Audio
    sample_rate: int = 24000
    hop_length: int = 300         # frames→samples: 300× overall
    # 74 f0 frames * 300 = 22200 samples
    # but ONNX output is also 22200: confirmed

    # Derived
    @property
    def istft_bins(self) -> int:
        return self.istft_n_fft // 2 + 1   # 11

    @property
    def bert_head_dim(self) -> int:
        return self.bert_hidden // self.bert_heads  # 64


DEFAULT_CONFIG = KittenConfig()
