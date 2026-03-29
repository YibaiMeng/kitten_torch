"""
ALBERT text encoder (reverse-engineered from ONNX).

Architecture:
  word_embeddings(178, 128) + position_embeddings(512, 128) + token_type_embeddings(2, 128)
  → LayerNorm(128) + Dropout
  → embedding_hidden_mapping_in: Linear(128, 768)
  → 12 × shared AlbertLayer (one set of weights, reused)
      - SelfAttention: Q/K/V(768→768), dense(768→768), LayerNorm
      - FFN: Linear(768→2048) + GELU + Linear(2048→768) + LayerNorm
  → bert_encoder: Linear(768, 128)   ← projects back to 128

Output: (batch, T, 128)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AlbertSelfAttention(nn.Module):
    def __init__(self, hidden: int = 768, heads: int = 12):
        super().__init__()
        self.heads = heads
        self.head_dim = hidden // heads
        self.scale = math.sqrt(self.head_dim)

        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.dense = nn.Linear(hidden, hidden)
        self.LayerNorm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, H = x.shape
        nh, dh = self.heads, self.head_dim

        q = self.query(x).view(B, T, nh, dh).transpose(1, 2)  # (B, nh, T, dh)
        k = self.key(x).view(B, T, nh, dh).transpose(1, 2)
        v = self.value(x).view(B, T, nh, dh).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / self.scale  # (B, nh, T, T)
        if mask is not None:
            attn = attn + mask
        attn = attn.softmax(dim=-1)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, H)
        out = self.dense(out)
        out = self.LayerNorm(out + x)
        return out


class AlbertLayer(nn.Module):
    def __init__(self, hidden: int = 768, heads: int = 12, intermediate: int = 2048):
        super().__init__()
        self.attention = AlbertSelfAttention(hidden, heads)
        self.ffn = nn.Linear(hidden, intermediate)
        self.ffn_output = nn.Linear(intermediate, hidden)
        self.full_layer_layer_norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        x = self.attention(x, mask)
        h = F.gelu(self.ffn(x))
        h = self.ffn_output(h)
        x = self.full_layer_layer_norm(h + x)
        return x


class Albert(nn.Module):
    """
    ALBERT with 12 shared-weight layers (a single AlbertLayer reused 12 times).
    """

    def __init__(
        self,
        vocab_size: int = 178,
        embedding_dim: int = 128,
        hidden: int = 768,
        heads: int = 12,
        intermediate: int = 2048,
        num_layers: int = 12,
        max_pos: int = 512,
        output_dim: int = 128,
    ):
        super().__init__()
        self.num_layers = num_layers

        # Embeddings (128-dim)
        self.embeddings = nn.ModuleDict({
            "word_embeddings": nn.Embedding(vocab_size, embedding_dim),
            "position_embeddings": nn.Embedding(max_pos, embedding_dim),
            "token_type_embeddings": nn.Embedding(2, embedding_dim),
        })
        self.emb_ln = nn.LayerNorm(embedding_dim)

        # Project embeddings to hidden
        self.embedding_hidden_mapping_in = nn.Linear(embedding_dim, hidden)

        # Single shared layer (reused num_layers times)
        self.albert_layer = AlbertLayer(hidden, heads, intermediate)

        # Project back to output_dim
        self.bert_encoder = nn.Linear(hidden, output_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (B, T) int64
            mask: optional (B, 1, 1, T) additive attention mask (large neg for padding)

        Returns:
            (B, T, output_dim=128)
        """
        B, T = input_ids.shape
        device = input_ids.device

        pos_ids = torch.arange(T, device=device).unsqueeze(0)  # (1, T)
        token_type_ids = torch.zeros(B, T, dtype=torch.long, device=device)

        x = (
            self.embeddings["word_embeddings"](input_ids)
            + self.embeddings["position_embeddings"](pos_ids)
            + self.embeddings["token_type_embeddings"](token_type_ids)
        )
        x = self.emb_ln(x)  # (B, T, 128)

        x = self.embedding_hidden_mapping_in(x)  # (B, T, 768)

        for _ in range(self.num_layers):
            x = self.albert_layer(x, mask)

        x = self.bert_encoder(x)  # (B, T, 128)
        return x
