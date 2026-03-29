"""
ONNX weight extraction and dequantization for KittenTTS nano.

The ONNX model uses dynamic quantization:
  - uint8 for Conv weights (DynamicQuantizeLinear + ConvInteger)
  - int8  for Linear/GEMM weights (MatMulInteger)
  - float32 for biases and scale parameters

Dequantization: fp32 = (quant - zero_point) * scale
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from onnx import numpy_helper


def _load_initializers(model_path: str | Path) -> dict[str, np.ndarray]:
    """Return all ONNX initializers as a name→ndarray dict."""
    model = onnx.load(str(model_path))
    return {init.name: numpy_helper.to_array(init) for init in model.graph.initializer}


def dequantize(
    quant: np.ndarray,
    scale: np.ndarray,
    zero_point: np.ndarray,
) -> np.ndarray:
    """Dequantize integer weights to float32."""
    return (quant.astype(np.float32) - zero_point.astype(np.float32)) * scale.astype(np.float32)


class ONNXWeights:
    """
    Loads and dequantizes all weights from the KittenTTS ONNX model.

    Usage:
        w = ONNXWeights(model_path)
        # raw (possibly quantized) initializer
        arr = w.raw("some/onnx/name")
        # fp32 dequantized weight
        t = w.get("some/onnx/name")
        # map of all initializers
        w.inits  # dict[str, np.ndarray]
    """

    def __init__(self, model_path: str | Path):
        self.inits = _load_initializers(model_path)
        # Build reverse lookup: base_name → (scale_name, zp_name) if quantized
        self._quant_map = self._build_quant_map()

    def _build_quant_map(self) -> dict[str, tuple[str, str]]:
        """
        For each weight tensor W, if W_scale and W_zero_point exist in initializers,
        record the mapping so get() can dequantize automatically.

        Handles two naming conventions:
          1. Exact: base = W, scale = W_scale, zp = W_zero_point
          2. Quantized suffix: W_quantized stored separately from W_scale / W_zero_point
             (KittenTTS pattern: weight_quantized + weight_scale + weight_zero_point)
        """
        qmap: dict[str, tuple[str, str]] = {}
        for name in self.inits:
            if name.endswith("_scale"):
                base = name[: -len("_scale")]
                zp_name = base + "_zero_point"
                # Convention 1: base tensor exists directly
                if base in self.inits and zp_name in self.inits:
                    qmap[base] = (name, zp_name)
                # Convention 2: base + "_quantized" tensor exists
                q_name = base + "_quantized"
                if q_name in self.inits and zp_name in self.inits:
                    qmap[q_name] = (name, zp_name)
        return qmap

    def raw(self, name: str) -> np.ndarray:
        if name not in self.inits:
            raise KeyError(f"Initializer not found: {name!r}. "
                           f"Available (sample): {list(self.inits)[:5]}")
        return self.inits[name]

    def get(self, name: str) -> torch.Tensor:
        """Return fp32 torch.Tensor, dequantizing if necessary."""
        arr = self.raw(name)
        if name in self._quant_map:
            scale_name, zp_name = self._quant_map[name]
            arr = dequantize(arr, self.inits[scale_name], self.inits[zp_name])
        return torch.from_numpy(arr.copy()).float()

    def get_np(self, name: str) -> np.ndarray:
        """Like get() but returns numpy array."""
        return self.get(name).numpy()

    # ------------------------------------------------------------------ #
    # Convenience helpers for common ONNX quantized patterns              #
    # ------------------------------------------------------------------ #

    def load_linear(
        self,
        weight_name: str,
        bias_name: str | None = None,
        transpose: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        Load a Linear layer's weight (and optional bias).
        ONNX GEMM uses (N, K) layout; nn.Linear expects (out, in) = same layout.
        `transpose=True` means the ONNX weight is already (out, in) after transposing.
        Actually ONNX MatMul stores (in_features, out_features) so we transpose by default.
        """
        w = self.get(weight_name)
        if transpose:
            w = w.T.contiguous()  # (in, out) → (out, in)
        b = self.get(bias_name) if bias_name and bias_name in self.inits else None
        return w, b

    def load_conv1d(
        self,
        weight_name: str,
        bias_name: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Load Conv1d weight (out_ch, in_ch, kernel) and bias."""
        w = self.get(weight_name)
        b = self.get(bias_name) if bias_name and bias_name in self.inits else None
        return w, b

    def names_matching(self, pattern: str) -> list[str]:
        """Return sorted initializer names matching a regex pattern."""
        rx = re.compile(pattern)
        return sorted(n for n in self.inits if rx.search(n))

    def __contains__(self, name: str) -> bool:
        return name in self.inits
