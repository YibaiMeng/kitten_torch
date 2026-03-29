"""
Load ONNX weights into the KittenTorch PyTorch model.

Maps ONNX initializer names → PyTorch module parameter names.
Handles:
  - Per-tensor quantization (int8/uint8 weights with scale/zero_point)
  - float16 weights (cast to float32)
  - LSTM weights in ONNX format → PyTorch format
  - Unnamed ONNX::MatMul_XXXX weights matched by order

ONNX → PyTorch LSTM weight conversion:
  ONNX DynamicQuantizeLSTM stores weights as:
    W_ih_quantized: (num_dir, input_size, 4*H)  — transposed from (num_dir, 4*H, input_size)
    W_hh_quantized: (num_dir, H, 4*H)            — transposed from (num_dir, 4*H, H)
    B: (num_dir, 8*H)                             — [b_ih | b_hh] per direction

  PyTorch LSTM expects:
    weight_ih_l{d}: (4*H, input_size)
    weight_hh_l{d}: (4*H, H)
    bias_ih_l{d}: (4*H,)
    bias_hh_l{d}: (4*H,)
    (for bidirectional: also _l{d}_reverse)
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .weight_loader import ONNXWeights


# ------------------------------------------------------------------ #
#  Dynamic ONNX tensor name discovery                                  #
#                                                                      #
#  ONNX assigns sequential IDs to unnamed nodes (onnx::LSTM_XXXX,     #
#  onnx::MatMul_XXXX). The IDs differ between model versions if the   #
#  graph topology changes, but their *order* (sorted numerically)     #
#  maps deterministically to model components.  Discovering them      #
#  dynamically makes the loader work with both v0.1 (ONNX1) and       #
#  v0.8 (ONNX2) without separate codepaths.                           #
# ------------------------------------------------------------------ #

def _find_lstm_groups(w: ONNXWeights) -> list[int]:
    """
    Return sorted list of LSTM bias group numbers.

    Each LSTM group N has initializers:
      onnx::LSTM_N          — bias (float32, shape (num_dir, 8H))
      onnx::LSTM_{N+1}_quantized — W_ih (int8)
      onnx::LSTM_{N+2}_quantized — W_hh (int8)

    There are 5 groups total (text_encoder, 2 predictor, duration, shared).
    """
    groups = sorted(
        int(m.group(1))
        for name in w.inits
        if (m := re.match(r'^onnx::LSTM_(\d+)$', name))
    )
    if len(groups) != 5:
        raise RuntimeError(
            f"Expected 5 LSTM bias groups, found {len(groups)}: {groups}. "
            "The ONNX model may have a different architecture than expected."
        )
    return groups


def _find_matmul_quantized(w: ONNXWeights) -> list[int]:
    """
    Return sorted list of quantized MatMul group numbers (onnx::MatMul_N_quantized).

    In v0.8-int8 there are 0 quantized MatMul entries (all unnamed MatMuls are float).
    """
    return sorted(
        int(m.group(1))
        for name in w.inits
        if (m := re.match(r'^onnx::MatMul_(\d+)_quantized$', name))
    )


def _find_matmul_fp(w: ONNXWeights) -> list[int]:
    """
    Return sorted list of float16/float32 MatMul initializer numbers.

    These are onnx::MatMul_N tensors that have *no* corresponding _quantized entry.
    In v0.8-int8 there are 10 in order:
      [0]   → bert.embedding_hidden_mapping_in
      [1-4] → BERT attention q/k/v/dense
      [5-6] → BERT ffn/ffn_output
      [7]   → BERT encoder
      [8]   → predictor.duration_proj
      [9]   → generator.sine_gen.l_linear
    """
    all_names = set(w.inits.keys())
    return sorted(
        int(m.group(1))
        for name in all_names
        if (m := re.match(r'^onnx::MatMul_(\d+)$', name))
        and f"onnx::MatMul_{m.group(1)}_quantized" not in all_names
    )


# ------------------------------------------------------------------ #
#  ONNX DynamicQuantizeLinear simulation                               #
# ------------------------------------------------------------------ #

def _onnx_dql(x: torch.Tensor) -> torch.Tensor:
    """
    Simulate ONNX DynamicQuantizeLinear: per-tensor uint8 (asymmetric) quantization.

    ONNX formula:
      scale = (max(x_max, 0) - min(x_min, 0)) / 255
      zero_point = clamp(round(-x_min / scale), 0, 255)
      y = clamp(round(x / scale) + zero_point, 0, 255)
      x_dq = (y - zero_point) * scale
    """
    x_max = float(x.max())
    x_min = float(x.min())
    scale = (max(x_max, 0.0) - min(x_min, 0.0)) / 255.0
    if scale < 1e-8:
        return x
    zp = min(255.0, max(0.0, round(-x_min / scale)))
    x_q = torch.clamp(torch.round(x / scale) + zp, 0.0, 255.0)
    return (x_q - zp) * scale


def _dql_hook(module: nn.Module, args: tuple) -> tuple:
    """Forward pre-hook: fake-quantize first input via ONNX DynamicQuantizeLinear."""
    return (_onnx_dql(args[0]),) + args[1:]


# ------------------------------------------------------------------ #
#  LSTM Weight Conversion                                              #
# ------------------------------------------------------------------ #

def _dequant_lstm_weight(w_q: np.ndarray, scale: np.ndarray, zp: np.ndarray) -> np.ndarray:
    """Per-direction dequantization for LSTM weights."""
    # w_q: (2, A, B), scale: (2,), zp: (2,)
    out = np.zeros_like(w_q, dtype=np.float32)
    for d in range(w_q.shape[0]):
        out[d] = (w_q[d].astype(np.float32) - float(zp[d])) * float(scale[d])
    return out


def _reorder_gates_onnx_to_pytorch(arr: np.ndarray, H: int) -> np.ndarray:
    """
    ONNX LSTM gate order:    [i, o, f, c]  (input, output, forget, cell)
    PyTorch LSTM gate order: [i, f, g, o]  (input, forget, cell, output)

    Reorder rows of weight (4H, D) or bias (4H,) from ONNX to PyTorch gate order.
    """
    i, o, f, c = arr[..., :H, :], arr[..., H:2*H, :], arr[..., 2*H:3*H, :], arr[..., 3*H:, :]
    return np.concatenate([i, f, c, o], axis=-2)


def _reorder_bias_onnx_to_pytorch(b: np.ndarray, H: int) -> np.ndarray:
    """Reorder 1-D bias vector (4H,) from ONNX [i,o,f,c] to PyTorch [i,f,g,o]."""
    i, o, f, c = b[:H], b[H:2*H], b[2*H:3*H], b[3*H:]
    return np.concatenate([i, f, c, o])


def _load_lstm_weights(
    module: nn.LSTM,
    w_ih_q: np.ndarray, w_ih_scale: np.ndarray, w_ih_zp: np.ndarray,
    w_hh_q: np.ndarray, w_hh_scale: np.ndarray, w_hh_zp: np.ndarray,
    bias: np.ndarray,
) -> None:
    """
    Load ONNX-format LSTM weights into PyTorch LSTM module.

    ONNX DynamicQuantizeLSTM stores:
      w_ih: (num_dir, input_size, 4*H)  — transposed from standard (num_dir, 4H, input_size)
      w_hh: (num_dir, H, 4*H)           — transposed from standard (num_dir, 4H, H)
      B:    (num_dir, 8*H)               — [b_ih | b_hh] per direction
      Gate order: [i, o, f, c] (ONNX convention)

    PyTorch LSTM expects:
      weight_ih_l0: (4*H, input_size), gate order [i, f, g, o]
      weight_hh_l0: (4*H, H),          gate order [i, f, g, o]
      bias_ih_l0/bias_hh_l0: (4*H,),   gate order [i, f, g, o]
    """
    w_ih = _dequant_lstm_weight(w_ih_q, w_ih_scale, w_ih_zp)  # (2, in, 4H)
    w_hh = _dequant_lstm_weight(w_hh_q, w_hh_scale, w_hh_zp)  # (2, H, 4H)
    # Transpose: (2, in, 4H) → (2, 4H, in) and (2, H, 4H) → (2, 4H, H)
    w_ih = w_ih.transpose(0, 2, 1)  # (2, 4H, in)
    w_hh = w_hh.transpose(0, 2, 1)  # (2, 4H, H)
    # B: (2, 8H) → split to b_ih (4H) and b_hh (4H) per direction (writable copies)
    H4 = w_ih.shape[1]  # 4*H
    H = H4 // 4
    b_ih = bias[:, :H4].copy()  # (2, 4H)
    b_hh = bias[:, H4:].copy()  # (2, 4H)

    # Reorder gates from ONNX [i,o,f,c] → PyTorch [i,f,g,o]
    for d in range(2):
        w_ih[d] = _reorder_gates_onnx_to_pytorch(w_ih[d:d+1], H)[0]
        w_hh[d] = _reorder_gates_onnx_to_pytorch(w_hh[d:d+1], H)[0]
        b_ih[d] = _reorder_bias_onnx_to_pytorch(b_ih[d], H)
        b_hh[d] = _reorder_bias_onnx_to_pytorch(b_hh[d], H)

    # Forward direction (index 0)
    module.weight_ih_l0.data.copy_(torch.from_numpy(w_ih[0]))
    module.weight_hh_l0.data.copy_(torch.from_numpy(w_hh[0]))
    module.bias_ih_l0.data.copy_(torch.from_numpy(b_ih[0]))
    module.bias_hh_l0.data.copy_(torch.from_numpy(b_hh[0]))

    if module.bidirectional:
        # Reverse direction (index 1)
        module.weight_ih_l0_reverse.data.copy_(torch.from_numpy(w_ih[1]))
        module.weight_hh_l0_reverse.data.copy_(torch.from_numpy(w_hh[1]))
        module.bias_ih_l0_reverse.data.copy_(torch.from_numpy(b_ih[1]))
        module.bias_hh_l0_reverse.data.copy_(torch.from_numpy(b_hh[1]))


def _load_quantized_linear(
    module: nn.Linear,
    weights: ONNXWeights,
    w_name: str,
    b_name: str | None = None,
) -> None:
    """Load a quantized Linear layer. ONNX stores weight as (in, out); Linear expects (out, in)."""
    w = weights.get(w_name).float()  # already dequantized by weights.get()
    # ONNX MatMulInteger: A @ W where W is (in_features, out_features)
    # nn.Linear: W is (out_features, in_features)
    module.weight.data.copy_(w.T)
    if b_name and b_name in weights:
        b = weights.raw(b_name)
        module.bias.data.copy_(torch.from_numpy(b.astype(np.float32)))
    # Simulate ONNX DynamicQuantizeLinear on activations at inference time
    module.register_forward_pre_hook(_dql_hook)


def _load_quantized_conv1d(
    module: nn.Conv1d,
    weights: ONNXWeights,
    w_name: str,
    b_name: str | None = None,
) -> None:
    """Load quantized Conv1d weights. ONNX Conv stores (out, in, k) directly."""
    w = weights.get(w_name).float()
    module.weight.data.copy_(w)
    if b_name and b_name in weights:
        b = weights.raw(b_name)
        module.bias.data.copy_(torch.from_numpy(b.astype(np.float32)))
    # Simulate ONNX DynamicQuantizeLinear on activations at inference time
    module.register_forward_pre_hook(_dql_hook)


def _load_conv1d_auto(
    module: nn.Conv1d,
    weights: ONNXWeights,
    prefix: str,
    b_name: str | None = None,
) -> None:
    """Load Conv1d, auto-detecting quantized (_quantized suffix) vs plain float."""
    if f"{prefix}.weight_quantized" in weights:
        _load_quantized_conv1d(module, weights, f"{prefix}.weight_quantized", b_name)
    else:
        module.weight.data.copy_(weights.get(f"{prefix}.weight"))
        if b_name and b_name in weights:
            module.bias.data.copy_(weights.get(b_name))


def _load_fp16_weight(module: nn.Module, attr: str, weights: ONNXWeights, name: str) -> None:
    """Load float16 initializer into a parameter, converting to float32."""
    arr = weights.raw(name).astype(np.float32)
    getattr(module, attr).data.copy_(torch.from_numpy(arr))


# ------------------------------------------------------------------ #
#  Main weight loading entry point                                     #
# ------------------------------------------------------------------ #

def load_weights(model: nn.Module, onnx_path: str | Path) -> None:
    """
    Load all ONNX weights into the KittenTTSTorch model.

    Args:
        model: KittenTTSTorch instance (with all submodules initialized)
        onnx_path: path to .onnx file
    """
    w = ONNXWeights(onnx_path)

    # Discover unnamed tensor IDs dynamically
    lstm_groups = _find_lstm_groups(w)       # 5 groups sorted
    matmul_fp   = _find_matmul_fp(w)         # 10 groups sorted

    _load_text_encoder(model.text_encoder, w, lstm_groups)
    _load_bert(model.bert, w, matmul_fp)
    _load_predictor(model.predictor, w, lstm_groups, matmul_fp)
    _load_decoder(model.decoder, w)
    _load_generator(model.decoder.generator, w, matmul_fp)

    # Apply dynamic int8 quantization to all LSTMs (matches ONNX DynamicQuantizeLSTM)
    import torch.ao.quantization as tq
    tq.quantize_dynamic(model, {nn.LSTM}, dtype=torch.qint8, inplace=True)


# ------------------------------------------------------------------ #
#  Text Encoder                                                        #
# ------------------------------------------------------------------ #

def _load_text_encoder(te, w: ONNXWeights, lstm_groups: list[int]) -> None:
    prefix = "kmodel.text_encoder"

    # Embedding
    te.embedding.weight.data.copy_(w.get(f"{prefix}.embedding.weight"))

    # CNN layers (v0.8: 2 blocks)
    for i in range(2):
        cp = f"{prefix}.cnn.{i}"
        _load_quantized_conv1d(
            te.cnn[i].conv, w,
            f"{cp}.0.weight_quantized",
            f"{cp}.0.bias",
        )
        # LayerNorm: gamma=weight, beta=bias
        te.cnn[i].norm.weight.data.copy_(w.get(f"{cp}.1.gamma"))
        te.cnn[i].norm.bias.data.copy_(w.get(f"{cp}.1.beta"))

    # BiLSTM — group[0] in ONNX order
    g0 = lstm_groups[0]
    _load_lstm_weights(
        te.lstm,
        w_ih_q=w.raw(f"onnx::LSTM_{g0 + 1}_quantized"),
        w_ih_scale=w.raw(f"onnx::LSTM_{g0 + 1}_scale"),
        w_ih_zp=w.raw(f"onnx::LSTM_{g0 + 1}_zero_point"),
        w_hh_q=w.raw(f"onnx::LSTM_{g0 + 2}_quantized"),
        w_hh_scale=w.raw(f"onnx::LSTM_{g0 + 2}_scale"),
        w_hh_zp=w.raw(f"onnx::LSTM_{g0 + 2}_zero_point"),
        bias=w.raw(f"onnx::LSTM_{g0}"),
    )
    # No text_proj in v0.8: decoder takes 128-dim features directly


# ------------------------------------------------------------------ #
#  ALBERT                                                              #
# ------------------------------------------------------------------ #

def _load_bert(bert, w: ONNXWeights, matmul_fp: list[int]) -> None:
    bp = "kmodel.bert"

    # Embeddings
    bert.embeddings["word_embeddings"].weight.data.copy_(w.get(f"{bp}.embeddings.word_embeddings.weight"))
    bert.embeddings["position_embeddings"].weight.data.copy_(w.get(f"{bp}.embeddings.position_embeddings.weight"))
    bert.embeddings["token_type_embeddings"].weight.data.copy_(w.get(f"{bp}.embeddings.token_type_embeddings.weight"))
    bert.emb_ln.weight.data.copy_(w.get(f"{bp}.embeddings.LayerNorm.weight"))
    bert.emb_ln.bias.data.copy_(w.get(f"{bp}.embeddings.LayerNorm.bias"))

    # embedding_hidden_mapping_in: Linear(128→768) — matmul_fp[0] (float16, stored as (in, out))
    arr = w.raw(f"onnx::MatMul_{matmul_fp[0]}").astype(np.float32)
    bert.embedding_hidden_mapping_in.weight.data.copy_(torch.from_numpy(arr.T))
    bert.embedding_hidden_mapping_in.bias.data.copy_(
        torch.from_numpy(w.raw(f"{bp}.encoder.embedding_hidden_mapping_in.bias").astype(np.float32))
    )

    # Albert shared layer attention weights (float16 in ONNX)
    # matmul_fp[1..4] → query, key, value, dense (each 768×768)
    alp = f"{bp}.encoder.albert_layer_groups.0.albert_layers.0"
    attn = bert.albert_layer.attention

    for param_name, fp_idx in [("query", 1), ("key", 2), ("value", 3), ("dense", 4)]:
        arr = w.raw(f"onnx::MatMul_{matmul_fp[fp_idx]}").astype(np.float32)  # (768, 768)
        getattr(attn, param_name).weight.data.copy_(torch.from_numpy(arr.T))

    for param_name, bias_name in [
        ("query", f"{alp}.attention.query.bias"),
        ("key",   f"{alp}.attention.key.bias"),
        ("value", f"{alp}.attention.value.bias"),
        ("dense", f"{alp}.attention.dense.bias"),
    ]:
        arr = w.raw(bias_name).astype(np.float32)
        getattr(attn, param_name).bias.data.copy_(torch.from_numpy(arr))

    # LayerNorm (attention)
    attn.LayerNorm.weight.data.copy_(w.get(f"{alp}.attention.LayerNorm.weight"))
    attn.LayerNorm.bias.data.copy_(w.get(f"{alp}.attention.LayerNorm.bias"))

    # FFN — matmul_fp[5..6]
    layer = bert.albert_layer
    arr = w.raw(f"onnx::MatMul_{matmul_fp[5]}").astype(np.float32)  # ffn (768, 2048)
    layer.ffn.weight.data.copy_(torch.from_numpy(arr.T))
    layer.ffn.bias.data.copy_(torch.from_numpy(w.raw(f"{alp}.ffn.bias").astype(np.float32)))

    arr = w.raw(f"onnx::MatMul_{matmul_fp[6]}").astype(np.float32)  # ffn_output (2048, 768)
    layer.ffn_output.weight.data.copy_(torch.from_numpy(arr.T))
    layer.ffn_output.bias.data.copy_(torch.from_numpy(w.raw(f"{alp}.ffn_output.bias").astype(np.float32)))

    layer.full_layer_layer_norm.weight.data.copy_(w.get(f"{alp}.full_layer_layer_norm.weight"))
    layer.full_layer_layer_norm.bias.data.copy_(w.get(f"{alp}.full_layer_layer_norm.bias"))

    # bert_encoder: Linear(768→128) — matmul_fp[7]
    arr = w.raw(f"onnx::MatMul_{matmul_fp[7]}").astype(np.float32)  # (768, 128)
    bert.bert_encoder.weight.data.copy_(torch.from_numpy(arr.T))
    bert.bert_encoder.bias.data.copy_(w.get("kmodel.bert_encoder.bias"))


# ------------------------------------------------------------------ #
#  Predictor                                                           #
# ------------------------------------------------------------------ #

def _load_adain1d(adain, w: ONNXWeights, prefix: str) -> None:
    """Load AdaIN1d: norm (weight+bias) + fc (weight+bias).

    Detects int8 (_quantized suffix) vs float weight storage:
    - int8: weight stored as (in, out) → needs transpose via _load_quantized_linear
    - float: weight stored as (out, in) → load directly
    """
    if f"{prefix}.norm.weight" in w:
        adain.norm.weight.data.copy_(w.get(f"{prefix}.norm.weight"))
    if f"{prefix}.norm.bias" in w:
        adain.norm.bias.data.copy_(w.get(f"{prefix}.norm.bias"))
    fc_w_quantized = f"{prefix}.fc.weight_quantized"
    if fc_w_quantized in w:
        _load_quantized_linear(adain.fc, w, fc_w_quantized, f"{prefix}.fc.bias")
    else:
        # float weight stored as (in, out) — transpose to (out, in) for nn.Linear
        arr = w.raw(f"{prefix}.fc.weight").astype(np.float32)
        adain.fc.weight.data.copy_(torch.from_numpy(arr.T))
        adain.fc.bias.data.copy_(torch.from_numpy(w.raw(f"{prefix}.fc.bias").astype(np.float32)))


def _load_pred_resblock(blk, w: ONNXWeights, prefix: str) -> None:
    """PredResBlock: norm1, conv1, norm2, conv2."""
    _load_adain1d(blk.norm1, w, f"{prefix}.norm1")
    _load_conv1d_auto(blk.conv1, w, f"{prefix}.conv1", f"{prefix}.conv1.bias")
    _load_adain1d(blk.norm2, w, f"{prefix}.norm2")
    _load_conv1d_auto(blk.conv2, w, f"{prefix}.conv2", f"{prefix}.conv2.bias")


def _load_pred_upsample_block(blk, w: ONNXWeights, prefix: str) -> None:
    """PredUpsampleBlock: pool, conv1x1, norm1, conv1, norm2, conv2."""
    # pool: float ConvTranspose1d
    pool_w = w.raw(f"{prefix}.pool.weight").astype(np.float32)  # (128, 1, 3)
    # ConvTranspose1d weight layout is (in_ch, out_ch/groups, k) for grouped
    # For depthwise (groups=in_ch): weight is (in_ch, 1, k)
    blk.pool.weight.data.copy_(torch.from_numpy(pool_w))
    blk.pool.bias.data.copy_(torch.from_numpy(w.raw(f"{prefix}.pool.bias").astype(np.float32)))

    _load_conv1d_auto(blk.conv1x1, w, f"{prefix}.conv1x1")
    _load_adain1d(blk.norm1, w, f"{prefix}.norm1")
    _load_conv1d_auto(blk.conv1, w, f"{prefix}.conv1", f"{prefix}.conv1.bias")
    _load_adain1d(blk.norm2, w, f"{prefix}.norm2")
    _load_conv1d_auto(blk.conv2, w, f"{prefix}.conv2", f"{prefix}.conv2.bias")


def _load_predictor(predictor, w: ONNXWeights, lstm_groups: list[int], matmul_fp: list[int]) -> None:
    # LSTM group assignments (by sorted order, v0.8-int8):
    #   lstm_groups[0]   → text_encoder BiLSTM (loaded in _load_text_encoder)
    #   lstm_groups[1-2] → predictor.text_encoder lstms[0,2] (2 BiLSTMs)
    #   lstm_groups[3]   → predictor.lstm (duration)
    #   lstm_groups[4]   → predictor.shared
    pp = "kmodel.predictor"

    # text_encoder: 2 BiLSTMs at lstms[0,2] + FCs at lstms[1,3]
    for layer_idx in range(2):
        g = lstm_groups[1 + layer_idx]
        lstm_idx = layer_idx * 2  # 0, 2
        fc_idx = lstm_idx + 1     # 1, 3

        _load_lstm_weights(
            predictor.text_encoder.lstms[lstm_idx],
            w_ih_q=w.raw(f"onnx::LSTM_{g + 1}_quantized"),
            w_ih_scale=w.raw(f"onnx::LSTM_{g + 1}_scale"),
            w_ih_zp=w.raw(f"onnx::LSTM_{g + 1}_zero_point"),
            w_hh_q=w.raw(f"onnx::LSTM_{g + 2}_quantized"),
            w_hh_scale=w.raw(f"onnx::LSTM_{g + 2}_scale"),
            w_hh_zp=w.raw(f"onnx::LSTM_{g + 2}_zero_point"),
            bias=w.raw(f"onnx::LSTM_{g}"),
        )

        fc_w_name = f"{pp}.text_encoder.lstms.{fc_idx}.fc.weight_quantized"
        if fc_w_name not in w:
            fc_w_name = f"{pp}.text_encoder.lstms.{fc_idx}.fc.weight"
        _load_quantized_linear(
            predictor.text_encoder.lstms[fc_idx], w,
            fc_w_name,
            f"{pp}.text_encoder.lstms.{fc_idx}.fc.bias",
        )

    # Duration LSTM — lstm_groups[3]
    g3 = lstm_groups[3]
    _load_lstm_weights(
        predictor.lstm,
        w_ih_q=w.raw(f"onnx::LSTM_{g3 + 1}_quantized"),
        w_ih_scale=w.raw(f"onnx::LSTM_{g3 + 1}_scale"),
        w_ih_zp=w.raw(f"onnx::LSTM_{g3 + 1}_zero_point"),
        w_hh_q=w.raw(f"onnx::LSTM_{g3 + 2}_quantized"),
        w_hh_scale=w.raw(f"onnx::LSTM_{g3 + 2}_scale"),
        w_hh_zp=w.raw(f"onnx::LSTM_{g3 + 2}_zero_point"),
        bias=w.raw(f"onnx::LSTM_{g3}"),
    )

    # Duration projection: Linear(128, 50) — matmul_fp[8]
    arr = w.raw(f"onnx::MatMul_{matmul_fp[8]}").astype(np.float32)
    predictor.duration_proj.weight.data.copy_(torch.from_numpy(arr.T))
    predictor.duration_proj.bias.data.copy_(w.get(f"{pp}.duration_proj.linear_layer.bias"))

    # Shared LSTM — lstm_groups[4]
    g4 = lstm_groups[4]
    _load_lstm_weights(
        predictor.shared,
        w_ih_q=w.raw(f"onnx::LSTM_{g4 + 1}_quantized"),
        w_ih_scale=w.raw(f"onnx::LSTM_{g4 + 1}_scale"),
        w_ih_zp=w.raw(f"onnx::LSTM_{g4 + 1}_zero_point"),
        w_hh_q=w.raw(f"onnx::LSTM_{g4 + 2}_quantized"),
        w_hh_scale=w.raw(f"onnx::LSTM_{g4 + 2}_scale"),
        w_hh_zp=w.raw(f"onnx::LSTM_{g4 + 2}_zero_point"),
        bias=w.raw(f"onnx::LSTM_{g4}"),
    )

    # F0 ResBlocks
    for i, blk in enumerate(predictor.F0):
        fp = f"{pp}.F0.{i}"
        if i == 1:
            _load_pred_upsample_block(blk, w, fp)
        else:
            _load_pred_resblock(blk, w, fp)
    predictor.F0_proj.weight.data.copy_(w.get(f"{pp}.F0_proj.weight"))
    predictor.F0_proj.bias.data.copy_(w.get(f"{pp}.F0_proj.bias"))

    # N ResBlocks
    for i, blk in enumerate(predictor.N):
        np_ = f"{pp}.N.{i}"
        if i == 1:
            _load_pred_upsample_block(blk, w, np_)
        else:
            _load_pred_resblock(blk, w, np_)
    predictor.N_proj.weight.data.copy_(w.get(f"{pp}.N_proj.weight"))
    predictor.N_proj.bias.data.copy_(w.get(f"{pp}.N_proj.bias"))

    # F0_conv/N_conv are in the decoder, not the predictor (ONNX: decoder.decoder.F0_conv)


# ------------------------------------------------------------------ #
#  Decoder                                                             #
# ------------------------------------------------------------------ #

def _load_adain_resblock(blk, w: ONNXWeights, prefix: str) -> None:
    """AdaINResBlock: conv1x1, norm1, conv1, norm2, conv2 (+ pool for upsample variant)."""
    _load_quantized_conv1d(blk.conv1x1, w, f"{prefix}.conv1x1.weight_quantized")
    _load_adain1d(blk.norm1, w, f"{prefix}.norm1")
    _load_quantized_conv1d(blk.conv1, w, f"{prefix}.conv1.weight_quantized", f"{prefix}.conv1.bias")
    _load_adain1d(blk.norm2, w, f"{prefix}.norm2")
    _load_quantized_conv1d(blk.conv2, w, f"{prefix}.conv2.weight_quantized", f"{prefix}.conv2.bias")
    if blk.upsample:
        pool_w = w.raw(f"{prefix}.pool.weight").astype(np.float32)
        blk.pool.weight.data.copy_(torch.from_numpy(pool_w))
        blk.pool.bias.data.copy_(torch.from_numpy(w.raw(f"{prefix}.pool.bias").astype(np.float32)))


def _load_decoder(decoder, w: ONNXWeights) -> None:
    dp = "kmodel.decoder"

    # asr_res: Conv1d(128, 64, k=1)
    _load_quantized_conv1d(decoder.asr_res, w,
                           f"{dp}.asr_res.0.weight_quantized")
    decoder.asr_res.bias.data.copy_(w.get(f"{dp}.asr_res.0.bias"))

    # F0_conv/N_conv: stride-2 conv downsample 2T→T (ONNX: decoder.decoder.F0_conv)
    decoder.F0_conv.weight.data.copy_(w.get(f"{dp}.F0_conv.weight"))
    decoder.F0_conv.bias.data.copy_(w.get(f"{dp}.F0_conv.bias"))
    decoder.N_conv.weight.data.copy_(w.get(f"{dp}.N_conv.weight"))
    decoder.N_conv.bias.data.copy_(w.get(f"{dp}.N_conv.bias"))

    # encode block
    _load_adain_resblock(decoder.encode, w, f"{dp}.encode")

    # decode blocks
    for i in range(4):
        _load_adain_resblock(decoder.decode[i], w, f"{dp}.decode.{i}")


# ------------------------------------------------------------------ #
#  Generator                                                           #
# ------------------------------------------------------------------ #

def _load_gen_resblock(blk, w: ONNXWeights, prefix: str) -> None:
    """Load GenResBlock (3 dilations of AdaIN + conv pairs)."""
    for i in range(3):
        _load_adain1d(blk.adain1[i], w, f"{prefix}.adain1.{i}")
        _load_adain1d(blk.adain2[i], w, f"{prefix}.adain2.{i}")
        blk.alpha1[i].data.copy_(w.get(f"{prefix}.alpha1.{i}"))
        blk.alpha2[i].data.copy_(w.get(f"{prefix}.alpha2.{i}"))
        _load_quantized_conv1d(blk.convs1[i], w, f"{prefix}.convs1.{i}.weight_quantized",
                               f"{prefix}.convs1.{i}.bias")
        _load_quantized_conv1d(blk.convs2[i], w, f"{prefix}.convs2.{i}.weight_quantized",
                               f"{prefix}.convs2.{i}.bias")


def _load_generator(gen, w: ONNXWeights, matmul_fp: list[int]) -> None:
    gp = "kmodel.decoder.generator"

    # SineGenerator linear: float32 (9, 1) — matmul_fp[9] (last in order)
    gen.sine_gen.l_linear.weight.data.copy_(
        torch.from_numpy(w.raw(f"onnx::MatMul_{matmul_fp[9]}").astype(np.float32).T)
    )
    # Sine linear bias might not exist; check
    # m_source.l_linear.bias: (1,)
    if f"{gp}.m_source.l_linear.bias" in w:
        gen.sine_gen.l_linear.bias.data.copy_(w.get(f"{gp}.m_source.l_linear.bias"))

    # ups (float16 ConvTranspose1d)
    for i, ups_mod in enumerate(gen.ups):
        arr_w = w.raw(f"{gp}.ups.{i}.weight").astype(np.float32)
        arr_b = w.raw(f"{gp}.ups.{i}.bias").astype(np.float32)
        # ConvTranspose1d weight shape: (in_ch, out_ch, k) in ONNX, (in_ch, out_ch/groups, k) in PyTorch
        ups_mod.weight.data.copy_(torch.from_numpy(arr_w))
        ups_mod.bias.data.copy_(torch.from_numpy(arr_b))

    # noise_convs
    for i, nc in enumerate(gen.noise_convs):
        _load_quantized_conv1d(nc, w, f"{gp}.noise_convs.{i}.weight_quantized",
                               f"{gp}.noise_convs.{i}.bias")

    # noise_res (AdaIN resblocks)
    for i, nrb in enumerate(gen.noise_res):
        _load_gen_resblock(nrb, w, f"{gp}.noise_res.{i}")

    # resblocks
    for i, rb in enumerate(gen.resblocks):
        _load_gen_resblock(rb, w, f"{gp}.resblocks.{i}")

    # conv_post
    _load_conv1d_auto(gen.conv_post, w, f"{gp}.conv_post", f"{gp}.conv_post.bias")

    # STFT filters
    for attr_name in ["weight_forward_real", "weight_forward_imag",
                      "weight_backward_real", "weight_backward_imag"]:
        arr = w.raw(f"{gp}.stft.{attr_name}").astype(np.float32)
        getattr(gen.stft, attr_name).data.copy_(torch.from_numpy(arr))
