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

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .weight_loader import ONNXWeights


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

    _load_text_encoder(model.text_encoder, w)
    _load_bert(model.bert, w)
    _load_predictor(model.predictor, w)
    _load_decoder(model.decoder, w)
    _load_generator(model.decoder.generator, w)

    # Apply dynamic int8 quantization to all LSTMs (matches ONNX DynamicQuantizeLSTM)
    import torch.ao.quantization as tq
    tq.quantize_dynamic(model, {nn.LSTM}, dtype=torch.qint8, inplace=True)


# ------------------------------------------------------------------ #
#  Text Encoder                                                        #
# ------------------------------------------------------------------ #

def _load_text_encoder(te, w: ONNXWeights) -> None:
    prefix = "kmodel.text_encoder"

    # Embedding
    te.embedding.weight.data.copy_(w.get(f"{prefix}.embedding.weight"))

    # CNN layers
    for i in range(6):
        cp = f"{prefix}.cnn.{i}"
        _load_quantized_conv1d(
            te.cnn[i].conv, w,
            f"{cp}.0.weight_quantized",
            f"{cp}.0.bias",
        )
        # LayerNorm: gamma=weight, beta=bias
        te.cnn[i].norm.weight.data.copy_(w.get(f"{cp}.1.gamma"))
        te.cnn[i].norm.bias.data.copy_(w.get(f"{cp}.1.beta"))

    # BiLSTM (LSTM group 7589)
    _load_lstm_weights(
        te.lstm,
        w_ih_q=w.raw("onnx::LSTM_7590_quantized"),
        w_ih_scale=w.raw("onnx::LSTM_7590_scale"),
        w_ih_zp=w.raw("onnx::LSTM_7590_zero_point"),
        w_hh_q=w.raw("onnx::LSTM_7591_quantized"),
        w_hh_scale=w.raw("onnx::LSTM_7591_scale"),
        w_hh_zp=w.raw("onnx::LSTM_7591_zero_point"),
        bias=w.raw("onnx::LSTM_7589"),
    )

    # text_proj: Linear(128, 512)
    # Weight: onnx::MatMul_7598_quantized (128, 512)
    _load_quantized_linear(
        te.text_proj, w,
        "onnx::MatMul_7598_quantized",
        f"{prefix}.text_proj.bias",
    )


# ------------------------------------------------------------------ #
#  ALBERT                                                              #
# ------------------------------------------------------------------ #

def _load_bert(bert, w: ONNXWeights) -> None:
    bp = "kmodel.bert"

    # Embeddings
    bert.embeddings["word_embeddings"].weight.data.copy_(w.get(f"{bp}.embeddings.word_embeddings.weight"))
    bert.embeddings["position_embeddings"].weight.data.copy_(w.get(f"{bp}.embeddings.position_embeddings.weight"))
    bert.embeddings["token_type_embeddings"].weight.data.copy_(w.get(f"{bp}.embeddings.token_type_embeddings.weight"))
    bert.emb_ln.weight.data.copy_(w.get(f"{bp}.embeddings.LayerNorm.weight"))
    bert.emb_ln.bias.data.copy_(w.get(f"{bp}.embeddings.LayerNorm.bias"))

    # embedding_hidden_mapping_in: Linear(128→768)
    # Weight: onnx::MatMul_7606_quantized (128, 768)
    _load_quantized_linear(
        bert.embedding_hidden_mapping_in, w,
        "onnx::MatMul_7606_quantized",
        f"{bp}.encoder.embedding_hidden_mapping_in.bias",
    )

    # Albert shared layer attention weights (float16 in ONNX)
    alp = f"{bp}.encoder.albert_layer_groups.0.albert_layers.0"
    attn = bert.albert_layer.attention

    # query/key/value/dense: float16 (768, 768) stored as (in, out)
    for param_name, onnx_name in [
        ("query", "onnx::MatMul_7607"),
        ("key", "onnx::MatMul_7610"),
        ("value", "onnx::MatMul_7613"),
        ("dense", "onnx::MatMul_7617"),
    ]:
        arr = w.raw(onnx_name).astype(np.float32)  # (768, 768)
        getattr(attn, param_name).weight.data.copy_(torch.from_numpy(arr.T))  # (out, in) = (768, 768)

    for param_name, bias_name in [
        ("query", f"{alp}.attention.query.bias"),
        ("key", f"{alp}.attention.key.bias"),
        ("value", f"{alp}.attention.value.bias"),
        ("dense", f"{alp}.attention.dense.bias"),
    ]:
        arr = w.raw(bias_name).astype(np.float32)
        getattr(attn, param_name).bias.data.copy_(torch.from_numpy(arr))

    # LayerNorm (attention)
    attn.LayerNorm.weight.data.copy_(w.get(f"{alp}.attention.LayerNorm.weight"))
    attn.LayerNorm.bias.data.copy_(w.get(f"{alp}.attention.LayerNorm.bias"))

    # FFN
    layer = bert.albert_layer
    # ffn: float16 (768, 2048) as (in, out)
    arr = w.raw("onnx::MatMul_7618").astype(np.float32)  # (768, 2048)
    layer.ffn.weight.data.copy_(torch.from_numpy(arr.T))  # (2048, 768)
    layer.ffn.bias.data.copy_(torch.from_numpy(w.raw(f"{alp}.ffn.bias").astype(np.float32)))

    # ffn_output: (2048, 768) as (in, out)
    arr = w.raw("onnx::MatMul_7619").astype(np.float32)  # (2048, 768)
    layer.ffn_output.weight.data.copy_(torch.from_numpy(arr.T))  # (768, 2048)
    layer.ffn_output.bias.data.copy_(torch.from_numpy(w.raw(f"{alp}.ffn_output.bias").astype(np.float32)))

    # full_layer_layer_norm
    layer.full_layer_layer_norm.weight.data.copy_(w.get(f"{alp}.full_layer_layer_norm.weight"))
    layer.full_layer_layer_norm.bias.data.copy_(w.get(f"{alp}.full_layer_layer_norm.bias"))

    # bert_encoder: Linear(768→128) — float16 (768, 128) as (in, out)
    arr = w.raw("onnx::MatMul_7763").astype(np.float32)  # (768, 128)
    bert.bert_encoder.weight.data.copy_(torch.from_numpy(arr.T))  # (128, 768)
    bert.bert_encoder.bias.data.copy_(w.get("kmodel.bert_encoder.bias"))


# ------------------------------------------------------------------ #
#  Predictor                                                           #
# ------------------------------------------------------------------ #

def _load_adain1d(adain, w: ONNXWeights, prefix: str) -> None:
    """Load AdaIN1d: norm (weight+bias) + fc (weight+bias)."""
    adain.norm.weight.data.copy_(w.get(f"{prefix}.norm.weight"))
    adain.norm.bias.data.copy_(w.get(f"{prefix}.norm.bias"))
    _load_quantized_linear(adain.fc, w, f"{prefix}.fc.weight_quantized", f"{prefix}.fc.bias")


def _load_pred_resblock(blk, w: ONNXWeights, prefix: str) -> None:
    """PredResBlock: norm1, conv1, norm2, conv2."""
    _load_adain1d(blk.norm1, w, f"{prefix}.norm1")
    _load_quantized_conv1d(blk.conv1, w, f"{prefix}.conv1.weight_quantized", f"{prefix}.conv1.bias")
    _load_adain1d(blk.norm2, w, f"{prefix}.norm2")
    _load_quantized_conv1d(blk.conv2, w, f"{prefix}.conv2.weight_quantized", f"{prefix}.conv2.bias")


def _load_pred_upsample_block(blk, w: ONNXWeights, prefix: str) -> None:
    """PredUpsampleBlock: pool, conv1x1, norm1, conv1, norm2, conv2."""
    # pool: float16 ConvTranspose1d
    pool_w = w.raw(f"{prefix}.pool.weight").astype(np.float32)  # (128, 1, 3)
    # ConvTranspose1d weight layout is (in_ch, out_ch/groups, k) for grouped
    # For depthwise (groups=in_ch): weight is (in_ch, 1, k)
    blk.pool.weight.data.copy_(torch.from_numpy(pool_w))
    blk.pool.bias.data.copy_(torch.from_numpy(w.raw(f"{prefix}.pool.bias").astype(np.float32)))

    _load_quantized_conv1d(blk.conv1x1, w, f"{prefix}.conv1x1.weight_quantized")
    _load_adain1d(blk.norm1, w, f"{prefix}.norm1")
    _load_quantized_conv1d(blk.conv1, w, f"{prefix}.conv1.weight_quantized", f"{prefix}.conv1.bias")
    _load_adain1d(blk.norm2, w, f"{prefix}.norm2")
    _load_quantized_conv1d(blk.conv2, w, f"{prefix}.conv2.weight_quantized", f"{prefix}.conv2.bias")


# LSTM group assignments (in ONNX order):
# 7589 → text_encoder BiLSTM (already loaded)
# 7816 → predictor.text_encoder lstms.0 (first pair, index 0)
# 7872 → predictor.text_encoder lstms.2
# 7928 → predictor.text_encoder lstms.4
# 7984 → predictor.text_encoder lstms.6
# 8040 → predictor.text_encoder lstms.8
# 8096 → predictor.text_encoder lstms.10
# 8151 → predictor.lstm (duration)
# 8212 → predictor.shared

_PRED_LSTM_GROUPS = [
    ("7816", "7817", "7818"),  # lstms.0
    ("7872", "7873", "7874"),  # lstms.2
    ("7928", "7929", "7930"),  # lstms.4
    ("7984", "7985", "7986"),  # lstms.6
    ("8040", "8041", "8042"),  # lstms.8
    ("8096", "8097", "8098"),  # lstms.10
]


def _load_predictor(predictor, w: ONNXWeights) -> None:
    pp = "kmodel.predictor"

    # text_encoder: 6 BiLSTMs at lstms[0,2,4,6,8,10] + FCs at lstms[1,3,5,7,9,11]
    for layer_idx, (b_group, wih_group, whh_group) in enumerate(_PRED_LSTM_GROUPS):
        lstm_idx = layer_idx * 2  # 0, 2, 4, 6, 8, 10
        fc_idx = lstm_idx + 1     # 1, 3, 5, 7, 9, 11
        lstm_mod = predictor.text_encoder.lstms[lstm_idx]
        fc_mod = predictor.text_encoder.lstms[fc_idx]

        _load_lstm_weights(
            lstm_mod,
            w_ih_q=w.raw(f"onnx::LSTM_{wih_group}_quantized"),
            w_ih_scale=w.raw(f"onnx::LSTM_{wih_group}_scale"),
            w_ih_zp=w.raw(f"onnx::LSTM_{wih_group}_zero_point"),
            w_hh_q=w.raw(f"onnx::LSTM_{whh_group}_quantized"),
            w_hh_scale=w.raw(f"onnx::LSTM_{whh_group}_scale"),
            w_hh_zp=w.raw(f"onnx::LSTM_{whh_group}_zero_point"),
            bias=w.raw(f"onnx::LSTM_{b_group}"),
        )

        # FC: lstms.{fc_idx}.fc
        fc_w_name = f"{pp}.text_encoder.lstms.{fc_idx}.fc.weight_quantized"
        if fc_w_name not in w:
            # lstms.11 is float16
            fc_w_name = f"{pp}.text_encoder.lstms.{fc_idx}.fc.weight"
        _load_quantized_linear(
            fc_mod, w,
            fc_w_name,
            f"{pp}.text_encoder.lstms.{fc_idx}.fc.bias",
        )

    # Duration LSTM (group 8151)
    _load_lstm_weights(
        predictor.lstm,
        w_ih_q=w.raw("onnx::LSTM_8152_quantized"),
        w_ih_scale=w.raw("onnx::LSTM_8152_scale"),
        w_ih_zp=w.raw("onnx::LSTM_8152_zero_point"),
        w_hh_q=w.raw("onnx::LSTM_8153_quantized"),
        w_hh_scale=w.raw("onnx::LSTM_8153_scale"),
        w_hh_zp=w.raw("onnx::LSTM_8153_zero_point"),
        bias=w.raw("onnx::LSTM_8151"),
    )

    # Duration projection: Linear(128, 50), weight is float16 (128, 50)
    arr = w.raw("onnx::MatMul_8154").astype(np.float32)  # (128, 50)
    predictor.duration_proj.weight.data.copy_(torch.from_numpy(arr.T))  # (50, 128)
    predictor.duration_proj.bias.data.copy_(w.get(f"{pp}.duration_proj.linear_layer.bias"))

    # Shared LSTM (group 8212)
    _load_lstm_weights(
        predictor.shared,
        w_ih_q=w.raw("onnx::LSTM_8213_quantized"),
        w_ih_scale=w.raw("onnx::LSTM_8213_scale"),
        w_ih_zp=w.raw("onnx::LSTM_8213_zero_point"),
        w_hh_q=w.raw("onnx::LSTM_8214_quantized"),
        w_hh_scale=w.raw("onnx::LSTM_8214_scale"),
        w_hh_zp=w.raw("onnx::LSTM_8214_zero_point"),
        bias=w.raw("onnx::LSTM_8212"),
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
    dp = "kmodel.decoder.decoder"

    # asr_res: Conv1d(512, 64, k=1)
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


def _load_generator(gen, w: ONNXWeights) -> None:
    gp = "kmodel.decoder.decoder.generator"

    # SineGenerator linear: float32 (9, 1)
    gen.sine_gen.l_linear.weight.data.copy_(
        torch.from_numpy(w.raw("onnx::MatMul_8321").astype(np.float32).T)  # (1, 9)
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

    # conv_post (float16)
    arr_w = w.raw(f"{gp}.conv_post.weight").astype(np.float32)
    arr_b = w.raw(f"{gp}.conv_post.bias").astype(np.float32)
    gen.conv_post.weight.data.copy_(torch.from_numpy(arr_w))
    gen.conv_post.bias.data.copy_(torch.from_numpy(arr_b))

    # STFT filters
    for attr_name in ["weight_forward_real", "weight_forward_imag",
                      "weight_backward_real", "weight_backward_imag"]:
        arr = w.raw(f"{gp}.stft.{attr_name}").astype(np.float32)
        getattr(gen.stft, attr_name).data.copy_(torch.from_numpy(arr))
