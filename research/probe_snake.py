"""
Compare ONNX vs PT for resblock.0 internals.

From probe_resblock2.py analysis:
  ONNX h (convs2.0 output) = -0.0903
  PT h (conv2_0 output)    = -0.2488
  → ~2.75x more negative in PT

This probe finds ONNX snake2.0 output (input to convs2.0).
If snake2.0 differs → bug in snake or upstream
If snake2.0 same → bug in conv2.0 (DQL or weights)
"""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import re

HF = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF}/kitten_tts_nano_v0_1.onnx"
TEXT = "Hello, how are you today?"
VOICE = "expr-voice-2-m"

from kitten_torch import KittenTTS
tts = KittenTTS(model_path=ONNX_PATH, voices_path=f"{HF}/voices.npz")
model = tts._model; model.eval()
voices_data = np.load(f"{HF}/voices.npz")
style_np = voices_data[VOICE][0:1].astype(np.float32)
style_pt = torch.from_numpy(style_np)
ids = tts._phonemize(TEXT)
input_ids = torch.tensor([ids], dtype=torch.long)

import kittentts as _kt, inspect
src = inspect.getsource(_kt.KittenTTS.__init__)
m = re.search(r"enumerate\(list\('(.+?)'\)\)", src, re.DOTALL)
WORD_INDEX = {s: i for i, s in enumerate(list(m.group(1)))}
phonemized = tts._phonemizer.phonemize([TEXT])[0]
normalized = ' '.join(re.findall(r'\w+|[^\w\s]', phonemized))
onnx_ids = [0] + [WORD_INDEX[c] for c in normalized if c in WORD_INDEX] + [0]
input_ids_np = np.array([onnx_ids], dtype=np.int64)
speed_np = np.array([1.0], dtype=np.float32)

# ── Find ONNX nodes in resblock.0 ─────────────────────────────────────────────
model_onnx = onnx.load(ONNX_PATH)

# Collect all nodes in resblock.0
rb0_nodes = [n for n in model_onnx.graph.node
             if "/decoder/generator/resblocks.0/" in n.name]

# Print ALL node names in resblock.0
# Key nodes from the structure analysis:
# snake1.0 output: /decoder/generator/resblocks.0/Add
# snake2.0 output: /decoder/generator/resblocks.0/Add_1
# DQL of snake2.0: /decoder/generator/resblocks.0/Add_1_output_0_QuantizeLinear
#   → quantized: Add_1_output_0_quantized, scale: Add_1_output_0_scale, zp: Add_1_output_0_zero_point
# convs2.0 output: /decoder/generator/resblocks.0/convs2.0/Conv_output_0_bias_add
# x_iter0 (residual add): /decoder/generator/resblocks.0/Add_2

# Also: x_into_resblocks = input consumed by iter0 adain1.0 InstanceNorm
# The InstanceNorm node:
for n in rb0_nodes:
    if n.op_type == "InstanceNormalization" and "adain1.0" in n.name:
        print(f"adain1.0 InstanceNorm inputs: {list(n.input)[:3]}")
        break

probe_targets = [
    ("snake1_0",   "/decoder/generator/resblocks.0/Add_output_0"),
    ("snake2_0",   "/decoder/generator/resblocks.0/Add_1_output_0"),
    ("convs1_0",   "/decoder/generator/resblocks.0/convs1.0/Conv_output_0"),
    ("convs2_0",   "/decoder/generator/resblocks.0/convs2.0/Conv_output_0"),
    ("x_after_0",  "/decoder/generator/resblocks.0/Add_2_output_0"),
    ("snake1_1",   "/decoder/generator/resblocks.0/Add_3_output_0"),
    ("snake2_1",   "/decoder/generator/resblocks.0/Add_4_output_0"),
    ("convs2_1",   "/decoder/generator/resblocks.0/convs2.1/Conv_output_0"),
    ("x_after_1",  "/decoder/generator/resblocks.0/Add_5_output_0"),
    ("snake2_2",   "/decoder/generator/resblocks.0/Add_6_output_0"),
    ("convs2_2",   "/decoder/generator/resblocks.0/convs2.2/Conv_output_0"),
    ("x_after_2",  "/decoder/generator/resblocks.0/Add_8_output_0"),
]

model_onnx2 = onnx.load(ONNX_PATH)
seen = set()
final_probes = []
for label, name in probe_targets:
    if name in seen:
        continue
    # Check element type
    elem_type = TensorProto.FLOAT
    for vi in model_onnx2.graph.value_info:
        if vi.name == name:
            elem_type = vi.type.tensor_type.elem_type
            break
    seen.add(name)
    final_probes.append((label, name, elem_type))
    vi_out = helper.make_tensor_value_info(name, elem_type, None)
    model_onnx2.graph.output.append(vi_out)

sess = ort.InferenceSession(model_onnx2.SerializeToString(), providers=["CPUExecutionProvider"])
outs = sess.run(None, {"input_ids": input_ids_np, "style": style_np, "speed": speed_np})

print("\n=== ONNX resblock.0 internals ===")
onnx_vals = {}
for i, (label, name, _) in enumerate(final_probes):
    val = outs[2 + i].astype(np.float32)  # skip audio(0) + durations(1)
    onnx_vals[label] = val
    print(f"  {label}: shape={tuple(val.shape)} mean={val.mean():.4f} std={val.std():.4f} max={np.abs(val).max():.4f}")

# ── PT: capture same intermediates ────────────────────────────────────────────
import types
pt_detail = {}
gen = model.decoder.generator

def make_detail_rb(rb):
    orig = rb.forward.__func__
    def verbose_fwd(self, x, s):
        pt_detail["x_input"] = x.detach().clone().float()
        for i in range(3):
            h = self.adain1[i](x, s)
            h = self.snake(h, self.alpha1[i])
            pt_detail[f"snake1_{i}"] = h.detach().clone().float()
            h = self.convs1[i](h)
            pt_detail[f"convs1_{i}"] = h.detach().clone().float()
            h = self.adain2[i](h, s)
            h = self.snake(h, self.alpha2[i])
            pt_detail[f"snake2_{i}"] = h.detach().clone().float()
            h = self.convs2[i](h)
            pt_detail[f"convs2_{i}"] = h.detach().clone().float()
            x = x + h
            pt_detail[f"x_after_{i}"] = x.detach().clone().float()
        return x
    rb.forward = types.MethodType(verbose_fwd, rb)
    return orig

orig = make_detail_rb(gen.resblocks[0])

with torch.inference_mode():
    _ = model(input_ids, style_pt, deterministic=True)

gen.resblocks[0].forward = types.MethodType(orig, gen.resblocks[0])

print("\n=== PT resblock.0 internals ===")
for k in ["snake1_0", "convs1_0", "snake2_0", "convs2_0", "x_after_0",
          "snake1_1", "snake2_1", "convs2_1", "x_after_1",
          "snake2_2", "convs2_2", "x_after_2"]:
    if k in pt_detail:
        v = pt_detail[k]
        print(f"  {k}: shape={tuple(v.shape)} mean={v.mean():.4f} std={v.std():.4f} max={v.abs().max():.4f}")

# ── Comparison ────────────────────────────────────────────────────────────────
print("\n=== ONNX vs PT comparison ===")
keys = ["snake1_0", "convs1_0", "snake2_0", "convs2_0", "x_after_0",
        "snake1_1", "snake2_1", "convs2_1", "x_after_1",
        "snake2_2", "convs2_2", "x_after_2"]
for k in keys:
    if k in onnx_vals and k in pt_detail:
        ov = onnx_vals[k]
        pv = pt_detail[k].numpy()
        T = min(ov.shape[-1], pv.shape[-1])
        diff = np.abs(ov[..., :T] - pv[..., :T])
        print(f"  {k}: ONNX={ov.mean():.4f} PT={pv.mean():.4f} diff_mean={diff.mean():.4f} diff_max={diff.max():.4f}")
    elif k in onnx_vals:
        print(f"  {k}: ONNX={onnx_vals[k].mean():.4f} (PT missing)")
    elif k in pt_detail:
        print(f"  {k}: PT={pt_detail[k].mean():.4f} (ONNX missing)")
