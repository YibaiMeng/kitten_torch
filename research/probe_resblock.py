"""
Probe GenResBlock internals: compare PT vs ONNX at each iteration step.
Goal: find where PT produces more negative output than ONNX.
"""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import torch
import torch.nn.functional as F
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

# ── Find ONNX resblock internal Add node outputs ──────────────────────────────
model_onnx = onnx.load(ONNX_PATH)

# Pattern: resblocks.{rb_idx}/Add_{n}_output_0
# Iteration residual adds are named Add_2, Add_5, Add_8 (every 3rd Add)
probe_targets = []

for rb_idx in range(2):  # stage1 resblocks only
    prefix = f"/decoder/generator/resblocks.{rb_idx}/"
    rb_adds = []
    for node in model_onnx.graph.node:
        if node.op_type == "Add" and prefix in node.name:
            for out in node.output:
                rb_adds.append((node.name, out))
    rb_adds.sort(key=lambda x: x[0])
    print(f"\nresblock.{rb_idx} Add nodes:")
    for name, out in rb_adds:
        print(f"  {name}: {out}")
    probe_targets.extend([(f"rb{rb_idx}_{name.split('/')[-1]}", out) for name, out in rb_adds])

# Also probe: ups.0 cast output (input to resblocks)
for node in model_onnx.graph.node:
    if "ups.0" in node.name and "Cast" in node.op_type:
        for out in node.output:
            probe_targets.append(("ups0_out", out))
        break

# And: resblock.0/Add_8 input (should == ups0_out + noise_rb0)
# Find the node that feeds into resblock.0 (the x before the resblocks)
# This is the Add node that adds ups.0 output and noise_res.0 output
for node in model_onnx.graph.node:
    if node.op_type == "Add" and "/decoder/generator/Add_" in node.name:
        # Check if this Add node is before any resblock
        is_before_resblock = all("resblocks" not in inp for inp in node.input)
        if is_before_resblock and "ups_0" in str(node.input) or "noise_res.0" in str(node.name):
            for out in node.output:
                probe_targets.append((f"gen_add_{node.name.split('/')[-1]}", out))

# Also: find the actual input to resblocks.0 first Conv
for node in model_onnx.graph.node:
    if f"/decoder/generator/resblocks.0/adain1.0" in node.name and node.op_type in ("Mul", "Add"):
        for inp in node.input:
            # The input that comes from OUTSIDE the resblock
            from_outside = True
            for n2 in model_onnx.graph.node:
                if f"/decoder/generator/resblocks.0/" in n2.name:
                    if inp in n2.output:
                        from_outside = False
                        break
            if from_outside and inp:
                probe_targets.append(("x_into_rb0", inp))
        break

print("\n=== Probe targets ===")
seen_names = set()
unique_probes = []
for label, name in probe_targets:
    if name and name not in seen_names:
        seen_names.add(name)
        unique_probes.append((label, name))
        print(f"  {label}: {name}")

# ── Run ONNX with probes ──────────────────────────────────────────────────────
model_onnx2 = onnx.load(ONNX_PATH)
for label, name in unique_probes:
    # Check type
    elem_type = TensorProto.FLOAT
    for vi in model_onnx2.graph.value_info:
        if vi.name == name:
            elem_type = vi.type.tensor_type.elem_type
            break
    vi = helper.make_tensor_value_info(name, elem_type, None)
    model_onnx2.graph.output.append(vi)

sess = ort.InferenceSession(model_onnx2.SerializeToString(), providers=["CPUExecutionProvider"])
outs = sess.run(None, {"input_ids": input_ids_np, "style": style_np, "speed": speed_np})
onnx_audio = outs[0]
print(f"\nONNX audio max: {np.abs(onnx_audio).max():.4f}")
print("\n=== ONNX resblock internals ===")
onnx_probes = {}
for i, (label, name) in enumerate(unique_probes):
    val = outs[2 + i].astype(np.float32)  # skip audio (0) and durations (1)
    onnx_probes[label] = val
    print(f"  {label}: shape={tuple(val.shape)} mean={val.mean():.4f} std={val.std():.4f} max={np.abs(val).max():.4f}")

# ── PT: hook into resblock internals ─────────────────────────────────────────
gen = model.decoder.generator
pt_captured = {}

# We need to hook INSIDE GenResBlock at each iteration
# Override forward temporarily to capture iteration outputs
original_forwards = {}

def make_verbose_resblock(rb_idx, rb):
    orig_forward = rb.forward.__func__

    def verbose_forward(self, x, s):
        for i in range(3):
            h = self.adain1[i](x, s)
            h = self.snake(h, self.alpha1[i])
            h = self.convs1[i](h)
            h = self.adain2[i](h, s)
            h = self.snake(h, self.alpha2[i])
            h = self.convs2[i](h)
            x = x + h
            pt_captured[f"rb{rb_idx}_iter{i}_x"] = x.detach().clone().float()
        return x

    import types
    rb.forward = types.MethodType(verbose_forward, rb)
    return orig_forward

orig0 = make_verbose_resblock(0, gen.resblocks[0])
orig1 = make_verbose_resblock(1, gen.resblocks[1])

with torch.inference_mode():
    audio_pt = model(input_ids, style_pt, deterministic=True)

# Restore
import types
gen.resblocks[0].forward = types.MethodType(orig0, gen.resblocks[0])
gen.resblocks[1].forward = types.MethodType(orig1, gen.resblocks[1])

print(f"\nPT audio max: {audio_pt.abs().max():.4f}")
print("\n=== PT resblock internals ===")
for k, v in pt_captured.items():
    print(f"  {k}: shape={tuple(v.shape)} mean={v.mean():.4f} std={v.std():.4f} max={v.abs().max():.4f}")

# ── Compare PT iter-by-iter with ONNX Add outputs ────────────────────────────
print("\n=== PT vs ONNX iteration comparison (resblock 0) ===")
# ONNX Add_2/5/8 correspond to iter 0/1/2
onnx_iter_labels = [k for k in onnx_probes if k.startswith("rb0_Add_")]
onnx_iter_labels.sort()
pt_iter_labels = [k for k in pt_captured if k.startswith("rb0_iter")]
pt_iter_labels.sort()

print(f"ONNX iter labels: {onnx_iter_labels}")
print(f"PT iter labels: {pt_iter_labels}")

if len(onnx_iter_labels) >= 3 and len(pt_iter_labels) >= 3:
    for i in range(3):
        ol = onnx_iter_labels[i]
        pl = pt_iter_labels[i]
        ov = onnx_probes[ol]
        pv = pt_captured[pl].numpy()
        T = min(ov.shape[-1], pv.shape[-1])
        diff = np.abs(ov[..., :T] - pv[..., :T])
        print(f"  iter{i}: ONNX mean={ov.mean():.4f} | PT mean={pv.mean():.4f} | diff max={diff.max():.4f} mean={diff.mean():.4f}")
else:
    print("Iteration labels don't match, printing raw comparison:")
    for ol in onnx_iter_labels:
        print(f"  ONNX {ol}: mean={onnx_probes[ol].mean():.4f}")
    for pl in pt_iter_labels:
        print(f"  PT {pl}: mean={pt_captured[pl].mean():.4f}")

print("\n=== PT vs ONNX iteration comparison (resblock 1) ===")
onnx_iter_labels1 = [k for k in onnx_probes if k.startswith("rb1_Add_")]
onnx_iter_labels1.sort()
pt_iter_labels1 = [k for k in pt_captured if k.startswith("rb1_iter")]
pt_iter_labels1.sort()

if len(onnx_iter_labels1) >= 3 and len(pt_iter_labels1) >= 3:
    for i in range(3):
        ol = onnx_iter_labels1[i]
        pl = pt_iter_labels1[i]
        ov = onnx_probes[ol]
        pv = pt_captured[pl].numpy()
        T = min(ov.shape[-1], pv.shape[-1])
        diff = np.abs(ov[..., :T] - pv[..., :T])
        print(f"  iter{i}: ONNX mean={ov.mean():.4f} | PT mean={pv.mean():.4f} | diff max={diff.max():.4f} mean={diff.mean():.4f}")
else:
    for ol in onnx_iter_labels1:
        print(f"  ONNX {ol}: mean={onnx_probes[ol].mean():.4f}")
    for pl in pt_iter_labels1:
        print(f"  PT {pl}: mean={pt_captured[pl].mean():.4f}")

# ── Also check AdaIN output + Snake output before first conv ──────────────────
print("\n=== Detailed PT resblock.0, iter 0 internals ===")
# Re-run with ultra-detailed hooks
pt_detail = {}

def make_detail_resblock(rb):
    def verbose_forward(self, x, s):
        pt_detail["x_input"] = x.detach().clone().float()
        for i in range(3):
            h = self.adain1[i](x, s)
            pt_detail[f"adain1_{i}"] = h.detach().clone().float()
            h = self.snake(h, self.alpha1[i])
            pt_detail[f"snake1_{i}"] = h.detach().clone().float()
            h = self.convs1[i](h)
            pt_detail[f"conv1_{i}"] = h.detach().clone().float()
            h = self.adain2[i](h, s)
            pt_detail[f"adain2_{i}"] = h.detach().clone().float()
            h = self.snake(h, self.alpha2[i])
            pt_detail[f"snake2_{i}"] = h.detach().clone().float()
            h = self.convs2[i](h)
            pt_detail[f"conv2_{i}"] = h.detach().clone().float()
            x = x + h
            pt_detail[f"x_after_{i}"] = x.detach().clone().float()
        return x
    import types
    rb.forward = types.MethodType(verbose_forward, rb)

make_detail_resblock(gen.resblocks[0])

with torch.inference_mode():
    _ = model(input_ids, style_pt, deterministic=True)

# Restore
import types
gen.resblocks[0].forward = types.MethodType(orig0, gen.resblocks[0])

for k in ["x_input", "adain1_0", "snake1_0", "conv1_0", "adain2_0", "snake2_0", "conv2_0", "x_after_0",
          "x_after_1", "x_after_2"]:
    if k in pt_detail:
        v = pt_detail[k]
        print(f"  {k}: mean={v.mean():.4f} std={v.std():.4f} max={v.abs().max():.4f}")
