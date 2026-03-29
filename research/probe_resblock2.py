"""
Injection probe: feed ONNX x_into_resblocks into PT resblock.0
to isolate divergence: inside resblock vs upstream (noise_res).
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
import types

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

# ── Find ONNX node: input to resblocks (Add of ups.0 + noise_rb0) ─────────────
model_onnx = onnx.load(ONNX_PATH)

# The x fed into resblocks.0 and resblocks.1 is the Add output:
#   ups.0 cast-output + noise_res.0 output
# Find Add nodes in generator scope but NOT in any sub-scope
gen_top_adds = []
for node in model_onnx.graph.node:
    name = node.name
    if (node.op_type == "Add"
        and "/decoder/generator/" in name
        and "/resblocks" not in name
        and "/noise_res" not in name
        and "/ups" not in name
        and "/noise_convs" not in name):
        gen_top_adds.append(node)

print("Generator top-level Add nodes (potential x = ups + noise):")
for n in gen_top_adds:
    print(f"  {n.name}: inputs={list(n.input)[:4]} → {list(n.output)[:2]}")

# Also look for the node whose output is consumed by BOTH resblocks.0 AND resblocks.1
# (since both use the same x in MRF pattern)
output_consumers = {}
for node in model_onnx.graph.node:
    for inp in node.input:
        if inp not in output_consumers:
            output_consumers[inp] = []
        output_consumers[inp].append(node.name)

# Find tensors consumed by resblocks.0 AND resblocks.1 (= the shared x input)
rb0_inputs = set()
rb1_inputs = set()
for node in model_onnx.graph.node:
    if "/decoder/generator/resblocks.0/" in node.name:
        for inp in node.input:
            rb0_inputs.add(inp)
    if "/decoder/generator/resblocks.1/" in node.name:
        for inp in node.input:
            rb1_inputs.add(inp)

shared_inputs = rb0_inputs & rb1_inputs
print(f"\nTensors consumed by BOTH resblock.0 and resblock.1 ({len(shared_inputs)} total):")
for s in list(shared_inputs)[:10]:
    print(f"  {s}")

# The input x should be a tensor with shape (1, 128, T) used by both
# Find the one that comes from OUTSIDE both resblocks
x_into_resblocks = None
for name in shared_inputs:
    # Check if this tensor is produced OUTSIDE resblocks scope
    producer_in_rb = False
    for node in model_onnx.graph.node:
        if name in node.output:
            if "/resblocks" in node.name:
                producer_in_rb = True
    if not producer_in_rb:
        x_into_resblocks = name
        print(f"\n→ x_into_resblocks: {name}")
        break

# Also find the residual Add outputs inside resblock.0 (x = x + h)
# These are the last Adds in each iteration (the ones with the largest sequence T)
rb0_add_nodes = []
for node in model_onnx.graph.node:
    if node.op_type == "Add" and "/decoder/generator/resblocks.0/" in node.name:
        rb0_add_nodes.append(node)

rb0_add_nodes.sort(key=lambda n: n.name)

# Now find which of these are the residual adds (x = x + h)
# Strategy: the residual add takes x (from previous iteration) and h (the new delta)
# The output of the residual add has shape (1, 128, T) and is consumed by the NEXT iteration

# Find the last 3 Add outputs from rb0 that have shape info
print("\nAll rb0 Add nodes:")
for n in rb0_add_nodes:
    print(f"  {n.name}: {list(n.input)[:3]} → {list(n.output)[:1]}")

# Build probe targets: x_into_resblocks + all rb0 Add outputs
probe_targets = []
if x_into_resblocks:
    probe_targets.append(("x_into_rb", x_into_resblocks))

# Add all rb0 and rb1 final residual-looking outputs
# The ONNX node Add_2/5/8 inside resblocks are the 3 residual adds (one per iteration)
for node in rb0_add_nodes:
    # Get the last part of the name
    node_short = node.name.split("/")[-1]
    for out in node.output:
        probe_targets.append((f"rb0_{node_short}", out))

# Also probe rb1 (to verify symmetry)
rb1_add_nodes = sorted([n for n in model_onnx.graph.node
                         if n.op_type == "Add" and "/decoder/generator/resblocks.1/" in n.name],
                        key=lambda n: n.name)
for node in rb1_add_nodes[-3:]:  # last 3 = iter residuals
    node_short = node.name.split("/")[-1]
    for out in node.output:
        probe_targets.append((f"rb1_{node_short}", out))

# ── Run ONNX ──────────────────────────────────────────────────────────────────
model_onnx2 = onnx.load(ONNX_PATH)
seen = set()
final_probes = []
for label, name in probe_targets:
    if name not in seen:
        seen.add(name)
        final_probes.append((label, name))
        elem_type = TensorProto.FLOAT
        for vi in model_onnx2.graph.value_info:
            if vi.name == name:
                elem_type = vi.type.tensor_type.elem_type
                break
        vi_out = helper.make_tensor_value_info(name, elem_type, None)
        model_onnx2.graph.output.append(vi_out)

sess = ort.InferenceSession(model_onnx2.SerializeToString(), providers=["CPUExecutionProvider"])
outs = sess.run(None, {"input_ids": input_ids_np, "style": style_np, "speed": speed_np})

print(f"\nONNX audio max: {np.abs(outs[0]).max():.4f}")
print("\n=== ONNX probes ===")
onnx_vals = {}
for i, (label, name) in enumerate(final_probes):
    val = outs[2 + i].astype(np.float32)
    onnx_vals[label] = val
    print(f"  {label}: shape={tuple(val.shape)} mean={val.mean():.4f} std={val.std():.4f} max={np.abs(val).max():.4f}")

# ── PT: capture x_into_resblocks ─────────────────────────────────────────────
pt_captured = {}

def hook_gen_forward(gen, orig_forward):
    """Override Generator.forward to capture x before resblocks."""
    def new_forward(self, features, f0_samples, style, deterministic=False):
        s = style[:, 128:]
        sine = self.sine_gen(f0_samples, deterministic)
        sine_stft = self.stft.forward_stft(sine)
        noise_0 = self.noise_convs[0](sine_stft)
        noise_1 = self.noise_convs[1](sine_stft)
        n0 = self.noise_res[0](noise_0, s)
        n1 = self.noise_res[1](noise_1, s)

        x = F.leaky_relu(features, 0.1)
        x = self.ups[0](x)
        from kitten_torch.modules.generator import _match_length
        n0 = _match_length(n0, x.shape[-1])
        x = x + n0
        pt_captured["x_into_rb"] = x.detach().clone().float()

        # resblock with iteration capture
        def run_rb_verbose(rb, rb_idx, x_in):
            x = x_in
            for i in range(3):
                h = rb.adain1[i](x, s)
                h = rb.snake(h, rb.alpha1[i])
                h = rb.convs1[i](h)
                h = rb.adain2[i](h, s)
                h = rb.snake(h, rb.alpha2[i])
                h = rb.convs2[i](h)
                x = x + h
                pt_captured[f"rb{rb_idx}_iter{i}"] = x.detach().clone().float()
                pt_captured[f"rb{rb_idx}_iter{i}_h"] = h.detach().clone().float()
            return x

        r0 = run_rb_verbose(self.resblocks[0], 0, x)
        r1 = run_rb_verbose(self.resblocks[1], 1, x)
        x = (r0 + r1) * 0.5

        x = F.leaky_relu(x, 0.1)
        x = self.ups[1](x)
        from kitten_torch.modules.generator import _match_length_pad
        x = _match_length_pad(x, n1.shape[-1])
        x = x + n1
        r2 = self.resblocks[2](x, s)
        r3 = self.resblocks[3](x, s)
        x = (r2 + r3) * 0.5
        x = F.leaky_relu(x, 0.01)
        x = self.conv_post(x)
        return self.stft.inverse_stft(x)

    gen.forward = types.MethodType(new_forward, gen)

gen = model.decoder.generator
orig_gen_forward = gen.forward.__func__
hook_gen_forward(gen, orig_gen_forward)

with torch.inference_mode():
    audio_pt = model(input_ids, style_pt, deterministic=True)

# Restore
gen.forward = types.MethodType(orig_gen_forward, gen)

print(f"\nPT audio max: {audio_pt.abs().max():.4f}")
print("\n=== PT resblock internals ===")
for k, v in pt_captured.items():
    print(f"  {k}: shape={tuple(v.shape)} mean={v.mean():.4f} std={v.std():.4f} max={v.abs().max():.4f}")

# ── Compare x_into_rb ────────────────────────────────────────────────────────
print("\n=== x_into_rb comparison ===")
if "x_into_rb" in onnx_vals and "x_into_rb" in pt_captured:
    ov = onnx_vals["x_into_rb"]
    pv = pt_captured["x_into_rb"].numpy()
    T = min(ov.shape[-1], pv.shape[-1])
    diff = np.abs(ov[..., :T] - pv[..., :T])
    print(f"  ONNX: mean={ov.mean():.4f} std={ov.std():.4f}")
    print(f"  PT:   mean={pv.mean():.4f} std={pv.std():.4f}")
    print(f"  diff: max={diff.max():.4f} mean={diff.mean():.4f}")

# ── Inject ONNX x into PT resblock.0, compare iter-by-iter ──────────────────
print("\n=== Inject ONNX x_into_rb → PT resblock.0 ===")
if "x_into_rb" in onnx_vals:
    x_onnx = torch.from_numpy(onnx_vals["x_into_rb"])
    s_pt = style_pt[:, 128:]
    rb = model.decoder.generator.resblocks[0]
    x = x_onnx.clone()

    with torch.inference_mode():
        for i in range(3):
            h = rb.adain1[i](x, s_pt)
            h = rb.snake(h, rb.alpha1[i])
            h = rb.convs1[i](h)
            h = rb.adain2[i](h, s_pt)
            h = rb.snake(h, rb.alpha2[i])
            h = rb.convs2[i](h)
            x = x + h
            print(f"  PT(on ONNX x) iter{i}: mean={x.float().mean():.4f} std={x.float().std():.4f}")

    # Compare with ONNX rb0 iter outputs
    print("\n  ONNX rb0 iter residuals (Add nodes with large shape):")
    for label, val in onnx_vals.items():
        if label.startswith("rb0_") and val.shape[-1] > 100:
            print(f"    {label}: mean={val.mean():.4f} std={val.std():.4f} shape={tuple(val.shape)}")
