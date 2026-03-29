"""
Targeted: find and probe the MRF combination nodes (r0+r1 and r2+r3)
and check if ONNX uses * 0.5 or just adds them.
"""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import torch
import onnx
import onnxruntime as ort
from onnx import helper, TensorProto
import re

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
VOICES_PATH = f"{HF_SNAPSHOT}/voices.npz"

TEXT = "Hello, how are you today?"
VOICE = "expr-voice-2-m"

from kitten_torch import KittenTTS
tts = KittenTTS(model_path=ONNX_PATH, voices_path=VOICES_PATH)

import kittentts as _kt, inspect
src = inspect.getsource(_kt.KittenTTS.__init__)
m = re.search(r"enumerate\(list\('(.+?)'\)\)", src, re.DOTALL)
WORD_INDEX = {s: i for i, s in enumerate(list(m.group(1)))}

voices_data = np.load(VOICES_PATH)
style_np = voices_data[VOICE][0:1].astype(np.float32)

phonemized = tts._phonemizer.phonemize([TEXT])[0]
normalized = ' '.join(re.findall(r"\w+|[^\w\s]", phonemized))
onnx_ids = [0]
for c in normalized:
    if c in WORD_INDEX:
        onnx_ids.append(WORD_INDEX[c])
onnx_ids.append(0)
input_ids_np = np.array([onnx_ids], dtype=np.int64)
speed_np = np.array([1.0], dtype=np.float32)

model_onnx = onnx.load(ONNX_PATH)

# Build graph maps
output_to_node = {}
node_inputs = {}
for node in model_onnx.graph.node:
    for out in node.output:
        output_to_node[out] = node

# Find the final output of each resblock by looking at what's CONSUMED
# by nodes OUTSIDE the resblock scope
def get_scope_outputs(scope_prefix):
    """Find tensors produced in scope but consumed outside."""
    in_scope = set()
    consumed_in_scope = set()
    for node in model_onnx.graph.node:
        if scope_prefix in node.name:
            for out in node.output:
                in_scope.add(out)
            for inp in node.input:
                consumed_in_scope.add(inp)
    # Also check nodes outside scope that consume in-scope outputs
    outside_consumed = set()
    for node in model_onnx.graph.node:
        if scope_prefix not in node.name:
            for inp in node.input:
                if inp in in_scope:
                    outside_consumed.add(inp)
    return in_scope, outside_consumed

# For each resblock, find its outputs consumed by outside nodes
print("=== Finding actual resblock outputs (consumed outside) ===")
rb_outputs = {}
for rb_idx in range(4):
    prefix = f"/decoder/generator/resblocks.{rb_idx}/"
    in_scope, outside = get_scope_outputs(prefix)
    print(f"\nresblock.{rb_idx} outputs consumed outside ({len(outside)} tensors):")
    for o in list(outside)[:5]:
        print(f"  {o}")
    rb_outputs[rb_idx] = list(outside)

# Find what consumes rb0 and rb1 outputs (MRF stage 1 combination)
print("\n=== Nodes consuming resblock.0 output ===")
rb0_outs = set(rb_outputs.get(0, []))
for node in model_onnx.graph.node:
    for inp in node.input:
        if inp in rb0_outs:
            print(f"  [{node.op_type}] {node.name}: {list(node.input)[:4]} → {list(node.output)[:2]}")

print("\n=== Nodes consuming resblock.1 output ===")
rb1_outs = set(rb_outputs.get(1, []))
for node in model_onnx.graph.node:
    for inp in node.input:
        if inp in rb1_outs:
            print(f"  [{node.op_type}] {node.name}: {list(node.input)[:4]} → {list(node.output)[:2]}")

print("\n=== Nodes consuming resblock.2 output ===")
rb2_outs = set(rb_outputs.get(2, []))
for node in model_onnx.graph.node:
    for inp in node.input:
        if inp in rb2_outs:
            print(f"  [{node.op_type}] {node.name}: {list(node.input)[:4]} → {list(node.output)[:2]}")

print("\n=== Nodes consuming resblock.3 output ===")
rb3_outs = set(rb_outputs.get(3, []))
for node in model_onnx.graph.node:
    for inp in node.input:
        if inp in rb3_outs:
            print(f"  [{node.op_type}] {node.name}: {list(node.input)[:4]} → {list(node.output)[:2]}")

# Now trace: what is the value of the MRF combination const (0.5 or something else)?
# Find all Mul nodes in the generator that have a constant input
print("\n=== Generator Mul nodes with scalar constants ===")
for node in model_onnx.graph.node:
    if "/decoder/generator/" in node.name and node.op_type == "Mul":
        # Check if any input is a constant initializer
        for inp in node.input:
            for init in model_onnx.graph.initializer:
                if init.name == inp:
                    import numpy as np
                    arr = np.frombuffer(init.raw_data, dtype=np.float32) if init.raw_data else np.array(init.float_data)
                    if arr.size <= 4:
                        print(f"  {node.name}: const={arr} → {list(node.output)[:1]}")

# Now probe the key tensors
probe_targets = []

# Add the nodes that consume rb0 and rb1 outputs (the MRF combination Add)
for rb_idx in range(4):
    rb_outs = rb_outputs.get(rb_idx, [])
    for rb_out in rb_outs:
        # Find what node consumes this
        for node in model_onnx.graph.node:
            for inp in node.input:
                if inp == rb_out and node.op_type in ("Add", "Mul", "Mean"):
                    # This is the MRF combination node
                    for out in node.output:
                        probe_targets.append((f"rb{rb_idx}_consumer_{node.op_type}", out))

# Deduplicate
seen = set()
unique_probes = []
for label, name in probe_targets:
    if name not in seen:
        seen.add(name)
        unique_probes.append((label, name))

print(f"\n=== Probe targets ({len(unique_probes)}) ===")
for label, name in unique_probes[:20]:
    print(f"  {label}: {name}")

# Also probe ups.0 cast output specifically
for node in model_onnx.graph.node:
    if "generator" in node.name and "ups.0" in node.name:
        for out in node.output:
            if out not in seen:
                seen.add(out)
                unique_probes.append((f"ups0_{node.op_type}", out))
                print(f"  ups0_{node.op_type}: {out}")

# Run ONNX probe
model_onnx2 = onnx.load(ONNX_PATH)
for label, name in unique_probes[:30]:
    vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
    model_onnx2.graph.output.append(vi)

try:
    sess = ort.InferenceSession(
        model_onnx2.SerializeToString(),
        providers=["CPUExecutionProvider"]
    )
    outputs = sess.run(None, {
        "input_ids": input_ids_np, "style": style_np, "speed": speed_np,
    })
    print(f"\nONNX audio max: {np.abs(outputs[0]).max():.4f}")
    print("\n=== ONNX probe results ===")
    for i, (label, name) in enumerate(unique_probes[:30]):
        val = outputs[1 + i].astype(np.float32)
        print(f"  {label} {tuple(val.shape)}: mean={val.mean():.4f}  std={val.std():.4f}  max={np.abs(val).max():.4f}")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()
    # Check types
    for label, name in unique_probes[:10]:
        for vi in model_onnx.graph.value_info:
            if vi.name == name:
                t = vi.type.tensor_type.elem_type
                print(f"  {label}: elem_type={t} (1=float32, 10=float16)")
