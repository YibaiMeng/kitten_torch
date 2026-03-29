"""
Targeted probe: find ONNX resblock final outputs and MRF combination nodes.
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

# Find the actual Add nodes that represent `x = x + h` inside resblocks
# Also find the Add nodes that represent `r0 + r1` (MRF combination)
# Strategy: find Add nodes whose inputs and outputs are all in the "resblocks" scope

# Build a node output → node mapping
output_to_node = {}
for node in model_onnx.graph.node:
    for out in node.output:
        output_to_node[out] = node

# Find Add nodes in the generator scope (not resblock-internal)
gen_add_nodes = []
gen_mul_nodes = []
for node in model_onnx.graph.node:
    if "/decoder/generator/" in node.name and "/resblocks" not in node.name and "/noise_res" not in node.name:
        if node.op_type == "Add":
            gen_add_nodes.append(node)
        elif node.op_type == "Mul":
            gen_mul_nodes.append(node)

print("=== Top-level generator Add nodes ===")
for node in gen_add_nodes[:20]:
    print(f"  {node.name}: inputs={list(node.input)[:3]} → outputs={list(node.output)[:2]}")

print("\n=== Top-level generator Mul nodes ===")
for node in gen_mul_nodes[:20]:
    print(f"  {node.name}: inputs={list(node.input)[:3]} → outputs={list(node.output)[:2]}")

# Find nodes after resblocks that combine r0+r1
# Look for Add nodes where one input contains "resblocks.0" or "resblocks.1"
mrf_combine_nodes = []
for node in model_onnx.graph.node:
    if node.op_type in ("Add", "Mean"):
        inps = list(node.input)
        # Does any input come from resblocks.0 or resblocks.1?
        for inp in inps:
            if "resblocks.0" in inp or "resblocks.1" in inp or "resblocks.2" in inp or "resblocks.3" in inp:
                mrf_combine_nodes.append((node.op_type, node.name, inps, list(node.output)))
                break

print("\n=== Nodes consuming resblock outputs ===")
for op, name, inps, outs in mrf_combine_nodes[:20]:
    print(f"  [{op}] {name}")
    print(f"    inputs: {inps[:4]}")
    print(f"    outputs: {outs[:2]}")

# Find the actual last operation inside each resblock
# The final op in resblocks.0 should be an Add (x = x + h) for the 3rd iteration
resblock_final_outputs = {}
for resblock_idx in range(4):
    prefix = f"/decoder/generator/resblocks.{resblock_idx}/"
    # Find all nodes in this resblock scope
    resblock_nodes = [n for n in model_onnx.graph.node
                      if prefix in n.name or n.name.startswith(prefix)]

    if not resblock_nodes:
        # Try different pattern
        prefix2 = f"kmodel.decoder.decoder.generator.resblocks.{resblock_idx}"
        resblock_nodes = [n for n in model_onnx.graph.node
                          if prefix2 in n.name]

    # Find nodes in this scope that have outputs NOT consumed by anything else in the same scope
    # These are the outputs of this resblock
    outputs_in_scope = set()
    for n in resblock_nodes:
        for out in n.output:
            outputs_in_scope.add(out)

    consumed_in_scope = set()
    for n in resblock_nodes:
        for inp in n.input:
            consumed_in_scope.add(inp)

    # Outputs produced in scope but consumed outside scope = resblock final outputs
    final = outputs_in_scope - consumed_in_scope
    resblock_final_outputs[resblock_idx] = list(final)
    print(f"\nresblock.{resblock_idx} final outputs (produced but not consumed within scope):")
    for f in list(final)[:5]:
        print(f"  {f}")

# Now probe the actual final outputs of resblocks + MRF combination
probe_names = []
for idx, finals in resblock_final_outputs.items():
    if finals:
        probe_names.append((f"resblock_{idx}_final", finals[0]))

# Also probe the MRF combine nodes
for op, name, inps, outs in mrf_combine_nodes[:4]:
    if outs:
        probe_names.append((f"mrf_combine_{name.split('/')[-1]}", outs[0]))

# Also probe ups.0 output (Cast_to_float32)
for node in model_onnx.graph.node:
    if "ups.0" in node.name and "Cast" in node.op_type:
        for out in node.output:
            probe_names.append(("ups0_cast_out", out))
        break

for node in model_onnx.graph.node:
    for out in node.output:
        if "ups.0" in out and "Cast_to_float32" in out:
            probe_names.append(("ups0_fp32", out))

print("\n=== Probe targets ===")
for label, name in probe_names:
    print(f"  {label}: {name}")

# Run ONNX with these as outputs
model_onnx2 = onnx.load(ONNX_PATH)
added = set()
final_probe = []
for label, name in probe_names:
    if name not in added:
        added.add(name)
        final_probe.append((label, name))
        vi = helper.make_tensor_value_info(name, TensorProto.FLOAT, None)
        model_onnx2.graph.output.append(vi)

try:
    sess = ort.InferenceSession(
        model_onnx2.SerializeToString(),
        providers=["CPUExecutionProvider"]
    )
    outputs = sess.run(None, {
        "input_ids": input_ids_np,
        "style": style_np,
        "speed": speed_np,
    })
    print(f"\nONNX audio max: {np.abs(outputs[0]).max():.4f}")
    print("\n=== ONNX probe results ===")
    for i, (label, name) in enumerate(final_probe):
        val = outputs[1 + i].astype(np.float32)
        print(f"  {label} {tuple(val.shape)}: mean={val.mean():.4f}  std={val.std():.4f}  max={np.abs(val).max():.4f}")
        print(f"    ({name})")
except Exception as e:
    print(f"\nError: {e}")
    import traceback
    traceback.print_exc()

    # Try identifying which outputs are float16
    print("\nChecking tensor types...")
    for label, name in final_probe:
        for vi in model_onnx.graph.value_info:
            if vi.name == name:
                print(f"  {label}: type={vi.type.tensor_type.elem_type}")
        for vi in model_onnx.graph.output:
            if vi.name == name:
                print(f"  {label} (graph output): type={vi.type.tensor_type.elem_type}")
