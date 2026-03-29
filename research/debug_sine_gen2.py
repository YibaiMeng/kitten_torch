"""Probe ONNX sine generator in detail - Mul_13, Mul_14, l_linear weight."""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import onnx
import onnxruntime as ort
import torch

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
VOICES_PATH = f"{HF_SNAPSHOT}/voices.npz"

model_proto = onnx.load(ONNX_PATH)

# Print all l_sin_gen initializer/constant names
print("=== l_sin_gen constants/initializers ===")
for init in model_proto.graph.initializer:
    if 'l_sin_gen' in init.name or 'm_source' in init.name:
        arr = np.array(onnx.numpy_helper.to_array(init))
        print(f"  {init.name!r}: {arr}")

print("\n=== l_linear weights ===")
for init in model_proto.graph.initializer:
    if 'l_linear' in init.name:
        arr = np.array(onnx.numpy_helper.to_array(init))
        print(f"  {init.name!r}: shape={arr.shape}, values={arr.flatten()}")

# Trace the l_sin_gen computation flow more carefully
print("\n=== l_sin_gen nodes (in order) ===")
gen_nodes = []
for node in model_proto.graph.node:
    if '/decoder/generator/m_source/' in node.name:
        gen_nodes.append(node)

for node in gen_nodes:
    print(f"  {node.op_type}: {node.name.split('/')[-1]!r}")
    print(f"    in={list(node.input)[:3]}")
    print(f"    out={list(node.output)[:2]}")

# Check Mul_13 inputs
print("\n=== Mul_13 node ===")
for node in model_proto.graph.node:
    if node.name == '/decoder/generator/m_source/l_sin_gen/Mul_13':
        print(f"  inputs: {list(node.input)}")

# Check what RandomNormalLike's output feeds
print("\n=== RandomNormalLike output flow ===")
for node in model_proto.graph.node:
    if node.name == '/decoder/generator/m_source/l_sin_gen/RandomNormalLike':
        print(f"  RandomNormalLike out: {list(node.output)}")
        rng_out = node.output[0]
        break

# Find what uses rng_out
for node in model_proto.graph.node:
    if 'l_sin_gen' in node.name and rng_out in node.input:
        print(f"  -> used by {node.op_type}: {node.name.split('/')[-1]}")
        print(f"     inputs={list(node.input)}")
