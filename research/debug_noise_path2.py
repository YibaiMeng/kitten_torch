"""
Find ONNX node names for: sine generator, forward STFT, noise_convs.
Then probe them and compare with PT.
"""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn.functional as F
from kitten_torch.model import build_model
from kitten_torch.modules.generator import _match_length, _match_length_pad

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
VOICES_PATH = f"{HF_SNAPSHOT}/voices.npz"

voices = np.load(VOICES_PATH)
voice_key = list(voices.keys())[0]
style_np = voices[voice_key][0:1].astype(np.float32)
input_ids_np = np.array([[0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 0]], dtype=np.int64)
speed = np.array([1.0], dtype=np.float32)

model_proto = onnx.load(ONNX_PATH)
vi_types = {vi.name: vi.type.tensor_type.elem_type for vi in model_proto.graph.value_info}

print("=== All generator nodes (non-noise_res) ===")
for node in model_proto.graph.node:
    name = node.name
    if '/decoder/generator/' not in name:
        continue
    if '/noise_res' in name:
        continue
    if '/resblocks' in name:
        continue
    print(f"  {node.op_type}: {name!r}")
    print(f"    out={list(node.output)}")
