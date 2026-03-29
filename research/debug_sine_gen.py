"""Probe ONNX sine generator step by step to find scale difference."""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import onnx
import onnxruntime as ort
import torch

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

probe_names = [
    # F0 upsampled
    "/decoder/generator/f0_upsamp/Resize_output_0",
    # Fundamental sin input phases
    "/decoder/generator/m_source/l_sin_gen/CumSum_output_0",   # accumulated phase
    "/decoder/generator/m_source/l_sin_gen/Sin_output_0",       # sin values
    "/decoder/generator/m_source/l_sin_gen/Mul_10_output_0",    # after Mul_10
    "/decoder/generator/m_source/l_sin_gen/Add_5_output_0",     # input to l_linear
    "/decoder/generator/m_source/l_linear/MatMul_output_0",     # after MatMul
    "/decoder/generator/m_source/l_linear/Add_output_0",        # after bias
    "/decoder/generator/m_source/l_tanh/Tanh_output_0",         # after Tanh
    # Forward STFT input
    "/decoder/generator/Unsqueeze_1_output_0",
    "/decoder/generator/Pad_output_0",
    "/decoder/generator/Conv_output_0",
    "/decoder/generator/Conv_1_output_0",
]

model_copy = onnx.ModelProto()
model_copy.CopyFrom(model_proto)
for name in probe_names:
    try:
        elem_type = vi_types.get(name, onnx.TensorProto.FLOAT)
        vi = onnx.helper.make_tensor_value_info(name, elem_type, None)
        model_copy.graph.output.append(vi)
    except:
        pass

sess = ort.InferenceSession(model_copy.SerializeToString(), providers=["CPUExecutionProvider"])
results = sess.run(None, {"input_ids": input_ids_np, "style": style_np, "speed": speed})
output_names = [o.name for o in sess.get_outputs()]
print("=== ONNX sine generator intermediates ===")
for name, result in zip(output_names, results):
    if name in set(probe_names):
        arr = np.asarray(result, dtype=np.float32)
        print(f"  {'/'.join(name.split('/')[-2:])}: shape={arr.shape}, range={arr.min():.6f} to {arr.max():.6f}, mean_abs={np.abs(arr).mean():.6f}")

# Check Mul_10 node inputs
print("\n=== Mul_10 node ===")
for node in model_proto.graph.node:
    if node.name == '/decoder/generator/m_source/l_sin_gen/Mul_10':
        print(f"  inputs: {list(node.input)}")
        for inp in node.input:
            # Check if it's a constant
            for init in model_proto.graph.initializer:
                if init.name == inp:
                    arr = np.array(onnx.numpy_helper.to_array(init))
                    print(f"  initializer {inp!r}: {arr}")
                    break

# Check Mul_14 node inputs
print("\n=== Mul_14 node ===")
for node in model_proto.graph.node:
    if node.name == '/decoder/generator/m_source/l_sin_gen/Mul_14':
        print(f"  inputs: {list(node.input)}")

# Check what Add_5 is
print("\n=== Add_5 node ===")
for node in model_proto.graph.node:
    if node.name == '/decoder/generator/m_source/l_sin_gen/Add_5':
        print(f"  inputs: {list(node.input)}")
        print(f"  outputs: {list(node.output)}")

# Check the l_linear weights
print("\n=== l_linear weights ===")
for init in model_proto.graph.initializer:
    if 'l_linear' in init.name and 'generator' in init.name:
        arr = np.array(onnx.numpy_helper.to_array(init))
        print(f"  {init.name!r}: shape={arr.shape}, range={arr.min():.6f} to {arr.max():.6f}")

# What does PT SineGenerator produce?
print("\n=== PT SineGenerator comparison ===")
from kitten_torch.model import build_model
model = build_model(ONNX_PATH)
model.eval()
gen = model.decoder.generator

# Use the ONNX F0
from kitten_torch.weight_loader import ONNXWeights
import onnxruntime as ort

sess2 = ort.InferenceSession(ONNX_PATH, providers=["CPUExecutionProvider"])
# Get ONNX f0_upsamp
import numpy as np
onnx_model = onnx.ModelProto()
onnx_model.CopyFrom(model_proto)
vi_f0 = onnx.helper.make_tensor_value_info(
    "/decoder/generator/f0_upsamp/Resize_output_0",
    onnx.TensorProto.FLOAT, None
)
onnx_model.graph.output.append(vi_f0)
sess_f0 = ort.InferenceSession(onnx_model.SerializeToString(), providers=["CPUExecutionProvider"])
results_f0 = sess_f0.run(None, {"input_ids": input_ids_np, "style": style_np, "speed": speed})
onnx_f0 = results_f0[-1]  # last added output
print(f"ONNX f0_upsamp: {onnx_f0.shape}, range={onnx_f0.min():.4f} to {onnx_f0.max():.4f}")

f0_t = torch.from_numpy(onnx_f0)
if f0_t.dim() == 2:
    f0_t = f0_t.unsqueeze(1)

with torch.no_grad():
    # manually compute sine gen steps
    B, _, N = f0_t.shape
    h = torch.arange(1, gen.sine_gen.n_h + 1).float().view(1, 9, 1)
    f0_h = f0_t * h  # (B, 9, N)
    phase_inc = 2.0 * 3.14159265 * f0_h / gen.sine_gen.sr
    phase = torch.cumsum(phase_inc, dim=-1)
    sines = torch.sin(phase)  # (B, 9, N)
    print(f"PT sines before l_linear: shape={sines.shape}, range={sines.min():.6f} to {sines.max():.6f}")
    out = gen.sine_gen.l_linear(sines.transpose(1, 2)).transpose(1, 2)
    print(f"PT l_linear output: shape={out.shape}, range={out.min():.6f} to {out.max():.6f}")
