"""Check Concat_1 input order: [mag, phase] or [phase, mag]?"""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import onnx
import onnxruntime as ort

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
VOICES_PATH = f"{HF_SNAPSHOT}/voices.npz"

model_proto = onnx.load(ONNX_PATH)

# Find Concat_1
for node in model_proto.graph.node:
    if node.name == '/decoder/generator/Concat_1':
        print(f"Concat_1 inputs: {list(node.input)}")
        print(f"Concat_1 outputs: {list(node.output)}")
        for attr in node.attribute:
            print(f"  axis={attr.i}")

# What are the producer nodes of Concat_1 inputs?
print("\nProducer of Sqrt output (magnitude):")
for node in model_proto.graph.node:
    if '/decoder/generator/Sqrt' == node.name:
        print(f"  {node.name}: out={list(node.output)}")

# What feeds into Concat_1?
print("\nNodes producing Concat_1 inputs:")
concat1_inputs = None
for node in model_proto.graph.node:
    if node.name == '/decoder/generator/Concat_1':
        concat1_inputs = list(node.input)
        break

if concat1_inputs:
    print(f"  Concat_1 takes inputs: {concat1_inputs}")
    for inp in concat1_inputs:
        for n in model_proto.graph.node:
            if inp in n.output:
                print(f"  '{inp}' produced by {n.op_type}: {n.name!r}")
                break

# Also check: does the LeakyRelu look correct?
print("\n=== LeakyRelu alpha values ===")
for node in model_proto.graph.node:
    if node.op_type == 'LeakyRelu' and '/decoder/generator/' in node.name:
        print(f"  {node.name!r}")
        for attr in node.attribute:
            print(f"    {attr.name} = {attr.f}")

# Also check sine gen tanh - is it really there and affecting?
voices = np.load(VOICES_PATH)
voice_key = list(voices.keys())[0]
style_np = voices[voice_key][0:1].astype(np.float32)
input_ids_np = np.array([[0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 0]], dtype=np.int64)
speed = np.array([1.0], dtype=np.float32)

vi_types = {vi.name: vi.type.tensor_type.elem_type for vi in model_proto.graph.value_info}
probe_names = [
    "/decoder/generator/Concat_1_output_0",
    "/decoder/generator/Sqrt_output_0",
    "/decoder/generator/Where_2_output_0",   # likely the atan2 result
    "/decoder/generator/Where_1_output_0",
    "/decoder/generator/Where_output_0",
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
onnx_vals = {}
for name, result in zip(output_names, results):
    if name in set(probe_names):
        arr = np.asarray(result, dtype=np.float32)
        onnx_vals[name] = arr
        print(f"\n{name.split('/')[-1]}: shape={arr.shape}")
        print(f"  range: {arr.min():.4f} to {arr.max():.4f}")

# Check which channels of Concat_1 are magnitude vs phase
c = onnx_vals.get("/decoder/generator/Concat_1_output_0")
s = onnx_vals.get("/decoder/generator/Sqrt_output_0")
if c is not None and s is not None:
    print(f"\nConcat_1[:,0:11,:] range: {c[:,0:11,:].min():.4f} to {c[:,0:11,:].max():.4f}")
    print(f"Concat_1[:,11:22,:] range: {c[:,11:22,:].min():.4f} to {c[:,11:22,:].max():.4f}")
    print(f"Sqrt (magnitude) range: {s.min():.4f} to {s.max():.4f}")
    # magnitude is always >=0
    if c[:,0:11,:].min() >= -0.001:
        print("=> channels [0:11] are MAGNITUDE (non-negative)")
    else:
        print("=> channels [0:11] are PHASE")
    if c[:,11:22,:].min() < -0.5:
        print("=> channels [11:22] are PHASE (negative values)")
    else:
        print("=> channels [11:22] are MAGNITUDE")
