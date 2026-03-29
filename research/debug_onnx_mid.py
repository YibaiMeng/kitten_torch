"""
Extract ONNX intermediate values at conv_post output, Slice_1, Slice_2, Exp, Sin outputs.
"""
import sys
sys.path.insert(0, "/home/yibaimeng_gmail_com/src/kitten")

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper

HF_SNAPSHOT = "/home/yibaimeng_gmail_com/.cache/huggingface/hub/models--KittenML--kitten-tts-nano-0.1/snapshots/7d99aae46867223af52d5ea16d076d8839ce1a2e"
ONNX_PATH = f"{HF_SNAPSHOT}/kitten_tts_nano_v0_1.onnx"
VOICES_PATH = f"{HF_SNAPSHOT}/voices.npz"

model_proto = onnx.load(ONNX_PATH)

voices = np.load(VOICES_PATH)
voice_key = list(voices.keys())[0]
style_np = voices[voice_key][0:1].astype(np.float32)
input_ids = np.array([[0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 0]], dtype=np.int64)
speed = np.array([1.0], dtype=np.float32)

# Get the slice constants
print("=== Slice constant values ===")
for node in model_proto.graph.node:
    if node.name in ("/decoder/generator/Slice_1", "/decoder/generator/Slice_2",
                     "/decoder/generator/Slice_3"):
        print(f"\n{node.name}: inputs={list(node.input)}")
        for inp in node.input:
            for n in model_proto.graph.node:
                if n.op_type == "Constant" and inp in n.output:
                    for attr in n.attribute:
                        if attr.name == "value":
                            arr = numpy_helper.to_array(attr.t)
                            print(f"  {inp} = {arr}")

# Get forward STFT forward section
print("\n=== forward STFT section ===")
for node in model_proto.graph.node:
    if node.op_type in ("Conv", "Div") and "generator" in node.name:
        if "conv_post" not in node.name and "quantize" not in node.name.lower():
            print(f"  {node.op_type}: {node.name!r}")
            print(f"    inputs: {list(node.input)[:3]}")
            print(f"    outputs: {list(node.output)}")

# Add key outputs to probe
probe_outputs = []
for node in model_proto.graph.node:
    if node.name in (
        "/decoder/generator/conv_post/Conv_output_0_Cast_to_float32_0",
        "/decoder/generator/Slice_1",
        "/decoder/generator/Slice_2",
        "/decoder/generator/Exp",
        "/decoder/generator/Sin",
        "/decoder/generator/Sin_1",
        "/decoder/generator/Cos",
        "/decoder/generator/Mul",
        "/decoder/generator/Mul_1",
        "/decoder/generator/ConvTranspose",
        "/decoder/generator/ConvTranspose_1",
        "/decoder/generator/Sub_1",
        "/decoder/generator/Div_1",
        "/decoder/generator/Div_2",
        "/decoder/generator/LeakyRelu_2",
    ):
        for out in node.output:
            probe_outputs.append(out)

print(f"\nProbing {len(probe_outputs)} outputs: {probe_outputs}")

# Run with these outputs
model_copy = onnx.ModelProto()
model_copy.CopyFrom(model_proto)
for out_name in probe_outputs:
    vi = onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.UNDEFINED, None)
    model_copy.graph.output.append(vi)

sess = ort.InferenceSession(model_copy.SerializeToString(), providers=["CPUExecutionProvider"])
results = sess.run(None, {
    "input_ids": input_ids,
    "style": style_np,
    "speed": speed,
})

output_names = [o.name for o in sess.get_outputs()]
probe_set = set(probe_outputs)
for name, result in zip(output_names, results):
    if name in probe_set:
        arr = np.asarray(result)
        print(f"\n{name!r}:")
        print(f"  shape={arr.shape}, dtype={arr.dtype}")
        print(f"  range: {arr.min():.4f} to {arr.max():.4f}")
        if arr.size <= 5:
            print(f"  values: {arr}")
