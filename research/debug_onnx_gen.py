"""
Probe ONNX generator intermediate values to find Div and Sin/Cos operations.
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

# Find ALL Div nodes in the generator
print("=== All Div nodes in generator ===")
for node in model_proto.graph.node:
    if node.op_type == "Div" and "generator" in node.name:
        print(f"  {node.name}: inputs={list(node.input)}, outputs={list(node.output)}")

# Find all nodes in the "last iSTFT section" - from conv_post to waveform
print("\n=== Detailed iSTFT section ===")
found_conv_post = False
for node in model_proto.graph.node:
    if "conv_post" in node.name:
        found_conv_post = True
    if found_conv_post:
        print(f"  {node.op_type}: name={node.name!r}")
        print(f"    inputs: {list(node.input)}")
        print(f"    outputs: {list(node.output)}")
        if node.op_type == "Cast" and "waveform" in node.output:
            break

# Check what initializer is used in the Div node
print("\n=== Div node initializers ===")
init_names = {init.name for init in model_proto.graph.initializer}
for node in model_proto.graph.node:
    if node.op_type == "Div" and "generator" in node.name:
        for inp in node.input:
            if inp in init_names:
                # Get the initializer value
                for init in model_proto.graph.initializer:
                    if init.name == inp:
                        arr = numpy_helper.to_array(init)
                        print(f"  Div constant {inp!r}: {arr}")

# Also check for Add nodes in MRF sum area
print("\n=== Generator MRF and Div nodes (stage 1 and 2) ===")
for node in model_proto.graph.node:
    if node.op_type in ("Add", "Div") and "generator" in node.name:
        # Look for Add_6, Div_2, Div_3 etc.
        if any(k in node.name for k in ["Add_5", "Add_6", "Add_7", "Add_8", "Div"]):
            print(f"  {node.op_type}: name={node.name!r}")
            print(f"    inputs: {list(node.input)}")
            print(f"    outputs: {list(node.output)}")

# Look for ALL Div nodes
print("\n=== ALL Div nodes ===")
for node in model_proto.graph.node:
    if node.op_type == "Div":
        print(f"  {node.name!r}: inputs={list(node.input)}, outputs={list(node.output)}")

# Now run with these intermediates as outputs to get actual values
print("\n=== Running ONNX with intermediate outputs ===")
voices = np.load(VOICES_PATH)
voice_key = list(voices.keys())[0]
style_np = voices[voice_key][0:1].astype(np.float32)
input_ids = np.array([[0, 10, 15, 20, 25, 30, 35, 40, 45, 50, 0]], dtype=np.int64)
speed = np.array([1.0], dtype=np.float32)

# Add key intermediate outputs to the model
intermediate_outputs = []
for node in model_proto.graph.node:
    if node.op_type == "Div" and "generator" in node.name:
        intermediate_outputs.extend(node.output)
    if node.op_type in ("LeakyRelu",) and "generator" in node.name:
        intermediate_outputs.extend(node.output)
    # Generator input (features from decoder)
    if "ups.0" in node.name and node.op_type == "ConvTranspose":
        intermediate_outputs.extend(node.input[:1])  # features input

# Also add the stage 1/2 Add outputs (resblock sum)
stage_adds = []
for node in model_proto.graph.node:
    if node.op_type == "Add" and "generator" in node.name:
        if any(f"Add_{k}" in node.name for k in range(3, 10)):
            stage_adds.append(node.output[0])

# Limit to unique outputs
intermediate_outputs = list(dict.fromkeys(intermediate_outputs))
print(f"Intermediate outputs to probe: {intermediate_outputs}")

# Extend the model
for out_name in intermediate_outputs[:10]:
    vi = onnx.helper.make_tensor_value_info(out_name, onnx.TensorProto.FLOAT, None)
    model_proto.graph.output.append(vi)

# Run
sess = ort.InferenceSession(model_proto.SerializeToString())
results = sess.run(None, {
    "input_ids": input_ids,
    "style": style_np,
    "speed": speed,
})

all_output_names = [o.name for o in sess.get_outputs()]
for i, (name, result) in enumerate(zip(all_output_names, results)):
    if name in intermediate_outputs:
        print(f"  {name!r}: shape={result.shape}, range={result.min():.4f} to {result.max():.4f}")
