import onnx
from collections import Counter

m = onnx.load("models/best_model_ptq_int8_shared_channel_val_7.onnx")
ops = Counter(n.op_type for n in m.graph.node)
print(ops)