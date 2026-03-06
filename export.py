"""
Export the one of quantized int8 model to tflite using litert-torch api
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
import litert_torch
import ai_edge_litert.interpreter
from ai_edge_quantizer import Quantizer
from pathlib import Path
import sys
import copy

# Ensure custom modules load
sys.path.append(str(Path(__file__).parent.resolve()))

from quantize import ExportableSeparableConvCNN, make_conv1d_from_bn, Preprocessor
from lib.model import SeparableConvCNN
from lib.train import MyDataset

def export_tflite(val_subject=1, use_quant=True, per_channel_quant=False):
    root_path = Path("./uci-har")
    
    prefix = "best_model_loso"
    if per_channel_quant:
        prefix += "_per_channel"
    model_path = Path(f"./models/{prefix}_val_{val_subject}.pth")
    
    out_prefix = "best_model_ptq_int8"
    if per_channel_quant:
        out_prefix += "_per_channel"
    output_tflite = Path(f"./models/{out_prefix}_val_{val_subject}.tflite")
    num_channels = 6
    
    print(f"Loading Base Float32 model from {model_path}...")
    original_model = SeparableConvCNN(num_classes=6, num_channels=num_channels, use_quant=use_quant, per_channel_quant=per_channel_quant)
    original_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    original_model.eval()

    # Extract Preprocessor Variables safely (works for scalar and vectors)
    if use_quant:
        k_val = original_model.quant.k.detach().clone()
        mu_val = original_model.quant.mu.detach().clone()
    else:
        k_val = None
        mu_val = None
        
    bn0_clone = copy.deepcopy(original_model.bn0)
    preprocessor = Preprocessor(bn0_clone, k_val, mu_val) if use_quant else bn0_clone

    # 1. Initialize the exportable model architecture natively (Float32) without BN overhead
    model = ExportableSeparableConvCNN(num_classes=6, num_channels=num_channels)
    
    model_state = model.state_dict()
    orig_state = original_model.state_dict()
    for key in orig_state.keys():
        if key in model_state:
            model_state[key].copy_(orig_state[key])
    model.load_state_dict(model_state)

    # Fold Batch Normalization into 1x1 Convolutions
    model.bn1_conv.load_state_dict(make_conv1d_from_bn(original_model.bn1).state_dict())
    model.bn2_conv.load_state_dict(make_conv1d_from_bn(original_model.bn2).state_dict())
    model.bn3_conv.load_state_dict(make_conv1d_from_bn(original_model.bn3).state_dict())
    model.bn4_conv.load_state_dict(make_conv1d_from_bn(original_model.bn4).state_dict())
    model.eval()

    # 2. Generate Trace Input & Convert to LiteRT (Float32)
    sample_input = torch.randn(1, num_channels, 128)
    print("Converting Float32 to TFLite via litert-torch...")
    tflite_f32 = litert_torch.convert(model, (sample_input,))
    f32_bytes = tflite_f32.tflite_model()

    # 3. Native Quantization using ai-edge-quantizer
    qt = Quantizer(f32_bytes)
    
    # Needs calibration data (use 3 training subjects just like in PyTorch PTQ)
    train_subject_path = root_path / "train" / "subject_train.txt"
    all_train_subjects = sorted(np.unique(np.loadtxt(train_subject_path, dtype=int)).tolist())
    train_subjects = [s for s in all_train_subjects if s != val_subject]
    calib_subjects = train_subjects[:3]
    
    calib_dataset = MyDataset(root_path, split='train', subject_ids=calib_subjects, use_gyro=True)
    g = torch.Generator().manual_seed(42)
    calib_loader = DataLoader(calib_dataset, batch_size=1, shuffle=True, generator=g)

    interpreter = ai_edge_litert.interpreter.Interpreter(model_content=f32_bytes)
    input_details = interpreter.get_signature_runner().get_input_details()
    input_name = list(input_details.keys())[0]

    def calib_data_gen():
        for i, (inputs, _) in enumerate(calib_loader):
            if i > 500:
                break
            processed = preprocessor(inputs)
            yield {input_name: processed.numpy()}
            
    calib_data = {"serving_default": calib_data_gen()}
    
    # Configure PTQ INT8
    qt.add_static_config(
        regex=".*",
        operation_name="*", # All ops
        activation_num_bits=8,
        weight_num_bits=8,
    )
    
    print("Calibrating INT8 Quantizer...")
    calib_result = qt.calibrate(calib_data)
    
    print("Quantizing to native INT8 format...")
    quantized_model = qt.quantize(calib_result)
    
    # 6. Export to disk
    quantized_model.export_model(str(output_tflite), overwrite=True)
    print(f"Exported successfully to {output_tflite}!")

if __name__ == "__main__":
    for val_subj in range(1, 22):
        print(f"\n======================================")
        print(f"Exporting TFLite for Fold {val_subj}/21")
        print(f"======================================")
        export_tflite(val_subject=val_subj)