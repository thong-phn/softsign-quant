"""
Leave-One-Subject-Out (LOSO) Post-Training Quantization Evaluation Script
Iterates through all 21 LOSO trained models.
For each fold:
- Loads the corresponding Float32 best model.
- Builds the Preprocessor and Exportable INT8 Model natively folding BatchNorms.
- Calibrates the INT8 parameters symmetrically against the 20 active Training subjects.
- Evaluates the final quantized architecture purely on the unseen Test data.
- Aggregates Quantized Test Accuracy and F1-Macro metrics across 21 folds.
"""
import torch
import torch.nn as nn
import torch.ao.quantization as ao_quantization
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from pathlib import Path
import numpy as np
import copy
import sys

# Ensure custom modules load
sys.path.append(str(Path(__file__).parent.resolve()))

from lib.model import SeparableConvCNN
from lib.wear_data import WearDataset
from quantize import Preprocessor, ExportableSeparableConvCNN, make_conv1d_from_bn, set_seed

def evaluate_quantized(preprocessor, model, data_loader, device):
    """
    Evaluates the trained Quantizer -> SeparableConvCNN pipeline.
    Returns Dictionary of Metrics (Accuracy, F1-Macro).
    """
    if hasattr(preprocessor, 'eval'):
        preprocessor.eval()
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            pre_processed = preprocessor(inputs)
            outputs = model(pre_processed)
            _, predicted = outputs.max(1)
            
            bs = labels.size(0)
            total += bs
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = 100. * correct / max(total, 1)
    f1_macro = f1_score(all_labels, all_preds, average='macro') * 100.0
    
    return {"accuracy": accuracy, "f1_macro": f1_macro}

def evaluate_baseline_f32(model, data_loader, device):
    """
    Evaluates original Float32 Native Model.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / max(total, 1)

import argparse

def main():
    parser = argparse.ArgumentParser(description="Evaluate Quantized Models")
    parser.add_argument("--quantization", type=str, choices=['no', 'softsign', 'gamma', 'linear'], default='softsign', help="Quantization layer to use")
    parser.add_argument("--per-channel-quant", action="store_true", help="Use per-channel quantization models")
    args = parser.parse_args()
    quantization = args.quantization
    per_channel_quant = args.per_channel_quant

    set_seed(42)

    project_root = Path(__file__).resolve().parent
    root_path = project_root / "wear"
    
    # Subjects Loading Matrix
    all_train_subjects = list(range(18))
    test_eval_subjects = list(range(18, 24))

    device = torch.device('cpu') 
    num_channels = 3
    num_classes = 8
    
    print(f"Total Available Training Subjects for Folds ({len(all_train_subjects)}): {all_train_subjects}")
    print(f"Fixed Unseen Test Subjects ({len(test_eval_subjects)}): {test_eval_subjects}")
    
    # Store aggregated validation statistics
    fold_results = []
    
    # Generator for reproducible DataLoader shuffling inside Calibration
    g = torch.Generator()
    g.manual_seed(42)

    # Global Test Dataset (constant through all formulations)
    test_dataset = WearDataset(root_path, subject_ids=test_eval_subjects)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    for val_subject in all_train_subjects:
        train_subjects = [s for s in all_train_subjects if s != val_subject]
        
        prefix_parts = ["wear_best_model_loso"]
        if quantization != 'softsign':
            prefix_parts.append(quantization)
        if per_channel_quant:
            prefix_parts.append("per_channel")
        prefix = "_".join(prefix_parts)
        
        model_path = project_root / "models" / f"{prefix}_val_{val_subject}.pth"
        
        print(f"\n{'='*50}")
        print(f"Quantizing Fold | Model: Val Subject {val_subject}")
        print(f"{'='*50}")

        if not model_path.exists():
            print(f"ERROR: Could not find trained model at {model_path}. Skipping.")
            continue

        # 1. Load Original Model
        original_model = SeparableConvCNN(num_classes=num_classes, num_channels=num_channels, quantization=quantization, per_channel_quant=per_channel_quant)
        original_model.load_state_dict(torch.load(model_path, map_location="cpu"))
        original_model.eval()

        quant_layer = original_model.quant if quantization != 'no' and original_model.quant is not None else None
        
        # 2. Build Preprocessor and Exportable Skeleton
        bn0_clone = copy.deepcopy(original_model.bn0)
        preprocessor = Preprocessor(bn0_clone, quant_layer)

        exportable_model = ExportableSeparableConvCNN(num_classes=num_classes, num_channels=num_channels)
        model_state = exportable_model.state_dict()
        orig_state = original_model.state_dict()

        for key in orig_state.keys():
            if key in model_state:
                model_state[key].copy_(orig_state[key])
        exportable_model.load_state_dict(model_state)

        exportable_model.bn1_conv.load_state_dict(make_conv1d_from_bn(original_model.bn1).state_dict())
        exportable_model.bn2_conv.load_state_dict(make_conv1d_from_bn(original_model.bn2).state_dict())
        exportable_model.bn3_conv.load_state_dict(make_conv1d_from_bn(original_model.bn3).state_dict())
        exportable_model.bn4_conv.load_state_dict(make_conv1d_from_bn(original_model.bn4).state_dict())
        exportable_model.eval()

        # 3. Calibration Dataloader Construction
        calib_subjects = train_subjects[:3] 
        calib_dataset = WearDataset(root_path, subject_ids=calib_subjects)
        calib_loader = DataLoader(calib_dataset, batch_size=64, shuffle=True, generator=g)

        # Baseline check before PTQ destruction
        acc_orig_f32 = evaluate_baseline_f32(original_model, test_loader, device=device)

        # 4. Trigger PTQ Compiler
        torch.backends.quantized.engine = 'qnnpack'
        exportable_model.qconfig = ao_quantization.get_default_qconfig('qnnpack')

        torch.ao.quantization.fuse_modules(exportable_model, [
            ['sep_conv1.pointwise', 'relu1'], 
            ['sep_conv2.pointwise', 'relu2'],
            ['sep_conv3.pointwise', 'relu3'],
            ['sep_conv4.pointwise', 'relu4'],
            ['fc1', 'relu5'],
        ], inplace=True) 

        ao_quantization.prepare(exportable_model, inplace=True)

        with torch.no_grad():
            for inputs, _ in calib_loader:
                pre_processed = preprocessor(inputs)
                exportable_model(pre_processed) 

        quantized_model = ao_quantization.convert(exportable_model, inplace=True)

        # 5. Native INT8 Benchmarks
        q_metrics = evaluate_quantized(preprocessor, quantized_model, test_loader, device=device)
        q_acc = q_metrics['accuracy']
        q_f1 = q_metrics['f1_macro']

        print(f" -> Baseline (F32) Test Acc : {acc_orig_f32:.2f}%")
        print(f" -> Quantized (INT8) Test Acc: {q_acc:.2f}%  |  Absolute Drop: {(acc_orig_f32 - q_acc):.2f}%")
        print(f" -> Quantized (INT8) Test F1 : {q_f1:.2f}%")

        fold_results.append({
            'val_subject': val_subject,
            'acc_f32': acc_orig_f32,
            'acc_int8': q_acc,
            'f1_int8': q_f1
        })
        
        # Save quantized model locally to disc based on fold ID
        out_prefix = "best_model_ptq_int8"
        if per_channel_quant:
            out_prefix += "_per_channel"
            
        torch.save(quantized_model.state_dict(), project_root / "models" / f"{out_prefix}_val_{val_subject}.pth")

    if len(fold_results) == 0:
        print("No metrics collected. Did the original trained Float32 models execute successfully?")
        sys.exit(0)

    # 6. Accumulate and Broadcast Total Pipeline Variances
    print(f"\n{'='*50}")
    print("QUANTIZED LOSO CROSS-VALIDATION COMPLETE")
    print(f"{'='*50}")

    acc_f32_list = [res['acc_f32'] for res in fold_results]
    acc_int8_list = [res['acc_int8'] for res in fold_results]
    f1_int8_list = [res['f1_int8'] for res in fold_results]

    mean_f32, std_f32 = np.mean(acc_f32_list), np.std(acc_f32_list)
    mean_int8_acc, std_int8_acc = np.mean(acc_int8_list), np.std(acc_int8_list)
    mean_int8_f1, std_int8_f1 = np.mean(f1_int8_list), np.std(f1_int8_list)

    print(f"Baseline F32 Test Accuracy (Across {len(fold_results)} folds) : {mean_f32:.2f}% ± {std_f32:.2f}%")
    print(f"Quantized INT8 Test Accuracy (Across {len(fold_results)} folds): {mean_int8_acc:.2f}% ± {std_int8_acc:.2f}%")
    print(f"Quantized INT8 Test F1-Macro (Across {len(fold_results)} folds): {mean_int8_f1:.2f}% ± {std_int8_f1:.2f}%")

    log_dir = project_root / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_name = "wear_quantize_loso_results"
    log_name += f"_{quantization}"
    if per_channel_quant:
        log_name += "_per_channel"
    log_name += ".log"
    
    log_path = log_dir / log_name

    with open(log_path, "w") as f:
        f.write(f"Baseline F32 Test Accuracy: {mean_f32:.2f}% +- {std_f32:.2f}%\n")
        f.write(f"Quantized INT8 Test Accuracy: {mean_int8_acc:.2f}% +- {std_int8_acc:.2f}%\n")
        f.write(f"Quantized INT8 Test F1-Macro: {mean_int8_f1:.2f}% +- {std_int8_f1:.2f}%\n\n")
        f.write("Detailed Fold Results:\n")
        for res in fold_results:
            f.write(f"Fold Val {res['val_subject']}: F32 Acc = {res['acc_f32']:.2f}%, INT8 Acc = {res['acc_int8']:.2f}%, INT8 F1 = {res['f1_int8']:.2f}%\n")

if __name__ == "__main__":
    main()
