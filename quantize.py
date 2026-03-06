import torch
import torch.nn as nn
import torch.ao.quantization as quantization
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
import copy
import sys

# Ensure custom modules load
sys.path.append(str(Path(__file__).parent.resolve()))

from lib.model import SeparableConvCNN, SeparableConv1d, SoftsignQuant
from lib.train import MyDataset
import random

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class QuantizableSeparableConv1d(nn.Module):
    """
    Quantization-friendly SeparableConv1d.
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

def make_conv1d_from_bn(bn: nn.BatchNorm1d) -> nn.Conv1d:
    """
    Converts a trained BatchNorm1d explicitly into a Conv1d layer (kernel_size=1)
    to bypass PyTorch's inability to quantize standalone BatchNorms.
    """
    channels = bn.num_features
    conv = nn.Conv1d(channels, channels, kernel_size=1, groups=channels, bias=True)
    
    eps = bn.eps
    mu = bn.running_mean
    var = bn.running_var
    gamma = bn.weight
    beta = bn.bias
    
    # Scale: gamma / sqrt(var + eps)
    scale = gamma / torch.sqrt(var + eps)
    # Shift: beta - mu * scale
    shift = beta - mu * scale
    
    # Fill the mock depthwise conv
    conv.weight.data = scale.view(channels, 1, 1).clone()
    conv.bias.data = shift.clone()
    return conv

class ExportableSeparableConvCNN(nn.Module):
    """
    The exact same SeparableConvCNN but:
    1. WITHOUT bn0 and SoftsignQuant (moved to preprocessing)
    2. WITH PyTorch Quantization Stubs
    3. EXPLICITLY replacing internal BatchNorm1d layers with 1x1 Convs 
       so QNNPack can quantize them securely.
    """
    def __init__(self, num_classes=6, num_channels=3, dropout=0.4):
        super().__init__()
        
        self.quant_stub = quantization.QuantStub()
        self.dequant_stub = quantization.DeQuantStub()
        
        self.sep_conv1 = QuantizableSeparableConv1d(num_channels, 32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        # This will become a 1x1 grouped Conv during weight porting!
        self.bn1_conv = nn.Conv1d(32, 32, kernel_size=1, groups=32, bias=True) 
        self.pool1 = nn.MaxPool1d(2)
        
        self.sep_conv2 = QuantizableSeparableConv1d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2_conv = nn.Conv1d(64, 64, kernel_size=1, groups=64, bias=True)
        self.pool2 = nn.MaxPool1d(2)
        
        self.sep_conv3 = QuantizableSeparableConv1d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.bn3_conv = nn.Conv1d(128, 128, kernel_size=1, groups=128, bias=True)
        self.pool3 = nn.MaxPool1d(2)
        
        self.sep_conv4 = QuantizableSeparableConv1d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.bn4_conv = nn.Conv1d(128, 128, kernel_size=1, groups=128, bias=True)
        self.pool4 = nn.MaxPool1d(2)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.flatten = Flatten()
        
        self.dropout = nn.Dropout(dropout)
        
        self.fc1 = nn.Linear(128, 64)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        x = self.quant_stub(x)
        
        x = self.sep_conv1(x)
        x = self.relu1(x)
        x = self.bn1_conv(x)
        x = self.pool1(x)
        
        x = self.sep_conv2(x)
        x = self.relu2(x)
        x = self.bn2_conv(x)
        x = self.pool2(x)
        
        x = self.sep_conv3(x)
        x = self.relu3(x)
        x = self.bn3_conv(x)
        x = self.pool3(x)
        
        x = self.sep_conv4(x)
        x = self.relu4(x)
        x = self.bn4_conv(x)
        x = self.pool4(x)
        
        x = self.global_avg_pool(x)
        x = self.flatten(x)
        x = self.dropout(x)
        
        x = self.fc1(x)
        x = self.relu5(x)
        x = self.fc2(x)
        
        x = self.dequant_stub(x)
        return x

class Preprocessor:
    """
    Simulates the external preprocessing pipeline.
    Raw Data -> BatchNorm (Normalization) -> Softsign 
    """
    def __init__(self, bn_layer, k, mu, bit_width=4):
        self.bn_mean = bn_layer.running_mean.view(1, -1, 1).detach()
        self.bn_var = bn_layer.running_var.view(1, -1, 1).detach()
        self.bn_weight = bn_layer.weight.view(1, -1, 1).detach()
        self.bn_bias = bn_layer.bias.view(1, -1, 1).detach()
        self.bn_eps = bn_layer.eps
        self.k = k
        self.mu = mu
        self.bit_width = bit_width
        self.S = (2**(bit_width - 1) - 1) / 2.0
        
    def __call__(self, x):
        # Apply trained BatchNorm math
        x = (x - self.bn_mean) / torch.sqrt(self.bn_var + self.bn_eps)
        x = x * self.bn_weight + self.bn_bias
        
        # Softsign
        z = self.k * (x - self.mu)
        d = 1.0 + torch.abs(z)
        y_raw = z / d
        y = y_raw * self.S
        
        max_val = 2**(self.bit_width - 1) - 1
        min_val = - (2**(self.bit_width - 1))
        
        y_round = torch.round(y)
        out = torch.clamp(y_round, min_val, max_val)
        return out

def evaluate(preprocessor, model, data_loader, device):
    """
    Trained Quantizer -> SeparableConvCNN
    """
    if hasattr(preprocessor, 'eval'):
        preprocessor.eval()
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            pre_processed = preprocessor(inputs)
            outputs = model(pre_processed)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    return 100. * correct / total

def evaluate_baseline(model, data_loader, device):
    """
    Baseline model: Input -> Batch Normalization -> Softsign -> SeparableConvCNN Model
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
    return 100. * correct / total

def main():
    set_seed(42)
    root_path = Path("./uci-har")
    model_path = Path("./models/best_model_subject1_val.pth")
    num_channels = 6 
    
    print("[STEP 1/5] Loading Best model (Trained Float32 Model)")
    original_model = SeparableConvCNN(num_classes=6, num_channels=num_channels)
    original_model.load_state_dict(torch.load(model_path, map_location="cpu"))
    original_model.eval()
    
    # Extract Softsign parameters
    k_val = original_model.quant.k.item()
    mu_val = original_model.quant.mu.item()
    print(f"Extracted Softsign k = {k_val:.4f}, mu = {mu_val:.4f}")
    
    # Bundle preprocessor
    bn0_clone = copy.deepcopy(original_model.bn0)
    preprocessor = Preprocessor(bn0_clone, k_val, mu_val)
    
    print("\n[STEP 2/5] Building Exportable Model (BN -> Conv replacement)")
    model = ExportableSeparableConvCNN(num_classes=6, num_channels=num_channels)
    
    # Manually port weights 
    model_state = model.state_dict()
    orig_state = original_model.state_dict()
    
    # Prefix mapping 
    for key in orig_state.keys():
        if key in model_state:
            model_state[key].copy_(orig_state[key])
            
    model.load_state_dict(model_state)
    
    # Manually overwrite the 'bnX_conv' blocks with the trained BN weights
    model.bn1_conv.load_state_dict(make_conv1d_from_bn(original_model.bn1).state_dict())
    model.bn2_conv.load_state_dict(make_conv1d_from_bn(original_model.bn2).state_dict())
    model.bn3_conv.load_state_dict(make_conv1d_from_bn(original_model.bn3).state_dict())
    model.bn4_conv.load_state_dict(make_conv1d_from_bn(original_model.bn4).state_dict())
    
    model.eval()
    
    print("\n3. Loading Datasets for Calibration & Test Evaluation...")
    device = torch.device('cpu') 
    
    # Calibration Set (Using Train to prevent data leakage!)
    train_subject_path = root_path / "train" / "subject_train.txt"
    train_subjects = sorted(np.unique(np.loadtxt(train_subject_path, dtype=int)).tolist())
    calib_subjects = train_subjects[:3] # We only need a few subjects to calibrate activations
    
    # Generator for reproducible DataLoader shuffling
    g = torch.Generator()
    g.manual_seed(42)
    
    calib_dataset = MyDataset(root_path, split='train', subject_ids=calib_subjects, use_gyro=True)
    calib_loader = DataLoader(calib_dataset, batch_size=64, shuffle=True, generator=g)
    
    # Evaluation Set (Test)
    test_subject_path = root_path / "test" / "subject_test.txt"
    test_eval_subjects = sorted(np.unique(np.loadtxt(test_subject_path, dtype=int)).tolist())
    
    test_dataset = MyDataset(root_path, split='test', subject_ids=test_eval_subjects, use_gyro=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Test our BN logic
    acc_orig = evaluate_baseline(original_model, test_loader, device=device)
    acc_export_f32 = evaluate(preprocessor, model, test_loader, device=device)
    
    print(f" -> Architecture verification. Original F32 Acc: {acc_orig:.2f}%. Export Architecture F32 Acc (BN folded to Conv): {acc_export_f32:.2f}%")
    assert abs(acc_orig - acc_export_f32) < 0.1, f"Model diverges. {acc_orig} vs {acc_export_f32}"
    
    print("\n4. Running Static Post-Training Quantization (PTQ)...")
    
    # Config for INT8 (qnnpack is better for mobile/arm/arm64)
    torch.backends.quantized.engine = 'qnnpack'
    model.qconfig = quantization.get_default_qconfig('qnnpack')
    
    # We can safely fuse the Depthwise -> Pointwise or Conv -> ReLU chains!
    torch.ao.quantization.fuse_modules(model, [
        ['sep_conv1.pointwise', 'relu1'], 
        ['sep_conv2.pointwise', 'relu2'],
        ['sep_conv3.pointwise', 'relu3'],
        ['sep_conv4.pointwise', 'relu4'],
        ['fc1', 'relu5'],
    ], inplace=True) 
    
    # Prepare model
    quantization.prepare(model, inplace=True)
    
    print(" -> Calibrating static INT8 intervals...")
    with torch.no_grad():
        for inputs, _ in calib_loader:
            pre_processed = preprocessor(inputs)
            model(pre_processed) 
            
    print(" -> Converting parameters to INT8...")
    quantized_model = quantization.convert(model, inplace=True)
    
    print("\n5. Benchmarking Fully Quantized Model...")
    acc_quantized = evaluate(preprocessor, quantized_model, test_loader, device=device)
    print(f" -> Quantized Model (INT8) Acc: {acc_quantized:.2f}%")
    print(f" -> Float32 Model (F32) Acc: {acc_export_f32:.2f}%")
    print(f" -> Absolute Accuracy Drop: {(acc_export_f32 - acc_quantized):.2f}%")
    
    # Save the static quantization payload
    torch.save(quantized_model.state_dict(), "./models/best_model_ptq_int8.pth")
    print("\nExported successful to ./models/best_model_ptq_int8.pth!")

if __name__ == "__main__":
    main()
