
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftsignQuant(nn.Module):
    """
    Learnable Softsign-based Quantization.
    """
    def __init__(self, bit_width=4, k_init=1.0, mu_init=0.0, num_channels=1, per_channel=False):
        super(SoftsignQuant, self).__init__()
        self.bit_width = bit_width
        self.S = (2**(bit_width - 1) - 1) / 2.0
        self.per_channel = per_channel
        
        # Learnable parameters
        if per_channel:
            self.k = nn.Parameter(torch.full((1, num_channels, 1), k_init, dtype=torch.float32))
            self.mu = nn.Parameter(torch.full((1, num_channels, 1), mu_init, dtype=torch.float32))
        else:
            self.k = nn.Parameter(torch.tensor(k_init, dtype=torch.float32))
            self.mu = nn.Parameter(torch.tensor(mu_init, dtype=torch.float32))
        
    def forward(self, x):
        # Softsign calculation (forward pass)
        z = self.k * (x - self.mu)
        d = 1.0 + torch.abs(z)
        y_raw = z / d
        y = y_raw * self.S
        
        # Hardware clamp 
        max_val = 2**(self.bit_width - 1) - 1
        min_val = - (2**(self.bit_width - 1))
        
        # Straight-through estimator for rounding and clamping
        y_round = torch.round(y)
        y_clamp = torch.clamp(y_round, min_val, max_val)
        
        # Detach gradient for rounding/clamping, but keep for y
        out = (y_clamp - y).detach() + y
        
        return out

class SeparableConv1d(nn.Module):
    """Depthwise Separable Convolution (Depthwise + Pointwise)"""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(SeparableConv1d, self).__init__()
        # Depthwise convolution
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=kernel_size,
            padding=padding, groups=in_channels, bias=False
        )
        # Pointwise convolution
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class SeparableConvCNN(nn.Module):
    """
    SeparableConv-based CNN
    Based on depthwise separable convolutions for efficiency
    """
    def __init__(self, num_classes=6, num_channels=3, freq_bins=65, dropout=0.4, use_quant=True, per_channel_quant=False):
        super(SeparableConvCNN, self).__init__()
        
        self.use_quant = use_quant
        self.per_channel_quant = per_channel_quant
        
        # Input shape: (batch, num_channels, 128) where num_channels is 3 (accel) or 6 (accel+gyro)
        
        # Stem block
        self.bn0 = nn.BatchNorm1d(num_channels)
        self.quant = SoftsignQuant(bit_width=4, num_channels=num_channels, per_channel=per_channel_quant)
        self.sep_conv1 = SeparableConv1d(num_channels, 32, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2)  # 31 -> 15
        
        # Separable conv blocks
        self.sep_conv2 = SeparableConv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2)  # 32 -> 16
        
        self.sep_conv3 = SeparableConv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)
        self.pool3 = nn.MaxPool1d(2)  # 16 -> 8
        
        self.sep_conv4 = SeparableConv1d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(2)  # 8 -> 4
        
        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # x: (batch, num_channels, 128)
        
        # Stem
        x = self.bn0(x)
        if self.use_quant:
            x = self.quant(x)
        x = F.relu(self.sep_conv1(x))
        x = self.bn1(x)
        x = self.pool1(x)
        
        # Block 2
        x = F.relu(self.sep_conv2(x))
        x = self.bn2(x)
        x = self.pool2(x)
        
        # Block 3
        x = F.relu(self.sep_conv3(x))
        x = self.bn3(x)
        x = self.pool3(x)
        
        # Block 4
        x = F.relu(self.sep_conv4(x))
        x = self.bn4(x)
        x = self.pool4(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)  # (batch, 128, 1)
        x = x.squeeze(-1)  # (batch, 128)
        
        # Classification head
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x

