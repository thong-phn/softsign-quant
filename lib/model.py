
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoftsignQuant(nn.Module):
    """
    Learnable Softsign-based Quantization with normalization and quantization to [0, 2**b - 1].
    """
    def __init__(self, bit_width=4, k_init=1.0, mu_init=0.0, num_channels=1, per_channel=False):
        super(SoftsignQuant, self).__init__()
        self.bit_width = bit_width
        self.n_levels = 2 ** bit_width
        self.per_channel = per_channel
        
        # Learnable parameters
        if per_channel:
            self.k = nn.Parameter(torch.full((1, num_channels, 1), k_init, dtype=torch.float32))
            self.mu = nn.Parameter(torch.full((1, num_channels, 1), mu_init, dtype=torch.float32))
        else:
            self.k = nn.Parameter(torch.tensor(k_init, dtype=torch.float32))
            self.mu = nn.Parameter(torch.tensor(mu_init, dtype=torch.float32))
        
    def forward(self, x):
        # Stage 1: Compute raw softsign
        z = self.k * (x - self.mu)
        d = 1.0 + torch.abs(z)
        raw_softsign = z / d
        
        # Stage 2: Scale to [-1, 1] using max_val at the boundary
        # max_val = k / (1 + k) is the maximum of softsign at the boundary
        max_val = self.k / (1.0 + self.k)
        max_val_safe = torch.clamp(torch.abs(max_val), min=1e-6)
        x_softsign = raw_softsign / max_val_safe
        
        # Clip to [-1, 1] for safety (handle floating point errors)
        x_softsign = torch.clamp(x_softsign, -1.0, 1.0)
        
        # Stage 3: Normalize softsign output to [0, 1]
        x_normalized = (x_softsign + 1.0) / 2.0
        
        # Stage 4: Quantization to [0, 2**b - 1]
        x_scaled = x_normalized * (self.n_levels - 1)
        x_quant = torch.round(x_scaled)
        
        # Clamp to valid range (for safety)
        x_quant = torch.clamp(x_quant, 0, self.n_levels - 1)
        
        # Straight-through estimator: forward uses quantized value, backward uses x_scaled gradient
        out = x_scaled + (x_quant - x_scaled).detach()
        
        return out

class UniformQuantizerSTE(nn.Module):
    """
    Uniform quantization with straight-through estimator.

    Args:
        n_bits: int
            Number of bits for quantization.
    Forward:
        x: torch.Tensor
            Input tensor to be quantized.
    """
    def __init__(self, n_bits):
        super().__init__()
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits

    def forward(self, x):
        # Scale x from [-1, 1] to [0, n_levels - 1]
        x_clipped = torch.clamp(x, -1, 1)
        x_scaled = (x_clipped + 1) * (self.n_levels - 1) / 2
        x_quant = torch.round(x_scaled)
        x_dequant = x_quant * 2 / (self.n_levels - 1) - 1

        # Straight-through estimator: use quantized value in forward, but pass gradients as if identity
        return x + (x_dequant - x).detach()
        

class gammaFunction(nn.Module):
    """
    Gamma function as proposed in paper. 

    Args:
        init: str
            Initialization type of gamma function ('id' for identity, 's_shaped' for s-shaped curve).
        offset: float
            Offset value for the gamma function.
    Forward:
        x_query: torch.Tensor
            Input tensor to be transformed by the gamma function.
    """
    def __init__(self, init="id", offset=0, num_channels=1, per_channel=False):
        super().__init__()
        shape = (1, num_channels, 1) if per_channel else (1,)
        
        if init == "id":
            self.gamma = nn.Parameter(torch.ones(shape))
        elif init == "s_shaped":
            self.gamma = nn.Parameter(0.4 * torch.ones(shape))
        
        self.offset = nn.Parameter(offset * torch.ones(shape))  

    def forward(self, x_query):
        x_query = torch.clamp(x_query, -1, 1)
        return torch.sign(x_query - self.offset) * (torch.abs(x_query - self.offset) + 1e-3) ** self.gamma

class GammaQuant(nn.Module):
    """
    GammaFunction + UniformQuantizerSTE
    """
    def __init__(self, bit_width=4, num_channels=1, per_channel=False, init="s_shaped", offset=0):
        super(GammaQuant, self).__init__()
        self.gamma_func = gammaFunction(init=init, offset=offset, num_channels=num_channels, per_channel=per_channel)
        self.quantizer = UniformQuantizerSTE(n_bits=bit_width)
        
    def forward(self, x):
        x = self.gamma_func(x)
        x = self.quantizer(x)
        return x

class LinearQuant(nn.Module):
    """
    Learnable Linear Quantization.
    """
    def __init__(self, bit_width=4, k_init=1.0, mu_init=0.0, num_channels=1, per_channel=False):
        super(LinearQuant, self).__init__()
        self.bit_width = bit_width
        self.per_channel = per_channel
        
        # Learnable parameters
        if per_channel:
            self.k = nn.Parameter(torch.full((1, num_channels, 1), k_init, dtype=torch.float32))
            self.mu = nn.Parameter(torch.full((1, num_channels, 1), mu_init, dtype=torch.float32))
        else:
            self.k = nn.Parameter(torch.tensor(k_init, dtype=torch.float32))
            self.mu = nn.Parameter(torch.tensor(mu_init, dtype=torch.float32))
            
    def forward(self, x):
        # Linear transform
        y = self.k * (x - self.mu)
        
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
    def __init__(self, num_classes=6, num_channels=3, freq_bins=65, dropout=0.4, quantization='softsign', per_channel_quant=False):
        super(SeparableConvCNN, self).__init__()
        
        self.quantization = quantization
        self.per_channel_quant = per_channel_quant
        
        # Input shape: (batch, num_channels, 128) where num_channels is 3 (accel) or 6 (accel+gyro)
        
        # Stem block
        # self.bn0 = nn.BatchNorm1d(num_channels)
        if self.quantization == 'softsign':
            self.quant = SoftsignQuant(bit_width=4, num_channels=num_channels, per_channel=per_channel_quant)
        elif self.quantization == 'gamma':
            self.quant = GammaQuant(bit_width=4, num_channels=num_channels, per_channel=per_channel_quant)
        elif self.quantization == 'linear':
            self.quant = LinearQuant(bit_width=4, num_channels=num_channels, per_channel=per_channel_quant)
        else:
            self.quant = None
            
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
        # x = self.bn0(x)
        if self.quantization != 'no' and self.quant is not None:
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

