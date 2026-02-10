import torch
import torch.nn as nn
from .bitnet import BitLinear

class PathfinderBlock(nn.Module):
    """
    Residual Block for Pathfinder:
    Conv1d (Spatial Mixing) + BitLinear (Channel Mixing/Projection)
    """
    def __init__(self, dim):
        super().__init__()
        # 1. Spatial Mixing: Conv1d 3x3
        # We can use standard Conv1d
        self.conv1 = nn.Conv1d(dim, dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(dim)
        
        # 2. Channel Mixing: BitLinear (acts like 1x1 Conv)
        self.proj = BitLinear(dim, dim)
        self.act = nn.GELU()

    def forward(self, x):
        # x: [B, C, L]
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        
        # BitLinear expects [..., in_features], so we transpose
        out = out.transpose(1, 2) # [B, L, C]
        out = self.proj(out)
        out = out.transpose(1, 2) # [B, C, L]
        
        out = self.act(out)
        
        return out + residual

class Pathfinder(nn.Module):
    """
    1D ResNet Distinguisher.
    Predicts if a partial state difference likely leads to a collision/preimage.
    """
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        
        # Fixed XOR-Net Layer
        # Computes features: X_i, X_i ^ X_{i+1}, X_i ^ X_{i+2}
        # We implement this as a fixed Conv1d
        self.xor_conv = nn.Conv1d(1, 3, kernel_size=3, padding=0, bias=False)
        # Weights: 
        # Filter 0: [0, 1, 0] -> X_i (Identity)
        # Filter 1: [1, 1, 0] -> X_i + X_{i+1} (XOR with neighbor)
        # Filter 2: [1, 0, 1] -> X_i + X_{i+2}
        # Note: In logic XOR is diff, in Reals it's roughly abs diff? 
        # Or we just learn it.
        # "First Layer: Fixed XOR-Net" implies we should set weights.
        with torch.no_grad():
            self.xor_conv.weight.zero_()
            self.xor_conv.weight[0, 0, 1] = 1.0 # Center
            self.xor_conv.weight[1, 0, 1] = 1.0; self.xor_conv.weight[1, 0, 2] = 1.0 # Center + Right
            self.xor_conv.weight[2, 0, 1] = 1.0; self.xor_conv.weight[2, 0, 0] = 1.0 # Center + Left
            
        # Initial projection to Model Dim
        self.input_proj = nn.Conv1d(3, self.dim, kernel_size=1)
        
        self.blocks = nn.ModuleList([
            PathfinderBlock(self.dim)
            for _ in range(config.pathfinder_depth)
        ])
        
        # Global Average Pooling
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = BitLinear(self.dim, 1)

    def forward(self, x_diff):
        # x_diff: [B, 256] (0/1 difference)
        # Treat as float for Conv
        x = x_diff.float().unsqueeze(1) # [B, 1, 256]
        
        # Pad for XOR Conv (since k=3, pad=0 reduces dim by 2)
        # or we just use padding=1 in conv
        x_padded = torch.nn.functional.pad(x, (1, 1))
        
        # XOR Features
        x = self.xor_conv(x_padded) # [B, 3, 256]
        
        # Project
        x = self.input_proj(x) # [B, Dim, 256]
        
        for block in self.blocks:
            x = block(x)
            
        x = self.pool(x).squeeze(-1) # [B, Dim]
        logits = self.head(x) # [B, 1]
        
        return torch.sigmoid(logits).squeeze(-1)
