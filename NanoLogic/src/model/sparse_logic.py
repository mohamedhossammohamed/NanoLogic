import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from .wiring import SHA256Wiring
from .bitnet import BitLinear

class SparseLogicBlock(nn.Module):
    """
    A single layer of the Sparse Logic Transformer.
    Uses Hard-Coded Wiring based on SHA-256 logic to mix information.
    """
    def __init__(self, dim, n_heads):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        
        # Get static wiring indices and masks (Global [256, k])
        # Returns dict mapping name -> (indices, mask)
        self.op_indices = SHA256Wiring.get_op_indices() 
        
        # Projections
        # Input features per bit:
        # 1. Identity (1)
        # 2. Sigma0 (Sum0) (3 neighbors)
        # 3. Sigma1 (Sum1) (3 neighbors)
        # 4. Sigma0_w (Msg Sched) (3 neighbors) [NEW]
        # 5. Sigma1_w (Msg Sched) (3 neighbors) [NEW]
        # 6. Vertical (8 neighbors)
        # 7. Carry Propagation (4 neighbors) [NEW]
        
        # Total gathered vectors per bit: 1 + 3 + 3 + 3 + 3 + 8 + 4 = 25 vectors.
        
        self.input_mix_dim = dim * (1 + 3 + 3 + 3 + 3 + 8 + 4)
        
        self.logic_gate = BitLinear(self.input_mix_dim, dim)
        
        # Pre-Norm Architecture: Need norm before Mixing and before MLP
        self.norm_logic = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim) 
        
        # BitConvSwiGLU (Replacing MLP)
        # 1. Expand 4x (BitLinear)
        # 2. Depthwise Conv (Local patterns)
        # 3. SiLU
        # 4. Project back (BitLinear)
        self.conv_swiglu = BitConvSwiGLU(dim, dim * 4)

    def forward(self, x):
        # Use gradient checkpointing during training to slash activation memory
        # Note: Checkpointing is now handled by the Recurrent Wrapper for the *entire* block
        # But we keep it here for flexibility if used stand-alone
        return self._forward_impl(x)

    def _forward_impl(self, x):
        # x: [B, 256, D]
        B, L, D = x.shape
        device = x.device
        
        # Ensure indices/masks are on device
        if self.op_indices['sigma0'][0].device != device:
             self.op_indices = {
                 k: (v[0].to(device), v[1].to(device)) 
                 for k, v in self.op_indices.items()
             }
             
        # Pre-Norm
        x_norm = self.norm_logic(x)

        # Helper to gather global neighbors with masking
        def gather_neighbors(op_data):
            indices, mask = op_data
            k = indices.shape[1]
            flat_indices = indices.view(-1)
            x_gathered = x_norm[:, flat_indices, :]
            x_gathered = x_gathered.view(B, 256, k, D)
            mask_b = mask.view(1, 256, k, 1)
            return x_gathered * mask_b

        s0_neigh = gather_neighbors(self.op_indices['sigma0'])
        s1_neigh = gather_neighbors(self.op_indices['sigma1'])
        s0w_neigh = gather_neighbors(self.op_indices['sigma0_w'])
        s1w_neigh = gather_neighbors(self.op_indices['sigma1_w'])
        v_neigh = gather_neighbors(self.op_indices['vertical'])
        c_neigh = gather_neighbors(self.op_indices['carry'])
        
        # Flatten
        s0_flat = s0_neigh.reshape(B, L, -1)
        s1_flat = s1_neigh.reshape(B, L, -1)
        s0w_flat = s0w_neigh.reshape(B, L, -1)
        s1w_flat = s1w_neigh.reshape(B, L, -1)
        v_flat = v_neigh.reshape(B, L, -1)
        c_flat = c_neigh.reshape(B, L, -1)
        
        # Concatenate
        combined = torch.cat([
            x_norm, 
            s0_flat, s1_flat, 
            s0w_flat, s1w_flat, 
            v_flat, c_flat
        ], dim=-1)
        
        # Project Logic Gate
        gate_out = self.logic_gate(combined)
        
        # Residual 1
        x = x + gate_out
        
        # Residual 2 + ConvSwiGLU
        # Pre-Norm happens inside ConvSwiGLU or here? 
        # Original code had norm_mlp. We'll do Pre-Norm here to match logic.
        x = x + self.conv_swiglu(self.norm_mlp(x))
        
        return x

class BitConvSwiGLU(nn.Module):
    """
    Replaces standard FeedForward with a convolutional SwiGLU block.
    Captures local bit patterns using depthwise convolution.
    """
    def __init__(self, d_model, d_hidden):
        super().__init__()
        # Expand
        self.w1 = BitLinear(d_model, d_hidden)
        # Depthwise Conv: Groups=d_hidden means each channel is convolved independently
        # Kernel 3, Padding 1 keeps sequence length same
        self.conv = nn.Conv1d(
            in_channels=d_hidden, 
            out_channels=d_hidden, 
            kernel_size=3, 
            padding=1, 
            groups=d_hidden
        )
        self.act = nn.SiLU()
        # Project
        self.w2 = BitLinear(d_hidden, d_model)

    def forward(self, x):
        # x: [B, L, D]
        # BitLinear expects [B, L, D]
        
        # 1. Expand
        x = self.w1(x) # [B, L, 4D]
        
        # 2. Conv1d expects [B, Channels, Length]
        # Transpose: [B, L, 4D] -> [B, 4D, L]
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2) # Back to [B, L, 4D]
        
        # 3. Activation
        x = self.act(x)
        
        # 4. Project
        x = self.w2(x)
        
        return x

class RecurrentSparseLogic(nn.Module):
    """
    A Super-Block that applies the same SparseLogicBlock iteratively.
    Equation: H_new = H_old + tanh(gate) * Block(H_old)
    """
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.loops = config.recurrent_loops
        
        # The Shared Block
        self.block = SparseLogicBlock(config.dim, config.n_heads)
        
        # Learnable Gate (Scalar), initialized to 0
        self.gate = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Loop with Gradient Checkpointing
        for _ in range(self.loops):
            if self.training:
                # Use checkpointing for memory efficiency
                delta = grad_checkpoint(self.block, x, use_reentrant=False)
            else:
                delta = self.block(x)
                
            # Gated Update
            x = x + torch.tanh(self.gate) * delta
            
        return x

class SparseLogicTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding
        self.embedding = nn.Embedding(2, config.dim) 
        self.pos_embed = nn.Parameter(torch.randn(1, 256, config.dim) * 0.02)
        
        # Recurrent Super-Block (Single Instance!)
        self.core = RecurrentSparseLogic(config)
        
        self.norm_f = nn.LayerNorm(config.dim)
        self.head = BitLinear(config.dim, 1)

    def forward(self, x):
        # x: [B, 256] (0/1 ints)
        x = self.embedding(x) + self.pos_embed
        
        # Run Recurrent Block
        x = self.core(x)
            
        x = self.norm_f(x)
        logits = self.head(x).squeeze(-1) # [B, 256]
        return logits
