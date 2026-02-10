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
        
        # Get static wiring indices (Global [256, k])
        self.op_indices = SHA256Wiring.get_op_indices() 
        
        # Projections
        # Input features per bit:
        # 1. Identity (1)
        # 2. Sigma0 (3 neighbors)
        # 3. Sigma1 (3 neighbors)
        # 4. Vertical (8 neighbors - corresponding bits in all words)
        
        # Total gathered vectors per bit: 1 + 3 + 3 + 8 = 15 vectors.
        # This provides the necessary information for Maj (3 inputs) and Ch (3 inputs) and Sum (3 inputs)
        # across words.
        
        self.input_mix_dim = dim * (1 + 3 + 3 + 8) 
        
        self.logic_gate = BitLinear(self.input_mix_dim, dim)
        
        # Pre-Norm Architecture: Need norm before Mixing and before MLP
        self.norm_logic = nn.LayerNorm(dim)
        self.norm_mlp = nn.LayerNorm(dim) # Renamed from self.norm for clarity
        
        self.mlp = nn.Sequential(
            BitLinear(dim, dim * 4),
            nn.GELU(),
            BitLinear(dim * 4, dim)
        )

    def forward(self, x):
        # Use gradient checkpointing during training to slash activation memory
        if self.training:
            return grad_checkpoint(self._forward_impl, x, use_reentrant=False)
        else:
            return self._forward_impl(x)

    def _forward_impl(self, x):
        # x: [B, 256, D]
        B, L, D = x.shape
        device = x.device
        
        # Indices need to be on device
        if self.op_indices['sigma0'].device != device:
             self.op_indices = {k: v.to(device) for k, v in self.op_indices.items()}
             
        # Pre-Norm
        x_norm = self.norm_logic(x)

        # Helper to gather global neighbors
        def gather_neighbors(indices):
            k = indices.shape[1]
            flat_indices = indices.view(-1)
            x_gathered = x_norm[:, flat_indices, :]
            return x_gathered.view(B, 256, k, D)

        s0_neigh = gather_neighbors(self.op_indices['sigma0']) # [B, 256, 3, D]
        s1_neigh = gather_neighbors(self.op_indices['sigma1']) # [B, 256, 3, D]
        v_neigh = gather_neighbors(self.op_indices['vertical']) # [B, 256, 8, D]
        
        # Flatten the neighbor dimension for Projection
        s0_flat = s0_neigh.reshape(B, L, -1) # [B, 256, 3*D]
        s1_flat = s1_neigh.reshape(B, L, -1) # [B, 256, 3*D]
        v_flat = v_neigh.reshape(B, L, -1)   # [B, 256, 8*D]
        
        # Concatenate: [Identity (Normed), Sigma0, Sigma1, Vertical]
        combined = torch.cat([x_norm, s0_flat, s1_flat, v_flat], dim=-1)
        
        # Project Logic Gate
        gate_out = self.logic_gate(combined)
        
        # Residual 1
        x = x + gate_out
        
        # Residual 2 + MLP
        x = x + self.mlp(self.norm_mlp(x))
        
        return x

class SparseLogicTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding: 256 bits (state) -> D
        # We treat each bit position as a token
        # Plus positional embedding for 0..255?
        # Or just 0/1 embedding?
        # Neuro-SHA: "Forces the model to only look at bits that actually matter"
        
        # We input the *Assignment Vector* (Logic State)
        self.embedding = nn.Embedding(2, config.dim) 
        self.pos_embed = nn.Parameter(torch.randn(1, 256, config.dim) * 0.02)
        
        self.blocks = nn.ModuleList([
            SparseLogicBlock(config.dim, config.n_heads)
            for _ in range(config.n_layers)
        ])
        
        self.norm_f = nn.LayerNorm(config.dim)
        # Output: Probability that this bit is "Correct" or "Active"?
        # Or "Glue Variable Scores"?
        self.head = BitLinear(config.dim, 1)

    def forward(self, x):
        # x: [B, 256] (0/1 ints)
        x = self.embedding(x) + self.pos_embed
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm_f(x)
        logits = self.head(x).squeeze(-1) # [B, 256]
        return logits
