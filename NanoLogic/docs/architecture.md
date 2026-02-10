# Sparse Logic Transformer — Architecture Deep-Dive

## Overview

The Sparse Logic Transformer is the core neural component of Neuro-SHA-M4. It replaces standard self-attention with a **hard-coded sparse wiring pattern** that mirrors the exact dependency graph of SHA-256.

## Why Not Standard Transformers?

| Property | Standard Transformer | Sparse Logic Transformer |
|----------|---------------------|--------------------------|
| Attention complexity | O(N²) — 65,536 ops | O(N·k) — 3,840 ops |
| Attention pattern | Learned (data-dependent) | Static (SHA-256 wiring) |
| Memory per layer | ~32MB (dense QKV) | ~4MB (sparse gather) |
| Inductive bias | Semantic similarity | Boolean logic gates |

SHA-256 is **not semantic** — it's a deterministic state machine. A standard Transformer wastes capacity learning attention patterns that we already know from the algorithm specification.

## The Wiring Diagram

Each of the 256 bit positions (8 words × 32 bits) gathers information from exactly **15 neighbors**:

```
Bit[i] in Word[w] attends to:
│
├── Identity:   Bit[i] itself (1 vector)
│
├── Σ₀ Group:   ROTR(2), ROTR(13), ROTR(22) of Bit[i] within Word[w]
│               → 3 vectors from intra-word rotation
│
├── Σ₁ Group:   ROTR(6), ROTR(11), ROTR(25) of Bit[i] within Word[w]
│               → 3 vectors from intra-word rotation
│
└── Vertical:   Bit[i] in Word[0], Word[1], ..., Word[7]
                → 8 vectors from inter-word (same position, different register)
```

**Total: 1 + 3 + 3 + 8 = 15 vectors per bit position.**

This is defined statically in [`wiring.py`](../src/model/wiring.py) via `SHA256Wiring.get_op_indices()`.

## Architecture Diagram

```
Input: [B, 256] (binary assignment vector, 0/1 integers)
  │
  ▼
┌─────────────────────────────┐
│  Embedding(2, dim)          │  Map 0/1 → D-dimensional vectors
│  + Positional Embedding     │  Learnable [1, 256, D]
└─────────────────────────────┘
  │
  ▼  (×24 layers)
┌─────────────────────────────────────────────────┐
│  SparseLogicBlock                               │
│                                                 │
│  1. Pre-Norm (LayerNorm)                        │
│  2. Gather neighbors via static wiring indices  │
│     ├── Σ₀ neighbors [B, 256, 3, D]             │
│     ├── Σ₁ neighbors [B, 256, 3, D]             │
│     └── Vertical     [B, 256, 8, D]             │
│  3. Concatenate: [B, 256, 15·D]                 │
│  4. BitLinear projection → [B, 256, D]          │
│  5. Residual connection                         │
│  6. Pre-Norm → MLP (BitLinear 4× expand) → Res  │
│                                                 │
│  ⚡ Gradient checkpointing during training      │
└─────────────────────────────────────────────────┘
  │
  ▼
┌─────────────────────────────┐
│  Final LayerNorm            │
│  BitLinear(D, 1) → squeeze  │  Per-bit logit: P(bit is correct)
└─────────────────────────────┘
  │
  ▼
Output: [B, 256] logits
```

## Key Design Decisions

### Pre-Norm (Not Post-Norm)

LayerNorm is applied **before** the logic gate and **before** the MLP, not after. This prevents signal explosion through the 24-layer stack — critical because BitNet quantization already introduces noise at each layer.

```python
# Pre-Norm: normalize BEFORE the logic gate
x_norm = self.norm_logic(x)
# ... gather neighbors from x_norm ...
gate_out = self.logic_gate(combined)
x = x + gate_out  # Residual on un-normed x
```

### Gradient Checkpointing

Each `SparseLogicBlock` uses `torch.utils.checkpoint` during training. This trades compute for memory — intermediate activations are recomputed during backward instead of stored. This is essential for fitting 24 layers in <0.3GB.

```python
def forward(self, x):
    if self.training:
        return grad_checkpoint(self._forward_impl, x, use_reentrant=False)
    else:
        return self._forward_impl(x)
```

### Vertical Wiring (Inter-Word Attention)

The vertical wiring connects the same bit position across all 8 SHA-256 working registers (a, b, c, d, e, f, g, h). This is critical because the `Maj(a, b, c)` and `Ch(e, f, g)` functions mix bits at the same position across different words.

```python
# For bit position `b`, vertical neighbors are:
# Word 0, bit b | Word 1, bit b | ... | Word 7, bit b
vertical_indices = []
for w in range(8):
    v_idx = w * 32 + bit_idx  # Same bit in each word
    vertical_indices.append(v_idx)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dim` | 256 | Hidden dimension per bit position |
| `n_layers` | 24 | Number of SparseLogicBlocks |
| `n_heads` | 8 | Parallel logic paths |
| `input_mix_dim` | 256 × 15 = 3,840 | Concatenated neighbor features |

## Parameter Count

- **Per layer**: ~4M parameters (logic gate + MLP, all BitLinear)
- **Total model**: ~36M parameters
- **Storage**: ~25MB (ternary packed weights)
- **Runtime RAM**: ~0.2GB on MPS
