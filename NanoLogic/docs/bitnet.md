# BitNet b1.58 — Ternary Quantization Layer

## Overview

BitNet b1.58 constrains all linear layer weights to the ternary set **{-1, 0, 1}**, reducing storage from 16 bits (FP16) to ~1.58 bits per parameter. This isn't just compression — it's an **inductive bias for boolean logic**.

## Why Ternary Weights Work for SHA-256

| Weight Value | Boolean Meaning | Logic Gate Analogy |
|:---:|---|---|
| `+1` | Pass this bit through | Wire / Buffer |
| `-1` | Invert this bit | NOT gate |
| `0` | Ignore this bit | Disconnected |

When multiple ternary neurons combine via summation + activation, they naturally form AND, OR, and XOR approximations — the exact gates that compose SHA-256.

## Forward Pass

```python
def forward(ctx, input, weight, bias=None):
    # 1. Weight Quantization: FP32 → {-1, 0, 1}
    w = weight - weight.mean()        # Center weights
    gamma = w.abs().mean()             # Scale factor
    w_scaled = w / (gamma + 1e-5)
    w_quant = w_scaled.round().clamp(-1, 1)  # Snap to ternary

    # 2. Input Quantization: FP32 → INT8
    input_scale = 127.0 / (input.abs().max(dim=-1, keepdim=True).values + 1e-5)
    input_quant = (input * input_scale).round().clamp(-128, 127) / input_scale

    # 3. Compute: MatMul with quantized operands
    output = F.linear(input_quant, w_quant * gamma, bias)
    return output
```

**Key**: The output is `input_quant @ (w_quant * gamma)`. The `gamma` scaling preserves the magnitude of the original weights while the ternary quantization preserves the direction.

## Backward Pass (Straight-Through Estimator)

The quantization functions `round()` and `clamp()` have zero gradient almost everywhere. We use the **Straight-Through Estimator (STE)** — gradients flow through the quantization as if it weren't there, but with proper scaling.

```python
def backward(ctx, grad_output):
    input, w_quant, gamma = ctx.saved_tensors

    # grad_input: scale by gamma (matches forward scale)
    grad_input = grad_output.matmul(w_quant) * gamma

    # grad_weight: scale by 1/gamma (compensate for quantization)
    grad_weight = grad_output.transpose(-2, -1).matmul(input) / (gamma + 1e-5)

    return grad_input, grad_weight, grad_bias
```

### ⚠️ Critical: The `1/gamma` Scaling

The `grad_weight` **must** be divided by `gamma`. Without this:
- The forward pass multiplies quantized weights by `gamma`
- The backward pass must undo this scaling for the weight update
- Missing this causes **gradient explosion** — weights get outsized updates

This was a critical bug found and fixed during the code audit.

## Memory Footprint

| Model Size | FP16 Storage | BitNet Storage | Reduction |
|:---:|:---:|:---:|:---:|
| 36M params | 72 MB | ~7 MB | **10×** |
| 100M params | 200 MB | ~20 MB | **10×** |
| 1.2B params | 2.4 GB | ~240 MB | **10×** |

## Integration

`BitLinear` is a drop-in replacement for `nn.Linear`:

```python
from src.model.bitnet import BitLinear

# Instead of:
# layer = nn.Linear(256, 256)

# Use:
layer = BitLinear(256, 256)
```

All projections in `SparseLogicBlock` and the output head use `BitLinear`. The only standard `nn.Linear` alternatives are the embedding layer and layer norms (which don't have weights in the traditional sense).

## References

- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
- [The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits](https://arxiv.org/abs/2402.17764)
