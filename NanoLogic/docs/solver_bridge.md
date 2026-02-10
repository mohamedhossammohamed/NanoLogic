# Neuro-Symbolic Solver Bridge

## Overview

The Solver Bridge connects the **neural training loop** (Python/MPS) with a **symbolic SAT solver** (Kissat/Z3) via shared memory. The neural network provides "intuition" (variable importance heatmaps) while the solver provides "rigor" (constraint propagation and proof).

## Architecture

```
┌──────────────────────────┐     Shared Memory     ┌──────────────────────────┐
│   Process A: Solver      │    (Zero-Copy IPC)     │  Process B: Neural       │
│   (CPU - CDCL Loop)      │◄──────────────────────►│  Oracle (MPS/ANE)        │
│                          │                        │                          │
│  1. Propagate            │   assignment_vector    │  1. Read assignment      │
│  2. Conflict → Learn     │  ──────────────────►   │  2. Sparse Transformer   │
│  3. Every 5K conflicts:  │                        │     inference (<5ms)     │
│     → Write assignment   │   priority_vector      │  3. Write priorities     │
│     → Read priorities    │  ◄──────────────────   │                          │
│  4. Update VSIDS scores  │                        │                          │
│     Score(v) = α·VSIDS   │                        │                          │
│              + β·Neural  │                        │                          │
└──────────────────────────┘                        └──────────────────────────┘
```

## Shared Memory Protocol

The bridge uses `multiprocessing.shared_memory.SharedMemory` for zero-copy data transfer.

### Buffer Layout (4MB)

```
Offset 0x00:  [Flag]  int32   — 0: Empty, 1: Assignment Ready, 2: Scores Ready
Offset 0x04:  [Size]  int32   — Payload size in bytes
Offset 0x08:  [Data]  bytes   — Variable-length payload
```

### Communication Flow

```python
# Solver writes assignment (Flag=1):
buffer[0:4] = int32(1)          # Flag: data ready
buffer[4:8] = int32(len(data))  # Size
buffer[8:]  = data              # Assignment vector (int8[])

# Neural Oracle reads, processes, writes scores (Flag=2):
buffer[0:4] = int32(2)          # Flag: result ready
buffer[4:8] = int32(len(scores))
buffer[8:]  = scores            # Priority scores (float32[])
```

## Neural Guidance: The Query Function

```python
def query_neural_guide(self, assignment_vector, model, device='cpu'):
    # 1. Convert solver assignment → 256-bit tensor
    state_tensor = assignment_to_tensor(assignment_vector)
    state_tensor = state_tensor[:, -256:]  # Current state only

    # 2. Inference (no gradients)
    with torch.no_grad():
        scores = model(state_tensor)  # [1, 256]

    # 3. Return as numpy priority map
    return scores.cpu().numpy().flatten()
```

The scores represent P(bit is correctly assigned). The solver uses these to update its VSIDS heuristic:

```
Score(v) = α · VSIDS_old(v) + β · Neural_pred(v)
```

Variables with high neural confidence get higher priority → the solver explores them first.

## Integration Strategy: NeuroGlue

The neural oracle is **not** called every step. It follows the "Conflict Budget" strategy:

1. Solver runs 5,000 CDCL conflicts with standard VSIDS
2. Solver pauses and snapshots current assignment
3. Neural oracle runs inference (~5ms on ANE)
4. Solver updates VSIDS scores with neural priorities
5. Solver resumes with "refocused" heuristic

This amortizes neural inference cost over thousands of symbolic steps.

## Current Status

All solver bridge components are implemented and functional.

### Integration Checklist

- [x] Shared memory allocation and cleanup
- [x] Read/write protocol
- [x] Assignment → tensor conversion (`cnf_utils.py`)
- [x] Neural inference pipeline
- [x] Z3 solver integration (`z3_sha256.py`)
- [x] VSIDS score injection hook (`bridge.py`)
- [x] Full end-to-end preimage search loop (`neuro_cdcl.py`)

---

## Z3 SHA-256 Encoder

[z3_sha256.py](../src/solver/z3_sha256.py) encodes the full SHA-256 compression function using Z3 bit-vectors.

### Key Functions

| Function | Description |
|---|---|
| `SHA256Solver.solve_preimage(hash_hex)` | Pure Z3 preimage search, no neural guidance |
| `SHA256Solver.solve_partial(hash_hex, hints, threshold)` | Neural-guided: fixes high-confidence bits before solving |

### Encoding Strategy

```
16 symbolic BitVec(32) message words W[0..15]
    → Message schedule expansion: W[16..63] via σ₀, σ₁
    → 64-round compression: Σ₀, Σ₁, Ch, Maj (all as Z3 bit-vector ops)
    → Output constrained to match target hash
```

Neural hints are injected as Z3 `Extract` constraints on individual bits:
```python
# Fix bit 42 to 1 (high neural confidence):
solver.add(Extract(bit_pos, bit_pos, W[word_idx]) == BitVecVal(1, 1))
```

---

## VSIDS Score Injection Hook

[bridge.py](../src/solver/bridge.py) `inject_vsids_scores()` blends neural confidence with Z3's decision heuristic.

```
Score(v) = α · VSIDS_old(v) + (1-α) · Neural_pred(v)
```

Since Z3 doesn't expose raw VSIDS scores, we approximate blending via **soft constraints** (assumptions):
- Scores above threshold (default 0.85) → fix bit to 1
- Scores below `1 - threshold` → fix bit to 0
- Middle range → let Z3 decide freely

The `alpha` parameter (default 0.3) scales neural confidence to prevent over-constraining.

---

## NeuroCDCL End-to-End Loop

[neuro_cdcl.py](../src/solver/neuro_cdcl.py) orchestrates the full search:

```
┌─── Iteration 0 ────────────────────────────────────┐
│  Z3 pure solve (no hints) → timeout (5s)           │
│  Extract partial assignment → Neural inference      │
│  Neural confidence → [0.92, 0.12, 0.87, ...]       │
│  Fix 47 high-confidence bits as constraints         │
└─────────────────────────────────────────────────────┘
         │
         ▼
┌─── Iteration 1 ────────────────────────────────────┐
│  Z3 solve_partial (47 bits fixed) → timeout (5s)   │
│  Tighten threshold: 0.85 → 0.87                    │
│  Neural re-inference → fix 63 bits                  │
└─────────────────────────────────────────────────────┘
         │
         ▼  ... repeat until SAT, UNSAT, or exhausted
```

### Configuration

| Parameter | Default | Description |
|---|---|---|
| `rounds` | 16 | SHA-256 rounds (16 for fast, 64 for full) |
| `max_iterations` | 10 | Maximum refinement loops |
| `confidence_threshold` | 0.85 | Initial threshold for fixing bits |
| `z3_timeout_ms` | 5000 | Z3 timeout per iteration |
| `confidence_growth` | 0.02 | Threshold increase per iteration |

### Usage

```python
from src.solver import NeuroCDCL, SHA256Solver

# Z3-only baseline (no neural model)
search = NeuroCDCL(model=None, rounds=16, max_iterations=5)
result = search.search("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")

# With trained model
model = NeuroCDCL.load_model("checkpoints/neuro_sha_step_5000.pt", config)
search = NeuroCDCL(model=model, device='mps', rounds=16)
result = search.search(target_hash_hex)
```
