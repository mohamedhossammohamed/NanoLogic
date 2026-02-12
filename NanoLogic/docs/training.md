# Training Pipeline — Curriculum, Optimizer, & Memory Guard

## Overview

The training pipeline is designed around three pillars:
1. **Curriculum Learning** — scale difficulty from 8 to 64 SHA-256 rounds, gated by accuracy
2. **Lion + GaLore Optimizer** — memory-efficient sign-based updates
3. **Memory Guard** — active RAM defense system

## Curriculum Scheduler

The model learns SHA-256 logic progressively, with **accuracy-gated promotion**:

| Phase | Rounds | Min Steps | Accuracy Gate | Goal |
|:---:|:---:|:---:|:---:|---|
| 0 | 8 | 500 | ≥ 95% | Learn basic boolean gate patterns |
| 1 | 16 | 2,000 | ≥ 95% | Local Σ₀/Σ₁ rotational logic |
| 2 | 32 | 5,000 | ≥ 95% | Extend to longer dependency chains |
| 3 | 64 | ∞ | — | Full SHA-256 (runs forever) |

### Promotion Conditions

Phase promotion requires **BOTH**:
1. ✅ Minimum steps completed in the current phase
2. ✅ Running average accuracy ≥ phase threshold (default 95%)

```python
# Promotion check (every step):
if current_step >= phase_min_steps[phase] and running_accuracy >= phase_threshold[phase]:
    promote_to_next_phase()
```

If the model hasn't reached 95% accuracy, it **stays** in the current phase indefinitely — no wasted compute on harder rounds the model can't handle yet.

### Configuration (config.py)

All curriculum parameters are configurable:

```python
@dataclass
class Config:
    # Starting round — always begin here, even on checkpoint resume
    start_round: int = 8
    
    # Phase schedule
    curriculum_rounds: List[int] = [8, 16, 32, 64]
    phase_min_steps: List[int] = [500, 2000, 5000, 0]
    
    # Per-phase accuracy gates (last is 1.0 = unreachable, runs forever)
    phase_accuracy_thresholds: List[float] = [0.95, 0.95, 0.95, 1.0]
```

To start training at 16 rounds instead of 8:
```python
config = Config(start_round=16)
```

To lower the accuracy gate for faster progression:
```python
config = Config(phase_accuracy_thresholds=[0.80, 0.85, 0.90, 1.0])
```

### Why Curriculum?

SHA-256's 64-round structure creates exponentially long dependency chains. Training on all 64 rounds from Step 0 would mean:
- Each training sample contains 64 state transitions
- The model must learn carry propagation across all rounds simultaneously
- Gradients from early rounds are diluted by 64× the chain length

By starting with 8 rounds, the model first masters the **basic logic** before scaling.

## Loss Function

**BCE with Logits** on per-bit predictions:

```python
class StateMatchingLoss:
    def forward(self, pred_state, target_state):
        # pred_state: [B, 256] logits
        # target_state: [B, 256] 0/1 binary
        return BCEWithLogitsLoss(pred_state, target_state.float())
```

The model predicts the **next SHA-256 state** given the current state. Each of the 256 bits is an independent binary classification.

**Random baseline**: BCE loss ≈ 0.693, Accuracy ≈ 50%.

## Data Generation (Shared Memory)

Data is streamed via `SharedMemoryLoader` to eliminate disk I/O bottlenecks:

- **Ring Buffer**: A pool of pre-allocated POSIX Shared Memory blocks.
- **CPU Producers**: Parallel worker processes generate SHA-256 traces using `SHA256Wiring.generate_trace`.
- **Zero-Copy**: Tensors are mapped directly from shared memory onto the main training process.
- **Auto-Sync**: Uses `multiprocessing.Queue` to manage empty/full buffer handovers.

```python
# Initialization in main.py:
loader = SharedMemoryLoader(batch_size=64, rounds=16, num_workers=4)

# Consumption:
inputs, targets = loader.get_batch(device='mps')
```
## Lion Optimizer with GaLore

### Lion (Evolved Sign Momentum)

Lion uses the **sign** of the gradient for updates, not the magnitude:

```python
update = exp_avg * β₁ + grad * (1 - β₁)
p.add_(sign(update), alpha=-lr)
```

**Why Lion for BitNet?**
- SHA-256 logic creates **wildly varying gradient magnitudes** (avalanche effect)
- `sign()` normalizes all gradient directions equally — acts as a robust regularizer
- Only stores **one momentum buffer** (not two like Adam) → 50% memory savings

### GaLore Low-Rank Projection

The momentum buffer is stored in **bfloat16** to halve memory further:

```python
state['exp_avg'] = torch.zeros_like(p, dtype=torch.bfloat16)
```

## Gradient Accumulation

Effective batch size = `batch_size × grad_accum_steps` = 2 × 32 = **64**.

```python
# Scale loss for accumulation
scaled_loss = loss / grad_accum
scaled_loss.backward()

# Only step every 32 micro-batches
if step % grad_accum == 0:
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()
```

**Gradient clipping** (`max_norm=1.0`) prevents gradient explosion from BitNet's STE backward pass.

## Memory Guard

The `MemoryGuard` actively monitors system RAM and intervenes if usage approaches the 10GB ceiling:

```
Every 10 steps:
  1. Poll psutil.virtual_memory()
  2. If used_gb > 10.0:
     → gc.collect()
     → torch.mps.empty_cache()
     → Re-check
     → If still high: sleep(30) for swap drain
```

### Emergency Protocol

```
[MEMORY GUARD] ⚠️  RAM Warning: 10.5GB / 10.0GB Limit
[MEMORY GUARD] 🧹 Triggering Garbage Collection & Cache Clear...
[MEMORY GUARD] ✅ Cooled down to 9.2GB
```

If GC doesn't help, the guard pauses training for 30 seconds to let macOS drain swap pages.

## Auto-Resume

The training loop supports fully automatic checkpoint resumption:

1. On **Ctrl+C**: saves `neuro_sha_final.pt` with model, optimizer, and scheduler state
2. On **restart**: scans `checkpoints/` for the most recent `.pt` file
3. Loads model weights and optimizer state, but **always starts at `config.start_round`**
4. Total step count is preserved for logging continuity

```python
# Auto-detected on startup:
🔄 Found checkpoint: checkpoints/neuro_sha_step_500.pt
   ✅ Resumed model weights | Total steps so far: 500
   📍 Starting at Phase 0 (8 rounds) per config.start_round=8
```

### Important: start_round Controls Phase

Even if a checkpoint was saved at phase 2 (32 rounds), restarting with `config.start_round = 8` will place the model back at phase 0. This ensures you can always control where training resumes from.

## Training Log

The training log is a single, append-only CSV at `logs/training.log`:

```csv
step,loss,accuracy,threshold,ram_gb,phase,rounds
10,0.733621,0.505225,0.95,0.25,0,8
20,0.732146,0.505981,0.95,0.30,0,8
```

The log is **never overwritten** — every restart appends to the same file.

## Mixed Precision (FP16)

The framework uses **FP16 Mixed Precision** with `torch.amp.GradScaler` optimized for Apple Silicon (MPS):

- **Autocast**: Forward pass runs in `float16` to double throughput and halve activation memory.
- **GradScaler**: Prevents gradient underflow by scaling loss before backward.
- **BitNet STE**: Gradients are unscaled before clipping to maintain stability in the ternary logic space.

## Checkpoint Format

```python
{
    'step': int,                    # Total training steps completed
    'model_state_dict': dict,       # SparseLogicTransformer weights
    'optimizer_state_dict': dict,   # Lion momentum buffers
    'scheduler_state_dict': dict,   # Curriculum state
    'scaler_state_dict': dict,      # GradScaler state (for consistency)
    'loss': float,                  # Loss at checkpoint time
}
```

Saved every **500 steps** to `checkpoints/neuro_sha_step_{N}.pt`.
