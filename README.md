<div align="center">

# 🧠 NEURO-SHA-M4

### *Neuro-Symbolic SHA-256 Cryptanalysis on Apple Silicon*

**Breaking SHA-256 logic on a MacBook Air M4 using < 10 GB RAM.**

[![Status](https://img.shields.io/badge/Status-Active%20Training-brightgreen)]()
[![License](https://img.shields.io/badge/License-Non--Commercial-red)](#license)
[![Platform](https://img.shields.io/badge/Platform-Apple%20M4-blue)]()
[![RAM](https://img.shields.io/badge/RAM%20Usage-0.2GB-success)]()

[Architecture](#architecture) · [Live Results](#live-training-results) · [Usage](#usage) · [CLI Demo](#neuro-cli) · [License](#license)

*Built by a Medical Student / Vibe Coder* · [𝕏 @MohamedHz72007](https://x.com/MohamedHz72007)

</div>

---

## What is This?

Neuro-SHA-M4 is a **neuro-symbolic framework** that learns the internal logic of SHA-256 and uses that knowledge to guide a SAT solver toward preimage solutions.

Instead of trying to "invert" SHA-256 with brute force (impossible) or with a naive neural network (also impossible), we do something smarter:

> **The neural network learns *where to look*. The symbolic solver proves *what's there*.**

The model currently achieves **62%+ bit-prediction accuracy** on SHA-256 internal state transitions — running on a single MacBook Air M4 with **0.2GB RAM usage**.

---

## Architecture

### 1. Sparse Logic Attention — O(N), Not O(N²)

Standard Transformers attend to *everything*. But SHA-256 is **sparse** — each bit only depends on a handful of neighbors defined by the `Σ₀`, `Σ₁`, `Maj`, and `Ch` functions.

We hard-code the attention mask to mirror the **exact wiring diagram** of SHA-256:

```
Bit[i] attends to:
  → Itself (identity)
  → ROTR(2,13,22) neighbors  (Σ₀ wiring)
  → ROTR(6,11,25) neighbors  (Σ₁ wiring)
  → Same bit across all 8 words (Vertical/Inter-word wiring)
```

**Result:** Instead of `256 × 256 = 65,536` attention weights per layer, we use `256 × 15 = 3,840`. That's a **17× reduction** — enabling 24 layers on a laptop.

### 2. BitNet b1.58 — Ternary Weights {-1, 0, 1}

Every linear layer uses **1.58-bit quantized weights**:

| Weight | Meaning |
|--------|---------|
| `+1` | Pass this bit |
| `-1` | Invert this bit (NOT) |
| `0` | Ignore this bit |

This isn't just compression — it's an **inductive bias for boolean logic**. The network naturally learns AND/OR/XOR gates without floating-point drift. A 36M parameter model fits in ~25MB.

### 3. Neuro-Symbolic Bridge — Guiding Z3 with Heatmaps

The trained model generates **variable importance heatmaps** that tell the SAT solver which bits to assign first:

```
Process A (Solver/CPU)          Process B (Neural Oracle/MPS)
       │                                │
       │── Assignment Vector ──────────→│
       │                                │── Sparse Logic Transformer
       │←── Priority Heatmap ──────────│
       │                                │
  CDCL Search                     BitNet Inference
  (Kissat/Z3)                      (<5ms latency)
```

The solver runs VSIDS for 5,000 conflicts, then queries the neural oracle. The oracle returns "glue variable" probabilities that refocus the solver on the structurally critical bits.

---

## Live Training Results

> Phase 1: 16-Round Logic Learning | MacBook Air M4 | 0.2GB RAM

| Step | Loss | Accuracy | RAM (GB) |
|-----:|-----:|---------:|---------:|
| 10 | 0.723 | 48.1% | 0.14 |
| 100 | 0.937 | 52.9% | 0.26 |
| 200 | 0.721 | 60.0% | 0.26 |
| 350 | 0.649 | 61.7% | 0.26 |
| 500 | 0.670 | 60.4% | 0.21 |
| 530 | 0.643 | **63.0%** | 0.09 |

The model is in active training, progressing through a 3-phase curriculum:
- **Phase 1** (Steps 0–1,000): 16-round SHA-256 logic
- **Phase 2** (Steps 1,000–5,000): 32-round extended chains
- **Phase 3** (Steps 5,000+): Full 64-round SHA-256

---

## Project Structure

```
NanoLogic/
├── main.py                     # Training entry point (auto-resume)
├── config.py                   # All hyperparameters
├── src/
│   ├── model/
│   │   ├── sparse_logic.py     # Sparse Logic Transformer (gradient checkpointing)
│   │   ├── bitnet.py           # BitNet b1.58 quantization layer
│   │   ├── wiring.py           # SHA-256 static wiring + trace generator
│   │   └── pathfinder.py       # ResNet-1D distinguisher
│   ├── optim/
│   │   └── lion_galore.py      # Lion optimizer with GaLore projection
│   ├── train/
│   │   ├── synthetic.py        # Lazy trace generator (zero-storage)
│   │   ├── curriculum.py       # 3-phase curriculum scheduler
│   │   └── loss.py             # BCE + Hamming distance loss
│   ├── solver/
│   │   ├── bridge.py           # Shared memory bridge (zero-copy)
│   │   └── cnf_utils.py        # SAT encoding utilities
│   └── utils/
│       └── monitor.py          # MemoryGuard (10GB ceiling)
├── tools/
│   └── neuro_cli.py            # Interactive demo CLI (rich)
├── checkpoints/                # Auto-saved every 500 steps
├── logs/                       # CSV training logs
├── LICENSE                     # PolyForm Noncommercial 1.0.0
└── COMMERCIAL_TERMS.md         # 60/40 profit-share for commercial use
```

---

## Usage

### Requirements

```bash
pip install torch psutil rich
```

### Train

```bash
cd NanoLogic
python3 main.py
```

Training auto-resumes from the latest checkpoint in `checkpoints/`. Press `Ctrl+C` to safely stop — progress is always saved.

### CLI Demo

```bash
# Watch AI vs Brute Force race
python3 tools/neuro_cli.py --mode race

# Interactive hash cracker dashboard
python3 tools/neuro_cli.py --mode crack
```

---

## The Vibe Note

> *This project is built by a medical student who codes between anatomy lectures. The constraint isn't a datacenter — it's a MacBook Air. The optimization isn't FLOPS — it's RAM. The goal isn't to break SHA-256 (yet) — it's to prove that a purpose-built logic engine, running on consumer silicon, can learn a structure that was designed to be unlearnable.*
>
> *If you're reading this and thinking "that's impossible" — good. That's the point.*
>
> — [@MohamedHz72007](https://x.com/MohamedHz72007)

---

## License

This project is licensed under the **[PolyForm Noncommercial License 1.0.0](LICENSE)**.

- ✅ Free for research, education, and personal use
- ✅ Free to modify and redistribute (non-commercially)
- ❌ **Commercial use is strictly prohibited** without a signed agreement

**Want to use this commercially?** See **[COMMERCIAL_TERMS.md](COMMERCIAL_TERMS.md)** or contact [@MohamedHz72007](https://x.com/MohamedHz72007).

---

## Citation

If you use this work in research, please cite:

```bibtex
@software{neurosham4_2026,
  author = {Mohamed Hossam},
  title = {NanoLogic: Neuro-Symbolic SHA-256 Cryptanalysis on Apple Silicon},
  year = {2026},
  url = {https://github.com/mohammedhossammohammed/NanoLogic}
}
```
# NanoLogic
