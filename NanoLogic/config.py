from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # ── Model Architecture ( The "Heavy Lifter" ) ───────────────────────
    # Strategy: Width > Depth. A wider bus (512) allows complex XOR/ROT 
    # logic to happen in a single step, while fewer layers (12) saves RAM.
    dim: int = 512              # WAS: 256. Doubled for logic capacity.
    n_layers: int = 24          # WAS: 24. Halved to fit in RAM with dim=512.
    n_heads: int = 16
               # Parallel logic paths (Vertical Wiring).
    vocab_size: int = 2         # Binary (0, 1) input. Output is ternary weights.

    # ── Wiring & Logic ──────────────────────────────────────────────────
    wiring_mode: str = "static_sha256"  # Hard-coded SHA-256 graph.
    rounds: int = 64            # The final goal.

    # ── Pathfinder (ResNet) ─────────────────────────────────────────────
    pathfinder_depth: int = 10  # Auxiliary network depth.
    
    # ── Memory Guard ( M4 Optimization ) ────────────────────────────────
    # Effective Batch Size = batch_size * grad_accum_steps = 64
    batch_size: int = 2         # Kept minimal to prevent activation explosion.
    grad_accum_steps: int = 64  # Accumulate 32 micro-batches before stepping.
    num_workers: int = 0        # 0 = Main Thread (Crucial for MacOS/MPS stability).
    pin_memory: bool = False    # Disabled to save physical RAM.

    # ── Training ( The "Precision" Protocol ) ───────────────────────────
    lr: float = 3e-5            # WAS: 1e-4. Slowed down 3x for Lion stability.
    weight_decay: float = 0.01  # Standard for Lion.
    warmup_steps: int = 1000    # Gentle wake-up for the optimizer.
    
    # Solver / Loop Limits
    max_solver_steps: int = 1000000000  # Effectively Infinite. (Ctrl+C to stop)

    # ── Curriculum ( The "Teacher" ) ────────────────────────────────────
    # Training starts at 8 rounds. It will NOT advance until accuracy >= 95%.
    start_round: int = 8
    
    # The Ladder: [8 -> 16 -> 32 -> 64]
    curriculum_rounds: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    
    # The Bar: Accuracy required to graduate from each phase
    # Phase 0 (8r): Must hit 95%
    # Phase 1 (16r): Must hit 85%
    # Phase 2 (32r): Must hit 75%
    # Phase 3 (64r): Runs forever (Threshold > 1.0)
    phase_accuracy_thresholds: List[float] = field(default_factory=lambda: [0.80, 0.70, 0.60, 0.55])
    
    # The Grind: Minimum steps to force in each phase before checking promotion
    phase_min_steps: List[int] = field(default_factory=lambda: [1000, 2000, 5000, 0])