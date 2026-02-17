from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # ── Architecture Selection ──────────────────────────────────────────
    architecture: str = "transformer"  # Options: "transformer", "biopcn"

    # ── Model Architecture ( The "Heavy Lifter" ) ───────────────────────
    # Strategy: Width > Depth. A wider bus (512) allows complex XOR/ROT 
    # logic to happen in a single step, while fewer layers (12) saves RAM.
    dim: int = 1024              # WAS: 256. Doubled for logic capacity.
    n_layers: int = 24          # WAS: 24. Halved to fit in RAM with dim=512.
    n_heads: int = 16
               # Parallel logic paths (Vertical Wiring).
    vocab_size: int = 2         # Binary (0, 1) input. Output is ternary weights.

    # ── BioPCN Hyperparameters (The "Living" Dynamics) ──────────────────
    pcn_settle_steps: int = 20         # Steps to "think" per batch
    pcn_alpha: float = 0.1             # Integration rate for state updates
    pcn_hebbian_lr: float = 0.001      # Learning rate for Hebbian plasticity
    pcn_temperature: float = 1.0       # Temperature for Langevin noise
    pcn_weight_threshold: float = 0.5  # Threshold for latent weight quantization

    # ── Wiring & Logic ──────────────────────────────────────────────────
    wiring_mode: str = "static_sha256"  # Hard-coded SHA-256 graph.
    wiring_mode: str = "static_sha256"  # Hard-coded SHA-256 graph.
    rounds: int = 64            # The final goal.
    
    # ── Recurrent Architecture (Research 4) ─────────────────────────────
    recurrent_loops: int = 12   # Number of iterations for the recurrent block.

    # ── Pathfinder (ResNet) ─────────────────────────────────────────────
    pathfinder_depth: int = 10  # Auxiliary network depth.
    
    # ── Memory Guard ( M4 Optimization ) ────────────────────────────────
    # Effective Batch Size = batch_size * grad_accum_steps = 64
    batch_size: int = 2         # Kept minimal to prevent activation explosion.
    grad_accum_steps: int = 64  # Accumulate 32 micro-batches before stepping.
    num_workers: int = 0        # 0 = Main Thread (Crucial for MacOS/MPS stability).
    pin_memory: bool = False    # Disabled to save physical RAM.
    use_checkpointing: bool = True # Enable gradient checkpointing for recurrent block.

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
    # Phase 0 (8r): Must hit 85%
    # Phase 1 (16r): Must hit 80%
    # Phase 2 (32r): Must hit 75%
    # Phase 3 (64r): Runs forever (Threshold > 0.7)
    phase_accuracy_thresholds: List[float] = field(default_factory=lambda: [0.85, 0.80, 0.80, 0.75])
    
    # The Grind: Minimum steps to force in each phase before checking promotion
    phase_min_steps: List[int] = field(default_factory=lambda: [1000, 2000, 5000, 10000])