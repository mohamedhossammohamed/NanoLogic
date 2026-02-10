from dataclasses import dataclass, field
from typing import List

@dataclass
class Config:
    # Model Architecture
    dim: int = 256              # Reduced from 512 to fit in MPS memory
    n_layers: int = 24          # Matches simplified logic depth
    n_heads: int = 8            # Standard attention heads (if used) or parallel logic paths
    vocab_size: int = 2         # Binary (0, 1) or Ternary (-1, 0, 1)

    # Wiring & Logic
    wiring_mode: str = "static_sha256"  # "static_sha256", "random", "learnable"
    rounds: int = 64            # Full SHA-256 rounds

    # Pathfinder (ResNet)
    pathfinder_depth: int = 10  # Number of Residual Blocks
    
    # Memory Guard
    grad_accum_steps: int = 32  # Effective batch size = batch_size * grad_accum = 64
    num_workers: int = 0        # Main Thread only (No fork overhead)
    pin_memory: bool = False    # Disable pinned memory for RAM savings

    # Training
    batch_size: int = 2         # Slashed from 8 to prevent activation explosion
    lr: float = 1e-4
    warmup_steps: int = 1000
    
    # Solver
    max_solver_steps: int = 10000

    # ── Curriculum ──────────────────────────────────────────────────────
    # Rounds schedule: training starts at start_round and doubles each phase.
    # Training does NOT advance to the next phase until accuracy >= the
    # corresponding threshold in phase_accuracy_thresholds.
    start_round: int = 8                                    # Override: always start here
    curriculum_rounds: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    phase_min_steps: List[int] = field(default_factory=lambda: [500, 2000, 5000, 0])
    phase_accuracy_thresholds: List[float] = field(default_factory=lambda: [0.95, 0.85, 0.75, 0.65])
    #   phase_accuracy_thresholds[i] = minimum accuracy to leave phase i
    #   phase_min_steps[i]           = minimum steps before checking accuracy
    #   Last phase threshold is 1.0 (unreachable) so it runs forever.
