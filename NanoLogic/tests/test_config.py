"""
Tests for Config
==================
Covers: default values, constraints, and type checks.
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config


class TestConfig:
    """Tests for Config dataclass defaults."""

    def test_dim_is_reasonable(self):
        """dim should be 256 or 512 max."""
        cfg = Config()
        assert cfg.dim <= 512, f"dim={cfg.dim} exceeds max 512"
        assert cfg.dim >= 64, f"dim={cfg.dim} is too small"

    def test_batch_size_small(self):
        """batch_size must be small for M4 memory."""
        cfg = Config()
        assert cfg.batch_size <= 16, f"batch_size={cfg.batch_size} too large for M4"

    def test_grad_accum_steps_exists(self):
        """Config has gradient accumulation setting."""
        cfg = Config()
        assert hasattr(cfg, 'grad_accum_steps'), "Missing grad_accum_steps field"
        assert cfg.grad_accum_steps >= 1

    def test_rounds_is_64(self):
        """SHA-256 has 64 rounds."""
        cfg = Config()
        assert cfg.rounds == 64

    def test_vocab_size_binary(self):
        """vocab_size should be 2 (binary: 0, 1)."""
        cfg = Config()
        assert cfg.vocab_size == 2

    def test_learning_rate_range(self):
        """Learning rate should be small for Lion optimizer."""
        cfg = Config()
        assert cfg.lr <= 1e-3, f"lr={cfg.lr} too high for Lion"
        assert cfg.lr > 0, "lr must be positive"

    def test_pathfinder_depth(self):
        """Pathfinder depth is set."""
        cfg = Config()
        assert hasattr(cfg, 'pathfinder_depth')
        assert cfg.pathfinder_depth > 0

    def test_max_solver_steps(self):
        """Solver steps is configured."""
        cfg = Config()
        assert cfg.max_solver_steps >= 1000

    def test_curriculum_rounds(self):
        """Curriculum schedule has 4 phases doubling from 8."""
        cfg = Config()
        assert cfg.curriculum_rounds == [8, 16, 32, 64]

    def test_start_round_default(self):
        """Default start_round is 8."""
        cfg = Config()
        assert cfg.start_round == 8

    def test_phase_accuracy_thresholds(self):
        """Accuracy thresholds are set per phase."""
        cfg = Config()
        assert len(cfg.phase_accuracy_thresholds) == len(cfg.curriculum_rounds)
        for t in cfg.phase_accuracy_thresholds[:-1]:
            assert 0 < t <= 1.0, f"Threshold {t} out of range"

    def test_phase_min_steps(self):
        """Minimum steps per phase are configured."""
        cfg = Config()
        assert len(cfg.phase_min_steps) == len(cfg.curriculum_rounds)

    def test_start_round_override(self):
        """start_round can be changed."""
        cfg = Config()
        cfg.start_round = 32
        assert cfg.start_round == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
