"""
Tests for Pathfinder (ResNet-1D Distinguisher)
================================================
Covers: forward pass, output shape, gradient flow,
        probability output range, and batch handling.
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from src.model.pathfinder import Pathfinder


class TestPathfinder:
    """Tests for the Pathfinder neural distinguisher."""

    @pytest.fixture
    def config(self):
        cfg = Config()
        cfg.pathfinder_depth = 3  # Smaller for tests
        cfg.dim = 64
        return cfg

    @pytest.fixture
    def pathfinder(self, config):
        return Pathfinder(config)

    def test_output_shape(self, pathfinder):
        """Output is [B] — single viability score per sample (sigmoid squeezed)."""
        x = torch.randn(4, 256)
        out = pathfinder(x)
        assert out.shape == (4,), f"Expected (4,), got {out.shape}"

    def test_single_sample(self, pathfinder):
        """Works with batch_size=1."""
        x = torch.randn(1, 256)
        out = pathfinder(x)
        assert out.shape == (1,)

    def test_output_probability_range(self, pathfinder):
        """Output is in [0, 1] (after sigmoid)."""
        x = torch.randn(4, 256)
        out = pathfinder(x)
        assert out.min() >= 0, f"Output below 0: {out.min()}"
        assert out.max() <= 1, f"Output above 1: {out.max()}"

    def test_output_finite(self, pathfinder):
        """Output is always finite."""
        x = torch.randn(4, 256)
        out = pathfinder(x)
        assert torch.isfinite(out).all()

    def test_gradient_flow(self, pathfinder):
        """Gradients flow through the Pathfinder."""
        x = torch.randn(4, 256, requires_grad=True)
        out = pathfinder(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.abs().sum() > 0

    def test_parameter_count(self, pathfinder):
        """Pathfinder is reasonably small."""
        n_params = sum(p.numel() for p in pathfinder.parameters())
        assert n_params > 0
        assert n_params < 5_000_000, f"Pathfinder too large: {n_params} params"

    def test_deterministic(self, pathfinder):
        """Same input gives same output."""
        pathfinder.eval()
        x = torch.randn(2, 256)
        out1 = pathfinder(x)
        out2 = pathfinder(x)
        assert torch.allclose(out1, out2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
