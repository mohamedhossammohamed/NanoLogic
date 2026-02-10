"""
Tests for Loss Functions
=========================
Covers: BCE mode, shape handling, gradient flow,
        random baseline, and perfect prediction.
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train.loss import HammingDistanceLoss, StateMatchingLoss


class TestHammingDistanceLoss:
    """Tests for HammingDistanceLoss."""

    def test_bce_mode_output(self):
        """BCE mode produces scalar loss."""
        loss_fn = HammingDistanceLoss(mode='bce')
        pred = torch.randn(4, 256)
        target = torch.randint(0, 2, (4, 256)).float()
        loss = loss_fn(pred, target)
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() > 0, "Loss should be positive"

    def test_perfect_prediction_low_loss(self):
        """Perfect prediction should have very low loss."""
        loss_fn = HammingDistanceLoss(mode='bce')
        target = torch.randint(0, 2, (4, 256)).float()
        pred = target * 20 - 10  # Maps 0→-10, 1→+10
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01, f"Perfect prediction loss too high: {loss.item()}"

    def test_random_prediction_baseline(self):
        """Random prediction should have loss ≈ 0.693 (ln(2))."""
        loss_fn = HammingDistanceLoss(mode='bce')
        pred = torch.zeros(100, 256)
        target = torch.randint(0, 2, (100, 256)).float()
        loss = loss_fn(pred, target)
        assert abs(loss.item() - 0.693) < 0.05, \
            f"Random baseline should be ~0.693, got {loss.item()}"

    def test_gradient_flow(self):
        """Gradients flow through the loss."""
        loss_fn = HammingDistanceLoss(mode='bce')
        pred = torch.randn(4, 256, requires_grad=True)
        target = torch.randint(0, 2, (4, 256)).float()
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.abs().sum() > 0


class TestStateMatchingLoss:
    """Tests for StateMatchingLoss (wrapper that returns scalar loss)."""

    def test_output_scalar(self):
        """Loss returns a scalar tensor."""
        loss_fn = StateMatchingLoss()
        pred = torch.randn(4, 256)
        target = torch.randint(0, 2, (4, 256)).float()
        loss = loss_fn(pred, target)
        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"

    def test_positive_loss(self):
        """Loss is always positive."""
        loss_fn = StateMatchingLoss()
        pred = torch.randn(4, 256)
        target = torch.randint(0, 2, (4, 256)).float()
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_gradient_flow(self):
        """Gradients flow through StateMatchingLoss."""
        loss_fn = StateMatchingLoss()
        pred = torch.randn(4, 256, requires_grad=True)
        target = torch.randint(0, 2, (4, 256)).float()
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert torch.isfinite(pred.grad).all()

    def test_perfect_prediction_low(self):
        """Perfect logits produce near-zero loss."""
        loss_fn = StateMatchingLoss()
        target = torch.ones(2, 256)
        pred = torch.ones(2, 256) * 10  # Strong positive logits
        loss = loss_fn(pred, target)
        assert loss.item() < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
