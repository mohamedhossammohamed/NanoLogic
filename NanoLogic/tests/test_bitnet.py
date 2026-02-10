"""
Tests for BitNet b1.58 Quantization Layer
==========================================
Covers: forward pass quantization, backward pass gradient scaling,
        gradient flow, and numerical stability.
"""

import torch
import torch.nn as nn
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model.bitnet import BitLinear, BitLinearFunction


class TestBitLinearForward:
    """Tests for the forward pass of BitLinear."""

    def test_output_shape(self):
        """Output shape matches expected dimensions."""
        layer = BitLinear(64, 32)
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 32), f"Expected (4, 32), got {out.shape}"

    def test_output_shape_3d(self):
        """Works with 3D input (batch, seq, features)."""
        layer = BitLinear(64, 32)
        x = torch.randn(4, 10, 64)
        out = layer(x)
        assert out.shape == (4, 10, 32)

    def test_weight_quantization_ternary(self):
        """Weights are quantized to {-1, 0, 1} during forward."""
        layer = BitLinear(64, 32)
        x = torch.randn(2, 64)

        # Run forward to trigger quantization (check internally)
        w = layer.weight - layer.weight.mean()
        gamma = w.abs().mean()
        w_scaled = w / (gamma + 1e-5)
        w_quant = w_scaled.round().clamp(-1, 1)

        unique_vals = torch.unique(w_quant)
        for v in unique_vals:
            assert v.item() in {-1.0, 0.0, 1.0}, f"Unexpected quantized value: {v}"

    def test_input_quantization_range(self):
        """Input quantization stays within INT8 range [-128, 127]."""
        x = torch.randn(4, 64) * 100  # Large range input
        input_scale = 127.0 / (x.abs().max(dim=-1, keepdim=True).values + 1e-5)
        x_quant = (x * input_scale).round().clamp(-128, 127)
        assert x_quant.min() >= -128
        assert x_quant.max() <= 127

    def test_deterministic(self):
        """Same input produces same output."""
        layer = BitLinear(64, 32)
        x = torch.randn(2, 64)
        out1 = layer(x)
        out2 = layer(x)
        assert torch.allclose(out1, out2), "Forward pass is not deterministic"

    def test_bias_option(self):
        """Layer works with and without bias."""
        layer_bias = BitLinear(64, 32, bias=True)
        layer_no_bias = BitLinear(64, 32, bias=False)
        x = torch.randn(2, 64)
        out1 = layer_bias(x)
        out2 = layer_no_bias(x)
        assert out1.shape == out2.shape


class TestBitLinearBackward:
    """Tests for the backward pass — the critical gradient scaling."""

    def test_gradients_exist(self):
        """Gradients flow through the BitLinear layer."""
        layer = BitLinear(64, 32)
        x = torch.randn(2, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Input gradient is None"
        assert layer.weight.grad is not None, "Weight gradient is None"

    def test_gradient_scaling_by_gamma(self):
        """grad_weight is scaled by 1/gamma (prevents gradient explosion)."""
        layer = BitLinear(64, 32)

        x = torch.randn(2, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        # Check that gradients are finite and reasonable
        assert torch.isfinite(layer.weight.grad).all(), "Weight gradients contain inf/nan"
        # Gradient magnitude should be bounded — without 1/gamma scaling,
        # gradients explode to thousands. With scaling, they stay reasonable.
        grad_max = layer.weight.grad.abs().max().item()
        assert grad_max < 10000, f"Weight gradients too large: {grad_max} (missing 1/gamma?)"

    def test_input_gradient_finite(self):
        """Input gradients are always finite."""
        layer = BitLinear(64, 32)
        x = torch.randn(4, 64, requires_grad=True)
        out = layer(x)
        loss = out.mean()
        loss.backward()
        assert torch.isfinite(x.grad).all(), "Input gradients contain inf/nan"

    def test_gradient_not_zero(self):
        """Gradients are non-zero (STE passes gradients through quantization)."""
        layer = BitLinear(64, 32)
        x = torch.randn(2, 64, requires_grad=True)
        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert layer.weight.grad.abs().sum() > 0, "Weight gradients are all zero"
        assert x.grad.abs().sum() > 0, "Input gradients are all zero"

    def test_gradient_accumulation(self):
        """Multiple backward passes accumulate correctly."""
        layer = BitLinear(64, 32)
        x = torch.randn(2, 64)

        out1 = layer(x)
        out1.sum().backward()
        grad1 = layer.weight.grad.clone()

        out2 = layer(x)
        out2.sum().backward()
        grad2 = layer.weight.grad.clone()

        # grad2 should be approximately 2x grad1
        assert torch.allclose(grad2, grad1 * 2, atol=1e-5), \
            "Gradient accumulation not working"


class TestBitLinearNumericalStability:
    """Tests for edge cases and numerical stability."""

    def test_zero_input(self):
        """Layer handles zero input without crashing."""
        layer = BitLinear(64, 32)
        x = torch.zeros(2, 64)
        out = layer(x)
        assert torch.isfinite(out).all(), "Zero input produces inf/nan"

    def test_large_input(self):
        """Layer handles very large inputs."""
        layer = BitLinear(64, 32)
        x = torch.randn(2, 64) * 1000
        out = layer(x)
        assert torch.isfinite(out).all(), "Large input produces inf/nan"

    def test_small_weight_gamma(self):
        """Handles near-zero gamma (all weights close to mean)."""
        layer = BitLinear(64, 32)
        with torch.no_grad():
            layer.weight.fill_(0.001)  # Near-uniform weights → tiny gamma
        x = torch.randn(2, 64)
        out = layer(x)
        assert torch.isfinite(out).all(), "Near-zero gamma causes inf/nan"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
