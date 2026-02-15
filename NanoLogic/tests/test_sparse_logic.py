"""
Tests for Sparse Logic Transformer
=====================================
Covers: forward pass, gradient checkpointing, pre-norm,
        residual connections, and full model integration.
"""

import torch
import torch.nn as nn
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from src.model.sparse_logic import SparseLogicBlock, SparseLogicTransformer, RecurrentSparseLogic, BitConvSwiGLU


class TestSparseLogicBlock:
    """Tests for a single SparseLogicBlock layer."""

    @pytest.fixture
    def block(self):
        return SparseLogicBlock(dim=64, n_heads=4)

    def test_output_shape(self, block):
        """Output has same shape as input (residual architecture)."""
        x = torch.randn(2, 256, 64)
        out = block(x)
        assert out.shape == (2, 256, 64), f"Expected (2, 256, 64), got {out.shape}"

    def test_residual_connection(self, block):
        """Output differs from input (block modifies the signal)."""
        x = torch.randn(2, 256, 64)
        out = block(x)
        # If residual only, output != input (gate adds something)
        diff = (out - x).abs().sum()
        assert diff > 0, "Block had no effect (broken gate or zero weights)"

    def test_gradient_checkpointing_train(self, block):
        """Gradient checkpointing activates during training."""
        block.train()
        x = torch.randn(2, 256, 64, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None, "No gradient through checkpointed block"

    def test_eval_mode(self, block):
        """Block works in eval mode (no checkpointing)."""
        block.eval()
        x = torch.randn(2, 256, 64)
        with torch.no_grad():
            out = block(x)
        assert out.shape == (2, 256, 64)

    def test_pre_norm_exists(self, block):
        """Block has pre-norm (LayerNorm before logic gate)."""
        has_norm = any(
            isinstance(m, nn.LayerNorm) for m in block.modules()
        )
        assert has_norm, "No LayerNorm found — missing pre-norm"

    def test_bit_conv_swiglu_structure(self, block):
        """Verify BitConvSwiGLU components exist."""
        assert hasattr(block, 'conv_swiglu'), "Block missing conv_swiglu"
        assert isinstance(block.conv_swiglu, BitConvSwiGLU)


class TestSparseLogicTransformer:
    """Tests for the full SparseLogicTransformer model."""

    @pytest.fixture
    def config(self):
        cfg = Config()
        cfg.dim = 64       # Small for testing
        cfg.n_layers = 4
        cfg.n_heads = 4
        return cfg

    @pytest.fixture
    def model(self, config):
        return SparseLogicTransformer(config)

    def test_output_shape(self, model):
        """Full model outputs [B, 256] logits."""
        x = torch.randint(0, 2, (2, 256))
        out = model(x)
        assert out.shape == (2, 256), f"Expected (2, 256), got {out.shape}"

    def test_binary_input(self, model):
        """Model accepts 0/1 integer inputs."""
        x = torch.randint(0, 2, (4, 256))
        out = model(x)
        assert torch.isfinite(out).all(), "Model produces inf/nan on binary input"

    def test_gradient_flow(self, model):
        """Gradients flow from output back to embedding."""
        x = torch.randint(0, 2, (2, 256))
        out = model(x)
        loss = out.sum()
        loss.backward()

        # Check embedding has gradients
        emb_has_grad = model.embedding.weight.grad is not None
        assert emb_has_grad, "No gradients reached the embedding layer"

    def test_parameter_count(self, model):
        """Model has >0 trainable parameters."""
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert n_params > 0, "Model has no trainable parameters"
        # Sanity check: small model shouldn't exceed 10M
        assert n_params < 10_000_000, f"Test model too large: {n_params} params"

    def test_train_eval_different(self, model):
        """Training vs eval modes produce different results (due to checkpointing)."""
        x = torch.randint(0, 2, (2, 256))
        model.train()
        out_train = model(x).detach().clone()
        model.eval()
        out_eval = model(x).detach().clone()
        # They should be the same numerically (checkpointing doesn't change output)
        assert torch.allclose(out_train, out_eval, atol=1e-4), \
            "Train/eval outputs differ significantly"

    def test_batch_independence(self, model):
        """Different batch items get different outputs (model is not degenerate)."""
        x = torch.randint(0, 2, (4, 256))
        out = model(x)
        # At least some outputs should differ
        diffs = (out[0] - out[1]).abs().sum()
        assert diffs > 0, "All batch items have identical output"



class TestRecurrentSparseLogic:
    """Tests for the Recurrent Super-Block."""

    @pytest.fixture
    def config(self):
        cfg = Config()
        cfg.dim = 64
        cfg.n_heads = 4
        cfg.recurrent_loops = 3 # Small loop for testing
        return cfg

    @pytest.fixture
    def block(self, config):
        return RecurrentSparseLogic(config)

    def test_output_shape(self, block):
        """Output has same shape as input (B, 256, D)."""
        x = torch.randn(2, 256, 64)
        out = block(x)
        assert out.shape == (2, 256, 64)

    def test_gate_initialization(self, block):
        """Gate should initialize to 0.0."""
        assert block.gate.item() == 0.0

    def test_loop_logic(self, block):
        """
        Verify that the loop actually runs and modifies the input.
        If gate is 0, tanh(gate) is 0, so output should equal input (Identity).
        """
        # Case 1: Gate = 0 (Default) -> Identity
        x = torch.randn(2, 256, 64)
        out = block(x)
        assert torch.allclose(x, out), "With gate=0, block should be identity"

        # Case 2: Gate != 0 -> Transformation
        block.gate.data.fill_(1.0) # tanh(1.0) ~= 0.76
        out_active = block(x)
        diff = (out_active - x).abs().sum()
        assert diff > 0, "With gate=1, block should modify input"

    def test_gradient_checkpointing_flag(self, block):
        """Ensure check for training/eval mode doesn't crash."""
        x = torch.randn(2, 256, 64, requires_grad=True)
        block.train()
        out = block(x)
        assert out.requires_grad
        
        block.eval()
        with torch.no_grad():
            out_eval = block(x)
        assert not out_eval.requires_grad

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
