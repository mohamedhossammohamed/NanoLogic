"""
Tests for Synthetic Data Generator
=====================================
Covers: lazy generation, data shapes, binary values,
        device placement, and memory behavior.
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.train.synthetic import infinite_trace_stream


class TestInfiniteTraceStream:
    """Tests for the lazy data generator."""

    def test_yields_tuples(self):
        """Generator yields (input, target) tuples."""
        gen = infinite_trace_stream(batch_size=2, rounds=8)
        batch = next(gen)
        assert isinstance(batch, tuple), f"Expected tuple, got {type(batch)}"
        assert len(batch) == 2, f"Expected 2 elements (input, target), got {len(batch)}"

    def test_input_shape(self):
        """Input has shape [B*rounds, 256]."""
        gen = infinite_trace_stream(batch_size=4, rounds=16)
        inputs, targets = next(gen)
        assert inputs.shape[1] == 256, f"Expected 256 bits, got {inputs.shape[1]}"
        # B*rounds samples
        assert inputs.shape[0] == 4 * 16, f"Expected {4*16} samples, got {inputs.shape[0]}"

    def test_target_shape(self):
        """Target has same shape as input."""
        gen = infinite_trace_stream(batch_size=4, rounds=16)
        inputs, targets = next(gen)
        assert inputs.shape == targets.shape, \
            f"Input/target shape mismatch: {inputs.shape} vs {targets.shape}"

    def test_binary_values(self):
        """All values are 0 or 1."""
        gen = infinite_trace_stream(batch_size=4, rounds=8)
        inputs, targets = next(gen)
        for name, tensor in [("inputs", inputs), ("targets", targets)]:
            unique = torch.unique(tensor)
            for v in unique.tolist():
                assert v in [0.0, 1.0], f"Non-binary value in {name}: {v}"

    def test_infinite_generation(self):
        """Can generate multiple batches without exhaustion."""
        gen = infinite_trace_stream(batch_size=2, rounds=8)
        for i in range(10):
            batch = next(gen)
            assert batch is not None, f"Generator exhausted at batch {i}"

    def test_different_batches(self):
        """Successive batches are different (random generation)."""
        gen = infinite_trace_stream(batch_size=2, rounds=8)
        batch1_inputs, _ = next(gen)
        batch2_inputs, _ = next(gen)
        diff = (batch1_inputs != batch2_inputs).float().sum()
        assert diff > 0, "Two successive batches are identical"

    def test_cpu_device(self):
        """Data is generated on CPU by default."""
        gen = infinite_trace_stream(batch_size=2, rounds=8, device='cpu')
        inputs, targets = next(gen)
        assert inputs.device.type == 'cpu'
        assert targets.device.type == 'cpu'

    def test_small_batch(self):
        """Works with batch_size=1."""
        gen = infinite_trace_stream(batch_size=1, rounds=4)
        inputs, targets = next(gen)
        assert inputs.shape[0] == 4  # 1 * 4 rounds


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
