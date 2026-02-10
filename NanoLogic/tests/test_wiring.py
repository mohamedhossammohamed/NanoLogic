"""
Tests for SHA-256 Wiring Diagram
=================================
Covers: wiring index generation, trace simulation,
        SHA-256 constants, and structural correctness.
"""

import torch
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.model.wiring import SHA256Wiring


class TestSHA256Constants:
    """Verify SHA-256 round constants are correct."""

    def test_k_length(self):
        """SHA-256 has exactly 64 round constants."""
        assert len(SHA256Wiring.K) == 64

    def test_k_first_value(self):
        """First round constant K[0] = 0x428a2f98."""
        assert SHA256Wiring.K[0].item() == 0x428a2f98

    def test_k_last_value(self):
        """Last round constant K[63] = 0xc67178f2."""
        assert SHA256Wiring.K[63].item() == 0xc67178f2


class TestWiringIndices:
    """Test the static wiring pattern generation."""

    def test_returns_dict(self):
        """get_op_indices returns a dict with expected keys."""
        result = SHA256Wiring.get_op_indices()
        assert isinstance(result, dict)
        assert 'sigma0' in result
        assert 'sigma1' in result
        assert 'vertical' in result

    def test_sigma0_shape(self):
        """Σ₀ indices: [256, 3] (3 rotation neighbors per bit)."""
        result = SHA256Wiring.get_op_indices()
        assert result['sigma0'].shape == (256, 3), \
            f"Expected (256, 3), got {result['sigma0'].shape}"

    def test_sigma1_shape(self):
        """Σ₁ indices: [256, 3]."""
        result = SHA256Wiring.get_op_indices()
        assert result['sigma1'].shape == (256, 3)

    def test_vertical_shape(self):
        """Vertical indices: [256, 8] (same bit in all 8 words)."""
        result = SHA256Wiring.get_op_indices()
        assert result['vertical'].shape == (256, 8)

    def test_indices_in_range(self):
        """All wiring indices must be in [0, 255]."""
        result = SHA256Wiring.get_op_indices()
        for name in ['sigma0', 'sigma1', 'vertical']:
            idx = result[name]
            assert idx.min() >= 0, f"{name} has negative index"
            assert idx.max() <= 255, f"{name} has out-of-range index: {idx.max()}"

    def test_sigma0_rotation_correctness(self):
        """
        For bit 0 of word 0: ROTR(2) → bit 30, ROTR(13) → bit 19, ROTR(22) → bit 10.
        All within word 0 (indices 0..31).
        """
        result = SHA256Wiring.get_op_indices()
        sigma0 = result['sigma0']
        neighbors = sigma0[0].tolist()
        expected = [(0 - 2) % 32, (0 - 13) % 32, (0 - 22) % 32]
        assert neighbors == expected, f"Σ₀[0] expected {expected}, got {neighbors}"

    def test_vertical_self_inclusion(self):
        """Vertical wiring for bit 0 in word 0 includes index 0."""
        result = SHA256Wiring.get_op_indices()
        vertical = result['vertical']
        # Bit 0, word 0 → index 0. Vertical includes all words for bit 0:
        # word 0 bit 0 = 0, word 1 bit 0 = 32, word 2 bit 0 = 64, ...
        expected = [w * 32 + 0 for w in range(8)]
        actual = vertical[0].tolist()
        assert actual == expected, f"Vertical[0] expected {expected}, got {actual}"


class TestTraceGeneration:
    """Test the SHA-256 trace simulator."""

    def test_trace_returns_tuple(self):
        """generate_trace returns (bits, W) tuple."""
        result = SHA256Wiring.generate_trace(batch_size=2, rounds=8)
        assert isinstance(result, tuple), f"Expected tuple, got {type(result)}"
        assert len(result) == 2

    def test_trace_bits_shape(self):
        """Bits have shape [B, rounds, 256]."""
        bits, W = SHA256Wiring.generate_trace(batch_size=4, rounds=16)
        assert bits.shape == (4, 16, 256), f"Expected (4, 16, 256), got {bits.shape}"

    def test_trace_schedule_shape(self):
        """Message schedule W has shape [B, 64]."""
        bits, W = SHA256Wiring.generate_trace(batch_size=4, rounds=16)
        assert W.shape == (4, 64), f"Expected (4, 64), got {W.shape}"

    def test_trace_binary(self):
        """All bit values are 0 or 1."""
        bits, _ = SHA256Wiring.generate_trace(batch_size=4, rounds=8)
        unique_vals = torch.unique(bits)
        for v in unique_vals.tolist():
            assert v in [0, 1], f"Non-binary value in trace: {v}"

    def test_trace_different_batches(self):
        """Different batch items produce different traces."""
        bits, _ = SHA256Wiring.generate_trace(batch_size=4, rounds=8)
        diffs = (bits[0] != bits[1]).float().sum()
        assert diffs > 0, "All batch items are identical"

    def test_trace_rounds_evolve(self):
        """State changes across rounds."""
        bits, _ = SHA256Wiring.generate_trace(batch_size=2, rounds=16)
        round_0 = bits[0, 0, :]
        round_15 = bits[0, -1, :]
        diffs = (round_0 != round_15).float().sum()
        assert diffs > 0, "State did not change across 16 rounds"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
