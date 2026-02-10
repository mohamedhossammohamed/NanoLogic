"""
Tests for Z3 SHA-256 Solver & NeuroCDCL
=========================================
Covers: Z3 encoding, reduced-round solving, neural-guided solving,
        VSIDS injection, and the end-to-end NeuroCDCL loop.
"""

import numpy as np
import hashlib
import pytest
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.solver.z3_sha256 import SHA256Solver
from src.solver.bridge import SolverBridge
from src.solver.neuro_cdcl import NeuroCDCL
from src.solver.cnf_utils import tensor_to_cnf


class TestSHA256Solver:
    """Tests for the Z3 SHA-256 constraint encoder."""

    def test_solver_creation(self):
        """Solver initializes with correct parameters."""
        solver = SHA256Solver(rounds=16, timeout_ms=5000)
        assert solver.rounds == 16
        assert solver.timeout_ms == 5000

    def test_rounds_capped_at_64(self):
        """Rounds are capped at 64 regardless of input."""
        solver = SHA256Solver(rounds=128)
        assert solver.rounds == 64

    def test_solve_preimage_returns_dict(self):
        """solve_preimage returns a dict with expected keys."""
        solver = SHA256Solver(rounds=16, timeout_ms=3000)
        # Use SHA-256 of empty string as target
        target = hashlib.sha256(b"").hexdigest()
        result = solver.solve_preimage(target)
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'time_ms' in result
        assert result['status'] in ('sat', 'unsat', 'timeout')

    def test_solve_preimage_invalid_hash_length(self):
        """Rejects hash strings that aren't 64 hex chars."""
        solver = SHA256Solver(rounds=16)
        with pytest.raises(AssertionError):
            solver.solve_preimage("deadbeef")  # Too short

    def test_solve_partial_with_hints(self):
        """solve_partial accepts neural hints without crashing."""
        solver = SHA256Solver(rounds=16, timeout_ms=2000)
        target = hashlib.sha256(b"test").hexdigest()
        hints = np.random.uniform(0, 1, size=256)
        result = solver.solve_partial(target, hints, confidence_threshold=0.9)
        assert 'bits_fixed' in result
        assert 'search_space_reduction' in result

    def test_solve_partial_no_high_confidence(self):
        """With all hints at 0.5, no bits should be fixed."""
        solver = SHA256Solver(rounds=16, timeout_ms=2000)
        target = hashlib.sha256(b"test").hexdigest()
        hints = np.full(256, 0.5)
        result = solver.solve_partial(target, hints, confidence_threshold=0.85)
        assert result['bits_fixed'] == 0

    def test_stats_tracking(self):
        """Stats are updated after solve calls."""
        solver = SHA256Solver(rounds=16, timeout_ms=1000)
        target = hashlib.sha256(b"x").hexdigest()
        solver.solve_preimage(target)
        stats = solver.get_stats()
        assert stats['solve_calls'] == 1
        assert stats['total_time_ms'] > 0


class TestVSIDSInjection:
    """Tests for the VSIDS score injection hook."""

    def test_inject_no_high_confidence(self):
        """Uniform scores (0.5) produce no assumptions."""
        import z3
        scores = np.full(256, 0.5)
        msg_vars = [z3.BitVec(f'W_{i}', 32) for i in range(8)]
        assumptions, count = SolverBridge.inject_vsids_scores(
            None, scores, msg_vars, threshold=0.85
        )
        assert count == 0
        assert len(assumptions) == 0

    def test_inject_high_confidence_bits(self):
        """Extreme scores produce assumptions."""
        import z3
        scores = np.full(256, 0.5)
        scores[0] = 0.99   # Very confident → fix to 1
        scores[32] = 0.01  # Very confident → fix to 0
        msg_vars = [z3.BitVec(f'W_{i}', 32) for i in range(8)]
        assumptions, count = SolverBridge.inject_vsids_scores(
            None, scores, msg_vars, alpha=0.0, threshold=0.85
        )
        assert count >= 1, f"Expected >=1 injected bits, got {count}"

    def test_alpha_controls_blend(self):
        """alpha=1.0 (pure Z3) should inject fewer bits than alpha=0.0 (pure neural)."""
        import z3
        scores = np.random.uniform(0.8, 1.0, size=256)
        msg_vars = [z3.BitVec(f'W_{i}', 32) for i in range(8)]

        _, count_neural = SolverBridge.inject_vsids_scores(
            None, scores, msg_vars, alpha=0.0, threshold=0.85
        )
        _, count_z3 = SolverBridge.inject_vsids_scores(
            None, scores, msg_vars, alpha=1.0, threshold=0.85
        )
        assert count_neural >= count_z3


class TestTensorToCnf:
    """Tests for logit → constraint conversion."""

    def test_high_confidence_generates_constraints(self):
        """Strong logits produce constraints."""
        import torch
        logits = torch.zeros(256)
        logits[0] = 5.0    # sigmoid(5) ≈ 0.993 → should produce constraint
        logits[1] = -5.0   # sigmoid(-5) ≈ 0.007 → should produce constraint
        constraints, probs = tensor_to_cnf(logits, threshold=0.85)
        assert len(constraints) >= 2
        assert probs.shape == (256,)

    def test_low_confidence_no_constraints(self):
        """Zero logits (sigmoid = 0.5) produce no constraints."""
        import torch
        logits = torch.zeros(256)
        constraints, _ = tensor_to_cnf(logits, threshold=0.85)
        assert len(constraints) == 0

    def test_constraint_format(self):
        """Constraints are (bit_index, value, confidence) tuples."""
        import torch
        logits = torch.ones(256) * 10
        constraints, _ = tensor_to_cnf(logits)
        for c in constraints:
            assert len(c) == 3
            idx, val, conf = c
            assert 0 <= idx < 256
            assert val in (0, 1)
            assert 0 < conf <= 1.0


class TestNeuroCDCL:
    """Tests for the end-to-end search loop."""

    def test_z3_only_baseline(self):
        """NeuroCDCL runs without a neural model (Z3-only)."""
        search = NeuroCDCL(
            model=None,
            rounds=16,
            max_iterations=2,
            z3_timeout_ms=2000,
        )
        target = hashlib.sha256(b"hello").hexdigest()
        result = search.search(target)
        assert 'status' in result
        assert 'iterations' in result
        assert result['iterations'] <= 2

    def test_search_returns_log(self):
        """Search returns per-iteration log."""
        search = NeuroCDCL(model=None, rounds=16, max_iterations=1, z3_timeout_ms=1000)
        target = hashlib.sha256(b"test").hexdigest()
        result = search.search(target)
        assert 'log' in result
        assert len(result['log']) >= 1
        assert 'time_ms' in result['log'][0]

    def test_total_time_tracked(self):
        """Total time is tracked."""
        search = NeuroCDCL(model=None, rounds=16, max_iterations=1, z3_timeout_ms=1000)
        target = hashlib.sha256(b"test").hexdigest()
        result = search.search(target)
        assert result['total_time_ms'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
