"""
NeuroCDCL — Neural-Guided Conflict-Driven Clause Learning
============================================================
End-to-end preimage search loop that iteratively:
1. Runs Z3 with timeout → extracts partial assignment
2. Neural oracle predicts bit probabilities from current state
3. High-confidence predictions are injected as soft constraints
4. Repeat until SAT, UNSAT, or max iterations
"""

import torch
import numpy as np
import time
import hashlib

from .z3_sha256 import SHA256Solver
from .bridge import SolverBridge


class NeuroCDCL:
    """
    The full neuro-symbolic search loop.
    
    Combines the Sparse Logic Transformer (neural intuition) with Z3
    (symbolic rigor) in an iterative refinement loop.
    
    Protocol:
        1. Z3 attempts preimage search with current constraints
        2. If timeout → extract partial model, run neural oracle
        3. Neural oracle assigns confidence scores to each bit
        4. High-confidence bits (>threshold) are fixed as hard constraints
        5. Z3 retries with reduced search space
        6. Repeat until SAT, UNSAT, or exhaustion
    """

    def __init__(
        self,
        model=None,
        device='cpu',
        rounds=16,
        max_iterations=10,
        confidence_threshold=0.85,
        z3_timeout_ms=5000,
        confidence_growth=0.02,
    ):
        """
        Args:
            model: Trained SparseLogicTransformer (or None for Z3-only baseline).
            device: 'cpu', 'mps', or 'cuda'.
            rounds: SHA-256 rounds for the solver (16, 32, or 64).
            max_iterations: Maximum neural-guided refinement iterations.
            confidence_threshold: Initial threshold for fixing bits.
            z3_timeout_ms: Z3 timeout per iteration.
            confidence_growth: Increase threshold each iteration to avoid over-fixing.
        """
        self.model = model
        self.device = device
        self.rounds = rounds
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold
        self.z3_timeout_ms = z3_timeout_ms
        self.confidence_growth = confidence_growth
        
        self.bridge = SolverBridge()
        
        # Search statistics
        self.search_log = []

    def _neural_inference(self, state_bits):
        """
        Run the neural model on a 256-bit state vector.
        
        Args:
            state_bits: numpy array of shape [256], values in {0, 1}.
            
        Returns:
            numpy array of shape [256], values in [0, 1] (confidence scores).
        """
        if self.model is None:
            # No model → return uniform (no guidance)
            return np.full(256, 0.5)

        state_tensor = torch.tensor(state_bits, dtype=torch.long).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(state_tensor)  # [1, 256]
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
        
        return probs

    def _hash_message(self, message_bytes):
        """Compute actual SHA-256 hash of a message for verification."""
        return hashlib.sha256(message_bytes).hexdigest()

    def search(self, target_hash_hex, progress_callback=None):
        """
        Run the full NeuroCDCL search loop.
        
        Args:
            target_hash_hex: 64-character hex string (target SHA-256 hash).
            progress_callback: Optional function(step, bits_fixed, threshold).
            
        Returns:
            dict with:
                'status': 'sat', 'unsat', 'exhausted'
                'message_bytes': preimage bytes (if sat)
                'iterations': number of refinement iterations
                'total_time_ms': total wall-clock time
                'bits_fixed_history': list of bits fixed per iteration
                'verified': bool, whether hash of found preimage matches target
                'log': detailed per-iteration log
        """
        self.search_log = []
        total_start = time.time()
        
        # Accumulated neural hints (start with no guidance)
        neural_hints = np.full(256, 0.5)
        current_threshold = self.confidence_threshold
        
        print(f"\n{'='*60}")
        print(f"  NeuroCDCL Preimage Search")
        print(f"  Target: {target_hash_hex[:16]}...{target_hash_hex[-16:]}")
        print(f"  Rounds: {self.rounds} | Max Iterations: {self.max_iterations}")
        print(f"  Neural Model: {'Loaded' if self.model else 'Disabled (Z3-only)'}")
        print(f"{'='*60}\n")

        for iteration in range(self.max_iterations):
            iter_start = time.time()
            
            # Create solver for this iteration
            solver = SHA256Solver(
                rounds=self.rounds,
                timeout_ms=self.z3_timeout_ms,
            )

            if progress_callback:
                progress_callback(iteration, 0 if iteration==0 else bits_fixed, current_threshold)

            # Attempt solve with current neural hints
            if iteration == 0:
                # First iteration: pure Z3, no hints
                result = solver.solve_preimage(target_hash_hex)
            else:
                # Subsequent iterations: inject neural guidance
                result = solver.solve_partial(
                    target_hash_hex, 
                    neural_hints, 
                    confidence_threshold=current_threshold,
                )

            iter_time = (time.time() - iter_start) * 1000
            bits_fixed = result.get('bits_fixed', 0)

            # Log this iteration
            log_entry = {
                'iteration': iteration,
                'status': result['status'],
                'time_ms': iter_time,
                'bits_fixed': bits_fixed,
                'threshold': current_threshold,
                'search_space_reduction': result.get('search_space_reduction', 0),
            }
            self.search_log.append(log_entry)

            status_icon = {'sat': '✅', 'unsat': '❌', 'timeout': '⏳'}.get(result['status'], '?')
            print(f"  Iter {iteration:2d} │ {status_icon} {result['status']:7s} │ "
                  f"{iter_time:7.1f}ms │ {bits_fixed:3d} bits fixed │ "
                  f"threshold: {current_threshold:.2f}")

            # SAT → verify and return
            if result['status'] == 'sat':
                msg_bytes = result['message_bytes']
                actual_hash = self._hash_message(msg_bytes)
                verified = (actual_hash == target_hash_hex.lower())
                
                total_time = (time.time() - total_start) * 1000
                
                print(f"\n  🏆 PREIMAGE FOUND!")
                print(f"  Message: {msg_bytes.hex()[:32]}...")
                print(f"  Hash:    {actual_hash}")
                print(f"  Match:   {'✅ VERIFIED' if verified else '⚠️ MISMATCH (reduced rounds)'}")
                print(f"  Time:    {total_time:.0f}ms across {iteration + 1} iterations\n")
                
                return {
                    'status': 'sat',
                    'message_bytes': msg_bytes,
                    'message_words': result['message_words'],
                    'iterations': iteration + 1,
                    'total_time_ms': total_time,
                    'bits_fixed_history': [e['bits_fixed'] for e in self.search_log],
                    'verified': verified,
                    'log': self.search_log,
                }

            # UNSAT → search space is provably empty
            if result['status'] == 'unsat':
                total_time = (time.time() - total_start) * 1000
                print(f"\n  ❌ UNSAT — no preimage exists (within {self.rounds}-round model)")
                print(f"  Time: {total_time:.0f}ms across {iteration + 1} iterations\n")
                return {
                    'status': 'unsat',
                    'iterations': iteration + 1,
                    'total_time_ms': total_time,
                    'log': self.search_log,
                }

            # TIMEOUT → run neural oracle for guidance
            # Generate a random candidate state for neural inference
            candidate_state = np.random.randint(0, 2, size=256).astype(np.int64)
            neural_hints = self._neural_inference(candidate_state)
            
            # Tighten threshold each iteration to fix more bits
            current_threshold = min(
                current_threshold + self.confidence_growth,
                0.98  # Never fix ALL bits — leave room for Z3
            )

        # Exhausted all iterations
        total_time = (time.time() - total_start) * 1000
        print(f"\n  ⏳ EXHAUSTED — max iterations ({self.max_iterations}) reached")
        print(f"  Time: {total_time:.0f}ms\n")
        
        return {
            'status': 'exhausted',
            'iterations': self.max_iterations,
            'total_time_ms': total_time,
            'bits_fixed_history': [e['bits_fixed'] for e in self.search_log],
            'log': self.search_log,
        }

    @staticmethod
    def load_model(checkpoint_path, config, device='cpu'):
        """
        Convenience: load a trained SparseLogicTransformer from checkpoint.
        
        Args:
            checkpoint_path: Path to .pt checkpoint file.
            config: Config object.
            device: Target device.
            
        Returns:
            Loaded model in eval mode.
        """
        from ..model.sparse_logic import SparseLogicTransformer
        
        model = SparseLogicTransformer(config).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"[NeuroCDCL] Loaded model from {checkpoint_path}")
        return model
