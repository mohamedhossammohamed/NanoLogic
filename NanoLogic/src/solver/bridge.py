import multiprocessing.shared_memory
import numpy as np
import torch
import time
from .cnf_utils import assignment_to_tensor

class SolverBridge:
    """
    Interface between the Python Training Loop (Neural) and the Symbolic Solver (C++/Kissat).
    Uses Shared Memory for zero-copy data transfer.
    """
    def __init__(self, shm_name="neuro_sha_bridge", size_bytes=1024*1024*4): # 4MB buffer
        self.shm_name = shm_name
        self.size = size_bytes
        self.shm = None
        self.connected = False
        
    def connect(self, create=True):
        try:
            if create:
                # Cleanup if exists
                try:
                    existing = multiprocessing.shared_memory.SharedMemory(name=self.shm_name)
                    existing.close()
                    existing.unlink()
                except:
                    pass
                self.shm = multiprocessing.shared_memory.SharedMemory(name=self.shm_name, create=True, size=self.size)
            else:
                self.shm = multiprocessing.shared_memory.SharedMemory(name=self.shm_name)
            self.connected = True
            print(f"[SolverBridge] Connected to Shared Memory: {self.shm_name}")
        except Exception as e:
            print(f"[SolverBridge] Connection Failed: {e}")
            self.connected = False

    def query_neural_guide(self, assignment_vector, model, device='cpu'):
        """
        Full round-trip:
        1. Receive Assignment (Argument)
        2. Convert to Tensor
        3. inference
        4. Return Scores
        """
        # 1. Convert
        # Assignment might be raw bytes from SHM or passed arg
        # Here we assume assignment_vector is a list/array passed from the 'Solver' loop simulation
        
        state_tensor = assignment_to_tensor(assignment_vector).unsqueeze(0).to(device) # [1, L]
        
        # 2. Inference
        with torch.no_grad():
            # Model expects [B, 256] or similar. 
            # Our tensor might be larger (16k bits).
            # The SparseLogicTransformer needs to handle the full sequence or chunks.
            # config.dim is small, but sequence length is large?
            # Wiring.py assumes 256 bits (8 words) state.
            # If we are doing full SHA, we need to handle the full trace or just the "Current Frontier".
            # Neuro-SHA-M4 focuses on "State-Matching", usually implied as the 256-bit state at round T.
            
            # For this prototype, we assume the input is the 256-bit state.
            if state_tensor.shape[1] > 256:
                 state_tensor = state_tensor[:, -256:] # Take last 256 bits (current state)
            
            scores = model(state_tensor) # [1, 256]
            
        return scores.cpu().numpy().flatten()
    
    def read_shared_state(self):
        """
        Reads the current state from shared memory.
        Format: [Flag (4 bytes)][Size (4 bytes)][Data...]
        """
        if not self.connected: return None
        
        # Simple protocol: check first byte. If 1, data is ready.
        # This is a spin-lock simulation.
        # For now, just read a chunk.
        buffer = self.shm.buf
        flag = np.frombuffer(buffer, dtype=np.int32, count=1)[0]
        
        if flag == 1:
            size = np.frombuffer(buffer, dtype=np.int32, count=1, offset=4)[0]
            data = np.frombuffer(buffer, dtype=np.int8, count=size, offset=8)
            return data
        return None

    def write_scores(self, scores):
        """
        Writes priority scores to shared memory.
        """
        if not self.connected: return
        
        data = scores.astype(np.float32)
        n_bytes = data.nbytes
        
        # Write [Flag=2 (Result Ready)][Size][Data]
        self.shm.buf[0:4] = np.array([2], dtype=np.int32).tobytes()
        self.shm.buf[4:8] = np.array([n_bytes], dtype=np.int32).tobytes()
        self.shm.buf[8:8+n_bytes] = data.tobytes()

    def close(self):
        if self.shm:
            self.shm.close()
            try:
                self.shm.unlink()
            except:
                pass

    @staticmethod
    def inject_vsids_scores(solver, neural_scores, message_vars, alpha=0.3, threshold=0.85):
        """
        VSIDS Score Injection Hook.
        
        Blends neural confidence scores with Z3's internal decision heuristic
        by adding soft constraints (assumptions) for high-confidence bits.
        
        Protocol:
            - Neural scores above `threshold` → fix bit to 1 (soft)
            - Neural scores below `1 - threshold` → fix bit to 0 (soft)
            - Scores in between → no constraint (let Z3 decide)
            
        The `alpha` parameter controls the blend weight:
            Score(v) = α · VSIDS_old(v) + (1-α) · Neural_pred(v)
        
        Since Z3 doesn't expose raw VSIDS scores, we approximate this by
        adding assumptions (retractable soft constraints) for high-confidence variables.
        
        Args:
            solver: z3.Solver instance.
            neural_scores: numpy array of shape [N] with values in [0, 1].
            message_vars: list of z3.BitVecRef (the symbolic message words).
            alpha: Blend weight (0 = pure neural, 1 = pure Z3). Default 0.3.
            threshold: Confidence threshold for adding constraints.
            
        Returns:
            assumptions: list of z3 boolean expressions to pass as assumptions.
            bits_injected: number of bits constrained by neural guidance.
        """
        import z3
        
        assumptions = []
        bits_injected = 0
        
        for bit_idx in range(min(len(neural_scores), 256)):
            conf = neural_scores[bit_idx]
            
            # Only inject if confidence exceeds threshold
            if conf <= threshold and conf >= (1.0 - threshold):
                continue
                
            word_idx = bit_idx // 32
            bit_pos = 31 - (bit_idx % 32)  # MSB-first
            
            if word_idx >= len(message_vars):
                continue
            
            # Scale confidence by (1 - alpha) to respect Z3's own heuristic
            effective_conf = conf * (1 - alpha) + 0.5 * alpha
            
            if effective_conf > threshold:
                # Fix bit to 1
                bit_constraint = z3.Extract(bit_pos, bit_pos, message_vars[word_idx]) == z3.BitVecVal(1, 1)
                assumptions.append(bit_constraint)
                bits_injected += 1
            elif effective_conf < (1.0 - threshold):
                # Fix bit to 0
                bit_constraint = z3.Extract(bit_pos, bit_pos, message_vars[word_idx]) == z3.BitVecVal(0, 1)
                assumptions.append(bit_constraint)
                bits_injected += 1
        
        return assumptions, bits_injected

