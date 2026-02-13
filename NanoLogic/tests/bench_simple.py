import torch
import sys
import os
import time
import hashlib

# Append path
sys.path.append(".")

from config import Config
from src.model.sparse_logic import SparseLogicTransformer
from src.solver.z3_sha256 import SHA256Solver
from src.solver.neuro_cdcl import NeuroCDCL

def run_bench():
    print("🚀 NanolLogic Sequential Benchmark")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    config = Config()
    model = SparseLogicTransformer(config).to(device)
    
    ckpt_path = "checkpoints/neuro_sha_step_22000.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print(f"Loaded {ckpt_path}")
    
    rounds = 8
    num_samples = 3
    timeout_ms = 5000 
    
    print(f"Config: {rounds} rounds, {num_samples} samples, {timeout_ms}ms timeout")
    
    for i in range(num_samples):
        target_bytes = os.urandom(32)
        target_hex = hashlib.sha256(target_bytes).hexdigest()
        print(f"\n--- Sample #{i+1}: {target_hex[:16]}... ---")
        
        # 1. Standard Z3
        s1 = SHA256Solver(rounds=rounds, timeout_ms=timeout_ms)
        t_start = time.time()
        res1 = s1.solve_preimage(target_hex)
        t_z3 = time.time() - t_start
        print(f"Standard Z3: {res1.get('status')} in {t_z3:.2f}s")
        
        # 2. Neuro Z3
        s2 = NeuroCDCL(model=model, device=device, rounds=rounds, max_iterations=3, z3_timeout_ms=timeout_ms//2)
        t_start = time.time()
        res2 = s2.search(target_hex)
        t_neuro = time.time() - t_start
        print(f"Neuro Z3: {res2.get('status')} in {t_neuro:.2f}s")

if __name__ == "__main__":
    run_bench()
