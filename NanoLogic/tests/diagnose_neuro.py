import torch
import sys
import os
import time

# Append path
sys.path.append(".")

from config import Config
from src.model.sparse_logic import SparseLogicTransformer
from src.solver.neuro_cdcl import NeuroCDCL

def test_neuro_cdcl():
    print("Testing NeuroCDCL Integration...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    config = Config()
    model = SparseLogicTransformer(config).to(device)
    
    # Load weights
    ckpt_path = "checkpoints/neuro_sha_step_22000.pt"
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print("Model loaded.")

    target_hex = "3b55dc51ed68289aed5f9b3f678c3fd43b55dc51ed68289aed5f9b3f678c3fd4"
    
    print("Initializing NeuroCDCL...")
    solver = NeuroCDCL(
        model=model, 
        device=device,
        rounds=8, 
        max_iterations=5, # Minimal iterations
        z3_timeout_ms=1000 
    )
    
    print("Starting Neuro-Guided Search...")
    start = time.time()
    res = solver.search(target_hex)
    end = time.time()
    
    print(f"Search completed in {end - start:.4f}s")
    print(f"Status: {res.get('status')}")

if __name__ == "__main__":
    try:
        test_neuro_cdcl()
        print("✅ NeuroCDCL test passed!")
    except Exception as e:
        print(f"❌ NeuroCDCL test failed: {e}")
