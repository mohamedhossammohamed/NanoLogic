import torch
import sys
import os
import time

# Append path
sys.path.append(".")

from config import Config
from src.model.sparse_logic import SparseLogicTransformer

def test_inference():
    print("Testing MPS Inference...")
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    config = Config()
    model = SparseLogicTransformer(config).to(device)
    
    # Load weights
    ckpt_path = "checkpoints/neuro_sha_step_22000.pt"
    if os.path.exists(ckpt_path):
        print(f"Loading checkpoint {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        print("Model loaded successfully.")
    else:
        print("Checkpoint not found, using random weights.")

    model.eval()
    
    print("Running forward pass...")
    with torch.no_grad():
        x = torch.randint(0, 2, (64, 256), dtype=torch.long, device=device)
        start = time.time()
        logits = model(x)
        end = time.time()
        print(f"Forward pass completed in {end - start:.4f}s")
        print(f"Output shape: {logits.shape}")
        print(f"Output mean: {logits.mean().item():.4f}")

if __name__ == "__main__":
    try:
        test_inference()
        print("✅ Inference test passed!")
    except Exception as e:
        print(f"❌ Inference test failed: {e}")
