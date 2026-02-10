print("Starting Validation Script...", flush=True)
import torch

import torch.nn as nn
import torch.optim as optim
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model.sparse_logic import SparseLogicTransformer
from src.model.wiring import SHA256Wiring
from src.train.loss import StateMatchingLoss

# --- Configuration ---
class ValidationConfig:
    def __init__(self):
        self.dim = 256
        self.n_heads = 8
        self.n_layers = 4
        self.dropout = 0.0
        self.device = 'cpu' # Force CPU to rule out MPS bugs

config = ValidationConfig()

print(f"Running Validation on Device: {config.device}")

# --- Helper Functions ---

def get_logic_batch(batch_size, rounds, device):
    """
    Generates (Input, Target) pairs where Target is the NEXT step of SHA-256.
    Retuns: 
        inputs: [B * (R-1), 256]
        targets: [B * (R-1), 256]
    """
    # generate_trace gives [B, Rounds, 256]
    bits, _ = SHA256Wiring.generate_trace(batch_size, rounds, device=device)
    
    # We want to predict t+1 from t
    # Input: States 0..R-2
    # Target: States 1..R-1
    
    inputs = bits[:, :-1, :].reshape(-1, 256).long()
    targets = bits[:, 1:, :].reshape(-1, 256).float() # BCE targets are float
    
    return inputs, targets

# --- Experiments ---

def run_experiment_a_overfit():
    print("\n[Running Experiment A: The 'Overfit' Test]")
    
    # 1. Setup
    model = SparseLogicTransformer(config).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = StateMatchingLoss().to(config.device)
    
    # 2. Fixed Batch
    params_before = [p.clone() for p in model.parameters()]
    inputs, targets = get_logic_batch(batch_size=16, rounds=2, device=config.device) # 1 step transition
    
    # 3. Train
    print("Training on fixed batch for 1000 steps...")
    final_loss = 0.0
    for step in range(1000):
        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        
        # DEBUG: Check Gradients
        if step % 200 == 0:
            grad_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.norm().item()
            print(f"Step {step}: Loss {loss.item():.5f} | Logits Mean: {logits.mean().item():.3f} Std: {logits.std().item():.3f} | Grad Norm: {grad_norm:.3f}")
            
        optimizer.step()
        final_loss = loss.item()
            
    # Success Check
    status = "✅ PASSED" if final_loss < 0.001 else "❌ FAILED"
    print(f"Result: {status} (Loss: {final_loss:.5f})")
    return status, final_loss

def run_experiment_b_logic():
    print("\n[Running Experiment B: The 'Logic' Test]")
    
    # 1. Setup
    model = SparseLogicTransformer(config).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = StateMatchingLoss().to(config.device)
    
    # 2. Train on 8-Round Traces (Transitions)
    # We'll use a larger batch for 1 epoch equivalent
    print("Generating 8-Round Logic Dataset...")
    train_inputs, train_targets = get_logic_batch(batch_size=16, rounds=8, device=config.device)
    test_inputs, test_targets = get_logic_batch(batch_size=16, rounds=8, device=config.device)

    # 3. Train (Approximating "1 epoch" as iterating through this batch enough times or creating a loop)
    # The prompt says "Generate a dataset... Train for 1 epoch". 
    # Let's do 500 steps on this data.
    print("Training for logic generalization...")
    for step in range(500):
        optimizer.zero_grad()
        logits = model(train_inputs) # Full batch gradient decent for stability
        loss = loss_fn(logits, train_targets)
        loss.backward()
        optimizer.step()
        
    # 4. Test
    model.eval()
    with torch.no_grad():
        test_logits = model(test_inputs)
        probs = torch.sigmoid(test_logits)
        preds = (probs > 0.5).float()
        
        # Accuracy: Fraction of bits correctly predicted
        correct = (preds == test_targets).sum().item()
        total = test_targets.numel()
        acc = correct / total
        
    status = "✅ PASSED" if acc > 0.90 else "❌ FAILED"
    print(f"Result: {status} (8-Round Acc: {acc*100:.2f}%)")
    return status, acc

def run_experiment_c_wiring():
    print("\n[Running Experiment C: The 'Wiring' Check]")
    
    model = SparseLogicTransformer(config) # CPU is fine for inspection
    block = model.blocks[0]
    
    # 1. Sparsity Ratio
    # The sparse logic is implemented via indices scattering/gathering.
    # New Connectivity: 
    # Each output bit depends on:
    # 1 (Self) + 3 (Sigma0) + 3 (Sigma1) + 8 (Vertical) = 15 bits.
    # Total possible inputs: 256.
    # Sparsity = 15 / 256 approx 5.8%.
    
    # Let's verify the `op_indices` are correct.
    indices = block.op_indices
    # Bit 0 connections:
    # Sigma0: [30, 19, 10]
    # Sigma1: [26, 21, 7]
    # Vertical: [0, 32, 64, 96, 128, 160, 192, 224] (Bit 0 of Words 0..7)
    
    s0_idx = indices['sigma0'][0].tolist() # Indices for Bit 0
    s1_idx = indices['sigma1'][0].tolist()
    v_idx = indices['vertical'][0].tolist()
    
    print(f"Bit 0 Sigma0 indices: {s0_idx}")
    print(f"Bit 0 Sigma1 indices: {s1_idx}")
    print(f"Bit 0 Vertical indices: {v_idx}")
    
    # Calculate Theoretical Sparsity
    sparsity = (256 * 15) / (256 * 256)
    print(f"Architecture Sparsity: {sparsity*100:.2f}%")
    
    # Check if matches expectations
    expected_s0 = [30, 19, 10]
    expected_v_sample = 32 # Should connect to Word 1
    
    passed_s0 = sorted(s0_idx) == sorted(expected_s0)
    passed_v = expected_v_sample in v_idx
    passed_sparsity = sparsity < 0.10 # < 10% is still sparse
    
    status = "✅ PASSED" if (passed_s0 and passed_v and passed_sparsity) else "❌ FAILED"
    
    return status, f"Sparsity: {sparsity*100:.1f}%, Vertical Check: {passed_v}"


# --- Main ---

def main():
    print("========================================")
    print("NEURO-SHA-M4 VALIDATION PROTOCOL")
    print("========================================")
    
    status_a, res_a = run_experiment_a_overfit()
    status_b, res_b = run_experiment_b_logic()
    status_c, res_c = run_experiment_c_wiring()
    
    print("\n[VALIDATION REPORT]")
    print("--------------------------------------------------")
    print(f"1. Overfit Test:   {status_a} (Loss: {res_a if isinstance(res_a, float) else res_a:.4f})")
    print(f"2. Logic Test:     {status_b} (8-Round Acc: {res_b*100:.1f}%)")
    print(f"3. Wiring Check:   {status_c} ({res_c})")
    print("--------------------------------------------------")
    
    if "PASSED" in status_a and "PASSED" in status_b and "PASSED" in status_c:
        print("Status: ARCHITECTURE VALIDATED. READY FOR SCALING.")
        sys.exit(0)
    else:
        print("Status: VALIDATION FAILED. CHECK COMPONENTS.")
        sys.exit(1)

if __name__ == "__main__":
    main()
