import torch
import torch.nn as nn
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from src.model.sparse_logic import SparseLogicTransformer

def diagnose_gate(checkpoint_path, force_open_val=None):
    print(f"📦 Loading checkpoint: {checkpoint_path}")
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return

    # Load Model State
    state_dict = ckpt['model_state_dict']
    
    # ── 1. Check Recurrent Gate ──────────────────────────────────────────
    gate_key = 'core.gate'
    if gate_key in state_dict:
        gate_val = state_dict[gate_key].item()
        print(f"\n🚪 Recurrent Gate Value: {gate_val:.6f}")
        
        # Apply Force Open if requested
        if force_open_val is not None:
            print(f"   🔧 Force-Opening Gate to {force_open_val}...")
            state_dict[gate_key].fill_(force_open_val)
            print(f"   ✅ New Gate Value: {state_dict[gate_key].item():.6f}")
            
        elif abs(gate_val) < 1e-6:
            print("   ⚠️  WARNING: Gate is effectively DEAD (0.0). Recurrence is disabled.")
            print("   💡 Use --force-open <value> to fix (e.g. 0.1 or 0.3).")
        else:
            print("   ✅ Gate is active (learning).")
    else:
        print(f"❌ '{gate_key}' not found in checkpoint!")

    # ── 2. Check BitConvSwiGLU Weights ───────────────────────────────────
    print("\n🕸️  BitConvSwiGLU Statistics:")
    
    # We look for the ConvSwiGLU in the core block
    # Prefix: core.block.conv_swiglu
    conv_prefix = "core.block.conv_swiglu"
    
    params_found = 0
    for key, tensor in state_dict.items():
        if conv_prefix in key and "weight" in key:
            mean = tensor.float().mean().item()
            std = tensor.float().std().item()
            zeros = (tensor == 0).sum().item()
            total = tensor.numel()
            sparsity = zeros / total
            
            print(f"   - {key:<40} | Mean: {mean:+.4f} | Std: {std:.4f} | Sparsity: {sparsity:.1%}")
            
            if std < 1e-6:
                print(f"     ⚠️  WARNING: Dead weights (Std ~ 0)!")
            
            params_found += 1
            
    if params_found == 0:
        print("   ❌ No BitConvSwiGLU weights found!")

    # ── Save if Modified ────────────────────────────────────────────────
    if force_open_val is not None:
        output_path = checkpoint_path.replace(".pt", "_fixed.pt")
        # Update checkpoint dictionary
        ckpt['model_state_dict'] = state_dict
        torch.save(ckpt, output_path)
        print(f"\n💾 Saved patched checkpoint to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose Recurrent Gate and ConvSwiGLU Weights")
    parser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    parser.add_argument("--force-open", type=float, default=None, help="Force gate to specific value (e.g. 0.3)")
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint):
        print(f"Error: Checkpoint '{args.checkpoint}' not found.")
        sys.exit(1)
        
    diagnose_gate(args.checkpoint, args.force_open)
