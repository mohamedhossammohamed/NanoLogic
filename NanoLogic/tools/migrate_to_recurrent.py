import torch
import torch.nn as nn
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from src.model.sparse_logic import SparseLogicTransformer

def migrate_checkpoint(input_path, output_path):
    print(f"📦 Loading legacy checkpoint: {input_path}")
    try:
        ckpt = torch.load(input_path, map_location="cpu")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        return

    old_state = ckpt['model_state_dict']
    
    # Initialize new model structure
    print("🏗️  Initializing new Recurrent Architecture...")
    config = Config()
    new_model = SparseLogicTransformer(config)
    new_state = new_model.state_dict()
    
    # ── Migration Logic ─────────────────────────────────────────────────
    
    # 1. Embeddings (Direct Copy)
    print("🔹 Migrating Embeddings...")
    new_state['embedding.weight'] = old_state['embedding.weight']
    
    if 'pos_embed' in old_state:
        new_state['pos_embed'] = old_state['pos_embed']
    else:
        print("⚠️  No pos_embed found in old checkpoint (initializing from scratch).")

    # 2. Logic Wiring (The "Brain")
    # We extract the middle layer (e.g., layer 12) as the seed for the recurrent block
    # Logic: The middle layer usually has the most stable "general" logic.
    source_layer_idx = 12
    print(f"🔹 Extracting Logic Block from Layer {source_layer_idx}...")
    
    prefix = f"blocks.{source_layer_idx}."
    target_prefix = "core.block."
    
    # Map: Logic Gate & Projections
    # Note: mixing logic (attention) is compatible, but MLP is NOT.
    
    # Logic Gate
    new_state[f'{target_prefix}logic_gate.weight'] = old_state[f'{prefix}logic_gate.weight']
    new_state[f'{target_prefix}logic_gate.bias'] = old_state[f'{prefix}logic_gate.bias']
    
    # Norms
    new_state[f'{target_prefix}norm_logic.weight'] = old_state[f'{prefix}norm_logic.weight']
    new_state[f'{target_prefix}norm_logic.bias'] = old_state[f'{prefix}norm_logic.bias']
    new_state[f'{target_prefix}norm_mlp.weight'] = old_state[f'{prefix}norm_mlp.weight']
    new_state[f'{target_prefix}norm_mlp.bias'] = old_state[f'{prefix}norm_mlp.bias']
    
    # 3. Handling ConvSwiGLU Mismatch
    # Old: MLP (Linear -> GELU -> Linear)
    # New: ConvSwiGLU (Linear -> Conv -> SiLU -> Linear)
    # ❌ Shapes do not match. We must discard old MLP weights.
    print("🔸 Discarding old MLP weights (incompatible with ConvSwiGLU).")
    print("   -> 'recurrent_block.block.conv_swiglu' will be random initialized.")
    
    # 4. Recurrent Gate
    # Initialize to 0 so the first few steps are just the identity/residual flow,
    # allowing the ConvSwiGLU to warm up without destabilizing the logic.
    new_state['core.gate'] = torch.zeros(1)
    print("🔹 Initialized Recurrent Gate to 0.0")

    # 5. Head
    print("🔹 Migrating Output Head...")
    new_state['norm_f.weight'] = old_state['norm_f.weight']
    new_state['norm_f.bias'] = old_state['norm_f.bias']
    new_state['head.weight'] = old_state['head.weight']
    new_state['head.bias'] = old_state['head.bias']
    
    # ── Save ────────────────────────────────────────────────────────────
    torch.save({
        'step': ckpt.get('step', 0),
        'model_state_dict': new_state,
        'optimizer_state_dict': None, # Invalidate optimizer state
        'scheduler_state_dict': None, # Reset scheduler
        'config': config,
        'migration_note': 'Migrated to NanoLogic Fixed-12'
    }, output_path)
    
    print(f"✅ Migration Complete! Saved to: {output_path}")
    print("⚠️  NOTE: Optimizer and Scheduler states were reset. Training will resume phase/lr.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Migrate Legacy Checkpoint to Fixed-12 Recurrent Architecture")
    parser.add_argument("--input", type=str, required=True, help="Path to old checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Path to save new checkpoint")
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)
        
    migrate_checkpoint(args.input, args.output)
