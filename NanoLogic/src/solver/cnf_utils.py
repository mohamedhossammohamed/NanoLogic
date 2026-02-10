import torch
import numpy as np

def assignment_to_tensor(assignment, rounds=64, dim=512):
    """
    Converts a flat assignment vector (from solver) to a structural tensor.
    
    assignment: List or Array of literals [1, -2, 3, ...] or binary [1, 0, 1, ...]
    
    If literals:
        positive = 1
        negative = 0
        0 (unassigned) = ? (maybe 0.5 or -1?)
        
    Neuro-SHA uses {0, 1} for BitNet. Unassigned variables might be handled 
    by a separate "mask" or "activity" input, or just 0 for now.
    """
    # Assuming assignment is a binary numpy array [0, 1, 0...] length N_vars
    # We map this to [ROUNDS, 8, 32]
    
    # Total bits = 64 * 8 * 32 = 16384 bits for full SHA.
    # But usually solvers work on a reduced round instance.
    
    # For now, simplistic padding/truncation
    total_bits = len(assignment)
    expected_bits = rounds * 8 * 32
    
    if total_bits < expected_bits:
        # Pad with 0
        pad = np.zeros(expected_bits - total_bits, dtype=assignment.dtype)
        assignment = np.concatenate([assignment, pad])
    elif total_bits > expected_bits:
        assignment = assignment[:expected_bits]
        
    return torch.tensor(assignment, dtype=torch.long)

def tensor_to_cnf(logits, threshold=0.85):
    """
    Convert model output logits to constraint hints for Z3.
    
    Takes the raw logits from SparseLogicTransformer and converts them
    into a list of (bit_index, predicted_value, confidence) tuples
    that can be injected as soft constraints.
    
    Args:
        logits: torch.Tensor of shape [256] (raw logits from model).
        threshold: Minimum confidence for constraint injection.
        
    Returns:
        constraints: list of (bit_index, value, confidence) tuples.
        neural_hints: numpy array of shape [256] for solve_partial().
    """
    import torch
    
    probs = torch.sigmoid(logits).detach().cpu().numpy()
    
    constraints = []
    for i, p in enumerate(probs):
        if p > threshold:
            constraints.append((i, 1, float(p)))
        elif p < (1.0 - threshold):
            constraints.append((i, 0, float(1.0 - p)))
    
    return constraints, probs
