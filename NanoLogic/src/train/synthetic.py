
import torch
import sys
import os

# Ensure the project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.wiring import SHA256Wiring

def infinite_trace_stream(batch_size=8, rounds=64, device='cpu'):
    """
    Lazy Generator for SHA-256 traces.
    Strictly yields ONE batch at a time to prevent RAM spikes.
    
    Args:
        batch_size (int): Number of traces per batch.
        rounds (int): Number of SHA-256 rounds to simulate.
        device (str): Device to generate data on (CPU recommended for memory).
        
    Yields:
        (inputs, targets): Tensors for training.
    """
    while True:
        # Generate full traces [B, R, 256]
        # We generate on CPU to avoid MPS allocator fragmentation
        states, _ = SHA256Wiring.generate_trace(batch_size, rounds=rounds+1, device=device)
        
        # Inputs: State[t]
        # Targets: State[t+1]
        # Flatten across rounds to treat each transition as an independent sample
        # shape: [B * R, 256]
        
        inputs = states[:, :-1, :].reshape(-1, 256)
        targets = states[:, 1:, :].reshape(-1, 256)
        
        yield inputs, targets
        
        # Explicitly delete to free memory immediately
        del states, inputs, targets
