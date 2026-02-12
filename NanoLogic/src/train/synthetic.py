
import torch
import sys
import os
import time
import glob
from multiprocessing import Process, Event

# Ensure the project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.model.wiring import SHA256Wiring

# Configuration for the buffer
BUFFER_DIR = "buffer_cache" # Relative to where main.py is run
MAX_BUFFER_SIZE = 100

def producer_worker(stop_event, batch_size, rounds):
    """Running on CPU Core: Generates data and saves to SSD."""
    # Ensure buffer directory exists
    os.makedirs(BUFFER_DIR, exist_ok=True)
    
    while not stop_event.is_set():
        # 1. Check if buffer is full
        # robust check to avoid crashing if dir is empty or deleted
        try:
            current_files = len(os.listdir(BUFFER_DIR))
        except FileNotFoundError:
            os.makedirs(BUFFER_DIR, exist_ok=True)
            current_files = 0
            
        if current_files >= MAX_BUFFER_SIZE:
            time.sleep(0.1) # Wait for GPU to eat some data
            continue

        # 2. Generate Data (The heavy CPU work)
        # We generate on CPU to avoid MPS allocator fragmentation
        # states shape: [B, R+1, 256]
        states, _ = SHA256Wiring.generate_trace(batch_size, rounds=rounds+1, device='cpu')
        
        # Prepare inputs and targets
        # Inputs: State[t]
        # Targets: State[t+1]
        # Flatten across rounds to treat each transition as an independent sample
        inputs = states[:, :-1, :].reshape(-1, 256)
        targets = states[:, 1:, :].reshape(-1, 256)
        
        # 3. Save to Disk (Fast on M4)
        timestamp = time.time_ns()
        # Use a unique filename with pid to avoid collisions if multiple workers start same time
        filename = os.path.join(BUFFER_DIR, f"batch_{os.getpid()}_{timestamp}.pt")
        
        # Save essentially a tuple or dict
        torch.save((inputs, targets), filename)
        
        # Explicit cleanup
        del states, inputs, targets

class DiskBufferLoader:
    def __init__(self, batch_size=64, rounds=64, num_workers=2):
        self.stop_event = Event()
        self.workers = []
        self.batch_size = batch_size
        self.rounds = rounds
        
        print(f"💿 Initializing DiskBufferLoader with {num_workers} workers...")
        print(f"   Buffer Dir: {os.path.abspath(BUFFER_DIR)}")

        # Cleanup old buffer on start
        if os.path.exists(BUFFER_DIR):
            for f in glob.glob(f"{BUFFER_DIR}/*.pt"):
                try:
                    os.remove(f)
                except OSError:
                    pass
        else:
            os.makedirs(BUFFER_DIR, exist_ok=True)

        # Start Producers
        for _ in range(num_workers):
            p = Process(target=producer_worker, args=(self.stop_event, batch_size, rounds))
            p.start()
            self.workers.append(p)
            
    def get_batch(self, device='cpu'):
        """Called by Training Loop: Loads from SSD and deletes."""
        while True:
            # Check for stop event just in case
            if self.stop_event.is_set():
                return None, None

            files = sorted(glob.glob(f"{BUFFER_DIR}/*.pt"))
            
            if not files:
                # print("⚠️  GPU Starved! Waiting for data...", end='\r')
                time.sleep(0.05)
                continue
            
            # Load and Delete
            target_file = files[0]
            try:
                # Load to CPU first
                data = torch.load(target_file, map_location='cpu')
                os.remove(target_file) # Cleanup immediately
                
                inputs, targets = data
                return inputs.to(device), targets.to(device)
                
            except (FileNotFoundError, RuntimeError, EOFError):
                # Another worker might have grabbed it, or partial write. Retry.
                continue

    def shutdown(self):
        print("💿 Shutting down DiskBufferLoader...")
        self.stop_event.set()
        for p in self.workers:
            if p.is_alive():
                p.terminate()
                p.join()
        
        # Optional: Cleanup buffer on exit
        # for f in glob.glob(f"{BUFFER_DIR}/*.pt"):
        #     os.remove(f)

def infinite_trace_stream(batch_size=8, rounds=64, device='cpu'):
    """
    DEPRECATED: Use DiskBufferLoader for high performance.
    
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
