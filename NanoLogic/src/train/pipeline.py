import torch
import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Event
from multiprocessing.shared_memory import SharedMemory
import numpy as np
import time
import os
import sys

# Add project root needed for wiring import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.model.wiring import SHA256Wiring

def worker_loop(stop_event, empty_q, full_q, shm_names, gen_batch, rounds, flat_size, buffer_size):
    """
    Worker logic:
    1. Get empty buffer index.
    2. Generate data.
    3. Write to shared memory.
    4. Put index in full queue.
    """
    # Attach to shared memory blocks by name
    shm_blocks = []
    for name in shm_names:
        try:
            shm_blocks.append(SharedMemory(name=name))
        except FileNotFoundError:
            # Parent might have died
            return
        
    while not stop_event.is_set():
        try:
            # Wait for empty buffer slot
            idx = empty_q.get(timeout=1.0)
        except:
            continue
            
        shm = shm_blocks[idx]
        
        # Generate Data (CPU)
        # states: [B, R+1, 256]
        try:
            states, _ = SHA256Wiring.generate_trace(gen_batch, rounds=rounds+1, device='cpu')
            
            inputs = states[:, :-1, :].reshape(-1, 256).numpy()
            targets = states[:, 1:, :].reshape(-1, 256).numpy()
            
            # Write to indices
            # Buffer layout: [Inputs (fat_size) | Targets (flat_size)]
            
            # Create numpy array wrapper around shared memory
            # Note: We must ensure dtypes match exactly (int64)
            arr = np.ndarray((2 * flat_size,), dtype=np.int64, buffer=shm.buf)
            
            # Copy data
            arr[:flat_size] = inputs.ravel()
            arr[flat_size:] = targets.ravel()
            
            full_q.put(idx)
            
        except Exception as e:
            # print(f"Worker Error: {e}")
            empty_q.put(idx) # Return buffer to pool if failed

class SharedMemoryLoader:
    """
    High-Performance Data Loader using POSIX Shared Memory.
    Bypasses Disk I/O to stream data from CPU generators to GPU consumer.
    
    Architecture:
    - Pool of pre-allocated SharedMemory blocks (Ring Buffer behavior).
    - 'empty_queue': Contains indices of buffers ready to be written to.
    - 'full_queue': Contains indices of buffers ready to be read from.
    """
    def __init__(self, batch_size=64, rounds=64, num_workers=4, buffer_count=20):
        self.batch_size = batch_size
        self.rounds = rounds
        self.num_workers = num_workers
        self.buffer_count = buffer_count
        
        self.samples_per_trace = rounds # transitions
        # We want total output samples ~= batch_size
        self.gen_batch_size = max(1, batch_size // self.samples_per_trace)
        self.actual_batch_size = self.gen_batch_size * self.samples_per_trace
        
        self.flat_size = self.actual_batch_size * 256
        self.dtype = np.int64
        self.itemsize = 8
        self.buffer_size = 2 * self.flat_size * self.itemsize # Inputs + Targets
        
        print(f"🧠 SharedMemoryLoader: Init")
        print(f"   Structure: {buffer_count} buffers x {self.buffer_size / 1e6:.2f} MB")
        print(f"   Batch: {self.gen_batch_size} traces * {self.samples_per_trace} rounds = {self.actual_batch_size} samples")

        # 1. Allocate Shared Memory
        self.shm_blocks = []
        for i in range(buffer_count):
            try:
                shm = SharedMemory(create=True, size=self.buffer_size)
                self.shm_blocks.append(shm)
            except OSError as e:
                print(f"❌ Failed to allocate shared memory: {e}")
                self.shutdown()
                raise e
                
        # 2. Queues
        self.empty_queue = Queue()
        self.full_queue = Queue()
        self.stop_event = Event()
        
        for i in range(buffer_count):
            self.empty_queue.put(i)
            
        # 3. Start Workers
        self.workers = []
        for _ in range(num_workers):
            p = Process(
                target=worker_loop,
                args=(
                    self.stop_event, 
                    self.empty_queue, 
                    self.full_queue, 
                    [shm.name for shm in self.shm_blocks],
                    self.gen_batch_size,
                    self.rounds,
                    self.flat_size,
                    self.buffer_size
                )
            )
            p.daemon = True
            p.start()
            self.workers.append(p)

        # 4. Pre-allocate Tensor Views (Zero-Copy Persistence)
        # We create the wrappers ONCE to avoid malloc/free race conditions in the hot loop.
        self.buffer_views = []
        print("   allocating persistent buffer views...")
        for shm in self.shm_blocks:
            # 1. Numpy wrapper around shared memory
            arr = np.ndarray((2 * self.flat_size,), dtype=np.int64, buffer=shm.buf)
            
            # 2. Torch wrapper (Zero-Copy)
            tensor_data = torch.from_numpy(arr)
            
            # 3. Pre-sliced views
            inputs_view = tensor_data[:self.flat_size].view(self.actual_batch_size, 256)
            targets_view = tensor_data[self.flat_size:].view(self.actual_batch_size, 256)
            
            self.buffer_views.append((inputs_view, targets_view))

    def get_batch(self, device='cpu'):
        """
        Returns (inputs, targets) tensors on device.
        """
        try:
            # Wait for data
            idx = self.full_queue.get(timeout=30.0) # 30s timeout to detect deadlocks
        except:
             return None, None
             
        # Retrieve pre-allocated views (Zero-Malloc)
        inputs_cpu, targets_cpu = self.buffer_views[idx]
        
        # Move to MPS/GPU (this causes a copy, unavoidable for device transfer)
        # valid non_blocking=True if pinned, but here we just do standard move
        inputs = inputs_cpu.to(device)
        targets = targets_cpu.to(device)
        
        # Release buffer back to workers
        self.empty_queue.put(idx)
        
        return inputs, targets

    def shutdown(self):
        self.stop_event.set()
        
        # Drain queues to let workers exit
        while not self.full_queue.empty(): self.full_queue.get()
        
        for p in self.workers:
            p.join(timeout=1.0)
            if p.is_alive():
                p.terminate()
                
        # Unlink memory
        for shm in self.shm_blocks:
            try:
                shm.close()
                shm.unlink()
            except:
                pass
        self.shm_blocks = []
        print("💿 SharedMemoryLoader shutdown complete.")
