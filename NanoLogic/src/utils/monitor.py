import psutil
import torch
import gc
import time
import os

class MemoryGuard:
    """
    Active memory defense system.
    Polls RAM usage and aggressively clears caches or pauses execution 
    if the system approaches the danger zone (Swap Thrashing).
    """
    def __init__(self, limit_gb: float = 14.0, poll_interval: int = 10):
        self.limit_gb = limit_gb
        self.poll_interval = poll_interval
        self.step_counter = 0
        self.process = psutil.Process(os.getpid())
        print(f"🛡️ Memory Guard Active: Limit {self.limit_gb}GB | Poll every {self.poll_interval} steps")

    @staticmethod
    def get_total_memory_usage():
        """
        Calculates total memory usage:
        1. RSS of Main Process
        2. RSS of all Child Processes (e.g. workers)
        3. MPS/GPU Memory (if available, though typically Unified)
        """
        current_process = psutil.Process(os.getpid())
        total_rss = current_process.memory_info().rss
        
        # Add children (workers)
        for child in current_process.children(recursive=True):
            try:
                total_rss += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
                
        # Convert to GB
        ram_gb = total_rss / (1024 ** 3)
        
        # MPS Memory (Driver Allocated)
        # On Apple Silicon, this is distinct from RSS for some allocations, 
        # or overlapping. We track it for completeness.
        mps_gb = 0.0
        if torch.backends.mps.is_available():
            try:
                # current_allocated_memory returns bytes
                mps_gb = torch.mps.current_allocated_memory() / (1024 ** 3)
            except:
                pass
                
        return ram_gb, mps_gb

    def check(self):
        """
        Call this every training step.
        """
        self.step_counter += 1
        if self.step_counter % self.poll_interval != 0:
            return

        # Check Total Usage (RAM + MPS estimate if separated)
        # For the active guard, we rely on psutil.virtual_memory().used 
        # because that's the system-wide truth for "Are we running out?"
        vm = psutil.virtual_memory()
        used_gb = vm.used / (1024 ** 3)
        
        if used_gb > self.limit_gb:
            print(f"\n[MEMORY GUARD] ⚠️ RAM Warning: {used_gb:.2f}GB / {self.limit_gb}GB Limit")
            self._emergency_brake(used_gb)

    def _emergency_brake(self, current_gb):
        """
        Executes the emergency protocol to reclaim RAM.
        """
        print("[MEMORY GUARD] 🧹 Triggering Garbage Collection & Cache Clear...")
        
        # 1. Force Python GC
        gc.collect()
        
        # 2. Clear MPS/CUDA Cache
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Re-check
        vm = psutil.virtual_memory()
        new_gb = vm.used / (1024 ** 3)
        
        if new_gb > self.limit_gb:
            print(f"[MEMORY GUARD] 🛑 RAM still high ({new_gb:.2f}GB). Pausing for SWAP drain...")
            # Wait for OS to swap out idle pages
            time.sleep(30)
            print("[MEMORY GUARD] ▶️ Resuming...")
        else:
            print(f"[MEMORY GUARD] ✅ Cooled down to {new_gb:.2f}GB")
