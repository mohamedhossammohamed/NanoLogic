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
    def __init__(self, limit_gb: float = 10.0, poll_interval: int = 10):
        self.limit_gb = limit_gb
        self.poll_interval = poll_interval
        self.step_counter = 0
        self.process = psutil.Process(os.getpid())
        print(f"🛡️ Memory Guard Active: Limit {self.limit_gb}GB | Poll every {self.poll_interval} steps")

    def check(self):
        """
        Call this every training step.
        """
        self.step_counter += 1
        if self.step_counter % self.poll_interval != 0:
            return

        # Check System Verification
        vm = psutil.virtual_memory()
        used_gb = vm.used / (1024 ** 3)
        
        # Check Process Specific (Optional, but good for debugging)
        # proc_gb = self.process.memory_info().rss / (1024 ** 3)

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
