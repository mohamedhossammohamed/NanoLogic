import torch
import torch.nn as nn
from tqdm import tqdm
import time
import gc
import os
import psutil
import sys
import logging

# Append path to ensure imports work
sys.path.append(".")

from config import Config
from src.model import SparseLogicTransformer, Pathfinder, SHA256Wiring
from src.optim import LionGaLore
from src.train import StateMatchingLoss, CurriculumScheduler, infinite_trace_stream
from src.train.pipeline import SharedMemoryLoader
from src.solver import SolverBridge
from src.utils.monitor import MemoryGuard
from src.utils.identity import ModelIdentity # [NEW]

# ── IMPORTS FOR MIXED PRECISION ──────────────────────────────────────────────
try:
    from torch.amp import autocast, GradScaler
    pass
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

# ── LOGGING UTILS ────────────────────────────────────────────────────────────
class TeeLogger(object):
    """
    Writes output to both the console (stdout) and a log file.
    Does NOT buffer; flushes immediately to capture crashes.
    """
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Redirects stderr too
class TeeErrorLogger(TeeLogger):
    def __init__(self, filename):
        self.terminal = sys.stderr
        self.log = open(filename, "a", encoding="utf-8")


def compute_accuracy(logits, targets):
    """Bit-level accuracy: fraction of correctly predicted bits."""
    preds = (logits > 0.0).long()
    return (preds == targets).float().mean().item()


def main():
    print("🧠 Initializing Neuro-SHA-M4 System...")

    # 1. Config & Identity
    config = Config()
    
    # [NEW] Model Identity System
    identity = ModelIdentity(config)
    print(f"🆔 Model Identity: {identity.model_id}")
    print(f"   Architecture: {config.dim}d / {config.n_layers}L / {config.n_heads}H")
    
    # Setup Isolated Workspace
    # logs/<id>/ and checkpoints/<id>/
    LOG_DIR, CHECKPOINT_DIR = identity.setup_workspace()
    print(f"📂 Workspace: {LOG_DIR}")
    
    # [NEW] Enable Tee Logger (Console -> File)
    console_log_path = os.path.join(LOG_DIR, "console.log")
    sys.stdout = TeeLogger(console_log_path)
    sys.stderr = TeeErrorLogger(console_log_path)
    print(f"📝 Console logs mirrored to: {console_log_path}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"✅ Device: {device}")
    
    # Enable mixed precision scaler
    scaler = None
    scaler_type = "Unknown"
    
    try:
        # Try new torch.amp API first
        from torch.amp import GradScaler as AmpGradScaler
        if device == 'mps':
            scaler = AmpGradScaler('mps')
            scaler_type = "MPS Native"
        else:
            scaler = AmpGradScaler()
            scaler_type = "Generic"
    except (ImportError, TypeError):
        # Fallback to legacy behavior
        scaler = GradScaler()
        scaler_type = "Legacy/CUDA (Warning Expected)"
        
    print(f"⚡ Mixed Precision (FP16): Enabled ({scaler_type})")

    # 2. Model
    model = SparseLogicTransformer(config).to(device)
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"🕸️ Model: Sparse Logic Transformer ({n_params:.1f}M Params)")

    pathfinder = Pathfinder(config).to(device)
    print("🗺️ Model: Pathfinder (ResNet)")

    # 3. Optimizer
    optimizer = LionGaLore(model.parameters(), lr=config.lr)
    print("🦁 Optimizer: LionGaLore")

    # 4. Solver Bridge
    bridge = SolverBridge()
    bridge.connect()
    print("🌉 Bridge: Ready (Waiting for Solver...)")

    # 5. Memory Guard
    memory_guard = MemoryGuard(limit_gb=14.0)

    # 6. Training Components
    criterion = StateMatchingLoss()
    scheduler = CurriculumScheduler(config)

    # Gradient Accumulation
    grad_accum = config.grad_accum_steps
    print(f"📦 Gradient Accumulation: {grad_accum} micro-batches (effective batch = {config.batch_size * grad_accum})")

    # ── AUTO-RESUME (Scoped to this Model ID) ────────────────────────────
    # Scan checkpoints/<id>/ for the latest saved state.
    resumed = False
    
    def get_step_from_filename(f):
        if f == "neuro_sha_final.pt":
            return float('inf')
        try:
            return int(f.split('_')[-1].split('.')[0])
        except (ValueError, IndexError):
            return -1

    ckpt_files = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")],
        key=get_step_from_filename,
        reverse=True
    ) if os.path.isdir(CHECKPOINT_DIR) else []

    if ckpt_files:
        resume_path = os.path.join(CHECKPOINT_DIR, ckpt_files[0])
        print(f"🔄 Found checkpoint: {resume_path}")
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            
            if 'pathfinder_state_dict' in ckpt:
                pathfinder.load_state_dict(ckpt['pathfinder_state_dict'])
                print("   ✅ Resumed Pathfinder weights")
            
            if ckpt.get('optimizer_state_dict') is not None:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                
            # Restore FULL scheduler state (phase, steps, total_steps)
            if ckpt.get('scheduler_state_dict') is not None:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                print(f"   ✅ Resumed scheduler state | Phase: {scheduler.current_phase} | Total steps: {scheduler.total_steps}")
            elif 'step' in ckpt:
                scheduler.total_steps = ckpt['step']
                print(f"   ✅ Resumed total steps: {scheduler.total_steps}")
            
            resumed = True
        except Exception as e:
            print(f"   ⚠️ Failed to load checkpoint ({e}). Starting fresh.")
    else:
        print("🆕 No checkpoint found for this configuration. Starting fresh.")

    # Print curriculum info
    thresholds_str = " | ".join(
        f"{r}r: {t:.0%}"
        for r, t in zip(config.curriculum_rounds, config.phase_accuracy_thresholds)
    )
    print(f"\n🚀 {'Resuming' if resumed else 'Starting'} Logic Learning (Deep Curriculum)...")
    print(f"   Phases: {' → '.join(str(r) for r in config.curriculum_rounds)} rounds")
    print(f"   Accuracy gates: {thresholds_str}")
    print(f"   Starting from: {scheduler.get_current_rounds()} rounds (config.start_round)")
    print(f"{'─'*90}")
    print(f"{'Step':>8} | {'Loss':>10} | {'Accuracy':>10} | {'Threshold':>10} | {'RAM (GB)':>10} | {'Phase':>6} | {'Rounds':>6}")
    print(f"{'─'*90}")

    model.train()

    # ── Log File (APPEND mode — never lose history) ──────────────────────
    # CSV Log now lives in logs/<id>/training.log
    log_path = os.path.join(LOG_DIR, "training.log")
    log_exists = os.path.isfile(log_path) and os.path.getsize(log_path) > 0
    log_file = open(log_path, "a")  # APPEND, not overwrite
    if not log_exists:
        log_file.write("step,loss,accuracy,threshold,ram_gb,phase,rounds\n")
        log_file.flush()

    # Accumulators
    running_loss = 0.0
    running_acc = 0.0
    log_steps_counter = 0 # Track actual steps for accurate averaging
    optimizer.zero_grad()

    # ── Initialize Shared Memory Loader ──────────────────────────────────
    current_rounds = scheduler.get_current_rounds()
    loader = SharedMemoryLoader(
        batch_size=config.batch_size, 
        rounds=current_rounds, 
        num_workers=4 # Use 4 parallel producers for M4 efficiency
    )

    try:
        while True:
            rounds = scheduler.get_current_rounds()

            # ── Generate Data (Shared Memory) ────────────────────────────
            # Blocks until data is available from shared memory
            inputs, targets = loader.get_batch(device=device)
            if inputs is None: # Shutdown signal
                break

            # ── Forward (Mixed Precision) ────────────────────────────────
            # Autocast context for FP16
            with autocast(device_type=device, dtype=torch.float16):
                logits = model(inputs)
                loss = criterion(logits, targets)
                # Scale loss for gradient accumulation
                scaled_loss = loss / grad_accum

            # Backward with Scaler
            scaler.scale(scaled_loss).backward()

            # Track metrics (unscaled)
            loss_val = loss.item()
            acc_val = compute_accuracy(logits.detach(), targets)
            running_loss += loss_val
            running_acc += acc_val
            log_steps_counter += 1

            # ── Step scheduler with accuracy (accuracy-gated promotion) ──
            _, phase_changed = scheduler.step(accuracy=acc_val)
            if phase_changed:
                new_rounds = scheduler.get_current_rounds()
                new_threshold = scheduler.get_accuracy_threshold()
                print(f"\n🔄 Phase Shift! Accuracy gate passed → now {new_rounds} rounds (next gate: {new_threshold:.0%})")
                
                # Restart Loader for new curriculum phase
                print("♻️  Restarting Data Loader for new curriculum phase...")
                
                # ── Safe Shutdown Sequence ──
                del inputs, targets, logits, loss, scaled_loss
                loader.shutdown()
                
                # Force cleanup of old shared memory views
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                
                loader = SharedMemoryLoader(
                    batch_size=config.batch_size, 
                    rounds=new_rounds, 
                    num_workers=4
                )

            # ── Optimizer Step (every grad_accum micro-batches) ──────────
            if scheduler.total_steps % grad_accum == 0:
                # Gradient clipping for safety (unscale first)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Scaler Step
                scaler.step(optimizer)
                scaler.update()
                
                optimizer.zero_grad()

            # ── Cleanup ──────────────────────────────────────────────────
            del inputs, targets, logits, loss, scaled_loss

            # Clear MPS cache periodically
            if scheduler.total_steps % 50 == 0:
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

            # Memory guard
            memory_guard.check()

            # ── Logging (every 10 steps) ─────────────────────────────────
            if scheduler.total_steps % 10 == 0 and log_steps_counter > 0:
                avg_loss = running_loss / log_steps_counter
                avg_acc = running_acc / log_steps_counter
                
                # Precise Memory Tracking
                cpu_gb, mps_gb = MemoryGuard.get_total_memory_usage()
                total_app_gb = cpu_gb + mps_gb
                
                threshold = scheduler.get_accuracy_threshold()

                # Print breakdown: Total (CPU | MPS)
                print(f"{scheduler.total_steps:>8} | {avg_loss:>10.4f} | {avg_acc:>9.2%} | {threshold:>9.0%} | {total_app_gb:>9.2f} ({cpu_gb:.1f}+{mps_gb:.1f}) | {scheduler.current_phase:>6} | {rounds:>6}")

                # CSV log (append)
                log_file.write(f"{scheduler.total_steps},{avg_loss:.6f},{avg_acc:.6f},{threshold:.2f},{total_app_gb:.2f},{scheduler.current_phase},{rounds}\n")
                log_file.flush()

                running_loss = 0.0
                running_acc = 0.0
                log_steps_counter = 0

            # ── Periodic Checkpoint (every 500 steps) ────────────────────
            if scheduler.total_steps % 500 == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"neuro_sha_step_{scheduler.total_steps}.pt")
                torch.save({
                    'step': scheduler.total_steps,
                    'model_state_dict': model.state_dict(),
                    'pathfinder_state_dict': pathfinder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss_val,
                    'scaler_state_dict': scaler.state_dict(), # [NEW] Save scaler state
                }, ckpt_path)
                print(f"  💾 Checkpoint saved: {ckpt_path}")
                
                # ── Checkpoint Cleanup (Keep Last 2) ──
                try:
                    all_ckpts = [f for f in os.listdir(CHECKPOINT_DIR) if f.startswith("neuro_sha_step_") and f.endswith(".pt")]
                    # Sort by step number
                    all_ckpts.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
                    
                    if len(all_ckpts) > 2:
                        to_delete = all_ckpts[:-2]
                        for f in to_delete:
                            os.remove(os.path.join(CHECKPOINT_DIR, f))
                            print(f"  🗑️  Pruned old checkpoint: {f}")
                except Exception as e:
                    print(f"  ⚠️  Checkpoint cleanup failed: {e}")

    except KeyboardInterrupt:
        print("\n🛑 Training Interrupted by User")

    # Final save
    final_path = os.path.join(CHECKPOINT_DIR, "neuro_sha_final.pt")
    torch.save({
        'step': scheduler.total_steps,
        'model_state_dict': model.state_dict(),
        'pathfinder_state_dict': pathfinder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(), # [NEW] Save scaler state
    }, final_path)
    print(f"💾 Final checkpoint saved: {final_path}")

    log_file.close()
    bridge.close()
    if 'loader' in locals():
        loader.shutdown()
    print(f"\n✅ Training Complete. {scheduler.total_steps} steps executed.")
    print(f"📊 Logs: {log_path}")
    print(f"📝 Console logs: {console_log_path}") # [NEW]


if __name__ == "__main__":
    main()
