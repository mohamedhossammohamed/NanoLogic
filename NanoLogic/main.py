import torch
import torch.nn as nn
from tqdm import tqdm
import time
import gc
import os
import psutil
import sys

# Append path to ensure imports work
sys.path.append(".")

from config import Config
from src.model import SparseLogicTransformer, Pathfinder, SHA256Wiring
from src.optim import LionGaLore
from src.train import StateMatchingLoss, CurriculumScheduler, infinite_trace_stream
from src.solver import SolverBridge
from src.utils.monitor import MemoryGuard

# ── Directories ──────────────────────────────────────────────────────────────
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


def compute_accuracy(logits, targets):
    """Bit-level accuracy: fraction of correctly predicted bits."""
    preds = (logits > 0.0).long()
    return (preds == targets).float().mean().item()


def main():
    print("🧠 Initializing Neuro-SHA-M4 System...")

    # 1. Config
    config = Config()
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"✅ Device: {device}")

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
    memory_guard = MemoryGuard(limit_gb=10.0)

    # 6. Training Components
    criterion = StateMatchingLoss()
    scheduler = CurriculumScheduler(config)

    # Gradient Accumulation
    grad_accum = config.grad_accum_steps
    print(f"📦 Gradient Accumulation: {grad_accum} micro-batches (effective batch = {config.batch_size * grad_accum})")

    # ── AUTO-RESUME ──────────────────────────────────────────────────────
    # Scan checkpoints/ for the latest saved state and resume automatically.
    # NOTE: config.start_round controls the starting phase — we load model
    # and optimizer weights but override the scheduler to start from the
    # configured round.
    resumed = False
    ckpt_files = sorted(
        [f for f in os.listdir(CHECKPOINT_DIR) if f.endswith(".pt")],
        key=lambda f: os.path.getmtime(os.path.join(CHECKPOINT_DIR, f)),
        reverse=True
    ) if os.path.isdir(CHECKPOINT_DIR) else []

    if ckpt_files:
        resume_path = os.path.join(CHECKPOINT_DIR, ckpt_files[0])
        print(f"🔄 Found checkpoint: {resume_path}")
        try:
            ckpt = torch.load(resume_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            # Restore total_steps for logging continuity but keep phase from config.start_round
            if 'scheduler_state_dict' in ckpt:
                scheduler.total_steps = ckpt['scheduler_state_dict'].get('total_steps', 0)
            elif 'step' in ckpt:
                scheduler.total_steps = ckpt['step']
            resumed = True
            print(f"   ✅ Resumed model weights | Total steps so far: {scheduler.total_steps}")
            print(f"   📍 Starting at Phase {scheduler.current_phase} ({scheduler.get_current_rounds()} rounds) per config.start_round={config.start_round}")
        except Exception as e:
            print(f"   ⚠️ Failed to load checkpoint ({e}). Starting fresh.")
    else:
        print("🆕 No checkpoint found. Starting fresh.")

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
    log_path = os.path.join(LOG_DIR, "training.log")
    log_exists = os.path.isfile(log_path) and os.path.getsize(log_path) > 0
    log_file = open(log_path, "a")  # APPEND, not overwrite
    if not log_exists:
        log_file.write("step,loss,accuracy,threshold,ram_gb,phase,rounds\n")
        log_file.flush()

    # Accumulators
    running_loss = 0.0
    running_acc = 0.0
    optimizer.zero_grad()

    try:
        while True:
            rounds = scheduler.get_current_rounds()

            # ── Generate Data (lazy, one micro-batch) ────────────────────
            data_gen = infinite_trace_stream(config.batch_size, rounds, "cpu")
            inputs_cpu, targets_cpu = next(data_gen)

            inputs = inputs_cpu.to(device)
            targets = targets_cpu.to(device)

            # ── Forward ──────────────────────────────────────────────────
            logits = model(inputs)
            loss = criterion(logits, targets)

            # Scale loss for gradient accumulation
            scaled_loss = loss / grad_accum
            scaled_loss.backward()

            # Track metrics (unscaled)
            loss_val = loss.item()
            acc_val = compute_accuracy(logits.detach(), targets)
            running_loss += loss_val
            running_acc += acc_val

            # ── Step scheduler with accuracy (accuracy-gated promotion) ──
            _, phase_changed = scheduler.step(accuracy=acc_val)
            if phase_changed:
                new_rounds = scheduler.get_current_rounds()
                new_threshold = scheduler.get_accuracy_threshold()
                print(f"\n🔄 Phase Shift! Accuracy gate passed → now {new_rounds} rounds (next gate: {new_threshold:.0%})")

            # ── Optimizer Step (every grad_accum micro-batches) ──────────
            if scheduler.total_steps % grad_accum == 0:
                # Gradient clipping for safety
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
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
            if scheduler.total_steps % 10 == 0:
                avg_loss = running_loss / 10
                avg_acc = running_acc / 10
                ram_gb = psutil.Process(os.getpid()).memory_info().rss / (1024 ** 3)
                threshold = scheduler.get_accuracy_threshold()

                print(f"{scheduler.total_steps:>8} | {avg_loss:>10.4f} | {avg_acc:>9.2%} | {threshold:>9.0%} | {ram_gb:>9.2f} | {scheduler.current_phase:>6} | {rounds:>6}")

                # CSV log (append)
                log_file.write(f"{scheduler.total_steps},{avg_loss:.6f},{avg_acc:.6f},{threshold:.2f},{ram_gb:.2f},{scheduler.current_phase},{rounds}\n")
                log_file.flush()

                running_loss = 0.0
                running_acc = 0.0

            # ── Periodic Checkpoint (every 500 steps) ────────────────────
            if scheduler.total_steps % 500 == 0:
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"neuro_sha_step_{scheduler.total_steps}.pt")
                torch.save({
                    'step': scheduler.total_steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss_val,
                }, ckpt_path)
                print(f"  💾 Checkpoint saved: {ckpt_path}")

    except KeyboardInterrupt:
        print("\n🛑 Training Interrupted by User")

    # Final save
    final_path = os.path.join(CHECKPOINT_DIR, "neuro_sha_final.pt")
    torch.save({
        'step': scheduler.total_steps,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, final_path)
    print(f"💾 Final checkpoint saved: {final_path}")

    log_file.close()
    bridge.close()
    print(f"\n✅ Training Complete. {scheduler.total_steps} steps executed.")
    print(f"📊 Logs: {log_path}")


if __name__ == "__main__":
    main()
