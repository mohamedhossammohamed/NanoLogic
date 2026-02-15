# GEMINI.md - NanoLogic (NEURO-SHA-M4)

This file provides instructional context for Gemini regarding the NanoLogic project.

## 🧠 Project Overview
**NanoLogic** (also known as NEURO-SHA-M4) is a neuro-symbolic framework designed for SHA-256 cryptanalysis. It is highly optimized for consumer hardware, specifically Apple Silicon (M4), achieving high performance with extremely low memory footprints (<0.3GB RAM).

### Core Philosophy
Instead of attempting a direct inversion of SHA-256, NanoLogic uses a **neural-guided symbolic search**:
1.  **Neural Component:** A Sparse Logic Transformer learns the internal state transitions and structural dependencies of SHA-256.
2.  **Symbolic Component:** A SAT solver (like Z3 or Kissat) uses the neural model's predictions as a heuristic (priority heatmaps) to focus its search on structurally critical bits.

### Main Technologies
-   **PyTorch (MPS):** Used for the neural model, leveraging Apple Silicon's GPU.
-   **BitNet b1.58:** Employs ternary weights `{-1, 0, 1}` to model boolean logic directly and reduce parameter size (quantization).
-   **Sparse Logic Attention:** Instead of dense attention, it uses hard-coded wiring based on the SHA-256 specification (`Σ₀`, `Σ₁`, `Maj`, `Ch`), reducing attention complexity from $O(N^2)$ to $O(N)$.
-   **LionGaLore:** Uses the Lion optimizer with GaLore (Gradient Low-Rank Projection) for memory-efficient training of large layers.
-   **SharedMemory I/O:** Replaced the disk-based buffer with POSIX Shared Memory, eliminating NVMe bottlenecks and maximizing M4 throughput.
-   **Mixed Precision (FP16):** Uses `GradScaler` and `autocast` to maintain numerical stability in the 1.58-bit quantized space while improving performance on MPS.
-   **NeuroCDCL:** A neural-guided SAT solver that uses the neural model's predictions as soft constraints/heuristics in the CDCL loop.
-   **MemoryGuard:** An active memory defense system that monitors RAM usage and prevents swap thrashing on limited-memory hardware.
-   **Recurrent Architecture (Fixed-12):** A looped super-block design that applies the same logic layer iteratively, reducing parameter count while increasing reasoning depth.
-   **Z3 Solver:** The symbolic engine bridged with the neural model.

---

## 🏗️ Architecture & Structure

-   `main.py`: The primary training entry point with auto-resume capability.
-   `nanologic.py`: The NanoLogic TUI (CyberSystem Edition) for interactive usage.
-   `config.py`: Centralized configuration for model hyperparameters, hardware optimization, and curriculum scheduling.
-   `src/model/`:
    -   `sparse_logic.py`: Implementation of the Sparse Logic Block and Transformer.
    -   `bitnet.py`: BitLinear layers with ternary weight quantization.
    -   `wiring.py`: The static wiring graph for SHA-256 and trace generators.
    -   `pathfinder.py`: Residual blocks for spatial and channel mixing in neural state transitions.
-   `src/train/`:
    -   `pipeline.py`: Shared Memory Loader (CPU producers -> GPU consumer).
    -   `curriculum.py`: Multi-phase training scheduler (e.g., 8 → 16 → 32 → 64 rounds).
    -   `loss.py`: Data generation and neuro-symbolic loss functions.
    -   `synthetic.py`: High-performance synthetic data generator for SHA-256 traces.
-   `src/solver/`:
    -   `bridge.py`: Shared memory bridge between the neural model and the SAT solver.
    -   `neuro_cdcl.py`: End-to-end preimage search loop integrating neural predictions with Z3.
    -   `z3_sha256.py`: Symbolic SHA-256 implementation using the Z3 API.
-   `src/optim/`:
    -   `lion_galore.py`: Memory-efficient optimizer implementation.
-   `src/utils/`:
    -   `monitor.py`: Implementation of MemoryGuard for system health monitoring.
-   `src/tools/` & `src/tui/`: Modular logic and UI components for the Textual-based dashboard.
-   `tools/`:
    -   `neuro_cli.py`: Interactive CLI for demonstrations and "AI vs. Brute Force" races.
    -   `benchmark_solver.py`: Benchmarking suite for Z3 vs. Neuro-Symbolic solver.
    -   `verify_metrics.py`: System integrity check for memory and data streams.
    -   `diagnose_gate.py`: Diagnostic tool for Recurrent Gate and BitConvSwiGLU weights.

---

## 🚀 Building and Running

### Prerequisites
-   Python 3.10+
-   Dependencies: `pip install torch psutil rich textual`
-   (Optional) Z3 solver for bridge functionality.

### Key Commands
-   **Training:**
    ```bash
    python3 main.py
    ```
    Training automatically resumes from `checkpoints/`. It follows a curriculum that promotes the model to more SHA-256 rounds only after reaching specific accuracy thresholds (e.g., 80% for 8 rounds).

-   **Interactive TUI:**
    ```bash
    # Launch the CyberSystem Dashboard
    python3 nanologic.py
    ```

-   **Interactive CLI:**
    ```bash
    # AI vs Brute Force race
    python3 tools/neuro_cli.py --mode race
    
    # Dashboard for hash cracking
    python3 tools/neuro_cli.py --mode crack
    ```

-   **Benchmarking & Verification:**
    ```bash
    # Run Z3 vs Neuro benchmark
    python3 tools/benchmark_solver.py --mode z3 --device cpu
    python3 tools/benchmark_solver.py --mode neuro --device cpu

    # Verify system metrics (RAM, Stream Integrity)
    python3 tools/verify_metrics.py
    ```

-   **Testing:**
    ```bash
    pytest
    ```
    The project includes a comprehensive test suite in the `tests/` directory covering bitnet, wiring, loss functions, and solver components.

---

## 🛠️ Development Conventions

1.  **Memory Optimization:** Always use `BitLinear` for dense layers and `grad_checkpointing` in `SparseLogicBlock` to keep RAM usage low.
2.  **Device-Agnostic Code:** While optimized for `mps`, the code should gracefully fall back to `cpu`.
3.  **Curriculum-Driven:** New features should respect the `CurriculumScheduler` logic to ensure the model doesn't "forget" early round logic when scaling.
4.  **Static Wiring:** Any changes to the SHA-256 logic must be reflected in `src/model/wiring.py` to maintain the $O(N)$ attention sparse mask.
5.  **Logging:** Training metrics are saved to `logs/training.log` in CSV format. Never overwrite this file; always append.

---

## 📝 License
This project is licensed under the **PolyForm Noncommercial License 1.0.0**. Commercial use is prohibited without separate agreement. See `LICENSE` and `COMMERCIAL_TERMS.md` for details.
