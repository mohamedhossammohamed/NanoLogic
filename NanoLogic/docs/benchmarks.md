# Solver Benchmarking & Verification

## Overview
This document outlines the methodology for benchmarking the Neuro-Symbolic Solver against the standard Z3 baseline, as well as the system verification protocols.

## Tools
All benchmark scripts are located in `tools/`:
- `benchmark_solver.py`: Runs the solver race.
- `verify_metrics.py`: Checks data integrity and memory reporting.
- `diagnose_gate.py`: Checks the recurrent gate status.

## Benchmark Methodology

### 1. Z3 Baseline (`--mode z3`)
- **Solver**: Standard Z3 Theorem Prover.
- **Problem**: SHA-256 Preimage Search (Reduced Rounds).
- **Constraints**: 
  - `rounds=8` (Standard for initial testing).
  - `timeout=10s` per instance.
- **Operation**: 
  - Generates 50 random message/hash pairs.
  - Attempts to invert the hash using Z3 BitVectors.

### 2. Neuro-Symbolic (`--mode neuro`)
- **Solver**: `NeuroCDCL` (Neural-Guided Conflict-Driven Clause Learning).
- **Model**: Latest `SparseLogicTransformer` checkpoint.
- **Operation**:
  1. **Prediction**: Neural model predicts probability of each message bit being 1.
  2. **Guidance**: High-confidence bits (>85%) are fixed as constraints in Z3.
  3. **Refinement**: If UNSAT/Timeout, the threshold is adjusted and retried (up to 10 iterations).

## Current Results (Feb 15, 2026)

| Solver | Rounds | Instances | Success Rate | Avg Time | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Z3** | 8 | 50 | 0% | N/A | Timed out (>10s) on all instances. |
| **Neuro**| 8 | 5 | 0% | N/A | Hybrid search also timed out. Model accuracy (~62%) insufficient for 8 rounds. |

> **Conclusion**: 8-round SHA-256 is harder than anticipated for the current model maturity. We need to either:
> 1. Continue training to >85% accuracy.
> 2. Benchmark on easier 4-round instances to validate the pipeline first.

## Verification Protocols

### Metrics Verification (`verify_metrics.py`)
Ensures the dashboard numbers are real.
- **Memory**: Checks `RSS` (System RAM) + `MPS` (GPU Memory).
- **Stream**: Hashes two consecutive batches to ensure `Batch A != Batch B` (Generator is not stuck).
- **Baseline Accuracy**: Compares model against:
  - Random Guess (~50%)
  - Majority Class (~50%)

### Gate Diagnosis (`diagnose_gate.py`)
Checks the health of the Recurrent Super-Block.
- **Healthy**: Gate > 0.0 (Learning).
- **Dead**: Gate == 0.0 (Identity).
- **Fix**: Force-open to 0.1 if dead.
