import sys
import time

# Append path
sys.path.append(".")

from src.solver.z3_sha256 import SHA256Solver

def test_z3():
    print("Testing Z3 Solver Integration...")
    rounds = 8
    target_hex = "0000000000000000000000000000000000000000000000000000000000000000"
    
    print(f"Initializing solver for {rounds} rounds...")
    solver = SHA256Solver(rounds=rounds, timeout_ms=5000)
    
    print("Starting solve (preimage of all zeros)...")
    start = time.time()
    res = solver.solve_preimage(target_hex)
    end = time.time()
    
    print(f"Solve completed in {end - start:.4f}s")
    print(f"Status: {res.get('status')}")

if __name__ == "__main__":
    try:
        test_z3()
        print("✅ Z3 test passed!")
    except Exception as e:
        print(f"❌ Z3 test failed: {e}")
