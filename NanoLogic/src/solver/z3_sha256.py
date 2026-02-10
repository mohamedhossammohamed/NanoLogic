"""
Z3 SHA-256 Constraint Encoder
==============================
Encodes SHA-256 compression function as Z3 bit-vector constraints.
Supports reduced-round solving and neural-guided partial solving.
"""

import z3
import numpy as np
import time

# SHA-256 Round Constants (first 32 bits of the fractional parts of cube roots of first 64 primes)
K = [
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5,
    0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
    0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc,
    0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
    0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
    0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3,
    0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5,
    0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
    0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
]

# SHA-256 Initial Hash Values
H0 = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
]


def _rotr32(x, n):
    """Right rotate a Z3 BitVecRef by n positions."""
    return z3.RotateRight(x, n)


def _shr32(x, n):
    """Logical right shift of a Z3 BitVecRef by n positions."""
    return z3.LShR(x, n)


def _sigma0(x):
    """σ₀(x) for message schedule = ROTR(7) ^ ROTR(18) ^ SHR(3)."""
    return _rotr32(x, 7) ^ _rotr32(x, 18) ^ _shr32(x, 3)


def _sigma1(x):
    """σ₁(x) for message schedule = ROTR(17) ^ ROTR(19) ^ SHR(10)."""
    return _rotr32(x, 17) ^ _rotr32(x, 19) ^ _shr32(x, 10)


def _sum0(x):
    """Σ₀(x) for compression = ROTR(2) ^ ROTR(13) ^ ROTR(22)."""
    return _rotr32(x, 2) ^ _rotr32(x, 13) ^ _rotr32(x, 22)


def _sum1(x):
    """Σ₁(x) for compression = ROTR(6) ^ ROTR(11) ^ ROTR(25)."""
    return _rotr32(x, 6) ^ _rotr32(x, 11) ^ _rotr32(x, 25)


def _ch(x, y, z_val):
    """Ch(x, y, z) = (x & y) ^ (~x & z)."""
    return (x & y) ^ (~x & z_val)


def _maj(x, y, z_val):
    """Maj(x, y, z) = (x & y) ^ (x & z) ^ (y & z)."""
    return (x & y) ^ (x & z_val) ^ (y & z_val)


class SHA256Solver:
    """
    Encodes SHA-256 as Z3 bit-vector constraints for preimage search.
    
    Supports:
        - Full or reduced-round SHA-256
        - Neural-guided solving (fixing high-confidence bits)
        - Timeout-based solving with statistics
    """

    def __init__(self, rounds=64, timeout_ms=60_000):
        """
        Args:
            rounds: Number of SHA-256 compression rounds (16, 32, or 64).
            timeout_ms: Z3 solver timeout in milliseconds.
        """
        self.rounds = min(rounds, 64)
        self.timeout_ms = timeout_ms
        self.stats = {
            'solve_calls': 0,
            'sat_count': 0,
            'unsat_count': 0,
            'timeout_count': 0,
            'total_time_ms': 0,
        }

    def _create_message_words(self):
        """Create 16 symbolic 32-bit message words W[0..15]."""
        return [z3.BitVec(f'W_{i}', 32) for i in range(16)]

    def _expand_message_schedule(self, W_init):
        """
        Expand the 16-word message to full 64-word schedule.
        W[t] = σ₁(W[t-2]) + W[t-7] + σ₀(W[t-15]) + W[t-16]  for t ≥ 16.
        """
        W = list(W_init)
        for t in range(16, self.rounds):
            w_new = _sigma1(W[t - 2]) + W[t - 7] + _sigma0(W[t - 15]) + W[t - 16]
            W.append(w_new)
        return W

    def _compression_loop(self, W):
        """
        Run the SHA-256 compression function for `self.rounds` rounds.
        
        Returns:
            (a, b, c, d, e, f, g, h): Final working variables as Z3 BitVecRef.
        """
        # Initialize working variables with H0
        a = z3.BitVecVal(H0[0], 32)
        b = z3.BitVecVal(H0[1], 32)
        c = z3.BitVecVal(H0[2], 32)
        d = z3.BitVecVal(H0[3], 32)
        e = z3.BitVecVal(H0[4], 32)
        f = z3.BitVecVal(H0[5], 32)
        g = z3.BitVecVal(H0[6], 32)
        h = z3.BitVecVal(H0[7], 32)

        for t in range(self.rounds):
            S1 = _sum1(e)
            ch = _ch(e, f, g)
            temp1 = h + S1 + ch + z3.BitVecVal(K[t], 32) + W[t]
            S0 = _sum0(a)
            maj = _maj(a, b, c)
            temp2 = S0 + maj

            h = g
            g = f
            f = e
            e = d + temp1
            d = c
            c = b
            b = a
            a = temp1 + temp2

        # Add initial hash values (mod 2^32 is implicit with BitVec(32))
        a = a + z3.BitVecVal(H0[0], 32)
        b = b + z3.BitVecVal(H0[1], 32)
        c = c + z3.BitVecVal(H0[2], 32)
        d = d + z3.BitVecVal(H0[3], 32)
        e = e + z3.BitVecVal(H0[4], 32)
        f = f + z3.BitVecVal(H0[5], 32)
        g = g + z3.BitVecVal(H0[6], 32)
        h = h + z3.BitVecVal(H0[7], 32)

        return (a, b, c, d, e, f, g, h)

    def solve_preimage(self, target_hash_hex):
        """
        Attempt to find a 512-bit message that hashes to the given target.
        
        Args:
            target_hash_hex: 64-character hex string (256-bit hash).
            
        Returns:
            dict with keys:
                'status': 'sat', 'unsat', or 'timeout'
                'message_words': list of 16 uint32 values (if sat)
                'message_bytes': bytes of the preimage (if sat)
                'time_ms': solve time
        """
        self.stats['solve_calls'] += 1
        start = time.time()

        # Parse target hash into 8 x 32-bit words
        assert len(target_hash_hex) == 64, f"Hash must be 64 hex chars, got {len(target_hash_hex)}"
        target_words = []
        for i in range(8):
            word_hex = target_hash_hex[i * 8:(i + 1) * 8]
            target_words.append(int(word_hex, 16))

        # Create solver
        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)

        # Create symbolic message
        W_init = self._create_message_words()

        # Expand message schedule
        W = self._expand_message_schedule(W_init)

        # Run compression
        out = self._compression_loop(W)

        # Constrain output to match target hash
        for i, target_val in enumerate(target_words):
            solver.add(out[i] == z3.BitVecVal(target_val, 32))

        # Solve
        result = solver.check()
        elapsed_ms = (time.time() - start) * 1000
        self.stats['total_time_ms'] += elapsed_ms

        if result == z3.sat:
            self.stats['sat_count'] += 1
            model = solver.model()
            msg_words = []
            msg_bytes = b''
            for w in W_init:
                val = model[w].as_long() if model[w] is not None else 0
                msg_words.append(val)
                msg_bytes += val.to_bytes(4, 'big')
            return {
                'status': 'sat',
                'message_words': msg_words,
                'message_bytes': msg_bytes,
                'time_ms': elapsed_ms,
            }
        elif result == z3.unsat:
            self.stats['unsat_count'] += 1
            return {'status': 'unsat', 'time_ms': elapsed_ms}
        else:
            self.stats['timeout_count'] += 1
            return {'status': 'timeout', 'time_ms': elapsed_ms}

    def solve_partial(self, target_hash_hex, neural_hints, confidence_threshold=0.85):
        """
        Neural-guided preimage search.
        
        Fixes high-confidence bits from neural predictions before solving,
        dramatically reducing the search space.
        
        Args:
            target_hash_hex: 64-character hex string.
            neural_hints: numpy array of shape [256] with values in [0, 1].
                         Each value represents P(bit_i = 1).
            confidence_threshold: Bits with confidence above this are fixed.
            
        Returns:
            Same dict as solve_preimage, plus:
                'bits_fixed': number of bits fixed by neural hints
                'search_space_reduction': percentage of search space eliminated
        """
        self.stats['solve_calls'] += 1
        start = time.time()

        # Parse target 
        assert len(target_hash_hex) == 64
        target_words = []
        for i in range(8):
            word_hex = target_hash_hex[i * 8:(i + 1) * 8]
            target_words.append(int(word_hex, 16))

        solver = z3.Solver()
        solver.set("timeout", self.timeout_ms)

        W_init = self._create_message_words()
        W = self._expand_message_schedule(W_init)
        out = self._compression_loop(W)

        # Target hash constraints
        for i, target_val in enumerate(target_words):
            solver.add(out[i] == z3.BitVecVal(target_val, 32))

        # Neural hints → fix high-confidence bits in the message
        bits_fixed = 0
        for bit_idx in range(min(len(neural_hints), 256)):
            conf = neural_hints[bit_idx]
            # Map bit_idx to (word_index, bit_position)
            word_idx = bit_idx // 32
            bit_pos = 31 - (bit_idx % 32)  # MSB-first

            if word_idx >= 16:
                continue  # Only fix message bits (W[0..15])

            if conf > confidence_threshold:
                # Fix this bit to 1
                solver.add(z3.Extract(bit_pos, bit_pos, W_init[word_idx]) == z3.BitVecVal(1, 1))
                bits_fixed += 1
            elif conf < (1.0 - confidence_threshold):
                # Fix this bit to 0
                solver.add(z3.Extract(bit_pos, bit_pos, W_init[word_idx]) == z3.BitVecVal(0, 1))
                bits_fixed += 1

        search_space_reduction = (bits_fixed / 512) * 100 if bits_fixed > 0 else 0

        result = solver.check()
        elapsed_ms = (time.time() - start) * 1000
        self.stats['total_time_ms'] += elapsed_ms

        if result == z3.sat:
            self.stats['sat_count'] += 1
            model = solver.model()
            msg_words = []
            msg_bytes = b''
            for w in W_init:
                val = model[w].as_long() if model[w] is not None else 0
                msg_words.append(val)
                msg_bytes += val.to_bytes(4, 'big')
            return {
                'status': 'sat',
                'message_words': msg_words,
                'message_bytes': msg_bytes,
                'time_ms': elapsed_ms,
                'bits_fixed': bits_fixed,
                'search_space_reduction': search_space_reduction,
            }
        elif result == z3.unsat:
            self.stats['unsat_count'] += 1
            return {
                'status': 'unsat',
                'time_ms': elapsed_ms,
                'bits_fixed': bits_fixed,
                'search_space_reduction': search_space_reduction,
            }
        else:
            self.stats['timeout_count'] += 1
            return {
                'status': 'timeout',
                'time_ms': elapsed_ms,
                'bits_fixed': bits_fixed,
                'search_space_reduction': search_space_reduction,
            }

    def get_stats(self):
        """Return solver statistics."""
        return dict(self.stats)
