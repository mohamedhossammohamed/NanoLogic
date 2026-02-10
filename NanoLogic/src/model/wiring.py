import torch

class SHA256Wiring:
    """
    Defines the static adjacency matrix (wiring diagram) of SHA-256.
    Includes a Differentiable/Torch-based Simulator for data generation.
    """
    
    K = torch.tensor([
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
    ], dtype=torch.int64)

    @staticmethod
    def rotr(x, n):
        return (x >> n) | (x << (32 - n)) & 0xFFFFFFFF
        
    @staticmethod
    def shr(x, n):
        return (x >> n)

    @staticmethod
    def ch(x, y, z):
        return (x & y) ^ (~x & z)

    @staticmethod
    def maj(x, y, z):
        return (x & y) ^ (x & z) ^ (y & z)

    @staticmethod
    def sum0(x):
        return SHA256Wiring.rotr(x, 2) ^ SHA256Wiring.rotr(x, 13) ^ SHA256Wiring.rotr(x, 22)

    @staticmethod
    def sum1(x):
        return SHA256Wiring.rotr(x, 6) ^ SHA256Wiring.rotr(x, 11) ^ SHA256Wiring.rotr(x, 25)

    @staticmethod
    def sigma0(x):
        return SHA256Wiring.rotr(x, 7) ^ SHA256Wiring.rotr(x, 18) ^ SHA256Wiring.shr(x, 3)

    @staticmethod
    def sigma1(x):
        return SHA256Wiring.rotr(x, 17) ^ SHA256Wiring.rotr(x, 19) ^ SHA256Wiring.shr(x, 10)

    @staticmethod
    def generate_trace(batch_size=1, rounds=16, device='cpu'):
        """
        Generates a synthetic trace of internal states.
        Returns: 
            states: [B, rounds, 8, 32] (Bits)
            messages: [B, 16] (Words)
        """
        # Random message [B, 16] (32-bit words)
        # Use int64 to avoid overflow issues during addition before masking
        M = torch.randint(0, 2**32, (batch_size, 16), dtype=torch.int64, device=device)
        
        # Initial Hash (Standard H0)
        H = torch.tensor([
            0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 
            0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
        ], dtype=torch.int64, device=device).repeat(batch_size, 1) # [B, 8]
        
        # Message Schedule W
        W = torch.zeros(batch_size, 64, dtype=torch.int64, device=device)
        W[:, :16] = M
        
        for t in range(16, 64):
            wt2 = W[:, t-2]
            wt7 = W[:, t-7]
            wt15 = W[:, t-15]
            wt16 = W[:, t-16]
            
            s1 = SHA256Wiring.sigma1(wt2)
            s0 = SHA256Wiring.sigma0(wt15)
            
            W[:, t] = (s1 + wt7 + s0 + wt16) & 0xFFFFFFFF
            
        # Compression Loop
        # We store state at each step: [B, Rounds, 8]
        states_words = torch.zeros(batch_size, rounds, 8, dtype=torch.int64, device=device)
        
        a, b, c, d, e, f, g, h = [H[:, i] for i in range(8)]
        
        K_dev = SHA256Wiring.K.to(device)
        
        for t in range(rounds):
            # Save current state
            states_words[:, t, 0] = a
            states_words[:, t, 1] = b
            states_words[:, t, 2] = c
            states_words[:, t, 3] = d
            states_words[:, t, 4] = e
            states_words[:, t, 5] = f
            states_words[:, t, 6] = g
            states_words[:, t, 7] = h
            
            # Update
            S1 = SHA256Wiring.sum1(e)
            Ch = SHA256Wiring.ch(e, f, g)
            temp1 = (h + S1 + Ch + K_dev[t] + W[:, t]) & 0xFFFFFFFF
            S0 = SHA256Wiring.sum0(a)
            Maj = SHA256Wiring.maj(a, b, c)
            temp2 = (S0 + Maj) & 0xFFFFFFFF
            
            h = g
            g = f
            f = e
            e = (d + temp1) & 0xFFFFFFFF
            d = c
            c = b
            b = a
            a = (temp1 + temp2) & 0xFFFFFFFF
            
        # Convert words to bits for training [B, Rounds, 8, 32]
        # Helper to unpack bits
        # [B, R, 8] -> [element]
        
        # Vectorized bit unpacking
        # Expand last dim
        bits = torch.zeros(batch_size, rounds, 256, dtype=torch.int64, device=device)
        
        for i in range(8):
            word = states_words[:, :, i] # [B, R]
            for bit in range(32):
                # Extract bit `bit` from word
                # (word >> bit) & 1. Note: Bit 0 is LSB usually, but SHA-256 is Big Endian? 
                # Standards usually print MSB first. 
                # Let's use (word >> (31-bit)) & 1 to match "Index 0 is MSB".
                extracted = (word >> (31 - bit)) & 1
                bits[:, :, i*32 + bit] = extracted
                
        return bits, W # Return bits and Schedule

    @staticmethod
    def get_op_indices(device='cpu'):
        """
        Returns GLOBAL indices [256, k] for specific SHA-256 operations.
        Used by SparseLogicLayer for direct bit gathering.
        """
        # Create global indices 0..255
        # Bit i is at index i.
        # Structure: 8 words of 32 bits.
        # Word w, Bit b -> Index = w*32 + b
        
        # 1. Sigma0/1 Rotational Indices (Intra-Word)
        # For bit `idx` (global), we find its word `w` and local bit `b`.
        # Then we apply rotation to `b` modulo 32.
        # Then convert back to global index: w*32 + b_rot
        
        global_idx = torch.arange(256, device=device)
        word_idx = global_idx // 32
        bit_idx = global_idx % 32
        
        def get_rotated_indices(shifts):
            # shifts: list of ints e.g. [2, 13, 22]
            indices = []
            for s in shifts:
                # Rotate right by s
                # (b - s) % 32
                b_rot = (bit_idx - s) % 32
                global_rot = word_idx * 32 + b_rot
                indices.append(global_rot)
            return torch.stack(indices, dim=1) # [256, len(shifts)]

        # Sigma0: ROTR 2, 13, 22
        sigma0_indices = get_rotated_indices([2, 13, 22])
        
        # Sigma1: ROTR 6, 11, 25
        sigma1_indices = get_rotated_indices([6, 11, 25])
        
        # Sigma0_w (Message Schedule): ROTR 7, 18, SHR 3
        # Use circular for SHR to keep dimensionality, model can learn to ignore wrapped bit if needed
        # or we implement masking. For now, just rotation.
        sigma0_w_indices = get_rotated_indices([7, 18, 3])
        
        # Sigma1_w: ROTR 17, 19, SHR 10
        sigma1_w_indices = get_rotated_indices([17, 19, 10])
        
        # 2. Vertical Indices (Inter-Word)
        # For a bit `b` in word `w`, we want to see bit `b` in all other words.
        # Total 8 words. We want 8 neighbors (including self, or excluding self).
        # Let's return all 8 versions of this bit position.
        # shape [256, 8]
        vertical_indices = []
        for w in range(8):
            # The index of bit `bit_idx` in word `w`
            v_idx = w * 32 + bit_idx
            vertical_indices.append(v_idx)
            
        vertical_indices = torch.stack(vertical_indices, dim=1) # [256, 8]

        return {
            'sigma0': sigma0_indices,
            'sigma1': sigma1_indices,
            'vertical': vertical_indices
        }
