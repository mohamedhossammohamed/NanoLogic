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
    def get_op_indices(device='cpu', carry_depth=4):
        """
        Returns GLOBAL indices [256, k] and masks [256, k] for specific SHA-256 operations.
        Used by SparseLogicLayer for direct bit gathering.
        
        New Logic:
        - Returns (indices, mask) tuple for each operation type.
        - indices: Global index of source bit. If masked, value doesn't matter (we use 0 or clamp).
        - mask: 1.0 if valid, 0.0 if masked (e.g. SHR shifted in zeros).
        """
        # Create global indices 0..255
        global_idx = torch.arange(256, device=device)
        word_idx = global_idx // 32
        bit_idx = global_idx % 32
        
        def get_rotated_indices(shifts):
            # shifts: list of ints e.g. [2, 13, 22]
            # Returns indices [256, k], mask [256, k] (all 1s)
            indices = []
            for s in shifts:
                # Rotate right by s: (b - s) % 32
                b_rot = (bit_idx - s) % 32
                global_rot = word_idx * 32 + b_rot
                indices.append(global_rot)
            
            idx_tensor = torch.stack(indices, dim=1)
            mask_tensor = torch.ones_like(idx_tensor, dtype=torch.float32)
            return idx_tensor, mask_tensor

        def get_shifted_indices(shifts, is_generated_shift=False):
            # shifts must correspond to operations. 
            # If is_generated_shift is True (for sigma_w), some ops are ROTR, some SHR.
            # We handle mixed operations by passing a list of (shift_amount, type='ROTR'|'SHR') tuples?
            # Or just infer based on standard SHA-256 ops?
            # Simpler: We know exactly what ops are required.
            pass

        # Helper for Sigma0/1 (Pure ROTR)
        # Sigma0: ROTR 2, 13, 22
        sigma0_idx, sigma0_mask = get_rotated_indices([2, 13, 22])
        
        # Sigma1: ROTR 6, 11, 25
        sigma1_idx, sigma1_mask = get_rotated_indices([6, 11, 25])
        
        # Use a more flexible helper for mixed operations (ROTR + SHR)
        def get_mixed_indices(ops):
            # ops: list of (shift, type) e.g. [(7, 'ROTR'), (18, 'ROTR'), (3, 'SHR')]
            indices = []
            masks = []
            
            for s, op_type in ops:
                if op_type == 'ROTR':
                    b_target = (bit_idx - s) % 32
                    mask = torch.ones(256, device=device)
                elif op_type == 'SHR':
                    # SHR n: Target bit b comes from b - n.
                    # If b - n < 0, it's a zero (masked).
                    # NOTE: bit_idx 0 is MSB in our convention?
                    # Wait, let's check standard. 
                    # If 0 is MSB: 1000.. >> 1 -> 0100..
                    # Bit 0 becomes 0. Bit 1 has value of old Bit 0.
                    # So Target b comes from Source b - s (if we index 0..31)
                    # Example: Target 1 comes from Source 0.
                    # If b - s < 0, valid bit.
                    # Wait. 
                    # MSB=0. LSB=31.
                    # x >> 3.
                    # Bit 0 (MSB) gets 0.
                    # Bit 1 gets 0.
                    # Bit 2 gets 0.
                    # Bit 3 gets Bit 0.
                    # So Target b comes from Source b - s.
                    # If b - s < 0, it wraps? No, it's 0.
                    b_target = bit_idx - s
                    # Mask is 1 if b_target >= 0, else 0
                    mask = (b_target >= 0).float()
                    # Clamp index to 0 to avoid -1 indexing (mask will kill it anyway)
                    b_target = torch.clamp(b_target, min=0)
                
                global_target = word_idx * 32 + b_target
                indices.append(global_target)
                masks.append(mask)
            
            return torch.stack(indices, dim=1), torch.stack(masks, dim=1)

        # Sigma0_w (Message Schedule): ROTR 7, 18, SHR 3
        sigma0_w_idx, sigma0_w_mask = get_mixed_indices([(7, 'ROTR'), (18, 'ROTR'), (3, 'SHR')])
        
        # Sigma1_w: ROTR 17, 19, SHR 10
        sigma1_w_idx, sigma1_w_mask = get_mixed_indices([(17, 'ROTR'), (19, 'ROTR'), (10, 'SHR')])
        
        # 2. Vertical Indices (Inter-Word) - All 1s mask
        vertical_indices = []
        for w in range(8):
            v_idx = w * 32 + bit_idx
            vertical_indices.append(v_idx)
        vertical_idx = torch.stack(vertical_indices, dim=1)
        vertical_mask = torch.ones_like(vertical_idx, dtype=torch.float32)


        # 3. Carry Propagation Indices (Ripple Carry)
        # Models addition dependency: i depends on i+1 (LSB side)
        # Default depth: 4 bits lookahead
        carry_indices = []
        carry_masks = []
        
        for k in range(1, carry_depth + 1):
            # Target i depends on Source i + k (towards LSB)
            # If i + k > 31, no incoming carry from there (it's outside word)
            b_src = bit_idx + k
            mask = (b_src < 32).float()
            
            # Clamp to 31 for safety
            b_src = torch.clamp(b_src, max=31)
            
            global_src = word_idx * 32 + b_src
            carry_indices.append(global_src)
            carry_masks.append(mask)
            
        carry_idx = torch.stack(carry_indices, dim=1)
        carry_mask = torch.stack(carry_masks, dim=1)

        return {
            'sigma0': (sigma0_idx, sigma0_mask),
            'sigma1': (sigma1_idx, sigma1_mask),
            'sigma0_w': (sigma0_w_idx, sigma0_w_mask), # Explicitly returning w indices
            'sigma1_w': (sigma1_w_idx, sigma1_w_mask),
            'vertical': (vertical_idx, vertical_mask),
            'carry': (carry_idx, carry_mask)
        }
