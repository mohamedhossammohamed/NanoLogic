import hashlib
import json
import os
from typing import Dict, Any

class ModelIdentity:
    """
    Manages Unique Model Identities.
    
    Generates a deterministic ID based on the model's structural configuration.
    This allows:
    1. Resuming the *correct* model even if multiple exist.
    2. Keeping logs separate for different experiments.
    3. Generating Model Cards automatically.
    """
    
    def __init__(self, config):
        self.config = config
        self.model_id = self._compute_id()
        self.specs = self._generate_specs()
        
    def _compute_id(self) -> str:
        """
        Computes a SHA256 hash of the critical structural parameters.
        Changing any of these means it's a effectively a new model.
        """
        # We start with a version prefix 'v1' to handle future hashing changes
        structure_str = f"v1-{self.config.dim}-{self.config.n_layers}-{self.config.n_heads}-{self.config.vocab_size}-{self.config.wiring_mode}-{self.config.rounds}-{self.config.recurrent_loops}"
        
        # 8-char hex digest is enough collisions resistance for local experiments
        return hashlib.sha256(structure_str.encode()).hexdigest()[:8]

    def _generate_specs(self) -> Dict[str, Any]:
        """
        Generates the Model Card metadata.
        """
        param_estimate = (
            # Embeddings
            (256 * self.config.dim) + 
            # Layers (BitLinear weights are technically ternary but stored as float/int8 in memory models)
            (self.config.n_layers * 3 * self.config.dim * self.config.dim) +
            # Heads (Logic wiring costs are mostly static but we count parameter storage)
            (self.config.n_layers * self.config.n_heads * self.config.dim) 
        ) / 1e6

        return {
            "id": self.model_id,
            "architecture": "Neuro-SHA-M4",
            "dim": self.config.dim,
            "layers": self.config.n_layers,
            "heads": self.config.n_heads,
            "wiring": self.config.wiring_mode,
            "rounds": self.config.rounds,
            "loops": self.config.recurrent_loops,
            "params_m": f"{param_estimate:.2f}M (Est)",
            "description": "Sparse Logic Transformer for SHA-256 Cryptanalysis"
        }

    def setup_workspace(self, base_log_dir="logs", base_ckpt_dir="checkpoints"):
        """
        Creates the isolated workspace for this model identity.
        Returns (log_dir, ckpt_dir)
        """
        log_dir = os.path.join(base_log_dir, self.model_id)
        ckpt_dir = os.path.join(base_ckpt_dir, self.model_id)
        
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # Save Model Card
        card_path = os.path.join(log_dir, "model_card.json")
        with open(card_path, "w") as f:
            json.dump(self.specs, f, indent=4)
            
        return log_dir, ckpt_dir
