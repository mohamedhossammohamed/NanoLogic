import torch
import torch.nn as nn
import torch.nn.functional as F

class HammingDistanceLoss(nn.Module):
    """
    Computes the Hamming Distance between predicted states and target states.
    For differentiable training, we use Soft Hamming Distance (L1 Loss) or BCE.
    """
    def __init__(self, mode='bce', reduction='mean'):
        super().__init__()
        self.mode = mode
        self.reduction = reduction
        
        if mode == 'bce':
            self.criterion = nn.BCEWithLogitsLoss(reduction=reduction)
        elif mode == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif mode == 'mse':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss mode: {mode}")

    def forward(self, pred, target):
        """
        pred: [B, N] (Logits or Probabilities depending on mode)
        target: [B, N] (0 or 1)
        """
        if self.mode == 'l1' or self.mode == 'mse':
            # expects probabilities
            pred = torch.sigmoid(pred)
            return self.criterion(pred, target.float())
        else:
            # BCE expects logits
            return self.criterion(pred, target.float())

class StateMatchingLoss(nn.Module):
    """
    Composite loss for Neuro-SHA-M4.
    Combines Hamming Distance on State with auxiliary losses if needed to guide the logic.
    """
    def __init__(self):
        super().__init__()
        self.hamming = HammingDistanceLoss(mode='bce')
        
    def forward(self, pred_state, target_state):
        # pred_state: [B, 256] logits
        # target_state: [B, 256] 0/1
        
        return self.hamming(pred_state, target_state)
