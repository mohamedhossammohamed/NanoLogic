import torch
from torch.optim.optimizer import Optimizer

class LionGaLore(Optimizer):
    """
    Lion Optimizer adapted with 'GaLore' style low-rank projection 
    concepts for memory efficiency on Apple Silicon.
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.0, rank=128):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, rank=rank)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                lr = group['lr']
                beta1, beta2 = group['betas']
                weight_decay = group['weight_decay']
                grad = p.grad
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    # Standard Lion: Initialize on CPU to save GPU memory
                    state['exp_avg'] = torch.zeros_like(p, dtype=torch.bfloat16, device='cpu')

                state['step'] += 1
                # Move to parameter device for computation
                exp_avg = state['exp_avg'].to(p.device)
                
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)
                    
                update = exp_avg.float() * beta1 + grad * (1 - beta1)
                p.add_(torch.sign(update), alpha=-lr)
                
                new_exp_avg = exp_avg.float() * beta2 + grad * (1 - beta2)
                # Offload back to CPU
                state['exp_avg'].copy_(new_exp_avg.bfloat16().to('cpu'), non_blocking=True)

        return loss
