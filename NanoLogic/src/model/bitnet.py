import torch
import torch.nn as nn
import torch.nn.functional as F

class BitLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        # 1. Weight Quantization: {-1, 0, 1}
        w = weight - weight.mean()
        gamma = w.abs().mean()
        w_scaled = w / (gamma + 1e-5)
        w_quant = w_scaled.round().clamp(-1, 1)
        
        # 2. Input Quantization: 8-bit
        input_scale = 127.0 / (input.abs().max(dim=-1, keepdim=True).values + 1e-5)
        input_quant = (input * input_scale).round().clamp(-128, 127) / input_scale
        
        ctx.save_for_backward(input, w_quant, gamma)
        
        output = F.linear(input_quant, w_quant * gamma, bias)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, w_quant, gamma = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            # Scale grad_input by gamma to match forward pass scale
            grad_input = grad_output.matmul(w_quant) * gamma
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.transpose(-2, -1).matmul(input) / (gamma + 1e-5)
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias

class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)

    def forward(self, input):
        return BitLinearFunction.apply(input, self.weight, self.bias)
