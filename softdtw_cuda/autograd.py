from __future__ import annotations

import torch
from torch.autograd import Function

from .utils.checks import check_D
from .cuda.launcher import softdtw_forward_cuda, softdtw_backward_cuda_log
from .cuda.launcher import softdtw_forward_cpu, softdtw_backward_cpu


class SoftDTWAutograd(Function):
    @staticmethod
    def forward(ctx, D: torch.Tensor, gamma: float, bandwidth: float | None):
        check_D(D)
        gamma_f = float(gamma)
        if gamma_f <= 0:
            raise ValueError(f"gamma must be > 0, got {gamma_f}")

        if bandwidth is None:
            bandwidth_f = -1.0
        else:
            bw = float(bandwidth)
            bandwidth_f = -1.0 if bw <= 0 else bw

        if D.is_cuda:
            out, R = softdtw_forward_cuda(D, gamma_f, bandwidth_f)
        else:
            out, R = softdtw_forward_cpu(D, gamma_f, bandwidth_f)

        ctx.save_for_backward(D, R.detach())
        ctx.gamma = gamma_f
        ctx.bandwidth = bandwidth_f
        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        D, R = ctx.saved_tensors
        gamma_f = ctx.gamma
        bandwidth_f = ctx.bandwidth

        if D.is_cuda:
            E = softdtw_backward_cuda_log(D, R, gamma_f, bandwidth_f)
        else:
            E = softdtw_backward_cpu(D, R, gamma_f, bandwidth_f)

        g = grad_output.reshape(-1).to(dtype=E.dtype).view(-1, 1, 1)
        grad_D = g * E
        return grad_D, None, None
