from __future__ import annotations

import torch
from torch.autograd import Function

from .cuda.launcher import (
    softdtw_forward_cuda_fused_sqeuclid,
    softdtw_backward_cuda_fused_sqeuclid,   # returns E (exp(logE))
)


class SoftDTWXYAutograd(Function):
    @staticmethod
    def forward(ctx, X: torch.Tensor, Y: torch.Tensor, gamma: float, bandwidth: float | None):
        # Forward CUDA fused: returns out (B,) and R (B,N+2,M+2)
        out, R = softdtw_forward_cuda_fused_sqeuclid(X, Y, float(gamma), -1.0 if bandwidth is None else float(bandwidth))

        # Save X,Y for gradient math; save detached R (no graph needed)
        ctx.save_for_backward(X, Y, R.detach())
        ctx.gamma = float(gamma)

        # Normalize bandwidth semantics: <=0 means disabled
        if bandwidth is None:
            ctx.bandwidth = -1.0
        else:
            bw = float(bandwidth)
            ctx.bandwidth = -1.0 if bw <= 0 else bw

        return out

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        X, Y, R = ctx.saved_tensors
        gamma = ctx.gamma
        bw = ctx.bandwidth

        # Compute E via fused log-space backward (Numba). Pass detached X/Y to be safe.
        E = softdtw_backward_cuda_fused_sqeuclid(X.detach(), Y.detach(), R, gamma, bw)  # (B,N,M)

        # Scale by upstream grad (B,) -> (B,1,1)
        g = grad_output.reshape(-1).to(device=X.device, dtype=X.dtype).view(-1, 1, 1)
        E = E * g

        # Reductions for sqeuclidean chain rule
        EX = E.sum(dim=2)  # (B,N)
        EY = E.sum(dim=1)  # (B,M)

        grad_X = 2.0 * (X * EX.unsqueeze(2) - torch.bmm(E, Y))                 # (B,N,D)
        grad_Y = 2.0 * (Y * EY.unsqueeze(2) - torch.bmm(E.transpose(1, 2), X)) # (B,M,D)

        return grad_X, grad_Y, None, None
