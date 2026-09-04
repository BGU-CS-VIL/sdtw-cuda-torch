"""Forward + backward agreement vs. INDEPENDENT references.

Two independent oracles, neither sharing code with the library's Numba kernels:

  1. ``naive_softdtw`` - a ~20-line pure-PyTorch dynamic program of the Cuturi &
     Blondel (2017) soft-DTW recursion with a squared-Euclidean local cost. Being
     differentiable, it yields both the value and (via autograd) the gradient, so
     it validates the forward *and* backward pass.
  2. ``tslearn.metrics.soft_dtw`` - a third-party implementation (skipped if not
     installed). Forward value only.

All comparisons use float64 on CPU.
"""
from __future__ import annotations

import pytest
import torch

from softdtw_cuda import softdtw

DTYPE = torch.float64

# (N, M, D, gamma)
CASES = [
    (5, 7, 3, 1.0),
    (8, 4, 1, 1.0),
    (6, 6, 2, 0.5),
    (7, 9, 2, 0.05),   # small gamma
    (30, 25, 4, 1.0),  # longer sequence
]


def naive_softdtw(x: torch.Tensor, y: torch.Tensor, gamma: float) -> torch.Tensor:
    """Independent pure-PyTorch soft-DTW (squared-euclidean). x:(N,D), y:(M,D) -> scalar."""
    N, M = x.shape[0], y.shape[0]
    dtype, device = x.dtype, x.device
    Dm = ((x[:, None, :] - y[None, :, :]) ** 2).sum(-1)  # (N, M)
    inf = torch.tensor(float("inf"), dtype=dtype, device=device)
    R = [[None] * (M + 1) for _ in range(N + 1)]
    R[0][0] = torch.zeros((), dtype=dtype, device=device)
    for i in range(1, N + 1):
        R[i][0] = inf
    for j in range(1, M + 1):
        R[0][j] = inf
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            r = torch.stack([R[i - 1][j - 1], R[i - 1][j], R[i][j - 1]])
            softmin = -gamma * torch.logsumexp(-r / gamma, dim=0)
            R[i][j] = Dm[i - 1, j - 1] + softmin
    return R[N][M]


def _randn(B, N, D, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(B, N, D, generator=g, dtype=DTYPE)


@pytest.mark.parametrize("N,M,D,gamma", CASES)
def test_forward_matches_naive(N, M, D, gamma):
    B = 3
    x = _randn(B, N, D, 1)
    y = _randn(B, M, D, 2)
    got = softdtw(x, y, gamma=gamma, fused=False)
    exp = torch.stack([naive_softdtw(x[b], y[b], gamma) for b in range(B)])
    assert torch.allclose(got, exp, atol=1e-8, rtol=1e-6), (got - exp).abs().max()


@pytest.mark.parametrize("N,M,D,gamma", CASES)
def test_backward_matches_naive(N, M, D, gamma):
    B = 3
    x = _randn(B, N, D, 1)
    y = _randn(B, M, D, 2)

    xa = x.clone().requires_grad_(True)
    ya = y.clone().requires_grad_(True)
    softdtw(xa, ya, gamma=gamma, fused=False).sum().backward()

    xb = x.clone().requires_grad_(True)
    yb = y.clone().requires_grad_(True)
    torch.stack([naive_softdtw(xb[b], yb[b], gamma) for b in range(B)]).sum().backward()

    assert torch.allclose(xa.grad, xb.grad, atol=1e-7, rtol=1e-5), (xa.grad - xb.grad).abs().max()
    assert torch.allclose(ya.grad, yb.grad, atol=1e-7, rtol=1e-5), (ya.grad - yb.grad).abs().max()


@pytest.mark.parametrize("N,M,D,gamma", CASES)
def test_forward_matches_tslearn(N, M, D, gamma):
    ts = pytest.importorskip("tslearn.metrics")
    B = 3
    x = _randn(B, N, D, 1)
    y = _randn(B, M, D, 2)
    got = softdtw(x, y, gamma=gamma, fused=False)
    exp = torch.tensor(
        [ts.soft_dtw(x[b].numpy(), y[b].numpy(), gamma=gamma) for b in range(B)],
        dtype=DTYPE,
    )
    assert torch.allclose(got, exp, atol=1e-8, rtol=1e-6), (got - exp).abs().max()
