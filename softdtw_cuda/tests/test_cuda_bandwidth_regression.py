"""Regression tests for the Sakoe-Chiba bandwidth NaN bug in the CUDA backward.

Before the fix, *all* CUDA backward kernels (fast, tiled, and fused) produced NaN
gradients whenever a bandwidth was active: out-of-band cells were not demoted to
-inf before in-band neighbours read them, yielding (-inf)+(+inf)=NaN in the
log-space accumulation. The CPU reference was correct. CI is CPU-only, so this
never surfaced. These tests assert the CUDA gradients are finite and match the
CPU reference under a bandwidth, for both the fused and unfused paths.
"""
from __future__ import annotations

import pytest
import torch

from softdtw_cuda import SoftDTW

# (N, M, D, bandwidth): mix of N==M and N!=M, D==1 and D>1
CASES = [(6, 6, 3, 2.0), (8, 5, 1, 3.0), (10, 10, 2, 4.0), (7, 9, 2, 3.0)]


def _grad(x, y, *, device, fused, bw):
    xa = x.to(device).clone().requires_grad_(True)
    ya = y.to(device).clone().requires_grad_(True)
    SoftDTW(gamma=1.0, bandwidth=bw, dist="sqeuclidean", fused=fused)(xa, ya).sum().backward()
    return xa.grad.cpu(), ya.grad.cpu()


@pytest.mark.cuda
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("N,M,D,bw", CASES)
def test_cuda_bandwidth_grad_matches_cpu(N, M, D, bw, fused):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    torch.manual_seed(0)
    x = torch.randn(2, N, D, dtype=torch.float64)
    y = torch.randn(2, M, D, dtype=torch.float64)

    gxc, gyc = _grad(x, y, device="cpu", fused=False, bw=bw)  # CPU ground truth
    gx, gy = _grad(x, y, device="cuda", fused=fused, bw=bw)

    assert torch.isfinite(gx).all() and torch.isfinite(gy).all(), "NaN/Inf gradient under bandwidth"
    assert torch.allclose(gx, gxc, atol=1e-10, rtol=1e-7), (gx - gxc).abs().max()
    assert torch.allclose(gy, gyc, atol=1e-10, rtol=1e-7), (gy - gyc).abs().max()
