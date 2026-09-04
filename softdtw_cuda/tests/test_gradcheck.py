"""Real ``torch.autograd.gradcheck`` tests (float64, finite differences).

These replace the previously misnamed ``test_gradcheck_tiny`` (which only asserted
``isfinite`` and never called ``gradcheck``). We check the custom backward passes
against finite differences over a spread of edge shapes:

  * N != M (both N<M and N>M)
  * D == 1 and D > 1
  * Sakoe-Chiba bandwidth on / off
  * small gamma
  * a longer sequence

The **unfused** path runs on CPU (no GPU needed). The **fused** path is CUDA-only,
so those tests skip when no GPU is present but are exercised by GPU CI.
"""
from __future__ import annotations

import pytest
import torch
from torch.autograd import gradcheck

from softdtw_cuda import SoftDTW
from softdtw_cuda.autograd import SoftDTWAutograd
from softdtw_cuda.autograd_xy import SoftDTWXYAutograd
from softdtw_cuda.distances import pairwise_distance

# gradcheck settings for float64
GC = dict(eps=1e-6, atol=1e-5, rtol=1e-3, raise_exception=True)

# (N, M, D, gamma, bandwidth) covering the required edge axes
XY_CASES = [
    (5, 7, 3, 1.0, None),   # N < M, D > 1
    (7, 4, 2, 1.0, None),   # N > M
    (6, 6, 1, 1.0, None),   # N == M, D == 1
    (5, 5, 2, 0.1, None),   # small gamma
    (6, 6, 3, 1.0, 2.0),    # bandwidth on, N == M
    (8, 5, 1, 1.0, 3.0),    # bandwidth on, N != M, D == 1
    (12, 10, 2, 0.5, None),  # longer sequence
]

# (B, N, M, gamma, bandwidth) for the D-matrix path (isolates the DP backward)
D_CASES = [
    (1, 5, 7, 1.0, None),
    (2, 4, 4, 1.0, None),   # batched
    (1, 6, 6, 0.1, None),   # small gamma
    (1, 6, 6, 1.0, 2.0),    # bandwidth
    (1, 8, 5, 0.5, 3.0),    # bandwidth, N != M
]


def _xy(N, M, D, *, device="cpu", seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(1, N, D, generator=g, dtype=torch.float64)
    y = torch.randn(1, M, D, generator=g, dtype=torch.float64)
    return x.to(device), y.to(device)


# ---------------------------------------------------------------------------
# Unfused: gradcheck the DP backward directly, on the distance matrix D.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("B,N,M,gamma,bandwidth", D_CASES)
def test_gradcheck_D_unfused_cpu(B, N, M, gamma, bandwidth):
    torch.manual_seed(0)
    x = torch.randn(B, N, 3, dtype=torch.float64)
    y = torch.randn(B, M, 3, dtype=torch.float64)
    D = pairwise_distance(x, y, dist="sqeuclidean").detach().requires_grad_(True)
    assert gradcheck(lambda d: SoftDTWAutograd.apply(d, gamma, bandwidth), (D,), **GC)


# ---------------------------------------------------------------------------
# Unfused: gradcheck end-to-end SoftDTW()(x, y) w.r.t. x and y (through distances).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("N,M,D,gamma,bandwidth", XY_CASES)
def test_gradcheck_xy_unfused_cpu(N, M, D, gamma, bandwidth):
    x, y = _xy(N, M, D)
    x.requires_grad_(True)
    y.requires_grad_(True)
    mod = SoftDTW(gamma=gamma, bandwidth=bandwidth, dist="sqeuclidean", fused=False)
    assert gradcheck(lambda a, b: mod(a, b), (x, y), **GC)


@pytest.mark.parametrize("N,gamma,bandwidth", [(6, 1.0, None), (5, 1.0, 2.0), (6, 0.1, None)])
def test_gradcheck_normalize_unfused_cpu(N, gamma, bandwidth):
    # normalized variant requires N == M (concatenation trick)
    x, y = _xy(N, N, 2)
    x.requires_grad_(True)
    y.requires_grad_(True)
    mod = SoftDTW(gamma=gamma, bandwidth=bandwidth, normalize=True, dist="sqeuclidean", fused=False)
    assert gradcheck(lambda a, b: mod(a, b), (x, y), **GC)


# ---------------------------------------------------------------------------
# Fused (CUDA-only): gradcheck the fused autograd and the end-to-end module.
# Skipped without a GPU; exercised by GPU CI.
# ---------------------------------------------------------------------------
@pytest.mark.cuda
@pytest.mark.parametrize("N,M,D,gamma,bandwidth", XY_CASES)
def test_gradcheck_xy_fused_cuda(N, M, D, gamma, bandwidth):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x, y = _xy(N, M, D, device="cuda")
    x.requires_grad_(True)
    y.requires_grad_(True)
    bw = -1.0 if bandwidth is None else float(bandwidth)
    assert gradcheck(lambda a, b: SoftDTWXYAutograd.apply(a, b, gamma, bw), (x, y), **GC)


@pytest.mark.cuda
@pytest.mark.parametrize("N,M,D,gamma,bandwidth", XY_CASES)
def test_gradcheck_module_fused_cuda(N, M, D, gamma, bandwidth):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    x, y = _xy(N, M, D, device="cuda")
    x.requires_grad_(True)
    y.requires_grad_(True)
    mod = SoftDTW(gamma=gamma, bandwidth=bandwidth, dist="sqeuclidean", fused=True)
    assert gradcheck(lambda a, b: mod(a, b), (x, y), **GC)
