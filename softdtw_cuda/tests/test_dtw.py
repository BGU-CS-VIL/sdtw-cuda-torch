"""Tests for the exact hard-DTW mode (DTW).

CPU tests validate the min-recurrence math against a pure-numpy reference and
run anywhere. CUDA tests validate the fused kernel (and fused-vs-D-based
equivalence, and DTW ~ SoftDTW(gamma->0)) and skip without a GPU.

Runnable directly (``python -m softdtw_cuda.tests.test_dtw``) or via pytest.
"""
from __future__ import annotations

import numpy as np
import torch
try:
    import pytest
except ImportError:                       # allow running as a plain script (no pytest)
    class _Mark:
        def __getattr__(self, _):
            return lambda *a, **k: (lambda f: f)
    class _Pytest:
        mark = _Mark()
    pytest = _Pytest()

from softdtw_cuda import DTW
from softdtw_cuda.cuda.launcher import dtw_forward_cpu


def _ref_hard_dtw(a: np.ndarray, b: np.ndarray) -> float:
    """Brute-force exact DTW, squared-euclidean local cost. a:(N,D) b:(M,D)."""
    N, M = a.shape[0], b.shape[0]
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)      # (N,M)
    R = np.full((N + 1, M + 1), np.inf, dtype=np.float64)
    R[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            R[i, j] = cost[i - 1, j - 1] + min(R[i - 1, j - 1], R[i - 1, j], R[i, j - 1])
    return float(R[N, M])


def _ref_matrix(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    Xn, Yn = X.cpu().numpy().astype(np.float64), Y.cpu().numpy().astype(np.float64)
    return torch.tensor([_ref_hard_dtw(Xn[b], Yn[b]) for b in range(Xn.shape[0])],
                        dtype=torch.float64)


SHAPES = [(4, 5, 5, 1), (4, 7, 5, 8), (3, 5, 9, 16), (2, 12, 12, 64)]  # (B,N,M,D)


@pytest.mark.parametrize("B,N,M,D", SHAPES)
def test_cpu_numba_ref_matches_bruteforce(B, N, M, D):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(B, N, D, generator=g, dtype=torch.float64)
    Y = torch.randn(B, M, D, generator=g, dtype=torch.float64)
    Dm = ((X[:, :, None, :] - Y[:, None, :, :]) ** 2).sum(-1)   # (B,N,M) costs
    out = dtw_forward_cpu(Dm, -1.0)
    ref = _ref_matrix(X, Y)
    assert torch.allclose(out, ref, rtol=1e-6, atol=1e-4), (out - ref).abs().max()


@pytest.mark.parametrize("B,N,M,D", SHAPES)
def test_module_cpu_matches_bruteforce(B, N, M, D):
    g = torch.Generator().manual_seed(1)
    X = torch.randn(B, N, D, generator=g, dtype=torch.float64)
    Y = torch.randn(B, M, D, generator=g, dtype=torch.float64)
    out = DTW(dist="sqeuclidean")(X, Y)          # CPU -> unfused min-DP
    ref = _ref_matrix(X, Y)
    assert torch.allclose(out, ref, rtol=1e-6, atol=1e-4), (out - ref).abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("B,N,M,D", SHAPES)
def test_cuda_fused_matches_ref(B, N, M, D):
    g = torch.Generator().manual_seed(2)
    X = torch.randn(B, N, D, generator=g).cuda()
    Y = torch.randn(B, M, D, generator=g).cuda()
    out = DTW(dist="sqeuclidean", fused=True)(X, Y).cpu()
    ref = _ref_matrix(X, Y).to(out.dtype)
    assert torch.allclose(out, ref, atol=1e-3, rtol=1e-4), (out - ref).abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_fused_matches_dbased():
    g = torch.Generator().manual_seed(3)
    X = torch.randn(3, 9, 32, generator=g).cuda()
    Y = torch.randn(3, 11, 32, generator=g).cuda()
    fused = DTW(dist="sqeuclidean", fused=True)(X, Y)
    dbased = DTW(dist="sqeuclidean", fused=False)(X, Y)
    assert torch.allclose(fused, dbased, atol=1e-3, rtol=1e-4), (fused - dbased).abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_hard_is_softdtw_gamma_to_zero():
    from softdtw_cuda import SoftDTW
    g = torch.Generator().manual_seed(4)
    X = torch.randn(4, 10, 16, generator=g).cuda()
    Y = torch.randn(4, 10, 16, generator=g).cuda()
    hard = DTW(dist="sqeuclidean", fused=True)(X, Y)
    soft = SoftDTW(gamma=1e-3, dist="sqeuclidean", fused=True)(X, Y)
    # small gamma approaches hard; loose tolerance (soft <= hard, gap ~ gamma*log3)
    assert torch.allclose(soft, hard, atol=5e-2), (soft - hard).abs().max()


def _ref_hard_dtw_bw(a: np.ndarray, b: np.ndarray, bw: float) -> float:
    """Brute-force exact DTW with a Sakoe-Chiba band |i-j| <= bw."""
    N, M = a.shape[0], b.shape[0]
    cost = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    R = np.full((N + 1, M + 1), np.inf, dtype=np.float64)
    R[0, 0] = 0.0
    for i in range(1, N + 1):
        for j in range(1, M + 1):
            if abs((i - 1) - (j - 1)) > bw:
                continue
            R[i, j] = cost[i - 1, j - 1] + min(R[i - 1, j - 1], R[i - 1, j], R[i, j - 1])
    return float(R[N, M])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("N,M", [(5, 5), (7, 5), (5, 9), (16, 4)])  # incl. N>M (buffer swap)
@pytest.mark.parametrize("bw", [1.0, 2.0, 4.0])
def test_cuda_fused_bandwidth_matches_ref(N, M, bw):
    B, D = 3, 8
    g = torch.Generator().manual_seed(5)
    X = torch.randn(B, N, D, generator=g)
    Y = torch.randn(B, M, D, generator=g)
    out = DTW(dist="sqeuclidean", fused=True, bandwidth=bw)(X.cuda(), Y.cuda()).cpu()
    Xn, Yn = X.numpy().astype(np.float64), Y.numpy().astype(np.float64)
    ref = torch.tensor([_ref_hard_dtw_bw(Xn[b], Yn[b], bw) for b in range(B)], dtype=torch.float64)
    assert torch.allclose(out.double(), ref, atol=1e-3, rtol=1e-4), (out.double() - ref).abs().max()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_fused_long_sequence():
    # Streaming path must run well past the 1024 per-block cap on GPU.
    g = torch.Generator().manual_seed(6)
    X = torch.randn(2, 2000, 4, generator=g).cuda()
    Y = torch.randn(2, 2300, 4, generator=g).cuda()
    out = DTW(dist="sqeuclidean", fused=True)(X, Y)
    assert out.shape == (2,) and torch.isfinite(out).all()


if __name__ == "__main__":
    passed = 0
    for (B, N, M, D) in SHAPES:
        g = torch.Generator().manual_seed(0)
        X = torch.randn(B, N, D, generator=g, dtype=torch.float64)
        Y = torch.randn(B, M, D, generator=g, dtype=torch.float64)
        Dm = ((X[:, :, None, :] - Y[:, None, :, :]) ** 2).sum(-1)
        ref = _ref_matrix(X, Y)
        cpu_numba = dtw_forward_cpu(Dm, -1.0)
        mod = DTW(dist="sqeuclidean")(X, Y)
        d1 = (cpu_numba - ref).abs().max().item()
        d2 = (mod - ref).abs().max().item()
        ok = torch.allclose(cpu_numba, ref, rtol=1e-6, atol=1e-4) and \
             torch.allclose(mod, ref, rtol=1e-6, atol=1e-4)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] B={B} N={N} M={M} D={D}: "
              f"numba-cpu maxdiff {d1:.2e} | module maxdiff {d2:.2e}")
    print(f"===== {passed}/{len(SHAPES)} CPU cases PASSED "
          f"(CUDA cases run on a GPU node via pytest) =====")
