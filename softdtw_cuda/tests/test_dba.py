"""Tests for classic DBA (dtw_barycenter).

Property tests (exactness on identical series, monotone total cost, inertia
improvement over the Euclidean mean, weight semantics) run anywhere on CPU;
a tslearn cross-check runs when tslearn is installed and a CUDA-device test
runs when a GPU is available.

Runnable directly (``python -m softdtw_cuda.tests.test_dba``) or via pytest.
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

from softdtw_cuda import DTW, dtw_barycenter


def _inertia(mu: torch.Tensor, X: torch.Tensor) -> float:
    """Sum of exact DTW(mu, x_i), squared-euclidean local cost, CPU."""
    B = X.shape[0]
    m = mu.unsqueeze(0).expand(B, -1, -1).contiguous()
    return float(DTW(dist="sqeuclidean", fused=False)(m, X).sum().item())


def _toy(B=6, N=40, F=3, seed=0) -> torch.Tensor:
    """Warped noisy copies of a shared smooth prototype."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, N)
    base = np.stack([np.sin(2 * np.pi * (t + p)) for p in np.linspace(0, 0.2, F)], 1)
    out = np.empty((B, N, F))
    for b in range(B):
        shift = rng.uniform(-0.08, 0.08)
        tb = np.clip(t + shift * np.sin(np.pi * t), 0, 1)
        for f in range(F):
            out[b, :, f] = np.interp(tb, t, base[:, f]) + 0.05 * rng.standard_normal(N)
    return torch.tensor(out, dtype=torch.float32)


def test_identical_series_is_fixed_point():
    x = _toy(B=1)[0]
    X = x.unsqueeze(0).repeat(5, 1, 1)
    mu = dtw_barycenter(X, max_iter=10)
    assert torch.allclose(mu, x, atol=1e-5), (mu - x).abs().max()


def test_total_cost_monotone_nonincreasing():
    X = _toy()
    costs = [_inertia(dtw_barycenter(X, max_iter=k, tol=0.0), X) for k in (1, 2, 4, 8)]
    for a, b in zip(costs, costs[1:]):
        assert b <= a + 1e-6 * max(abs(a), 1.0), costs


def test_improves_over_euclidean_mean():
    X = _toy()
    mean = X.mean(0)
    mu = dtw_barycenter(X, max_iter=30)
    assert _inertia(mu, X) <= _inertia(mean, X) + 1e-9


def test_zero_weights_equal_subset():
    X = _toy(B=4)
    w = torch.tensor([1.0, 1.0, 0.0, 0.0])
    mu_w = dtw_barycenter(X, weights=w, max_iter=15)
    mu_s = dtw_barycenter(X[:2], max_iter=15)
    assert torch.allclose(mu_w, mu_s, atol=1e-6), (mu_w - mu_s).abs().max()


def test_init_sets_barycenter_length():
    X = _toy(N=40)
    init = X.mean(0)[::2].contiguous()            # T=20
    mu = dtw_barycenter(X, init=init, max_iter=10)
    assert mu.shape == (20, X.shape[2])


def test_bandwidth_runs_and_stays_finite():
    X = _toy()
    mu = dtw_barycenter(X, bandwidth=10, max_iter=10)
    assert torch.isfinite(mu).all()


def test_against_tslearn_if_available():
    try:
        from tslearn.barycenters import dtw_barycenter_averaging
    except ImportError:
        pytest.skip("tslearn not installed")
        return
    X = _toy(B=5, N=32, F=2, seed=1)
    init = X.mean(0)
    ours = dtw_barycenter(X, init=init, max_iter=30, tol=0.0)
    theirs = torch.tensor(
        dtw_barycenter_averaging(X.numpy().astype(np.float64),
                                 init_barycenter=init.numpy().astype(np.float64),
                                 max_iter=30, tol=0.0),
        dtype=torch.float32)
    # Path tie-breaks may differ; compare quality (inertia), not coordinates.
    i_ours, i_theirs = _inertia(ours, X), _inertia(theirs, X)
    assert i_ours <= i_theirs * 1.02 + 1e-9, (i_ours, i_theirs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_cuda_device_matches_cpu():
    X = _toy()
    mu_cpu = dtw_barycenter(X, max_iter=10)
    mu_gpu = dtw_barycenter(X.cuda(), device="cuda", max_iter=10).cpu()
    assert torch.allclose(mu_cpu, mu_gpu, atol=1e-4), (mu_cpu - mu_gpu).abs().max()


if __name__ == "__main__":
    for fn in [test_identical_series_is_fixed_point,
               test_total_cost_monotone_nonincreasing,
               test_improves_over_euclidean_mean,
               test_zero_weights_equal_subset,
               test_init_sets_barycenter_length,
               test_bandwidth_runs_and_stays_finite,
               test_against_tslearn_if_available]:
        fn()
        print(f"[ok] {fn.__name__}")
    if torch.cuda.is_available():
        test_cuda_device_matches_cpu()
        print("[ok] test_cuda_device_matches_cpu")
    else:
        print("[skip] test_cuda_device_matches_cpu (no CUDA)")
    print("ALL PASSED")
