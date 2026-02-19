import torch
import pytest
from softdtw_cuda import SoftDTW


@pytest.mark.cuda
def test_log_backward_stability_long_small_gamma():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)

    B = 2
    N = 128
    M = 128
    D = 3
    gamma = 0.01   # intentionally small → unstable in naive backward

    x = torch.randn(B, N, D, device="cuda", requires_grad=True)
    y = torch.randn(B, M, D, device="cuda")

    sdtw = SoftDTW(gamma=gamma, dist="sqeuclidean")

    out = sdtw(x, y)

    # forward must be finite
    assert torch.isfinite(out).all(), "Forward produced non-finite values"

    loss = out.sum()
    loss.backward()

    grad = x.grad

    # backward must be finite
    assert torch.isfinite(grad).all(), "Gradient contains NaN/Inf"

    # gradients should not collapse to zero
    grad_norm = grad.abs().mean()
    assert grad_norm > 1e-6, f"Gradient vanished (mean abs = {grad_norm})"
