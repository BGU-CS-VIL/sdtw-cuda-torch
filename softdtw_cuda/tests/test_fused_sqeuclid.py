import torch

from softdtw_cuda import SoftDTW


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        return True
    return False


def test_fused_forward_matches_unfused_small():
    if _skip_if_no_cuda():
        return

    torch.manual_seed(0)
    B, N, M, D = 2, 12, 9, 3
    x = torch.randn(B, N, D, device="cuda", requires_grad=True)
    y = torch.randn(B, M, D, device="cuda", requires_grad=True)

    sdtw_fused = SoftDTW(gamma=1.0, dist="sqeuclidean", fused=True, normalize=False)
    sdtw_unfused = SoftDTW(gamma=1.0, dist="sqeuclidean", fused=False, normalize=False)

    out_fused = sdtw_fused(x, y)
    out_unfused = sdtw_unfused(x, y)

    # Forward values should match closely
    assert torch.allclose(out_fused, out_unfused, atol=1e-4, rtol=1e-4)


def test_fused_backward_matches_unfused_tiny():
    if _skip_if_no_cuda():
        return

    torch.manual_seed(0)
    B, N, M, D = 1, 5, 4, 2

    x1 = torch.randn(B, N, D, device="cuda", requires_grad=True)
    y1 = torch.randn(B, M, D, device="cuda", requires_grad=True)

    # Clone for unfused path so we compare gradients apples-to-apples
    x2 = x1.detach().clone().requires_grad_(True)
    y2 = y1.detach().clone().requires_grad_(True)

    sdtw_fused = SoftDTW(gamma=1.0, dist="sqeuclidean", fused=True, normalize=False)
    sdtw_unfused = SoftDTW(gamma=1.0, dist="sqeuclidean", fused=False, normalize=False)

    # Fused grads
    loss_fused = sdtw_fused(x1, y1).sum()
    loss_fused.backward()
    gx_fused = x1.grad.detach().clone()
    gy_fused = y1.grad.detach().clone()

    # Unfused grads (through materialized D + sqeuclidean)
    loss_unfused = sdtw_unfused(x2, y2).sum()
    loss_unfused.backward()
    gx_unfused = x2.grad.detach().clone()
    gy_unfused = y2.grad.detach().clone()

    # Gradients should match. Tolerances slightly looser due to exp/log numerical differences.
    assert torch.allclose(gx_fused, gx_unfused, atol=5e-3, rtol=5e-3)
    assert torch.allclose(gy_fused, gy_unfused, atol=5e-3, rtol=5e-3)


def test_fused_normalize_runs_and_matches_unfused_small():
    if _skip_if_no_cuda():
        return

    torch.manual_seed(0)
    B, N, M, D = 2, 10, 10, 3
    x = torch.randn(B, N, D, device="cuda", requires_grad=True)
    y = torch.randn(B, M, D, device="cuda", requires_grad=True)

    sdtw_fused = SoftDTW(gamma=1.0, dist="sqeuclidean", fused=True, normalize=True)
    sdtw_unfused = SoftDTW(gamma=1.0, dist="sqeuclidean", fused=False, normalize=True)

    out_fused = sdtw_fused(x, y)
    out_unfused = sdtw_unfused(x, y)

    assert torch.allclose(out_fused, out_unfused, atol=1e-4, rtol=1e-4)
