import torch
import pytest
from softdtw_cuda import SoftDTW

def test_softdtw_cuda_runs_longer_than_1024():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    torch.manual_seed(0)
    x = torch.randn(1, 5000, 2, device="cuda", requires_grad=True)
    y = torch.randn(1, 5000, 2, device="cuda")

    sdtw = SoftDTW(gamma=1.0, dist="sqeuclidean", fused=True)
    out = sdtw(x, y)
    assert torch.isfinite(out).all()

    out.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().mean() > 0
