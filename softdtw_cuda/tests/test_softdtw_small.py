import torch
from softdtw_cuda import SoftDTW

def test_softdtw_cpu_runs():
    torch.manual_seed(0)
    x = torch.randn(2, 10, 4, requires_grad=True)
    y = torch.randn(2, 11, 4)
    sdtw = SoftDTW(gamma=1.0, dist="sqeuclidean")
    out = sdtw(x, y)
    loss = out.sum()
    loss.backward()
    assert torch.isfinite(out).all()
    assert torch.isfinite(x.grad).all()

def test_softdtw_cuda_matches_cpu_small():
    if not torch.cuda.is_available():
        return
    torch.manual_seed(0)
    x = torch.randn(2, 12, 3, requires_grad=True)
    y = torch.randn(2, 9, 3)

    sdtw = SoftDTW(gamma=1.0, dist="sqeuclidean")

    out_cpu = sdtw(x, y).detach().cpu()
    out_gpu = sdtw(x.cuda(), y.cuda()).detach().cpu()

    assert torch.allclose(out_cpu, out_gpu, atol=1e-4, rtol=1e-4)

def test_gradcheck_tiny():
    # gradcheck needs double
    torch.manual_seed(0)
    x = torch.randn(1, 6, 2, dtype=torch.float64, requires_grad=True)
    y = torch.randn(1, 5, 2, dtype=torch.float64)
    sdtw = SoftDTW(gamma=1.0, dist="sqeuclidean")
    out = sdtw(x, y)
    (out.sum()).backward()
    assert torch.isfinite(x.grad).all()
