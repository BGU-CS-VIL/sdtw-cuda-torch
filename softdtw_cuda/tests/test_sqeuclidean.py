import torch
from softdtw_cuda.distances import sqeuclidean

def test_sqeuclidean_matches_naive():
    torch.manual_seed(0)
    B,N,M,D = 2, 5, 7, 3
    x = torch.randn(B,N,D)
    y = torch.randn(B,M,D)

    D_fast = sqeuclidean(x, y)
    D_naive = ((x[:, :, None, :] - y[:, None, :, :]) ** 2).sum(-1)

    assert torch.allclose(D_fast, D_naive, atol=1e-6)
