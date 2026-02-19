from __future__ import annotations

import torch


def sqeuclidean(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Efficient squared Euclidean distance:
      D[b,i,j] = ||x[b,i]-y[b,j]||^2

    x: (B,N,D), y: (B,M,D)
    returns: (B,N,M)
    """
    if x.dim() != 3 or y.dim() != 3:
        raise ValueError(f"Expected x,y as (B,N,D)/(B,M,D). Got {tuple(x.shape)} and {tuple(y.shape)}")
    if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[2]:
        raise ValueError(f"Batch/features mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")

    # (B,N)
    x2 = (x * x).sum(dim=-1)
    # (B,M)
    y2 = (y * y).sum(dim=-1)
    # (B,N,M)
    xy = torch.bmm(x, y.transpose(1, 2))
    D = x2.unsqueeze(2) + y2.unsqueeze(1) - 2.0 * xy

    # Numerical cleanup (fp roundoff can produce tiny negatives)
    return D.clamp_min(0.0)


def pairwise_distance(x: torch.Tensor, y: torch.Tensor, *, dist: str) -> torch.Tensor:
    dist = dist.lower()
    if dist in ("sqeuclidean", "sq_euclidean", "squared_euclidean"):
        return sqeuclidean(x, y)
    raise ValueError(f"Unknown dist='{dist}'. Supported: 'sqeuclidean'")
