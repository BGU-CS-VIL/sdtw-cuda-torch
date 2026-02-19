from __future__ import annotations

import torch
from .module import SoftDTW


def softdtw(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    gamma: float = 1.0,
    bandwidth: float | None = None,
    normalize: bool = False,
    dist: str = "sqeuclidean",
    fused: bool | None = None,
) -> torch.Tensor:
    """
    Convenience functional API.

    x: (B,N,D) or (N,D)
    y: (B,M,D) or (M,D)
    fused: None (auto), True (require fused), False (never fused)
    returns: (B,)
    """
    return SoftDTW(gamma=gamma, bandwidth=bandwidth, normalize=normalize, dist=dist, fused=fused)(x, y)
