from __future__ import annotations

import torch
from torch import nn

from .distances import pairwise_distance
from .cuda.launcher import (
    dtw_forward_cuda_fused_sqeuclid,
    dtw_forward_cuda,
    dtw_forward_cpu,
)


class DTW(nn.Module):
    """Exact (hard) Dynamic Time Warping distance — the gamma->0 limit of SoftDTW.

    Reuses the same anti-diagonal CUDA machinery as :class:`SoftDTW`, but the
    per-cell aggregation is a true ``min`` instead of soft-min, giving the
    *exact* DTW distance with no gamma to tune and no exp/log per cell. In
    fused mode the forward streams the DP over three rotating anti-diagonals,
    so peak memory is O(B*min(N,M)) with no (B,N,M) cost tensor or DP table;
    the unfused path materializes the (B,N,M) local-cost matrix.

    Forward-only: hard-DTW is non-differentiable at the optimal path, so this
    module runs under ``torch.no_grad()`` and does not build an autograd graph.
    It is intended for evaluation / retrieval (e.g. 1-NN-DTW).

    On L2-normalized inputs, squared-euclidean = 2*(1 - cosine), so
    ``DTW(dist='sqeuclidean')`` reproduces cosine-DTW up to a positive
    affine factor (argmin-invariant) — i.e. exact cosine 1-NN-DTW.

    Args (mirror :class:`SoftDTW`):
        dist:      'sqeuclidean' (default) or 'cosine' (unfused only).
        fused:     None -> auto (fused when CUDA + sqeuclidean; auto ignores D);
                   True -> require fused (error otherwise);
                   False -> never fused: build the (B,N,M) cost with one bmm,
                       then the min-DP. Fused recomputes cost O(D) per cell, so
                       for large D (e.g. 1-NN-DTW on foundation features) prefer
                       fused=False for much better speed.
        bandwidth: Sakoe-Chiba band. None or <= 0 disables the constraint.
    """

    def __init__(self, dist: str = "sqeuclidean", fused: bool | None = None,
                 bandwidth: float | None = None):
        super().__init__()
        self.dist = dist
        self.fused = fused
        self.bandwidth = -1.0 if bandwidth is None else float(bandwidth)

    def _use_fused(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        fused_ok = (
            x.is_cuda and y.is_cuda
            and self.dist.lower() in ("sqeuclidean", "sq_euclidean", "squared_euclidean")
        )
        if self.fused is True and not fused_ok:
            raise ValueError("fused=True requires CUDA tensors and dist='sqeuclidean'.")
        if self.fused is False:
            return False
        return fused_ok

    @torch.no_grad()
    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Accept (N,D) and (M,D)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if y.dim() == 2:
            y = y.unsqueeze(0)
        if x.dim() != 3 or y.dim() != 3:
            raise ValueError(
                f"Expected x,y to have shape (B,N,D)/(B,M,D) (or unbatched (N,D)). "
                f"Got x={tuple(x.shape)}, y={tuple(y.shape)}"
            )
        if x.shape[0] != y.shape[0]:
            raise ValueError(f"Batch sizes must match. Got {x.shape[0]} vs {y.shape[0]}")
        if x.shape[2] != y.shape[2]:
            raise ValueError(f"Feature dims must match. Got {x.shape[2]} vs {y.shape[2]}")
        if x.shape[1] == 0 or y.shape[1] == 0:
            raise ValueError(f"Sequence lengths must be > 0. Got N={x.shape[1]}, M={y.shape[1]}.")

        if self._use_fused(x, y):
            return dtw_forward_cuda_fused_sqeuclid(x, y, self.bandwidth)

        D_xy = pairwise_distance(x, y, dist=self.dist)
        if D_xy.is_cuda:
            return dtw_forward_cuda(D_xy, self.bandwidth)
        return dtw_forward_cpu(D_xy, self.bandwidth)


def dtw(x: torch.Tensor, y: torch.Tensor, dist: str = "sqeuclidean",
            fused: bool | None = None, bandwidth: float | None = None) -> torch.Tensor:
    """Functional exact hard-DTW distance. See :class:`DTW`."""
    return DTW(dist=dist, fused=fused, bandwidth=bandwidth)(x, y)
