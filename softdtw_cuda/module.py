from __future__ import annotations

import math
import torch
from torch import nn

from .distances import pairwise_distance
from .autograd import SoftDTWAutograd
from .autograd_xy import SoftDTWXYAutograd  # <-- new fused autograd


class SoftDTW(nn.Module):
    """
    User-facing module.

    - dist: currently supports "sqeuclidean"
    - normalize: SoftDTW(x,y) - 0.5*(SoftDTW(x,x)+SoftDTW(y,y))
    - fused:
        None  -> auto (use fused only when possible)
        True  -> require fused (error if not possible)
        False -> never fused (always materialize D and use D-based autograd)
    """

    def __init__(
        self,
        *,
        gamma: float = 1.0,
        bandwidth: float | None = None,
        normalize: bool = False,
        dist: str = "sqeuclidean",
        fused: bool | None = None,
    ):
        super().__init__()
        self.gamma = float(gamma)
        if self.gamma <= 0:
            raise ValueError(f"gamma must be > 0, got {self.gamma}")
        if not math.isfinite(self.gamma):
            raise ValueError(f"gamma must be finite, got {self.gamma}")

        # treat None or <=0 as disabled
        if bandwidth is None:
            self.bandwidth = None
        else:
            bw = float(bandwidth)
            self.bandwidth = None if bw <= 0 else bw

        self.normalize = bool(normalize)
        self.dist = str(dist)
        self.fused = fused

    def _use_fused(self, x: torch.Tensor, y: torch.Tensor) -> bool:
        # Only supported for sqeuclidean on CUDA (for now)
        fused_ok = (
            self.dist.lower() in ("sqeuclidean", "sq_euclidean", "squared_euclidean")
            and x.is_cuda
            and y.is_cuda
            and x.device == y.device
        )
        if self.fused is True and not fused_ok:
            raise ValueError("fused=True requires CUDA tensors and dist='sqeuclidean'.")
        if self.fused is False:
            return False
        # auto or forced True
        return fused_ok

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Accept (N,D) and (M,D)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if y.dim() == 2:
            y = y.unsqueeze(0)

        if x.dim() != 3 or y.dim() != 3:
            raise ValueError(
                f"Expected x,y to have shape (B,N,D) and (B,M,D) (or unbatched (N,D)). "
                f"Got x={tuple(x.shape)}, y={tuple(y.shape)}"
            )
        
        if self.normalize and x.shape[1] != y.shape[1]:
            raise ValueError(
                f"normalize=True currently requires equal sequence lengths (N==M) because it uses the "
                f"concatenation trick. Got N={x.shape[1]}, M={y.shape[1]}."
            )
        
        bx, _, dx = x.shape
        by, _, dy = y.shape
        if dx != dy:
            raise ValueError(f"Feature dims must match. Got x.shape[-1]={dx}, y.shape[-1]={dy}")
        if bx != by:
            raise ValueError(f"Batch sizes must match. Got x.shape[0]={bx}, y.shape[0]={by}")
        if x.shape[1] == 0 or y.shape[1] == 0:
            raise ValueError(
                f"Sequence lengths must be > 0. Got N={x.shape[1]}, M={y.shape[1]}."
            )

        use_fused = self._use_fused(x, y)

        # ---- Normalization mode ----
        if self.normalize:
            # Stack everything up as in canonical normalization trick
            x_cat = torch.cat([x, x, y], dim=0)
            y_cat = torch.cat([y, x, y], dim=0)

            if use_fused:
                out = SoftDTWXYAutograd.apply(x_cat, y_cat, self.gamma, self.bandwidth)
            else:
                D = pairwise_distance(x_cat, y_cat, dist=self.dist)
                out = SoftDTWAutograd.apply(D, self.gamma, self.bandwidth)

            out_xy, out_xx, out_yy = out.split(bx, dim=0)
            return out_xy - 0.5 * (out_xx + out_yy)

        # ---- Non-normalized ----
        if use_fused:
            return SoftDTWXYAutograd.apply(x, y, self.gamma, self.bandwidth)

        D_xy = pairwise_distance(x, y, dist=self.dist)
        return SoftDTWAutograd.apply(D_xy, self.gamma, self.bandwidth)
