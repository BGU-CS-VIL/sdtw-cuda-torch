from __future__ import annotations

import torch


def check_D(D: torch.Tensor) -> None:
    if not isinstance(D, torch.Tensor):
        raise TypeError("D must be a torch.Tensor")
    if D.dim() != 3:
        raise ValueError(f"D must have shape (B,N,M). Got {tuple(D.shape)}")
    if D.dtype not in (torch.float32, torch.float64, torch.float16, torch.bfloat16):
        raise TypeError(f"Unsupported dtype {D.dtype}")
    if not D.is_contiguous():
        # We'll make contiguous in launcher; still warn early if you want strictness
        pass
