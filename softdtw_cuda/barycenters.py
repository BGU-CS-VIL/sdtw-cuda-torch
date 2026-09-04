"""
SoftDTW Barycenter Averaging

Implements time series averaging using soft Dynamic Time Warping geometry.
Based on the method from Cuturi & Blondel (ICML 2017).

Reference: https://github.com/tslearn-team/tslearn/blob/main/tslearn/barycenters/softdtw.py
"""

from __future__ import annotations

import time
import torch
from .module import SoftDTW


def softdtw_barycenter(
    X: torch.Tensor,
    *,
    gamma: float = 1.0,
    weights: torch.Tensor | None = None,
    max_iter: int = 100,
    lr: float = 0.1,
    init: torch.Tensor | None = None,
    device: str | torch.device | None = None,
    verbose: bool = False,
    fused: bool | None = None,
    early_stopping: bool = True,
    patience: int = 10,
    tol: float = 1e-5,
) -> torch.Tensor:
    """
    Compute a SoftDTW barycenter (time series average) through optimization.

    This function finds the barycenter that minimizes the weighted sum of SoftDTW
    distances to all input time series using gradient-based optimization.

    Args:
        X: Input time series of shape (B, N, D) where:
           - B: batch size (number of sequences)
           - N: sequence length
           - D: feature dimension
        gamma: SoftDTW regularization parameter. Default: 1.0
        weights: Optional weights for each sequence, shape (B,). Default: uniform
        max_iter: Maximum optimization iterations. Default: 100
        lr: Learning rate for optimization. Default: 0.1
        init: Initial barycenter, shape (N, D). If None, uses weighted mean. Default: None
        device: Device to compute on. If None, uses X's device. Default: None
        verbose: Print iteration progress and timing. Default: False
        fused: Fused mode selection. Default: None (auto-select)
           - None: Auto-select (use fused if CUDA available)
           - True: Require fused mode (error if not available)
           - False: Never use fused mode (always use standard distance matrix)
        early_stopping: Stop early if loss plateaus. Default: True
        patience: Iterations without improvement before stopping. Default: 10
        tol: Absolute improvement threshold for early stopping. Default: 1e-5
           Note: Uses absolute improvement (best_loss - loss_val > tol), which handles
           negative SoftDTW values correctly

    Returns:
        Barycenter of shape (N, D)

    Example:
        >>> X = torch.randn(16, 100, 3, device="cuda")  # 16 sequences of length 100, dim 3
        >>> barycenter = softdtw_barycenter(X, gamma=1.0, max_iter=50, verbose=True)
        >>> barycenter.shape
        torch.Size([100, 3])

        >>> # Force fused mode for memory efficiency
        >>> barycenter_fused = softdtw_barycenter(X, fused=True)

        >>> # Force unfused mode for predictable performance
        >>> barycenter_unfused = softdtw_barycenter(X, fused=False)
    """
    device = device or X.device

    # Move X to target device first
    X = X.to(device)
    B, N, D = X.shape

    # Normalize weights
    if weights is None:
        weights = torch.ones(B, device=device) / B
    else:
        weights = weights.to(device)
        weights = weights / weights.sum()

    # Initialize barycenter with weighted mean (better than unweighted mean)
    if init is None:
        barycenter = (X * weights.view(B, 1, 1)).sum(dim=0).clone()
    else:
        barycenter = init.clone().to(device)

    # Ensure barycenter requires gradients
    barycenter = barycenter.requires_grad_(True)

    # Create SoftDTW loss function
    loss_fn = SoftDTW(gamma=gamma, normalize=False, fused=fused)

    # Optimizer
    optimizer = torch.optim.Adam([barycenter], lr=lr)

    # Learning rate scheduler: cosine annealing for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max_iter, eta_min=lr * 0.1
    )

    # Start timing
    opt_start_time = time.time()
    best_loss = float('inf')
    patience_counter = 0

    # Optimization loop
    for iteration in range(max_iter):
        # Synchronize before timing for accurate CUDA measurements
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        optimizer.zero_grad()

        # Expand barycenter to batch size for comparison
        barycenter_batch = barycenter.unsqueeze(0).expand(B, -1, -1)

        # Compute SoftDTW loss to all sequences
        distances = loss_fn(barycenter_batch, X)  # shape: (B,)

        # Weighted loss (can be negative due to SoftDTW soft-min aggregation)
        loss = (weights * distances).sum()

        # Backprop and optimization step
        loss.backward()

        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_([barycenter], max_norm=1.0)

        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        improvement = float('nan')  # Track improvement for logging

        # Early stopping: track absolute improvement (works for negative losses)
        if early_stopping:
            improvement = best_loss - loss_val
            if improvement > tol:  # Absolute improvement (works for negative losses)
                best_loss = loss_val
                patience_counter = 0
            else:
                patience_counter += 1

            # Stop if no improvement for 'patience' iterations (after warmup)
            if patience_counter >= patience and iteration > max_iter // 2:
                if verbose:
                    print(
                        f"Early stopping at iteration {iteration + 1} "
                        f"(no improvement for {patience} iterations)"
                    )
                break

        # Optional: Print progress with timing
        if verbose and (iteration + 1) % 20 == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.time() - opt_start_time
            lr_val = optimizer.param_groups[0]['lr']

            # Format improvement string (may be NaN if early_stopping disabled)
            if early_stopping:
                improvement_str = f"{improvement:9.6f}"
            else:
                improvement_str = "     N/A "

            print(
                f"Iteration {iteration + 1:3d}/{max_iter} | "
                f"Loss: {loss_val:9.6f} | "
                f"Improvement: {improvement_str} | "
                f"LR: {lr_val:.2e}"
            )

    return barycenter.detach()


def softdtw_barycenter_cpu(
    X: torch.Tensor,
    *,
    gamma: float = 1.0,
    weights: torch.Tensor | None = None,
    max_iter: int = 100,
    lr: float = 0.1,
    init: torch.Tensor | None = None,
    verbose: bool = False,
    fused: bool | None = None,
    early_stopping: bool = True,
    patience: int = 10,
    tol: float = 1e-5,
) -> torch.Tensor:
    """
    Compute a SoftDTW barycenter on CPU (convenience wrapper).

    Args:
        X: Input time series of shape (B, N, D)
        gamma: SoftDTW regularization parameter. Default: 1.0
        weights: Optional weights for each sequence. Default: uniform
        max_iter: Maximum optimization iterations. Default: 100
        lr: Learning rate for optimization. Default: 0.01
        init: Initial barycenter. If None, uses weighted mean. Default: None
        verbose: Print iteration progress and timing. Default: False
        fused: Fused mode selection. Default: None (auto-select)
        early_stopping: Stop early if loss plateaus. Default: True
        patience: Iterations without improvement before stopping. Default: 10
        tol: Improvement threshold for early stopping. Default: 1e-5

    Returns:
        Barycenter of shape (N, D)
    """
    return softdtw_barycenter(
        X,
        gamma=gamma,
        weights=weights,
        max_iter=max_iter,
        lr=lr,
        init=init,
        device="cpu",
        verbose=verbose,
        fused=fused,
        early_stopping=early_stopping,
        patience=patience,
        tol=tol,
    )


# ---------------------------------------------------------------------------
# Classic DBA (DTW Barycenter Averaging) — exact DTW, Petitjean et al. 2011
# ---------------------------------------------------------------------------

import numpy as np
from numba import njit

from .distances import pairwise_distance


@njit(cache=True)
def _dba_iteration(D_all, X, w):
    """One DBA pass: full DP + backtrack per series, weighted accumulation.

    D_all: (B, T, N) float64 local cost (barycenter rows x series cols).
    X:     (B, N, F) float64 series values.
    w:     (B,) float64 weights.
    Returns (sums (T,F), counts (T,), total_cost). Ties in the backtrack
    prefer the diagonal (match), then insertion (i-1), then deletion (j-1).
    """
    B, T, N = D_all.shape
    F = X.shape[2]
    sums = np.zeros((T, F))
    counts = np.zeros(T)
    total = 0.0
    R = np.empty((T + 1, N + 1))
    for b in range(B):
        R[:] = np.inf
        R[0, 0] = 0.0
        for i in range(1, T + 1):
            for j in range(1, N + 1):
                r0 = R[i - 1, j - 1]
                r1 = R[i - 1, j]
                r2 = R[i, j - 1]
                rmin = r0
                if r1 < rmin:
                    rmin = r1
                if r2 < rmin:
                    rmin = r2
                R[i, j] = D_all[b, i - 1, j - 1] + rmin
        total += w[b] * R[T, N]
        i, j = T, N
        while True:
            for f in range(F):
                sums[i - 1, f] += w[b] * X[b, j - 1, f]
            counts[i - 1] += w[b]
            if i == 1 and j == 1:
                break
            if i == 1:
                j -= 1
                continue
            if j == 1:
                i -= 1
                continue
            r0 = R[i - 1, j - 1]
            r1 = R[i - 1, j]
            r2 = R[i, j - 1]
            if r0 <= r1 and r0 <= r2:
                i -= 1
                j -= 1
            elif r1 <= r2:
                i -= 1
            else:
                j -= 1
    return sums, counts, total


def dtw_barycenter(
    X: torch.Tensor,
    *,
    max_iter: int = 30,
    tol: float = 1e-5,
    weights: torch.Tensor | None = None,
    init: torch.Tensor | None = None,
    bandwidth: float | None = None,
    device: str | torch.device | None = None,
    verbose: bool = False,
) -> torch.Tensor:
    """Classic DBA (DTW Barycenter Averaging; Petitjean et al., 2011) under
    exact DTW — the hard counterpart of :func:`softdtw_barycenter`.

    Alternates (1) exact-DTW alignment of every series to the current
    barycenter and (2) per-index weighted arithmetic mean of the aligned
    values, until the total DTW cost stops improving. The local cost is fixed
    to squared euclidean: the arithmetic-mean update is the exact minimizer
    of the alignment objective only under that cost.

    Unlike :func:`softdtw_barycenter` (gradient descent, differentiable),
    DBA is a fixed-point scheme on hard alignments: monotonically
    non-increasing total cost, no learning rate, no gamma.

    Implementation: local-cost matrices are computed with torch on `device`
    (GPU-capable, the O(B*T*N*F) part); the sequential DP + backtrack runs in
    a numba-jitted CPU loop (float64). The DP materializes one (T+1, N+1)
    matrix at a time — DBA needs the full matrix for backtracking, so the
    fused flat-memory kernel of :class:`DTW` does not apply here.

    Args:
        X: (B, N, F) series (equal lengths).
        max_iter: maximum DBA iterations.
        tol: relative total-cost improvement below which to stop.
        weights: (B,) nonnegative per-series weights (default: uniform).
        init: (T, F) initial barycenter (default: weighted Euclidean mean,
            T = N). Its length T sets the barycenter length.
        bandwidth: Sakoe-Chiba band on |i - j·T/N| (None disables).
        device: where the cost matrices are computed (default: X.device).
        verbose: print per-iteration total cost.

    Returns:
        (T, F) barycenter, on X's original device and dtype.
    """
    if X.dim() != 3:
        raise ValueError(f"Expected X of shape (B, N, F). Got {tuple(X.shape)}")
    B, N, F = X.shape
    if B == 0 or N == 0:
        raise ValueError(f"Empty input: B={B}, N={N}.")
    out_device, out_dtype = X.device, X.dtype
    dev = torch.device(device) if device is not None else X.device
    Xd = X.detach().to(dev, dtype=torch.float32)

    if weights is None:
        w = torch.ones(B, dtype=torch.float64)
    else:
        w = weights.detach().to("cpu", dtype=torch.float64).reshape(-1)
        if w.numel() != B:
            raise ValueError(f"weights must have {B} entries. Got {w.numel()}.")
        if (w < 0).any() or w.sum() <= 0:
            raise ValueError("weights must be nonnegative with positive sum.")
    w_np = w.numpy()
    w_dev = w.to(dev, dtype=torch.float32)

    if init is None:
        mu = (Xd * w_dev[:, None, None]).sum(0) / w_dev.sum()          # (N, F)
    else:
        if init.dim() != 2 or init.shape[1] != F:
            raise ValueError(f"init must be (T, {F}). Got {tuple(init.shape)}")
        mu = init.detach().to(dev, dtype=torch.float32).clone()
    T = mu.shape[0]

    band_mask = None
    if bandwidth is not None and bandwidth > 0:
        ii = torch.arange(T, device=dev, dtype=torch.float32)[:, None]
        jj = torch.arange(N, device=dev, dtype=torch.float32)[None, :]
        band_mask = (ii - jj * (T / max(N, 1))).abs() > float(bandwidth)  # (T, N)

    X_np = Xd.to("cpu", dtype=torch.float64).numpy()
    prev_total = float("inf")
    for it in range(max_iter):
        D_all = pairwise_distance(mu.unsqueeze(0).expand(B, -1, -1), Xd,
                                  dist="sqeuclidean")                  # (B, T, N)
        if band_mask is not None:
            D_all = D_all.masked_fill(band_mask, float("inf"))
        D_np = D_all.to("cpu", dtype=torch.float64).numpy()
        sums, counts, total = _dba_iteration(D_np, X_np, w_np)
        if verbose:
            print(f"[dba] iter {it:3d}  total_cost={total:.6f}")
        mu = torch.from_numpy(sums / counts[:, None]).to(dev, dtype=torch.float32)
        if prev_total - total <= tol * max(abs(prev_total), 1.0):
            break
        prev_total = total
    return mu.to(out_device, dtype=out_dtype)
