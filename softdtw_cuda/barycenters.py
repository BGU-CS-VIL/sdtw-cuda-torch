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
