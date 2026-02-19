from __future__ import annotations

import math
import numpy as np
import torch
from numba import cuda
from numba import jit, prange

from .kernels import softdtw_forward_kernel, softdtw_forward_diag_cuda
from .kernels import softdtw_backward_log_cuda, softdtw_backward_log_diag_cuda
from .kernels import softdtw_forward_diag_sqeuclid_cuda
from .kernels import softdtw_backward_log_diag_sqeuclid_cuda

# GLOBALS
TPB_LONG = 256


# HELPERS
def _diag_bounds(p: int, N: int, M: int) -> tuple[int, int]:
    i_min = max(0, p - (M - 1))
    i_max = min(N - 1, p)
    return i_min, i_max



def _threads_and_passes(N: int, M: int) -> tuple[int, int]:
    tpb = max(N, M)
    n_passes = 2 * tpb - 1
    return tpb, n_passes


# MAIN - on-the-fly D
def softdtw_forward_cuda_fused_sqeuclid(X: torch.Tensor, Y: torch.Tensor, gamma: float, bandwidth: float):
    """
    Fused SoftDTW forward for squared-euclidean distance that does NOT materialize D (B,N,M).

    X: (B,N,D), Y: (B,M,D) CUDA tensors
    Returns: (out: (B,), R: (B,N+2,M+2))
    """
    if not (X.is_cuda and Y.is_cuda):
        raise ValueError("Expected CUDA tensors X and Y")
    if X.dim() != 3 or Y.dim() != 3:
        raise ValueError(f"Expected X,Y as (B,N,D)/(B,M,D). Got {tuple(X.shape)} and {tuple(Y.shape)}")
    if X.shape[0] != Y.shape[0] or X.shape[2] != Y.shape[2]:
        raise ValueError(f"Batch/features mismatch: {tuple(X.shape)} vs {tuple(Y.shape)}")

    # Detach before passing to numba
    X_ = X.detach().contiguous()
    Y_ = Y.detach().contiguous()

    B, N, D = X_.shape
    M = Y_.shape[1]

    # Allocate DP table
    R = torch.full((B, N + 2, M + 2), math.inf, device=X_.device, dtype=X_.dtype)
    R[:, 0, 0] = 0.0

    X_ca = cuda.as_cuda_array(X_)
    Y_ca = cuda.as_cuda_array(Y_)
    R_ca = cuda.as_cuda_array(R)

    inv_bw = float(bandwidth)  # can be -1.0 to disable

    # Anti-diagonals over unpadded (i,j): p = i + j, i in [0,N-1], j in [0,M-1]
    for p in range(N + M - 1):
        i_min = max(0, p - (M - 1))
        i_max = min(N - 1, p)
        if i_max < i_min:
            continue
        diag_len = i_max - i_min + 1
        grid_x = (diag_len + TPB_LONG - 1) // TPB_LONG

        # grid=(grid_x, B), so batch = blockIdx.y in kernel
        softdtw_forward_diag_sqeuclid_cuda[(grid_x, B), TPB_LONG](
            X_ca,
            Y_ca,
            R_ca,
            float(gamma),
            inv_bw,
            N,
            M,
            D,
            p,
        )

    out = R[:, -2, -2].contiguous()
    return out, R


def softdtw_backward_cuda_fused_sqeuclid(X: torch.Tensor, Y: torch.Tensor, R: torch.Tensor, gamma: float, bandwidth: float):
    """
    Fused SoftDTW backward (log-space) for squared-euclidean distance that does NOT materialize D_pad.

    Inputs:
      X: (B,N,D) CUDA
      Y: (B,M,D) CUDA
      R: (B,N+2,M+2) CUDA (from forward)
    Returns:
      E: (B,N,M) CUDA  (E = d SoftDTW / d D  in linear space, via exp(logE))
    """
    if not (X.is_cuda and Y.is_cuda and R.is_cuda):
        raise ValueError("Expected CUDA tensors X, Y, R")
    if X.dim() != 3 or Y.dim() != 3:
        raise ValueError(f"Expected X,Y as (B,N,D)/(B,M,D). Got {tuple(X.shape)} and {tuple(Y.shape)}")
    if X.shape[0] != Y.shape[0] or X.shape[2] != Y.shape[2]:
        raise ValueError(f"Batch/features mismatch: {tuple(X.shape)} vs {tuple(Y.shape)}")

    # Detach before passing to numba
    X_ = X.detach().contiguous()
    Y_ = Y.detach().contiguous()

    B, N, D = X_.shape
    M = Y_.shape[1]

    if R.shape != (B, N + 2, M + 2):
        raise ValueError(f"Expected R shape {(B, N+2, M+2)}, got {tuple(R.shape)}")

    R_ = R.contiguous()

    # ---------- boundary conditions for R ----------
    R_work = R_.clone()
    R_work[:, :, -1] = -math.inf
    R_work[:, -1, :] = -math.inf
    R_work[:, -1, -1] = R_work[:, -2, -2]

    # ---------- init logE ----------
    logE = torch.full((B, N + 2, M + 2), -math.inf, device=X_.device, dtype=X_.dtype)
    logE[:, -1, -1] = 0.0  # log(1)

    X_ca = cuda.as_cuda_array(X_)
    Y_ca = cuda.as_cuda_array(Y_)
    Rw_ca = cuda.as_cuda_array(R_work)
    logE_ca = cuda.as_cuda_array(logE)

    inv_gamma = float(1.0 / gamma)
    bw = float(bandwidth)

    # Reverse anti-diagonals over unpadded indices p = i + j, starting from (N-1)+(M-1)-1 = N+M-2 down to 0
    for p in range(N + M - 2, -1, -1):
        i_min = max(0, p - (M - 1))
        i_max = min(N - 1, p)
        if i_max < i_min:
            continue
        diag_len = i_max - i_min + 1
        grid_x = (diag_len + TPB_LONG - 1) // TPB_LONG

        softdtw_backward_log_diag_sqeuclid_cuda[(grid_x, B), TPB_LONG](
            X_ca,
            Y_ca,
            Rw_ca,
            logE_ca,
            inv_gamma,
            bw,
            N,
            M,
            D,
            p,
        )

    # crop + exp
    E = torch.exp(logE[:, 1:N + 1, 1:M + 1]).contiguous()
    return E



# MAIN - Full D Matrix
def softdtw_forward_cuda(D: torch.Tensor, gamma: float, bandwidth: float):
    if not D.is_cuda:
        raise ValueError("Expected CUDA tensor D")

    D_ = D.detach().contiguous()
    B, N, M = D_.shape
    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    # Allocate DP table
    R = torch.full((B, N + 2, M + 2), math.inf, device=D_.device, dtype=D_.dtype)
    R[:, 0, 0] = 0.0
    
    # --- Fast path: one block per batch element ---
    tpb, n_passes = _threads_and_passes(N, M)
    USE_FAST_PATH = (tpb <= 1024)

    if USE_FAST_PATH:
        softdtw_forward_kernel[B, tpb](
            cuda.as_cuda_array(D_),
            float(gamma),
            float(bandwidth),
            N,
            M,
            n_passes,
            cuda.as_cuda_array(R),
        )
        out = R[:, -2, -2].contiguous()
        return out, R

    # --- Long sequence path: tiled anti-diagonal launches ---
    

    D_ca = cuda.as_cuda_array(D_)
    R_ca = cuda.as_cuda_array(R)

    # Iterate anti-diagonals in unpadded (i,j) coords over D (shape N x M)
    for p in range(N + M - 1):
        i_min, i_max = _diag_bounds(p, N, M)
        if i_max < i_min:
            continue
        diag_len = i_max - i_min + 1
        grid_x = (diag_len + TPB_LONG - 1) // TPB_LONG

        # grid=(grid_x, B) so batch index is blockIdx.y inside kernel
        softdtw_forward_diag_cuda[(grid_x, B), TPB_LONG](
            D_ca,
            R_ca,
            float(gamma),
            float(bandwidth),
            N,
            M,
            p,
        )

    out = R[:, -2, -2].contiguous()
    return out, R


def softdtw_backward_cuda_log(D: torch.Tensor, R: torch.Tensor, gamma: float, bandwidth: float):
    if not D.is_cuda:
        raise ValueError("Expected CUDA tensor D")

    D_ = D.detach().contiguous()
    B, N, M = D_.shape
    R = R.contiguous()

    if gamma <= 0:
        raise ValueError(f"gamma must be > 0, got {gamma}")

    # ---------- pad D ----------
    D_pad = torch.zeros((B, N + 2, M + 2), device=D_.device, dtype=D_.dtype)
    D_pad[:, 1:N + 1, 1:M + 1] = D_

    # ---------- boundary conditions for R ----------
    R_work = R.clone()
    R_work[:, :, -1] = -math.inf
    R_work[:, -1, :] = -math.inf
    R_work[:, -1, -1] = R_work[:, -2, -2]

    # ---------- init logE ----------
    logE = torch.full((B, N + 2, M + 2), -math.inf, device=D_.device, dtype=D_.dtype)
    logE[:, -1, -1] = 0.0  # log(1)

    # ---------- choose fast vs tiled ----------
    tpb, n_passes = _threads_and_passes(N, M)
    USE_FAST_PATH = (tpb <= 1024)

    if USE_FAST_PATH:
        # fast path: your existing diagonal backward kernel (single block per batch)
        softdtw_backward_log_cuda[B, tpb](
            cuda.as_cuda_array(D_pad),
            cuda.as_cuda_array(R_work),
            float(1.0 / gamma),
            float(bandwidth),
            N,
            M,
            n_passes,
            cuda.as_cuda_array(logE),
        )
    else:
        # tiled path: launch one kernel per anti-diagonal in reverse order

        Dp_ca = cuda.as_cuda_array(D_pad)
        Rw_ca = cuda.as_cuda_array(R_work)
        logE_ca = cuda.as_cuda_array(logE)

        inv_gamma = float(1.0 / gamma)
        bw = float(bandwidth)
        if bw <= 0:
            bw = -1.0

        # unpadded indices (i,j) are 0..N-1, 0..M-1, diagonals p = i+j
        for p in range(N + M - 2, -1, -1):
            i_min, i_max = _diag_bounds(p, N, M)
            if i_max < i_min:
                continue
            diag_len = i_max - i_min + 1
            grid_x = (diag_len + TPB_LONG - 1) // TPB_LONG

            softdtw_backward_log_diag_cuda[(grid_x, B), TPB_LONG](
                Dp_ca,
                Rw_ca,
                logE_ca,
                inv_gamma,
                bw,
                N,
                M,
                p,
            )

    # crop + exp
    E = torch.exp(logE[:, 1:N + 1, 1:M + 1]).contiguous()
    return E




# ---- CPU reference (optional but useful for tests) ----

@jit(nopython=True, parallel=True)
def _softdtw_forward_cpu_np(D: np.ndarray, gamma: float, bandwidth: float):
    B, N, M = D.shape
    R = np.ones((B, N + 2, M + 2), dtype=D.dtype) * np.inf
    R[:, 0, 0] = 0.0
    for b in prange(B):
        for j in range(1, M + 1):
            for i in range(1, N + 1):
                if 0 < bandwidth < abs(i - j):
                    continue
                r0 = -R[b, i - 1, j - 1] / gamma
                r1 = -R[b, i - 1, j] / gamma
                r2 = -R[b, i, j - 1] / gamma
                rmax = max(max(r0, r1), r2)
                rsum = np.exp(r0 - rmax) + np.exp(r1 - rmax) + np.exp(r2 - rmax)
                softmin = -gamma * (np.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
    return R


@jit(nopython=True, parallel=True)
def _softdtw_backward_cpu_np(D_: np.ndarray, R: np.ndarray, gamma: float, bandwidth: float):
    B, N, M = D_.shape
    D = np.zeros((B, N + 2, M + 2), dtype=D_.dtype)
    D[:, 1:N + 1, 1:M + 1] = D_

    E = np.zeros((B, N + 2, M + 2), dtype=D_.dtype)
    E[:, -1, -1] = 1.0

    R[:, :, -1] = -np.inf
    R[:, -1, :] = -np.inf
    R[:, -1, -1] = R[:, -2, -2]

    for b in prange(B):
        for j in range(M, 0, -1):
            for i in range(N, 0, -1):
                if np.isinf(R[b, i, j]):
                    R[b, i, j] = -np.inf
                if 0 < bandwidth < abs(i - j):
                    continue
                a0 = (R[b, i + 1, j] - R[b, i, j] - D[b, i + 1, j]) / gamma
                b0 = (R[b, i, j + 1] - R[b, i, j] - D[b, i, j + 1]) / gamma
                c0 = (R[b, i + 1, j + 1] - R[b, i, j] - D[b, i + 1, j + 1]) / gamma
                a = np.exp(a0); bb = np.exp(b0); c = np.exp(c0)
                E[b, i, j] = E[b, i + 1, j] * a + E[b, i, j + 1] * bb + E[b, i + 1, j + 1] * c

    return E[:, 1:N + 1, 1:M + 1]


def softdtw_forward_cpu(D: torch.Tensor, gamma: float, bandwidth: float):
    D_np = D.detach().cpu().numpy()
    R_np = _softdtw_forward_cpu_np(D_np, float(gamma), float(bandwidth))
    R = torch.from_numpy(R_np).to(D.device).type_as(D)
    out = R[:, -2, -2].contiguous()
    return out, R


def softdtw_backward_cpu(D: torch.Tensor, R: torch.Tensor, gamma: float, bandwidth: float):
    D_np = D.detach().cpu().numpy()
    R_np = R.detach().cpu().numpy().copy()  # .copy() prevents in-place mutation of saved autograd tensor
    E_np = _softdtw_backward_cpu_np(D_np, R_np, float(gamma), float(bandwidth))
    return torch.from_numpy(E_np).to(D.device).type_as(D).contiguous()
