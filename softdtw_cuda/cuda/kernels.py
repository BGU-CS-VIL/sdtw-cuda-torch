from __future__ import annotations

import math
from numba import cuda


@cuda.jit
def softdtw_forward_diag_sqeuclid_cuda(X, Y, R, gamma, bandwidth, N, M, D, p):
    b = cuda.blockIdx.y
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    i_min = max(0, p - (M - 1))
    i_max = min(N - 1, p)
    diag_len = i_max - i_min + 1
    if t >= diag_len:
        return

    i = i_min + t
    j = p - i

    ip = i + 1
    jp = j + 1

    if bandwidth > 0 and abs(i - j) > bandwidth:
        return

    # cost = ||X[b,i,:] - Y[b,j,:]||^2
    cost = 0.0
    for k in range(D):
        diff = X[b, i, k] - Y[b, j, k]
        cost += diff * diff

    inv_gamma = 1.0 / gamma

    r0 = -R[b, ip - 1, jp - 1] * inv_gamma
    r1 = -R[b, ip - 1, jp]     * inv_gamma
    r2 = -R[b, ip,     jp - 1] * inv_gamma

    rmax = r0
    if r1 > rmax: rmax = r1
    if r2 > rmax: rmax = r2

    rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
    softmin = -gamma * (math.log(rsum) + rmax)

    R[b, ip, jp] = cost + softmin


@cuda.jit
def softdtw_backward_log_diag_sqeuclid_cuda(X, Y, R, logE, inv_gamma, bandwidth, N, M, D, p):
    b = cuda.blockIdx.y
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    i_min = max(0, p - (M - 1))
    i_max = min(N - 1, p)
    diag_len = i_max - i_min + 1
    if t >= diag_len:
        return

    i = i_min + t
    j = p - i

    ip = i + 1
    jp = j + 1

    if bandwidth > 0 and abs(i - j) > bandwidth:
        return

    Rij = R[b, ip, jp]
    if math.isinf(Rij):
        Rij = -math.inf

    # costs for transitions:
    # D_pad[ip+1, jp]   corresponds to ||X[i+1] - Y[j]||^2
    # D_pad[ip, jp+1]   corresponds to ||X[i]   - Y[j+1]||^2
    # D_pad[ip+1, jp+1] corresponds to ||X[i+1] - Y[j+1]||^2

    # cost_down: (i+1, j)
    cost_down = 0.0
    if i + 1 < N:
        for k in range(D):
            diff = X[b, i + 1, k] - Y[b, j, k]
            cost_down += diff * diff
    else:
        # this state will be invalid anyway due to R boundary = -inf
        cost_down = 0.0

    # cost_right: (i, j+1)
    cost_right = 0.0
    if j + 1 < M:
        for k in range(D):
            diff = X[b, i, k] - Y[b, j + 1, k]
            cost_right += diff * diff
    else:
        cost_right = 0.0

    # cost_diag: (i+1, j+1)
    cost_diag = 0.0
    if (i + 1 < N) and (j + 1 < M):
        for k in range(D):
            diff = X[b, i + 1, k] - Y[b, j + 1, k]
            cost_diag += diff * diff
    else:
        cost_diag = 0.0

    la = (R[b, ip + 1, jp]     - Rij - cost_down)  * inv_gamma
    lb = (R[b, ip,     jp + 1] - Rij - cost_right) * inv_gamma
    lc = (R[b, ip + 1, jp + 1] - Rij - cost_diag)  * inv_gamma

    t1 = logE[b, ip + 1, jp]     + la
    t2 = logE[b, ip,     jp + 1] + lb
    t3 = logE[b, ip + 1, jp + 1] + lc

    # reuse your helper if you want (recommended):
    m = t1
    if t2 > m: m = t2
    if t3 > m: m = t3

    if m == -math.inf:
        logE[b, ip, jp] = -math.inf
    else:
        logE[b, ip, jp] = m + math.log(math.exp(t1 - m) + math.exp(t2 - m) + math.exp(t3 - m))



@cuda.jit
def softdtw_forward_kernel(D, gamma, bandwidth, max_i, max_j, n_passes, R):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    I = tid
    inv_gamma = 1.0 / gamma

    for p in range(n_passes):
        J = max(0, min(p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        if I + J == p and (I < max_i and J < max_j):
            if not (abs(i - j) > bandwidth > 0):
                r0 = -R[b, i - 1, j - 1] * inv_gamma
                r1 = -R[b, i - 1, j] * inv_gamma
                r2 = -R[b, i, j - 1] * inv_gamma
                rmax = max(max(r0, r1), r2)
                rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
                softmin = -gamma * (math.log(rsum) + rmax)
                R[b, i, j] = D[b, i - 1, j - 1] + softmin
        cuda.syncthreads()

@cuda.jit
def softdtw_forward_diag_cuda(D, R, gamma, bandwidth, N, M, p):
    b = cuda.blockIdx.y  # batch in Y
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    # diagonal bounds in unpadded coordinates
    i_min = max(0, p - (M - 1))
    i_max = min(N - 1, p)

    diag_len = i_max - i_min + 1
    if t >= diag_len:
        return

    i = i_min + t
    j = p - i

    ip = i + 1
    jp = j + 1

    # bandwidth pruning (in padded coords uses ip/jp, but difference same)
    if bandwidth > 0 and abs(ip - jp) > bandwidth:
        return

    inv_gamma = 1.0 / gamma

    r0 = -R[b, ip - 1, jp - 1] * inv_gamma
    r1 = -R[b, ip - 1, jp]     * inv_gamma
    r2 = -R[b, ip,     jp - 1] * inv_gamma

    rmax = r0
    if r1 > rmax: rmax = r1
    if r2 > rmax: rmax = r2

    rsum = math.exp(r0 - rmax) + math.exp(r1 - rmax) + math.exp(r2 - rmax)
    softmin = -gamma * (math.log(rsum) + rmax)

    R[b, ip, jp] = D[b, i, j] + softmin



@cuda.jit
def softdtw_backward_kernel_legacy(D_pad, R, inv_gamma, bandwidth, max_i, max_j, n_passes, E):
    b = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    I = tid

    for p in range(n_passes):
        rev_p = n_passes - p - 1
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        if I + J == rev_p and (I < max_i and J < max_j):
            if math.isinf(R[b, i, j]):
                R[b, i, j] = -math.inf

            if not (abs(i - j) > bandwidth > 0):
                # NOTE: this is the baseline (numerically unsafe). We'll replace with stabilized/log-space soon.
                a = math.exp((R[b, i + 1, j] - R[b, i, j] - D_pad[b, i + 1, j]) * inv_gamma)
                bb = math.exp((R[b, i, j + 1] - R[b, i, j] - D_pad[b, i, j + 1]) * inv_gamma)
                c = math.exp((R[b, i + 1, j + 1] - R[b, i, j] - D_pad[b, i + 1, j + 1]) * inv_gamma)
                E[b, i, j] = E[b, i + 1, j] * a + E[b, i, j + 1] * bb + E[b, i + 1, j + 1] * c
        cuda.syncthreads()

@cuda.jit(device=True, inline=True)
def _logsumexp3(a, b, c):
    m = a
    if b > m: m = b
    if c > m: m = c
    if m == -math.inf:
        return -math.inf
    return m + math.log(math.exp(a - m) + math.exp(b - m) + math.exp(c - m))


@cuda.jit
def softdtw_backward_log_cuda(D, R, inv_gamma, bandwidth, max_i, max_j, n_passes, logE):
    """
    D: (B, N+2, M+2) padded
    R: (B, N+2, M+2) padded (with boundary conditions already set)
    logE: (B, N+2, M+2) padded, initialized to -inf with logE[:,-1,-1]=0
    """
    k = cuda.blockIdx.x
    tid = cuda.threadIdx.x

    I = tid

    for p in range(n_passes):
        rev_p = n_passes - p - 1
        J = max(0, min(rev_p - tid, max_j - 1))

        i = I + 1
        j = J + 1

        if I + J == rev_p and (I < max_i and J < max_j):

            # pruning
            if not (abs(i - j) > bandwidth > 0):

                Rij = R[k, i, j]
                if math.isinf(Rij):
                    Rij = -math.inf

                # log transition weights (no exp here!)
                la = (R[k, i + 1, j]     - Rij - D[k, i + 1, j])     * inv_gamma
                lb = (R[k, i, j + 1]     - Rij - D[k, i, j + 1])     * inv_gamma
                lc = (R[k, i + 1, j + 1] - Rij - D[k, i + 1, j + 1]) * inv_gamma

                t1 = logE[k, i + 1, j]     + la
                t2 = logE[k, i, j + 1]     + lb
                t3 = logE[k, i + 1, j + 1] + lc

                logE[k, i, j] = _logsumexp3(t1, t2, t3)

        cuda.syncthreads()

@cuda.jit
def softdtw_backward_log_diag_cuda(Dp, R, logE, inv_gamma, bandwidth, N, M, p):
    b = cuda.blockIdx.y
    t = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x

    i_min = max(0, p - (M - 1))
    i_max = min(N - 1, p)
    diag_len = i_max - i_min + 1
    if t >= diag_len:
        return

    i = i_min + t
    j = p - i

    ip = i + 1
    jp = j + 1

    # pruning
    if bandwidth > 0 and abs(i - j) > bandwidth:
        return

    Rij = R[b, ip, jp]
    if math.isinf(Rij):
        Rij = -math.inf

    la = (R[b, ip + 1, jp]     - Rij - Dp[b, ip + 1, jp])     * inv_gamma
    lb = (R[b, ip, jp + 1]     - Rij - Dp[b, ip, jp + 1])     * inv_gamma
    lc = (R[b, ip + 1, jp + 1] - Rij - Dp[b, ip + 1, jp + 1]) * inv_gamma

    t1 = logE[b, ip + 1, jp]     + la
    t2 = logE[b, ip, jp + 1]     + lb
    t3 = logE[b, ip + 1, jp + 1] + lc

    m = t1
    if t2 > m: m = t2
    if t3 > m: m = t3

    if m == -math.inf:
        logE[b, ip, jp] = -math.inf
    else:
        logE[b, ip, jp] = _logsumexp3(t1, t2, t3)



