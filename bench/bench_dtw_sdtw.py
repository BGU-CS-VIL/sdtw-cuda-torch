"""Strengths benchmark: SoftDTW & exact DTW (ours, fused/unfused) vs Maghoumi.

Maghoumi implements SoftDTW only, and only on GPU for max(N, M) <= 1024, so it
appears in the SoftDTW rows and is marked N/A once the sequence passes the 1024
cap (its CUDA path silently falls back to CPU there). The grid is chosen to make
the library's strengths legible:

  * Fused SoftDTW    -> large peak-memory reduction at high D (no (B,N,M) cost
                        tensor), and it keeps running past N=1024 where Maghoumi
                        cannot stay on the GPU.
  * Unfused SoftDTW  -> the fast path; competitive with / faster than Maghoumi
                        where Maghoumi runs.
  * Exact DTW (fused)-> forward-only, streams three anti-diagonals, so peak
                        memory is O(B*min(N,M)) and stays flat as N grows.
  * Exact DTW (unfused) is shown as the contrast that materializes (B,N,M).

Metrics per (config, method):
  * peak GPU memory (MB), deterministic
  * runtime mean +/- std over `REPS` timed reps after `WARMUP` warmups (ms)

SoftDTW is timed forward+backward (loss usage); DTW is forward-only (its intended
evaluation/retrieval usage). Run on a single CUDA GPU.
"""
from __future__ import annotations

import json
import math
import os
import statistics
import sys

import pandas as pd
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)                      # for the vendored Maghoumi baseline

from softdtw_cuda import SoftDTW, DTW
from soft_dtw_cuda import SoftDTW as MaghoumiSoftDTW

GAMMA = 1.0
WARMUP = 2
REPS = 7
MAGHOUMI_MAX_LEN = 1024                            # its GPU path caps here

# (B, N, D); sequences are square (M = N).
CONFIGS = [
    # length scaling at D=64 (fused memory win + long-sequence capability)
    (16, 256, 64),
    (16, 512, 64),
    (16, 1024, 64),
    (16, 2048, 64),
    (16, 4096, 64),
    # feature-dim scaling at N=512 (fused win grows with D)
    (16, 512, 1),
    (16, 512, 16),
    (16, 512, 128),
    # a heavier batch at the Maghoumi cap
    (32, 1024, 64),
]


def _bench(step, warmup=WARMUP, reps=REPS):
    """Return (mean_ms, std_ms, peak_mb) or (nan, nan, nan) on OOM/failure."""
    try:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        for _ in range(warmup):
            step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(reps):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            s.record()
            step()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
        peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
        std = statistics.pstdev(times) if len(times) > 1 else 0.0
        return statistics.mean(times), std, peak
    except RuntimeError as ex:                     # typically CUDA OOM
        torch.cuda.empty_cache()
        print(f"    [skip] {type(ex).__name__}: {str(ex)[:80]}")
        return math.nan, math.nan, math.nan


def _soft_step(mod, X, Y):
    """Forward + backward wrt X (SoftDTW as a training loss)."""
    def step():
        x = X.detach().clone().requires_grad_(True)
        loss = mod(x, Y).sum()
        loss.backward()
    return step


def _dtw_step(mod, X, Y):
    """Forward-only exact DTW (evaluation / retrieval)."""
    def step():
        mod(X, Y)
    return step


def run_config(B, N, D):
    M = N
    X = torch.randn(B, N, D, device="cuda")
    Y = torch.randn(B, M, D, device="cuda")

    row = {"B": B, "N": N, "D": D}

    # SoftDTW - Maghoumi (GPU only for max(N,M) <= 1024)
    if max(N, M) <= MAGHOUMI_MAX_LEN:
        mag = MaghoumiSoftDTW(use_cuda=True, gamma=GAMMA, normalize=False, bandwidth=None)
        ms, sd, mb = _bench(_soft_step(mag, X, Y))
    else:
        ms, sd, mb = math.nan, math.nan, math.nan   # N/A: no GPU path past 1024
    row.update(sdtw_maghoumi_ms=ms, sdtw_maghoumi_std=sd, sdtw_maghoumi_mb=mb)

    # SoftDTW - ours unfused / fused
    for tag, fused in (("unfused", False), ("fused", True)):
        mod = SoftDTW(gamma=GAMMA, dist="sqeuclidean", fused=fused, normalize=False)
        ms, sd, mb = _bench(_soft_step(mod, X, Y))
        row.update({f"sdtw_{tag}_ms": ms, f"sdtw_{tag}_std": sd, f"sdtw_{tag}_mb": mb})

    # Exact DTW - ours unfused / fused (forward-only)
    for tag, fused in (("unfused", False), ("fused", True)):
        mod = DTW(dist="sqeuclidean", fused=fused)
        ms, sd, mb = _bench(_dtw_step(mod, X, Y))
        row.update({f"dtw_{tag}_ms": ms, f"dtw_{tag}_std": sd, f"dtw_{tag}_mb": mb})

    del X, Y
    torch.cuda.empty_cache()
    return row


def _fmt(ms, sd):
    if math.isnan(ms):
        return "N/A"
    return f"{ms:.1f}±{sd:.1f}"


def _fmt_mb(mb):
    return "N/A" if math.isnan(mb) else f"{mb:.0f}"


def to_markdown(df, meta):
    lines = []
    lines.append(f"# SoftDTW & exact DTW vs Maghoumi ({meta['device']})")
    lines.append("")
    lines.append(f"torch {meta['torch']}, CUDA {meta['cuda']}, gamma={GAMMA}, "
                 f"{REPS} reps (mean±std ms) after {WARMUP} warmups. "
                 f"SoftDTW = forward+backward; DTW = forward-only.")
    lines.append("")
    lines.append("### Peak GPU memory (MB)")
    lines.append("| B | N | D | SoftDTW Maghoumi | SoftDTW unfused | SoftDTW fused | "
                 "DTW unfused | DTW fused |")
    lines.append("|--:|--:|--:|--:|--:|--:|--:|--:|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r.B:.0f} | {r.N:.0f} | {r.D:.0f} | {_fmt_mb(r.sdtw_maghoumi_mb)} | "
            f"{_fmt_mb(r.sdtw_unfused_mb)} | {_fmt_mb(r.sdtw_fused_mb)} | "
            f"{_fmt_mb(r.dtw_unfused_mb)} | {_fmt_mb(r.dtw_fused_mb)} |")
    lines.append("")
    lines.append("### Runtime (ms, mean±std)")
    lines.append("| B | N | D | SoftDTW Maghoumi | SoftDTW unfused | SoftDTW fused | "
                 "DTW unfused | DTW fused |")
    lines.append("|--:|--:|--:|--:|--:|--:|--:|--:|")
    for _, r in df.iterrows():
        lines.append(
            f"| {r.B:.0f} | {r.N:.0f} | {r.D:.0f} | "
            f"{_fmt(r.sdtw_maghoumi_ms, r.sdtw_maghoumi_std)} | "
            f"{_fmt(r.sdtw_unfused_ms, r.sdtw_unfused_std)} | "
            f"{_fmt(r.sdtw_fused_ms, r.sdtw_fused_std)} | "
            f"{_fmt(r.dtw_unfused_ms, r.dtw_unfused_std)} | "
            f"{_fmt(r.dtw_fused_ms, r.dtw_fused_std)} |")
    lines.append("")
    return "\n".join(lines)


def main():
    assert torch.cuda.is_available(), "CUDA GPU required"
    torch.manual_seed(0)
    meta = {
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gamma": GAMMA,
        "reps": REPS,
        "warmup": WARMUP,
    }
    print(f"Device: {meta['device']} | torch {meta['torch']} | CUDA {meta['cuda']}")

    rows = []
    for (B, N, D) in CONFIGS:
        print(f"Running B={B:3d} N={N:5d} D={D:4d} ...")
        rows.append(run_config(B, N, D))

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print("\n===== RAW RESULTS =====\n")
    print(df.to_string(index=False))

    df.to_csv(os.path.join(HERE, "bench_dtw_sdtw.csv"), index=False)
    with open(os.path.join(HERE, "bench_dtw_sdtw_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    md = to_markdown(df, meta)
    with open(os.path.join(HERE, "bench_dtw_sdtw_table.md"), "w") as f:
        f.write(md)
    print("\n" + md)
    print("Saved: bench_dtw_sdtw.csv, bench_dtw_sdtw_table.md, bench_dtw_sdtw_meta.json")


if __name__ == "__main__":
    main()
