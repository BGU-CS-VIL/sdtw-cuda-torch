import math
import os

import matplotlib.pyplot as plt
import pandas as pd
import torch

from softdtw_cuda import SoftDTW

# ---------------------------------------------------------------------
# Import Maghoumi implementation (you copy-pasted soft_dtw_cuda.py)
# ---------------------------------------------------------------------
from soft_dtw_cuda import SoftDTW as MaghoumiSoftDTW

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------
# Utils
# ---------------------------------------------------------------------
def peak_mb():
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def time_ms(fn, iters=5):
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    starter.record()
    for _ in range(iters):
        fn()
    ender.record()
    torch.cuda.synchronize()
    return starter.elapsed_time(ender) / iters


def run_one(fn_step):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()

    # warmup (JIT, kernel caching, etc.)
    fn_step()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    ms = time_ms(fn_step)
    mb = peak_mb()
    return mb, ms


def _make_step(sdtw, X, Y):
    """Return a zero-arg step function that closes over sdtw, X, Y."""
    def step():
        x = X.detach().clone().requires_grad_(True)
        y = Y.detach().clone().requires_grad_(True)
        loss = sdtw(x, y).sum()
        loss.backward()
    return step


# ---------------------------------------------------------------------
# Per-config benchmark (runs each of the three variants once)
# ---------------------------------------------------------------------
def run_config(B, N, D, gamma, bandwidth):
    M = N
    X = torch.randn(B, N, D, device="cuda", requires_grad=True)
    Y = torch.randn(B, M, D, device="cuda", requires_grad=True)

    # -----------------------------------------------------------------
    # Maghoumi – limited to max(N, M) <= 1024
    # -----------------------------------------------------------------
    mag_mb, mag_ms = math.nan, math.nan
    if max(N, M) <= 1024:
        try:
            mag = MaghoumiSoftDTW(
                use_cuda=True, gamma=gamma, normalize=False, bandwidth=bandwidth
            )
            step_mag = _make_step(mag, X, Y)
            mag_mb, mag_ms = run_one(step_mag)
        except RuntimeError:
            pass  # OOM or other CUDA failure → remains nan

    # -----------------------------------------------------------------
    # Ours (unfused)
    # -----------------------------------------------------------------
    ours_unfused = SoftDTW(
        gamma=gamma, bandwidth=bandwidth, dist="sqeuclidean",
        fused=False, normalize=False,
    )
    ours_unfused_mb, ours_unfused_ms = run_one(_make_step(ours_unfused, X, Y))

    # -----------------------------------------------------------------
    # Ours (fused)
    # -----------------------------------------------------------------
    ours_fused = SoftDTW(
        gamma=gamma, bandwidth=bandwidth, dist="sqeuclidean",
        fused=True, normalize=False,
    )
    ours_fused_mb, ours_fused_ms = run_one(_make_step(ours_fused, X, Y))

    return dict(
        B=B, N=N, D=D,
        maghoumi_mb=mag_mb,
        ours_unfused_mb=ours_unfused_mb,
        ours_fused_mb=ours_fused_mb,
        maghoumi_ms=mag_ms,
        ours_unfused_ms=ours_unfused_ms,
        ours_fused_ms=ours_fused_ms,
        fused_saving_mb=(
            mag_mb - ours_fused_mb if not math.isnan(mag_mb) else math.nan
        ),
        fused_saving_pct=(
            100.0 * (mag_mb - ours_fused_mb) / mag_mb
            if not math.isnan(mag_mb) else math.nan
        ),
    )


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------
COLORS  = {"Maghoumi": "tab:red",   "Unfused": "tab:blue",  "Fused": "tab:green"}
MARKERS = {"Maghoumi": "s",         "Unfused": "^",         "Fused": "o"}


def _plot_metric(ax, xs, results_list, metric_key, xlabel, ylabel, title, nan_label_suffix=""):
    for label in ("Maghoumi", "Unfused", "Fused"):
        key = {
            "Maghoumi": f"maghoumi_{metric_key}",
            "Unfused":  f"ours_unfused_{metric_key}",
            "Fused":    f"ours_fused_{metric_key}",
        }[label]
        ys = [r[key] for r in results_list]
        # Build a display label; annotate if some values are nan
        disp = label
        if any(math.isnan(v) for v in ys):
            disp += nan_label_suffix
        ax.plot(xs, ys, label=disp, color=COLORS[label], marker=MARKERS[label],
                linewidth=1.8, markersize=6)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.4)


# ---------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------
def main():
    assert torch.cuda.is_available()
    torch.manual_seed(0)

    gamma     = 1.0
    bandwidth = None

    # -----------------------------------------------------------------
    # Build the full set of unique (B, N, D) configs (no duplicates)
    # -----------------------------------------------------------------

    # Original experiment grid (kept unchanged)
    orig_grid = [
        (B, N, D)
        for B in [16, 32]
        for N in [128, 512, 1024, 2048]
        for D in [1, 64]
    ]

    # Vary sequence length  – B=32, D=128
    vary_L_grid = [(32, L, 128) for L in [64, 128, 256, 512, 1024, 2048]]

    # Vary feature dimension – B=32, N=256
    vary_D_grid = [(32, 256, D) for D in [8, 16, 32, 64, 128, 256]]

    # Union, preserving first-seen order
    seen, all_configs = set(), []
    for cfg in orig_grid + vary_L_grid + vary_D_grid:
        if cfg not in seen:
            seen.add(cfg)
            all_configs.append(cfg)

    # -----------------------------------------------------------------
    # Run every unique config exactly once
    # -----------------------------------------------------------------
    results: dict[tuple, dict] = {}
    for B, N, D in all_configs:
        print(f"Running B={B:3d}, N={N:5d}, D={D:4d} …")
        results[(B, N, D)] = run_config(B, N, D, gamma, bandwidth)

    # -----------------------------------------------------------------
    # Original table & CSV (unchanged behaviour)
    # -----------------------------------------------------------------
    orig_rows = [results[cfg] for cfg in orig_grid]
    df = pd.DataFrame(orig_rows)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", None)
    print("\n===== BENCHMARK RESULTS =====\n")
    print(df)

    out_csv = os.path.join(HERE, "softdtw_memory_benchmark.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nSaved results to: {out_csv}")

    # -----------------------------------------------------------------
    # Plots  (2 × 2: memory & runtime × vary-L & vary-D)
    # -----------------------------------------------------------------
    vary_L_rows = [results[cfg] for cfg in vary_L_grid]
    vary_D_rows = [results[cfg] for cfg in vary_D_grid]
    Ls = [cfg[1] for cfg in vary_L_grid]
    Ds = [cfg[2] for cfg in vary_D_grid]

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    fig.suptitle("SoftDTW Benchmark  (B = 32)", fontsize=13, fontweight="bold")

    # Row 0 – Peak GPU RAM
    _plot_metric(
        axes[0, 0], Ls, vary_L_rows,
        metric_key="mb",
        xlabel="Sequence Length (L)",
        ylabel="Peak GPU RAM (MB)",
        title="Memory vs. Length  (D = 128)",
        nan_label_suffix=" [N/A > 1024]",
    )
    _plot_metric(
        axes[0, 1], Ds, vary_D_rows,
        metric_key="mb",
        xlabel="Feature Dimension (D)",
        ylabel="Peak GPU RAM (MB)",
        title="Memory vs. Dimension  (L = 256)",
    )

    # Row 1 – Runtime
    _plot_metric(
        axes[1, 0], Ls, vary_L_rows,
        metric_key="ms",
        xlabel="Sequence Length (L)",
        ylabel="Runtime (ms)",
        title="Runtime vs. Length  (D = 128)",
        nan_label_suffix=" [N/A > 1024]",
    )
    _plot_metric(
        axes[1, 1], Ds, vary_D_rows,
        metric_key="ms",
        xlabel="Feature Dimension (D)",
        ylabel="Runtime (ms)",
        title="Runtime vs. Dimension  (L = 256)",
    )

    plt.tight_layout()
    out_plot = os.path.join(HERE, "benchmark_plots.png")
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    print(f"Saved plots to:   {out_plot}")
    plt.close()


if __name__ == "__main__":
    main()
