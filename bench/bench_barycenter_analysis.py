"""Analysis table: SoftDTW barycenter (softdba) vs classic DBA (dtw_barycenter).

This is an *analysis* comparison, not a like-for-like speed contest: the two
methods optimize different objectives (a smooth, differentiable soft-DTW energy
vs. a hard-alignment fixed point). To make them comparable we cross-evaluate
every resulting barycenter on ONE shared yardstick -- the mean exact-DTW cost to
the set -- and also report each method's own soft-DTW energy.

Per method (softdba fused, softdba unfused, DBA) and per (K, L, D):
  * wall time (s), peak GPU memory (MB)
  * mean exact-DTW cost of the barycenter to the K inputs  (shared yardstick)
  * mean soft-DTW energy to the K inputs                   (soft objective)

Run on a single CUDA GPU.
"""
from __future__ import annotations

import json
import math
import os
import time

import pandas as pd
import torch

from softdtw_cuda import SoftDTW, DTW, softdtw_barycenter, dtw_barycenter

HERE = os.path.dirname(os.path.abspath(__file__))

GAMMA = 1.0
SOFT_ITERS = 100
DBA_ITERS = 30
CONFIGS = [(32, 128, 1), (32, 512, 1)]     # (K, L, D)


def make_data(K, L, D, seed=0):
    """K warped, noisy sine sequences of length L (a simple averaging testbed)."""
    g = torch.Generator().manual_seed(seed)
    t = torch.linspace(0, 4 * math.pi, L)
    seqs = []
    for _ in range(K):
        phase = torch.rand(1, generator=g).item() * 2 * math.pi
        warp = 0.8 + 0.4 * torch.rand(1, generator=g).item()
        amp = 0.7 + 0.6 * torch.rand(1, generator=g).item()
        base = amp * torch.sin(warp * t + phase)
        noise = 0.05 * torch.randn(L, generator=g)
        seq = (base + noise).unsqueeze(-1).repeat(1, D)          # (L, D)
        seqs.append(seq)
    return torch.stack(seqs, 0).cuda()                            # (K, L, D)


def eval_barycenter(bary, X):
    """Mean exact-DTW cost (shared yardstick) and mean soft-DTW energy to the set."""
    K = X.shape[0]
    b = bary.unsqueeze(0).expand(K, -1, -1).contiguous()
    with torch.no_grad():
        dtw_cost = DTW(dist="sqeuclidean", fused=True)(b, X).mean().item()
        soft = SoftDTW(gamma=GAMMA, dist="sqeuclidean", fused=True)(b, X).mean().item()
    return dtw_cost, soft


def run_method(name, fn, X):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.time()
    bary = fn(X)
    torch.cuda.synchronize()
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    dtw_cost, soft = eval_barycenter(bary, X)
    return {"method": name, "wall_s": wall, "peak_mb": peak,
            "mean_dtw_cost": dtw_cost, "mean_softdtw": soft}


def build_methods():
    return [
        ("softdba_fused",
         lambda X: softdtw_barycenter(X, gamma=GAMMA, max_iter=SOFT_ITERS, lr=0.1,
                                      fused=True, early_stopping=False)),
        ("softdba_unfused",
         lambda X: softdtw_barycenter(X, gamma=GAMMA, max_iter=SOFT_ITERS, lr=0.1,
                                      fused=False, early_stopping=False)),
        ("dba_exact",
         lambda X: dtw_barycenter(X, max_iter=DBA_ITERS)),
    ]


def warmup():
    """Trigger numba JIT (softdtw fused/unfused kernels + the njit DBA loop) and
    the scoring kernels on a tiny input, so timed runs exclude compilation."""
    print("Warming up (JIT compile) ...")
    Xw = make_data(4, 16, 1, seed=999)
    for name, fn in build_methods():
        try:
            bary = fn(Xw)
            eval_barycenter(bary, Xw)
        except Exception as ex:                    # pragma: no cover
            print(f"  warmup skip {name}: {ex}")
    torch.cuda.synchronize()


def run_config(K, L, D):
    X = make_data(K, L, D)
    rows = []
    for name, fn in build_methods():
        print(f"    {name} ...")
        r = run_method(name, fn, X)
        r.update(K=K, L=L, D=D)
        rows.append(r)
    del X
    torch.cuda.empty_cache()
    return rows


def to_markdown(df, meta):
    lines = [f"# Barycenter analysis: softdba vs classic DBA ({meta['device']})", "",
             f"torch {meta['torch']}, CUDA {meta['cuda']}, gamma={GAMMA}, "
             f"softdba {SOFT_ITERS} iters, DBA {DBA_ITERS} iters. "
             "`mean_dtw_cost` is the shared yardstick (lower = tighter alignment).", ""]
    for (K, L, D), sub in df.groupby(["K", "L", "D"]):
        lines.append(f"### K={K}, L={L}, D={D}")
        lines.append("| Method | Wall (s) | Peak mem (MB) | Mean DTW cost | Mean soft-DTW |")
        lines.append("|---|--:|--:|--:|--:|")
        for _, r in sub.iterrows():
            lines.append(f"| {r.method} | {r.wall_s:.2f} | {r.peak_mb:.0f} | "
                         f"{r.mean_dtw_cost:.3f} | {r.mean_softdtw:.3f} |")
        lines.append("")
    return "\n".join(lines)


def main():
    assert torch.cuda.is_available(), "CUDA GPU required"
    torch.manual_seed(0)
    meta = {"device": torch.cuda.get_device_name(0),
            "torch": torch.__version__, "cuda": torch.version.cuda, "gamma": GAMMA}
    print(f"Device: {meta['device']} | torch {meta['torch']} | CUDA {meta['cuda']}")

    warmup()

    rows = []
    for (K, L, D) in CONFIGS:
        print(f"Running K={K} L={L} D={D} ...")
        rows.extend(run_config(K, L, D))

    df = pd.DataFrame(rows)
    print("\n===== RAW RESULTS =====\n")
    print(df.to_string(index=False))
    df.to_csv(os.path.join(HERE, "bench_barycenter_analysis.csv"), index=False)
    with open(os.path.join(HERE, "bench_barycenter_analysis_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    md = to_markdown(df, meta)
    with open(os.path.join(HERE, "bench_barycenter_analysis_table.md"), "w") as f:
        f.write(md)
    print("\n" + md)
    print("Saved: bench_barycenter_analysis.csv, _table.md, _meta.json")


if __name__ == "__main__":
    main()
