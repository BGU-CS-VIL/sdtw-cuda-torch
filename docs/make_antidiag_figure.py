"""Anti-diagonal wavefront + 3-buffer streaming figure for the fused DTW kernel."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
from matplotlib.lines import Line2D

# palette (light, README-friendly)
GRAY = "#DAD8CF"; TEAL = "#7FD3B6"; PURP = "#B7B2F0"; CORAL = "#EFA98C"; PEND = "#F3F2EE"
E_GRAY="#9a988f"; E_TEAL="#0F6E56"; E_PURP="#534AB7"; E_CORAL="#993C1D"; E_PEND="#cfcdc6"
INK = "#2C2C2A"

N, M = 5, 7                      # rows i (0..4), cols j (0..6)
P = 5                            # active anti-diagonal p = i + j
def cls(i, j):
    p = i + j
    if p < P - 2:  return GRAY, E_GRAY
    if p == P - 2: return TEAL, E_TEAL
    if p == P - 1: return PURP, E_PURP
    if p == P:     return CORAL, E_CORAL
    return PEND, E_PEND

fig = plt.figure(figsize=(11.6, 5.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.16)
axg = fig.add_subplot(gs[0]); axb = fig.add_subplot(gs[1])

# ---- left: DP grid / wavefront ----
for i in range(N):
    for j in range(M):
        fc, ec = cls(i, j)
        axg.add_patch(Rectangle((j - 0.46, -i - 0.46), 0.92, 0.92,
                                facecolor=fc, edgecolor=ec, linewidth=1.2))
# index labels
for j in range(M):
    axg.text(j, 0.75, str(j), ha="center", va="bottom", fontsize=9, color=INK)
for i in range(N):
    axg.text(-0.72, -i, str(i), ha="right", va="center", fontsize=9, color=INK)
axg.text(3, 1.25, "cols  j  (M = 7)", ha="center", fontsize=10, color=INK)
axg.text(-1.15, -2, "rows  i  (N = 5)", ha="center", va="center", rotation=90, fontsize=10, color=INK)

# target cell (2,3) + its three predecessors
tgt = (2, 3)
axg.add_patch(Rectangle((tgt[1]-0.46, -tgt[0]-0.46), 0.92, 0.92,
                        facecolor="none", edgecolor=INK, linewidth=2.4))
preds = [((1, 2), E_TEAL), ((1, 3), E_PURP), ((2, 2), E_PURP)]
tc = (tgt[1], -tgt[0])
for (pi, pj), col in preds:
    pc = (pj, -pi)
    axg.add_patch(FancyArrowPatch(pc, tc, arrowstyle="-|>", mutation_scale=13,
                                  shrinkA=16, shrinkB=18, lw=1.8, color=col))
axg.text(3.0, -2.0, "min\nof 3", ha="center", va="center", fontsize=8.5,
         color=INK, zorder=5)

axg.set_title("Anti-diagonal wavefront:  cells on  p = i + j  are independent\n"
              "→ one GPU thread each (tiled across blocks, no 1024-thread cap)",
              fontsize=11)
axg.set_xlim(-1.5, M - 0.3); axg.set_ylim(-N + 0.3, 1.7)
axg.set_aspect("equal"); axg.axis("off")

# legend under the grid
leg = [Line2D([0],[0], marker="s", ls="", ms=11, mfc=CORAL, mec=E_CORAL, label="p — active wavefront"),
       Line2D([0],[0], marker="s", ls="", ms=11, mfc=PURP, mec=E_PURP, label="p − 1  (prev1)"),
       Line2D([0],[0], marker="s", ls="", ms=11, mfc=TEAL, mec=E_TEAL, label="p − 2  (prev2)"),
       Line2D([0],[0], marker="s", ls="", ms=11, mfc=GRAY, mec=E_GRAY, label="computed"),
       Line2D([0],[0], marker="s", ls="", ms=11, mfc=PEND, mec=E_PEND, label="pending")]
axg.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.16),
           ncol=3, frameon=False, fontsize=9, handletextpad=0.4, columnspacing=1.2)

# ---- right: three rolling buffers ----
rows = [("prev2  (p−2)", TEAL, E_TEAL, 2), ("prev1  (p−1)", PURP, E_PURP, 1),
        ("curr   (p)",   CORAL, E_CORAL, 0)]
for label, fc, ec, y in rows:
    for k in range(N):                       # N = min(N, M) slots
        axb.add_patch(Rectangle((k - 0.44, y - 0.34), 0.88, 0.68,
                                facecolor=fc, edgecolor=ec, linewidth=1.2))
    axb.text(-0.9, y, label, ha="right", va="center", fontsize=9.5, color=INK)
# rotation arrow (curr -> prev1 -> prev2)
axb.add_patch(FancyArrowPatch((N + 0.15, 0), (N + 0.15, 1.85), arrowstyle="-|>",
                              mutation_scale=15, lw=1.6, color=INK,
                              connectionstyle="arc3,rad=-0.35"))
axb.text(N + 0.75, 1.0, "rotate\nafter each\ndiagonal", ha="left", va="center",
         fontsize=8.6, color=INK)
axb.text(2, -0.95, "each buffer:  min(N, M)  slots", ha="center", fontsize=9, color=INK)
axb.set_title("Streaming: keep only three anti-diagonals\n"
              "→  O(B · min(N, M))  peak memory  (no  (B, N, M)  DP table)", fontsize=11)
axb.set_xlim(-2.4, N + 1.9); axb.set_ylim(-1.5, 2.75)
axb.set_aspect("equal"); axb.axis("off")

import os; fig.savefig(os.path.join(os.path.dirname(__file__), "antidiag_streaming.png"), dpi=200, bbox_inches="tight", facecolor="white")
print("saved antidiag_fig.png")
