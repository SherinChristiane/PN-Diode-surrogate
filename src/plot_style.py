"""Shared matplotlib style so every figure in the repo looks consistent."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
BLUE = "#2a78d6"     # physics-constrained surrogate
ORANGE = "#eb6834"   # plain MLP baseline
AQUA = "#1baf7a"     # analytic / reference
TRUTH = "#0b0b0b"    # DEVSIM simulation

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "lines.linewidth": 2,
    "legend.frameon": False,
    "font.size": 10,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
})
