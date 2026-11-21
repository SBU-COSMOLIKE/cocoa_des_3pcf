"""Redraw Gaussian/lognormal fourth-order tests from the adjacent small table.

The teaching notebook generates the table from 32,000 independent experiments
per distribution, each with 128 vectors. No simulation maps are loaded here.
"""
from pathlib import Path
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = False
matplotlib.rcParams['ytick.right'] = False
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = '1.0'
matplotlib.rcParams['axes.labelsize'] = 'medium'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'
matplotlib.rcParams["figure.dpi"] = 90

folder = Path(__file__).resolve().parent
data = np.loadtxt(fname=folder / "data.txt")
colors = ["#0072B2", "#E69F00", "#CC79A7", "#000000"]
markers = ["o", "s", "^", "D"]
labels = ["Wishart 2", "Wishart 4", "Distinct pairs 2", "Distinct pairs 4"]
fig, axes = plt.subplots(
    nrows=1,
    ncols=2,
    figsize=(10, 4.2),
    constrained_layout=True,
)
for method, label in enumerate(labels):
    rows = data[data[:, 1] == method]
    positions = rows[:, 0] + (method - 1.5) * 0.13
    axes[0].errorbar(
        x=positions,
        y=rows[:, 2],
        yerr=2. * rows[:, 3],
        fmt=markers[method],
        color=colors[method],
        capsize=3,
        label=label,
    )
    axes[1].plot(
        positions,
        rows[:, 4],
        marker=markers[method],
        linestyle="none",
        color=colors[method],
    )
axes[0].axhline(y=0., color="gray", linestyle=":")
axes[0].set_ylabel("Mean error in first diagonal entry")
axes[0].set_title("Bars: twice the Monte Carlo standard error")
legend = axes[0].legend(
    loc="lower center",
    bbox_to_anchor=(.5, 1.16),
    ncol=2,
    fontsize=9,
    frameon=False,
)
axes[1].set_yscale("log")
axes[1].set_ylabel("Relative RMS matrix error")
axes[1].set_title("Scatter of individual 128-vector estimates")
for ax in axes:
    ax.set_xticks(
        ticks=np.arange(3),
        labels=["Gaussian", "Lognormal\n0.25", "Lognormal\n0.5"],
    )
fig.canvas.draw()
assert not legend.get_window_extent().overlaps(axes[0].get_window_extent())
fig.savefig(fname=folder / "fourth_order.pdf")
fig.savefig(fname=folder / "fourth_order.png", dpi=300)
plt.close(fig)
