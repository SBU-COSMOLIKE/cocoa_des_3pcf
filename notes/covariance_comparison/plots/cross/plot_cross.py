"""Redraw the comparison from the adjacent, recorded text table.

The table stores measured summary statistics, not theoretical predictions.
The PDF note and companion notebook explain the selected maps and errors.
Run this file from any directory; only its adjacent data.txt is needed.
The output PDF is used by the note and the PNG permits visual inspection.
"""

from pathlib import Path
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

# Figure style of the whole notebook: the shared plotting functions never change
# rcParams, so these lines set every font, tick and grid default.
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
values = np.loadtxt(fname=folder / "data.txt")
# Columns alternate means and their standard errors, each multiplied by
# one million. The eight same-run pairs share the primary maps in the auto
# curve. The different-run control pairs those maps with eight other runs.
fig, ax = plt.subplots(figsize=(5.4, 2.6))
for column, name, color, marker, width, style in (
    (1, "Same map", "#0072B2", "o", 2.4, "-"),
    (3, "Same run, new draw", "#E69F00", "s", 1.8, "--"),
    (5, "Different run", "#CC79A7", "D", 1.3, ":"),
):
    ax.errorbar(
        x=values[:, 0],
        y=values[:, column],
        yerr=values[:, column + 1],
        color=color,
        marker=marker,
        linewidth=width,
        linestyle=style,
        capsize=3,
        label=name,
    )
ax.axhline(y=0., color="0.5", linestyle=":", linewidth=1.)
ax.set_xlabel("Pair separation [arcmin]", fontsize=11)
ax.set_ylabel(r"$10^6\,\xi_+$", fontsize=11)
ax.tick_params(axis="both", labelsize=10)
# A dedicated strip keeps every legend entry above the data. Check that
# placement in display coordinates, rather than trusting a chosen corner.
legend = ax.legend(
    loc="lower center",
    bbox_to_anchor=(0.5, 1.02),
    ncol=3,
    frameon=False,
    fontsize=9,
)
fig.tight_layout()
fig.canvas.draw()
assert legend.get_window_extent().y0 > ax.get_window_extent().y1
fig.savefig(fname=folder / "cross.pdf")
fig.savefig(fname=folder / "cross.png", dpi=300)
plt.close(fig)
