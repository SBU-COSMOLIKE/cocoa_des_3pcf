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
# Each row is one angular bin. The last column is the leave-one-run-out
# standard error of the variance ratio; it excludes reference uncertainty.
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.errorbar(
    x=values[:, 0],
    y=values[:, 1],
    yerr=values[:, 2],
    color="#0072B2",
    marker="o",
    linewidth=2.4,
    capsize=3,
)
# Equality is a reference line, identified by the caption rather than a legend.
ax.axhline(y=1., color="0.5", linestyle=":", linewidth=1.)
ax.set_xlabel("Pair separation [arcmin]", fontsize=11)
ax.set_ylabel("Variance / released variance", fontsize=11)
ax.tick_params(axis="both", labelsize=10)
fig.tight_layout()
fig.savefig(fname=folder / "variance.pdf")
fig.savefig(fname=folder / "variance.png", dpi=300)
plt.close(fig)
