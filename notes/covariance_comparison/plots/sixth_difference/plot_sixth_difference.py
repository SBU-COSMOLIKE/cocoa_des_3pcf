"""Redraw this covariance comparison from its two adjacent text matrices.

Each matrix uses the same row definitions. This file needs only NumPy and
Matplotlib; it does not load maps, a notebook, or the CosmoLike library.
The note and notebook explain how the input matrices were estimated.
Run this file from any directory to regenerate its PDF and PNG.
"""

from pathlib import Path
import numpy as np
import matplotlib
from matplotlib import pyplot as plt

# Use the same figure defaults as the project notebooks so that standalone
# redrawing preserves their fonts, ticks and line weights.
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
first = np.loadtxt(fname=folder / "first.txt")
second = np.loadtxt(fname=folder / "second.txt")
assert first.shape == second.shape
assert np.all(np.diag(first) > 0.)
assert np.all(np.diag(second) > 0.)
# Divide by diagonal standard-deviation products. Dividing by individual
# off-diagonal covariance entries would explode where correlations cross zero.
scale = np.sqrt(np.outer(np.diag(second), np.diag(second)))
difference = 100. * (first - second) / scale
# Keep second- and sixth-order color scales and histogram bins identical.
limit = 27.
assert np.max(a=np.abs(difference)) <= limit
fig = plt.figure(figsize=(5.2, 6.2), constrained_layout=True)
grid = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[1., 0.42])
ax = fig.add_subplot(grid[0])
artist = ax.imshow(
    X=difference,
    origin="lower",
    cmap="RdBu_r",
    vmin=-limit,
    vmax=limit,
    interpolation="nearest",
)
fig.colorbar(mappable=artist, ax=ax, label="Normalized covariance difference [%]")
histogram = fig.add_subplot(grid[1])
# The histogram includes all entries, including symmetric duplicates. It
# describes this matrix; its bars are not independent statistical samples.
histogram.hist(
    x=difference.ravel(),
    bins=np.linspace(start=-limit, stop=limit, num=33),
    histtype="step",
    color="#0072B2",
    linewidth=2.4,
)
histogram.set_xlabel("Normalized covariance difference [%]", fontsize=12)
histogram.set_ylabel("Matrix entries", fontsize=12)
ax.set_title(label="Sixth-order Eifler minus direct simulations", fontsize=12)
ax.set_xticks(ticks=[3.5, 11.5], labels=['Squared apertures', 'Cubed apertures'], rotation=35, ha="right")
ax.set_yticks(ticks=[3.5, 11.5], labels=['Squared apertures', 'Cubed apertures'])
# Eight squared-moment rows precede eight cubed-moment rows. The boundary
# marks a change of observable, not a change of matrix normalization.
ax.axhline(y=7.5, color="0.25", linewidth=0.6)
ax.axvline(x=7.5, color="0.25", linewidth=0.6)
ax.tick_params(axis="both", labelsize=11)
fig.savefig(fname=folder / "sixth_difference.pdf")
fig.savefig(fname=folder / "sixth_difference.png", dpi=300)
plt.close(fig)
