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
# Normalize EACH matrix with its OWN diagonal. The two halves therefore
# compare correlation coefficients rather than raw covariances of mixed units.
first_correlation = first / np.sqrt(np.outer(np.diag(first), np.diag(first)))
second_correlation = second / np.sqrt(np.outer(np.diag(second), np.diag(second)))
combined = first_correlation.copy()
upper_indices = np.triu_indices(n=len(first), k=1)
combined[upper_indices] = second_correlation[upper_indices]
fig, ax = plt.subplots(figsize=(5.4, 4.8), constrained_layout=True)
artist = ax.imshow(
    X=combined,
    origin="lower",
    cmap="RdBu_r",
    vmin=-1.,
    vmax=1.,
    interpolation="nearest",
)
fig.colorbar(mappable=artist, ax=ax, label="Correlation coefficient")
# With origin='lower', row>column appears ABOVE-LEFT of the diagonal on
# the displayed image. Name the visible halves, not the matrix-index halves.
ax.set_title(label="Upper-left: Second-order Eifler\nLower-right: Direct simulations", fontsize=12)
ax.set_xticks(ticks=[3.5, 11.5], labels=['Squared apertures', 'Cubed apertures'], rotation=35, ha="right")
ax.set_yticks(ticks=[3.5, 11.5], labels=['Squared apertures', 'Cubed apertures'])
# Eight squared-moment rows precede eight cubed-moment rows. The boundary
# marks a change of observable, not a change of matrix normalization.
ax.axhline(y=7.5, color="0.25", linewidth=0.6)
ax.axvline(x=7.5, color="0.25", linewidth=0.6)
ax.tick_params(axis="both", labelsize=11)
fig.savefig(fname=folder / "all_triangles.pdf")
fig.savefig(fname=folder / "all_triangles.png", dpi=300)
plt.close(fig)
