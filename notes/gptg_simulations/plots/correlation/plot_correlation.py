"""Redraw the pilot comparison using only the adjacent text records.

The tables are frozen outputs of the map/precision scripts in the note.
No map loading or fit is performed here. See the note for their assumptions.
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

reference = np.loadtxt(fname=folder / 'reference.txt')
generated = np.loadtxt(fname=folder / 'generated.txt')
old_scale = np.sqrt(np.diag(reference))
new_scale = np.sqrt(np.diag(generated))
old = reference / old_scale[:, None] / old_scale[None, :]
new = generated / new_scale[:, None] / new_scale[None, :]
# With origin=lower, row>=column is the displayed upper-left triangle.
# Generated maps occupy that half; the full supplied ensemble is lower-right.
combined = np.where(np.tril(np.ones((16, 16), dtype=bool)), new, old)
fig, ax = plt.subplots(figsize=(4.2, 3.6), constrained_layout=True)
artist = ax.imshow(X=combined, origin='lower', cmap='RdBu_r', vmin=-1., vmax=1.)
fig.colorbar(mappable=artist, ax=ax, label='Correlation')
ax.axhline(y=7.5, color='black', linewidth=.6)
ax.axvline(x=7.5, color='black', linewidth=.6)
ax.set_xticks(ticks=[3.5, 11.5], labels=['Squared apertures', 'Cubed apertures'])
ax.set_yticks(ticks=[3.5, 11.5], labels=['Squared apertures', 'Cubed apertures'])
ax.tick_params(axis='both', labelsize=8)
ax.set_title('Upper-left: generated\nLower-right: Rafael full ensemble', fontsize=10)
NAME = 'correlation.pdf'

fig.savefig(fname=folder / NAME, dpi=300)
fig.savefig(fname=folder / NAME.replace('.pdf', '.png'), dpi=300)
plt.close(fig)
