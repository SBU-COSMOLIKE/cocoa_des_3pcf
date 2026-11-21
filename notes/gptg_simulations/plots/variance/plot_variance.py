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
ratio = np.diag(generated) / np.diag(reference)
fig, ax = plt.subplots(figsize=(4.4, 3.4))
ax.plot(np.arange(8), ratio[:8], color='#0072B2', marker='o',
        linewidth=2.4, label='Squared apertures')
ax.plot(np.arange(8), ratio[8:], color='#E69F00', marker='s',
        linewidth=1.8, linestyle='--', label='Cubed apertures')
ax.axhline(y=1., color='gray', linestyle=':')
ax.set_xticks(ticks=np.arange(8), labels=['1/30','1/60','2/30','2/60',
                                       '3/30','3/60','4/30','4/60'])
ax.tick_params(axis='x', labelsize=8)
ax.set_xlabel('Source bin / radius [arcmin]')
ax.set_ylabel('Variance: generated / Rafael full ensemble')
ax.set_ylim(bottom=.5, top=1.55)
legend = ax.legend(loc='lower center', bbox_to_anchor=(.5, 1.02), frameon=False)
fig.canvas.draw()
assert not legend.get_window_extent().overlaps(ax.get_window_extent())
NAME = 'variance.pdf'

fig.savefig(fname=folder / NAME, dpi=300)
fig.savefig(fname=folder / NAME.replace('.pdf', '.png'), dpi=300)
plt.close(fig)
