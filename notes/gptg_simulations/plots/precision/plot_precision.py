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

data = np.loadtxt(fname=folder / 'data.txt')
fig, ax = plt.subplots(figsize=(4.5, 3.4))
# The distinct-pair construction has no Gaussian-distribution assumption.
# Dotted companion curves deliberately apply Wishart formulas to these data.
colors = ['#0072B2', '#E69F00']
for group, count in enumerate((128,512)):
    for offset in (0,3):
        rows = data[(data[:,0] == count) & (data[:,1] >= offset)
                    & (data[:,1] < offset+3)]
        name = 'Gaussian formula' if offset == 0 else 'Distinct pairs'
        style = ':' if offset == 0 else '-'
        ax.plot([2,4,6], rows[:,4], color=colors[group],
                marker=['o','s'][group], linewidth=[2.4,1.8][group],
                linestyle=style, label=f'{count}, {name}')
ax.set_xticks(ticks=[2,4,6])
ax.set_xlabel('Precision expansion order')
ax.set_ylabel('Relative RMS error about exact inverse')
ax.set_ylim(bottom=.05, top=.29)
legend = ax.legend(loc='lower center', bbox_to_anchor=(.5,1.01),
                   ncol=2, fontsize=8, frameon=False)
fig.canvas.draw()
assert not legend.get_window_extent().overlaps(ax.get_window_extent())
NAME = 'precision.pdf'

fig.savefig(fname=folder / NAME, dpi=300)
fig.savefig(fname=folder / NAME.replace('.pdf', '.png'), dpi=300)
plt.close(fig)
