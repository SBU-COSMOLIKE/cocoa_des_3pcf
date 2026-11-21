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
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(4.2, 6.1), sharex=True)
fig.subplots_adjust(hspace=.08, top=.89, left=.18, right=.89)
for source, ax in enumerate(axes):
    for parity in (0, 1):
        rows = data[(data[:, 2] == source) & (data[:, 3] == source)
                    & (data[:, 4] == parity)]
        ell = np.sqrt(rows[:, 0]*rows[:, 1])
        ax.plot(ell, rows[:, 6]/rows[:, 5], color=['#0072B2','#E69F00'][parity],
                linewidth=[2.4,1.8][parity], marker=['o','s'][parity],
                markersize=3, linestyle=['-','--'][parity], label=['E','B'][parity])
    ax.axhline(y=1., color='gray', linestyle=':')
    ax.set_ylim(bottom=.55, top=1.25)
    ax.set_xscale('log')
    ax.set_ylabel(f'Bin {source+1}')
    ax.yaxis.set_label_position('right')
legend = axes[0].legend(loc='lower center', bbox_to_anchor=(.5, 1.04),
                       ncol=2, frameon=False)
fig.supylabel('Masked auto-power: generated / calibration maps', x=.025)
axes[-1].set_xlabel(r'Multipole $\ell$')
fig.canvas.draw()
assert not legend.get_window_extent().overlaps(axes[0].get_window_extent())
NAME = 'spectra.pdf'

fig.savefig(fname=folder / NAME, dpi=300)
fig.savefig(fname=folder / NAME.replace('.pdf', '.png'), dpi=300)
plt.close(fig)
