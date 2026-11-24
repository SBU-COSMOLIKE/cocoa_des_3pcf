"""Redraw bispectrum_shapes.pdf from data.json: how a triangle's shape
enters the tree-level matter bispectrum.

Drawn is the reduced bispectrum Q = B_tree / [P1 P2 + P2 P3 + P3 P1]
for a power-law toy spectrum P propto k^(-1.5), against the angle
between the wavevectors k1 and k2, for three side ratios k2/k1.  The
enhancement toward aligned configurations (angle near 0 or pi) and the
shallow minimum near equilateral shapes are the generic signatures of
gravitational collapse.  data.json is written by the companion
notebook aperture_mass_statistic.ipynb (its Section 9).
Run from any folder: python plot_bispectrum_shapes.py
"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Figure style of the whole note: the house rcParams block, so every
# font, tick and grid default matches the other project figures.
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

HERE = os.path.dirname(os.path.abspath(__file__))

# One style per side ratio: Okabe-Ito colors with redundant widths and
# line styles, readable in grayscale.
COLORS = ['#0072B2', '#E69F00', '#CC79A7']
WIDTHS = [2.4, 1.8, 1.3]
STYLES = ['solid', 'dashed', 'dashdot']

with open(os.path.join(HERE, 'data.json')) as stream:
    data = json.load(stream)

angle = np.asarray(data['angle_over_pi'])
curves = np.asarray(data['reduced_bispectrum'])  # [nratio, nangle]
ratios = data['side_ratios']

fig, ax = plt.subplots(figsize=(6.0, 4.0))
for i, ratio in enumerate(ratios):
    ax.plot(angle, curves[i], color=COLORS[i], linewidth=WIDTHS[i],
            linestyle=STYLES[i], label=rf'$k_2/k_1 = {ratio:g}$')
ax.set_xlabel(r'angle between $k_1$ and $k_2$, in units of $\pi$')
ax.set_ylabel(r'$Q(k_1, k_2, k_3)$')
ax.set_xlim(0.0, 1.0)
# Fit the y range to the curves with 10% padding.
top, bottom = float(curves.max()), float(curves.min())
pad = 0.1 * (top - bottom)
ax.set_ylim(bottom - pad, top + pad)
ax.legend(frameon=False, fontsize=11, loc='upper center')

for extension in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'bispectrum_shapes.{extension}'),
                dpi=300)
plt.close(fig)
