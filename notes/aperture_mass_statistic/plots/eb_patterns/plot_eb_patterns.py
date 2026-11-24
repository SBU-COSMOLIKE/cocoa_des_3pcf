"""Redraw eb_patterns.pdf from data.json: the E and B polarization
patterns of a shear field around a point.

Left panel: the E pattern, headless sticks tangent to circles around
the center, which is what lensing by an overdensity produces.  Right
panel: the B pattern, the same sticks rotated by 45 degrees into a
swirl, which no scalar lensing potential can produce.  Each stick
shows the local stretching axis; sticks are headless because a shear
has an orientation but no direction.  data.json is written by the
companion notebook aperture_mass_statistic.ipynb (its Section 9).
Run from any folder: python plot_eb_patterns.py
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
matplotlib.rcParams['axes.grid'] = False
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'

HERE = os.path.dirname(os.path.abspath(__file__))

BLUE, ORANGE = '#0072B2', '#E69F00'

with open(os.path.join(HERE, 'data.json')) as stream:
    data = json.load(stream)

rings = np.asarray(data['ring_radii'])
nstick = int(data['sticks_per_ring'])
half = float(data['stick_half_length'])

fig, axes = plt.subplots(1, 2, figsize=(8.0, 4.0))
fig.subplots_adjust(wspace=0.08)
titles = ['E pattern (tangential)', 'B pattern (swirl)']
offsets = [np.pi / 2.0, np.pi / 4.0]   # stick angle relative to the
colors = [BLUE, ORANGE]                # radial direction
for ax, title, offset, color in zip(axes, titles, offsets, colors):
    ax.plot([0.0], [0.0], marker='o', color='black', markersize=5)
    for radius in rings:
        for position in 2.0 * np.pi * np.arange(nstick) / nstick:
            x = radius * np.cos(position)
            y = radius * np.sin(position)
            stick = position + offset
            dx = half * np.cos(stick)
            dy = half * np.sin(stick)
            ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color,
                    linewidth=2.0, solid_capstyle='round')
    ax.set_title(title, fontsize=12)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    limit = float(rings.max()) + 2.5 * half
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    for spine in ax.spines.values():
        spine.set_visible(False)

for extension in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'eb_patterns.{extension}'),
                dpi=300)
plt.close(fig)
