"""Redraw ia_bmode_geometry.pdf from data.json: the E-type and B-type
stretch patterns a shape mode can carry along one wavevector.

Each panel shows one Fourier mode travelling along the arrow k: at
every position the local shape distortion is a headless stick whose
length follows the wave's amplitude.  Left panel: an E-type stretch,
aligned with k where the wave is positive and perpendicular where it
is negative; the first-order tidal response can only produce this
pattern.  Right panel: a B-type stretch, the same amplitudes at 45
degrees to k, which requires a second-order kernel.  data.json is
written by the companion notebook aperture_mass_statistic.ipynb (its
Section 9).  Run from any folder: python plot_ia_bmode_geometry.py
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

positions = np.asarray(data['positions'])      # x along the wave
rows = np.asarray(data['rows'])                # y offsets of the rows
amplitude = np.cos(2.0 * np.pi * positions / float(data['wavelength']))
half = float(data['stick_half_length'])

fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4))
fig.subplots_adjust(wspace=0.08)
titles = ['E-type stretch', 'B-type stretch']
colors = [BLUE, ORANGE]
tilts = [0.0, np.pi / 4.0]             # stick angle at positive phase
for ax, title, color, tilt in zip(axes, titles, colors, tilts):
    for y in rows:
        for x, value in zip(positions, amplitude):
            # Positive phase: stretch at the tilt angle; negative
            # phase: at the perpendicular angle.  Length follows the
            # wave's amplitude.
            angle = tilt if value >= 0.0 else tilt + np.pi / 2.0
            length = half * abs(value)
            dx = length * np.cos(angle)
            dy = length * np.sin(angle)
            ax.plot([x - dx, x + dx], [y - dy, y + dy], color=color,
                    linewidth=2.2, solid_capstyle='round')
    # The wavevector arrow below the sticks.
    arrow_y = float(rows.min()) - 3.2 * half
    ax.annotate('', xy=(positions.max(), arrow_y),
                xytext=(positions.min(), arrow_y),
                arrowprops=dict(arrowstyle='->', color='black',
                                linewidth=1.4))
    ax.annotate(r'$k$', (positions.max(), arrow_y),
                textcoords='offset points', xytext=(2, 8), fontsize=13)
    ax.set_title(title, fontsize=12)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylim(arrow_y - 2.0 * half, float(rows.max()) + 2.5 * half)
    ax.set_xlim(positions.min() - 3.0 * half,
                positions.max() + 3.0 * half)
    for spine in ax.spines.values():
        spine.set_visible(False)

for extension in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'ia_bmode_geometry.{extension}'),
                dpi=300)
plt.close(fig)
