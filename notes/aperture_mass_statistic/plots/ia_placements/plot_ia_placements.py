"""Redraw ia_placements.pdf from data.json: the alignment operator
split resolved by galaxy placement at one radius.

For each of the eight placements (labelled by which ordered legs carry
an intrinsic shape) the markers show the absolute contribution of each
alignment operator to the Map3 row of the lowest-redshift triplet at
R = 14 arcmin, divided by the gravity-only term, on a logarithmic
axis.  Open markers flag negative contributions.  The gravity operator
lives entirely in GGG and is not drawn; placements whose contribution
is exactly zero for an operator carry no marker.  data.json is written
by the companion notebook aperture_mass_statistic.ipynb (its
Section 7).  Run from any folder: python plot_ia_placements.py
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
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['axes.labelsize'] = 'medium'
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'

HERE = os.path.dirname(os.path.abspath(__file__))

def assert_legend_clear(fig, ax, legend):
    """Stop if the legend box overlaps any drawn line or marker.

    The figure rules require this check in code: a legend must never
    touch the data, and a visual check alone does not survive new
    data.json contents.
    """
    fig.canvas.draw()
    box = legend.get_window_extent()
    for line in ax.get_lines():
        points = line.get_xydata()
        if points.size == 0:
            continue
        pixels = ax.transData.transform(points)
        inside = ((pixels[:, 0] >= box.x0) & (pixels[:, 0] <= box.x1)
                  & (pixels[:, 1] >= box.y0) & (pixels[:, 1] <= box.y1))
        if bool(np.any(inside)):
            raise RuntimeError(
                'the legend overlaps a drawn curve; move the legend or '
                'widen the axis range in this script')



# One style per operator: Okabe-Ito colors with distinct markers, so
# the categories separate without color.
COLORS = ['#0072B2', '#E69F00', '#CC79A7']
MARKERS = ['o', 's', 'D']

with open(os.path.join(HERE, 'data.json')) as stream:
    data = json.load(stream)

placements = data['placements']                  # 8 labels
names = data['operator_names']                   # drawn operators
values = np.asarray(data['ratio_to_gravity'])    # [noperator, 8], signed

positions = np.arange(len(placements), dtype=float)
offsets = np.linspace(-0.22, 0.22, len(names))

fig, ax = plt.subplots(figsize=(6.4, 4.0))
for i, name in enumerate(names):
    row = values[i]
    for j, value in enumerate(row):
        if value == 0.0:
            continue
        filled = value > 0.0
        ax.plot([positions[j] + offsets[i]], [abs(value)],
                marker=MARKERS[i], markersize=7, linestyle='none',
                color=COLORS[i],
                markerfacecolor=COLORS[i] if filled else 'none',
                markeredgecolor=COLORS[i])
    # One legend handle per operator, with the drawn (filled) marker.
    ax.plot([], [], marker=MARKERS[i], markersize=7, linestyle='none',
            color=COLORS[i], label=name)
ax.set_yscale('log')
ax.set_xticks(positions)
ax.set_xticklabels(placements)
ax.set_xlabel('placement of the intrinsic legs')
ax.set_ylabel(r'$|$operator$| \; / \;$ gravity-only')
legend = ax.legend(frameon=False, fontsize=11, loc='upper right')

assert_legend_clear(fig, ax, legend)

for extension in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'ia_placements.{extension}'),
                dpi=300)
plt.close(fig)
