"""Redraw map3_operators.pdf from data.json: the intrinsic-alignment
operator split of the Map3 prediction at the TATT point of
EXAMPLE_EVALUATE2.yaml.

One panel: each curve is one physical operator's contribution to one
source triplet's Map3 rows -- summed over the eight galaxy placements
(GGG ... III) and divided by the gravity-only term -- against the
filter radius.  The gravity-only curve itself would be the constant 1
and is left out; the caption of the note names it.  data.json is
written by the companion notebook aperture_mass_statistic.ipynb (its
Section 7) from nw.map3_components().
Run from any folder: python plot_map3_operators.py
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



# One style per operator: Okabe-Ito colors, redundant widths, styles
# and markers, so the curves separate in grayscale too.
COLORS = ['#0072B2', '#E69F00', '#CC79A7', '#56B4E9']
WIDTHS = [2.4, 1.8, 1.3, 0.9]
STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
MARKERS = ['o', 's', 'D', '^']

with open(os.path.join(HERE, 'data.json')) as stream:
    data = json.load(stream)

radii = np.asarray(data['radii_arcmin'])
names = data['operator_names']                 # drawn operators
ratios = np.asarray(data['ratio_to_gravity'])  # [noperator, nradius]

fig, ax = plt.subplots(figsize=(6.0, 4.0))
for i, name in enumerate(names):
    ax.plot(radii, ratios[i], color=COLORS[i], linewidth=WIDTHS[i],
            linestyle=STYLES[i], marker=MARKERS[i], markersize=5,
            label=name)
ax.axhline(0.0, color='black', linewidth=0.7)
ax.set_xscale('log')
# The measured radii are the only x values; label them directly.
ax.set_xticks(radii)
ax.set_xticklabels([f'{radius:g}' for radius in radii])
ax.minorticks_off()
ax.set_xlabel(r'$R$ [arcmin]')
ax.set_ylabel(data['ylabel'])
# Fit the y range to the curves, 0 included, about 10% padding.
top = max(0.0, float(ratios.max()))
bottom = min(0.0, float(ratios.min()))
pad = 0.1 * (top - bottom)
ax.set_ylim(bottom - pad, top + pad)
legend = ax.legend(frameon=False, fontsize=11,
                   loc=data.get('legend_loc', 'best'))

assert_legend_clear(fig, ax, legend)

for extension in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'map3_operators.{extension}'),
                dpi=300)
plt.close(fig)
