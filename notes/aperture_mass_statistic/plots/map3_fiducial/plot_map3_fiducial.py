"""Redraw map3_fiducial.pdf from data.json: the third-order
aperture-mass prediction of the des_3pcf likelihood at the fiducial
point of EXAMPLE_EVALUATE2.yaml.

One panel: Map3 against the filter radius R (the four measured radii),
one curve per drawn source-bin triplet, with the values multiplied by
the power of ten stored in data.json so the y axis shows numbers of
order one.  data.json is written by the companion notebook
aperture_mass_statistic.ipynb (its Section 5) from nw.map3().
Run from any folder: python plot_map3_fiducial.py
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



# One style per drawn triplet: Okabe-Ito colors, redundant widths,
# styles and markers, so the curves separate in grayscale too.
COLORS = ['#0072B2', '#E69F00', '#CC79A7', '#56B4E9']
WIDTHS = [2.4, 1.8, 1.3, 0.9]
STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
MARKERS = ['o', 's', 'D', '^']

with open(os.path.join(HERE, 'data.json')) as stream:
    data = json.load(stream)

radii = np.asarray(data['radii_arcmin'])
labels = data['triplet_labels']
values = np.asarray(data['scaled_values'])     # [ntriplet, nradius]

fig, ax = plt.subplots(figsize=(6.0, 4.0))
for i, label in enumerate(labels):
    ax.plot(radii, values[i], color=COLORS[i], linewidth=WIDTHS[i],
            linestyle=STYLES[i], marker=MARKERS[i], markersize=5,
            label=label)
ax.set_xscale('log')
# The measured radii are the only x values; label them directly.
ax.set_xticks(radii)
ax.set_xticklabels([f'{radius:g}' for radius in radii])
ax.minorticks_off()
ax.set_xlabel(r'$R$ [arcmin]')
ax.set_ylabel(data['ylabel'])
# Fit the y range to the curves with about 10% padding, 0 included
# when the curves come close to it.
top = float(values.max())
bottom = min(0.0, float(values.min()))
pad = 0.1 * (top - bottom)
ax.set_ylim(bottom - (pad if bottom < 0.0 else 0.0), top + pad)
legend = ax.legend(frameon=False, fontsize=11, loc='upper right',
                   title='source triplet', title_fontsize=11)

assert_legend_clear(fig, ax, legend)

for extension in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'map3_fiducial.{extension}'),
                dpi=300)
plt.close(fig)
