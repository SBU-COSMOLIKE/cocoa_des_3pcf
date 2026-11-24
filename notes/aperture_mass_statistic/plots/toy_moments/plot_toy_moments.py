"""Redraw toy_moments.pdf from data.json: aperture-mass moments
measured on the notebook's simulated maps against the filter radius.

Top panel: the variance of the aperture-mass map of the Gaussian toy
convergence field (points with block-estimated uncertainties) against
the power-spectrum prediction evaluated on the same discrete Fourier
grid (line), on a logarithmic y axis.  Bottom panel: the third moment
of the aperture-mass map, scaled by its rms cubed, for the Gaussian
map (consistent with zero) and for the quadratic non-Gaussian toy map
(clearly nonzero).  data.json is written by the companion notebook
aperture_mass_statistic.ipynb (its Section 3).
Run from any folder: python plot_toy_moments.py
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



# Okabe-Ito colors (no red with green); every category also separates
# by marker and line presence, so grayscale stays readable.
COLORS = ['#0072B2', '#E69F00', '#CC79A7']
MARKERS = ['o', 's', 'D']

with open(os.path.join(HERE, 'data.json')) as stream:
    data = json.load(stream)

radii = np.asarray(data['radii_arcmin'])
map2_measured = np.asarray(data['map2_measured'])
map2_error = np.asarray(data['map2_error'])
map2_predicted = np.asarray(data['map2_predicted'])
skew_gauss = np.asarray(data['skew_gauss'])
skew_gauss_error = np.asarray(data['skew_gauss_error'])
skew_toy = np.asarray(data['skew_toy'])
skew_toy_error = np.asarray(data['skew_toy_error'])

fig, axes = plt.subplots(2, 1, figsize=(6.0, 6.4))
fig.subplots_adjust(hspace=0.3)

# Top: measured variance against the grid prediction.
axes[0].plot(radii, map2_predicted, color=COLORS[0], linewidth=1.6,
             label='power-spectrum prediction')
axes[0].errorbar(radii, map2_measured, yerr=map2_error, fmt=MARKERS[0],
                 color='black', markersize=5, capsize=3,
                 linestyle='none', label='measured on the map')
axes[0].set_xscale('log')
axes[0].set_yscale('log')
axes[0].set_xticks(radii)
axes[0].set_xticklabels([f'{radius:g}' for radius in radii])
axes[0].minorticks_off()
axes[0].set_xlabel(r'$R$ [arcmin]')
axes[0].set_ylabel(r'$\langle M_{\mathrm{ap}}^2 \rangle$')
legend_top = axes[0].legend(frameon=False, fontsize=11,
                            loc='lower left')

# Bottom: the scaled third moments; offset the two series slightly in
# radius so the error bars do not overlap.
axes[1].axhline(0.0, color='black', linewidth=0.7)
axes[1].errorbar(radii * 0.97, skew_gauss, yerr=skew_gauss_error,
                 fmt=MARKERS[1], color=COLORS[0], markersize=5,
                 capsize=3, linestyle='none', label='Gaussian map')
axes[1].errorbar(radii * 1.03, skew_toy, yerr=skew_toy_error,
                 fmt=MARKERS[2], color=COLORS[1], markersize=5,
                 capsize=3, linestyle='none', label='non-Gaussian toy')
axes[1].set_xscale('log')
axes[1].set_xticks(radii)
axes[1].set_xticklabels([f'{radius:g}' for radius in radii])
axes[1].minorticks_off()
axes[1].set_xlabel(r'$R$ [arcmin]')
axes[1].set_ylabel(
    r'$\langle M_{\mathrm{ap}}^3 \rangle / \sigma_{M}^3$')
# Fit the y range to the points, 0 included, 10% padding.
values = np.concatenate([skew_gauss + skew_gauss_error,
                         skew_gauss - skew_gauss_error,
                         skew_toy + skew_toy_error,
                         skew_toy - skew_toy_error, [0.0]])
pad = 0.1 * (values.max() - values.min())
axes[1].set_ylim(values.min() - pad, values.max() + pad)
legend_bottom = axes[1].legend(frameon=False, fontsize=11,
                               loc='upper right')

assert_legend_clear(fig, axes[0], legend_top)
assert_legend_clear(fig, axes[1], legend_bottom)

for extension in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'toy_moments.{extension}'),
                dpi=300)
plt.close(fig)
