"""Redraw triangle_coverage.pdf from data.json: the side plane of the
side-angle-side (SAS) triangle grid behind the DES Y3 Map3 rows.

One panel: the (d2, d3) plane on logarithmic axes, with one grid line
per side-bin edge (20 logarithmic bins from 0.5 to 80 arcmin), the
9 x 9 block kept by the 8 arcmin lower-edge scale cut shaded, and the
four aperture radii marked on the axes.  Every kept side pair carries
the number of opening-angle bins printed inside the shaded block.
data.json is written by the companion notebook
aperture_mass_statistic.ipynb (its Section 5) from the likelihood keys.
Run from any folder: python plot_triangle_coverage.py
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

with open(os.path.join(HERE, 'data.json')) as stream:
    data = json.load(stream)

edges = np.asarray(data['edges_arcmin'])       # the 21 side-bin edges
cut = float(data['cut_arcmin'])                # lower-edge scale cut
radii = data['radii_arcmin']                   # the four filter radii
kept = np.asarray(data['kept_lower_edges_arcmin'])
phi_bins = int(data['phi_bins'])

fig, ax = plt.subplots(figsize=(5.4, 5.0))

# The kept block: every side pair whose bins both survive the cut.
# The shaded square runs from the first kept lower edge to the last
# upper edge of the grid.
ax.fill_between([kept[0], edges[-1]], kept[0], edges[-1],
                color='0.90', zorder=0)

# One thin line per bin edge, both axes: the measured grid.
for edge in edges:
    ax.axvline(edge, color='0.6', linewidth=0.4, zorder=1)
    ax.axhline(edge, color='0.6', linewidth=0.4, zorder=1)

# The scale cut, drawn as one heavier line per axis.
ax.axvline(cut, color='#D55E00', linewidth=1.8, zorder=2)
ax.axhline(cut, color='#D55E00', linewidth=1.8, zorder=2)

# The four aperture radii, marked on both axes just inside the frame.
for radius in radii:
    ax.plot([radius], [edges[0] * 1.07], marker='^', color='#0072B2',
            markersize=7, clip_on=False, zorder=3)
    ax.plot([edges[0] * 1.07], [radius], marker='>', color='#0072B2',
            markersize=7, clip_on=False, zorder=3)

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(edges[0], edges[-1])
ax.set_ylim(edges[0], edges[-1])
ax.set_xlabel(r'$d_2$ [arcmin]')
ax.set_ylabel(r'$d_3$ [arcmin]')
ticks = [0.5, 2, 8, 20, 80]
ax.set_xticks(ticks)
ax.set_xticklabels([f'{t:g}' for t in ticks])
ax.set_yticks(ticks)
ax.set_yticklabels([f'{t:g}' for t in ticks])
ax.minorticks_off()

# The counts, written inside the regions they describe.
nkept = kept.size
ax.text(np.sqrt(kept[0] * edges[-1]), np.sqrt(kept[0] * edges[-1]),
        f'kept: {nkept} x {nkept} side bins\n'
        rf'$\times$ {phi_bins} angle bins each',
        ha='center', va='center', fontsize=11)
ax.text(np.sqrt(edges[0] * cut), np.sqrt(edges[0] * cut),
        'cut', ha='center', va='center', fontsize=11,
        color='#D55E00')
ax.text(np.sqrt(edges[0] * cut) * 2.6, edges[0] * 1.35,
        'aperture radii', ha='left', va='center', fontsize=10,
        color='#0072B2')

for extension in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'triangle_coverage.{extension}'),
                dpi=300)
plt.close(fig)
