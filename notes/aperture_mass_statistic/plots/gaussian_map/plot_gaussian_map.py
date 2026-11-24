"""Redraw gaussian_map.pdf from data.json: the simulated Gaussian
convergence map of the companion notebook and its aperture-mass map.

Left panel: the periodic Gaussian convergence map kappa on a 10 degree
box (downsampled for storage).  Right panel: the aperture-mass map at
R = 14 arcmin, obtained by multiplying the map's Fourier modes by the
compensated window U(ell R) = (ell R)^2/2 exp(-(ell R)^2/2); the
band-pass action of the window is the visual point.  data.json is
written by the companion notebook aperture_mass_statistic.ipynb (its
Section 3).  Run from any folder: python plot_gaussian_map.py
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

with open(os.path.join(HERE, 'data.json')) as stream:
    data = json.load(stream)

box = float(data['box_deg'])
kappa = np.asarray(data['kappa'])
aperture = np.asarray(data['map_r14'])
radius = data['radius_arcmin']

# Two maps side by side, each with its own symmetric color range (the
# fields are zero-mean) and its own colorbar.  cividis is a
# colorblind-safe sequential colormap.
fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2))
fig.subplots_adjust(wspace=0.18)
titles = [r'$\kappa$',
          rf'$M_{{\mathrm{{ap}}}}$ at $R = {radius:g}$ arcmin']
for ax, field, title in zip(axes, [kappa, aperture], titles):
    limit = float(np.max(np.abs(field)))
    image = ax.imshow(field, origin='lower', cmap='cividis',
                      vmin=-limit, vmax=limit,
                      extent=[0.0, box, 0.0, box])
    ax.set_xlabel(r'$\theta_x$ [deg]')
    ax.set_title(title, fontsize=12)
    fig.colorbar(image, ax=ax, shrink=0.82)
axes[0].set_ylabel(r'$\theta_y$ [deg]')

for extension in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'gaussian_map.{extension}'),
                dpi=300)
plt.close(fig)
