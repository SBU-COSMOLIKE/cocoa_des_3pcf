"""Redraw aperture_filter.pdf from data.json: the compensated aperture
filter of the DES Y3 Map3 measurement at its four radii.

Top panel: the real-space weight U_R(theta) of the note's Eq. (filter
definition), multiplied by R^2 so the four radii share one y axis (the
filter's amplitude scales as 1/R^2; the multiplication leaves a pure
shape).  Bottom panel: the Fourier window U(ell R) = (ell R)^2 / 2 *
exp(-(ell R)^2 / 2) against the multipole ell; each curve peaks at
ell = sqrt(2)/R with peak value 1/e.  data.json is written by the
companion notebook aperture_mass_statistic.ipynb (its Section 2).
Run from any folder: python plot_aperture_filter.py
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

# The four radii, one style each: Okabe-Ito colors (no red with green),
# every category encoded redundantly by color, line width and style.
COLORS = ['#0072B2', '#E69F00', '#CC79A7', '#56B4E9']
WIDTHS = [2.4, 1.8, 1.3, 0.9]
STYLES = ['solid', 'dashed', 'dashdot', 'dotted']

with open(os.path.join(HERE, 'data.json')) as stream:
    data = json.load(stream)

radii = data['radii_arcmin']
theta = np.asarray(data['theta_arcmin'])
u_r2 = np.asarray(data['U_times_R2'])          # [nradius, ntheta]
ell = np.asarray(data['ell'])
uhat = np.asarray(data['Uhat'])                # [nradius, nell]

fig, axes = plt.subplots(2, 1, figsize=(6.0, 6.4))
fig.subplots_adjust(hspace=0.34)

# Top: the real-space weight, positive inside sqrt(2) R, negative
# outside, integrating to zero (the compensation property).
for i, radius in enumerate(radii):
    axes[0].plot(theta, u_r2[i], color=COLORS[i], linewidth=WIDTHS[i],
                 linestyle=STYLES[i], label=rf"$R = {radius:g}'$")
axes[0].axhline(0.0, color='black', linewidth=0.7)
axes[0].set_xlabel(r'$\theta$ [arcmin]')
axes[0].set_ylabel(r'$R^2\, U_R(\theta)$')
axes[0].set_xlim(0.0, float(theta[-1]))
# Fit the y range to the curves: peak 1/(2 pi) at theta = 0, small
# negative trough; about 10% padding, no symmetric white space.
top = float(u_r2.max())
bottom = float(u_r2.min())
axes[0].set_ylim(bottom - 0.1 * (top - bottom), top + 0.1 * (top - bottom))
axes[0].legend(frameon=False, fontsize=11, loc='upper right')

# Bottom: the Fourier window on a logarithmic multipole axis; the
# dotted drop lines mark each curve's peak at ell = sqrt(2)/R.
for i, radius in enumerate(radii):
    axes[1].plot(ell, uhat[i], color=COLORS[i], linewidth=WIDTHS[i],
                 linestyle=STYLES[i])
    peak_ell = data['peak_ell'][i]
    axes[1].plot([peak_ell, peak_ell], [0.0, np.exp(-1.0)],
                 color=COLORS[i], linewidth=0.7, linestyle=':')
axes[1].set_xscale('log')
axes[1].set_xlabel(r'$\ell$')
axes[1].set_ylabel(r'$\widehat{U}(\ell R)$')
axes[1].set_xlim(float(ell[0]), float(ell[-1]))
axes[1].set_ylim(0.0, np.exp(-1.0) * 1.1)

for extension in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'aperture_filter.{extension}'),
                dpi=300)
plt.close(fig)
