"""Redraw triangle_geometry.pdf from data.json: the side-angle-side
labelling of a triangle and its centroid construction.

Left panel: a triangle with the two sides d2 (vertex 1 to vertex 2)
and d3 (vertex 1 to vertex 3) and the opening angle phi between them.
Right panel: the same triangle with its centroid, the dashed offsets
r_a from the vertices to the centroid, and a short headless stick at
each vertex showing the direction toward which that vertex's shear is
rotated (the centroid projection).  data.json is written by the
companion notebook aperture_mass_statistic.ipynb (its Section 9).
Run from any folder: python plot_triangle_geometry.py
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

BLUE, ORANGE, PURPLE = '#0072B2', '#E69F00', '#CC79A7'

with open(os.path.join(HERE, 'data.json')) as stream:
    data = json.load(stream)

vertices = np.asarray(data['vertices'])        # [3, 2], vertex 1 first
centroid = vertices.mean(axis=0)
closed = np.vstack([vertices, vertices[:1]])

fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0))
fig.subplots_adjust(wspace=0.08)
for ax in axes:
    ax.plot(closed[:, 0], closed[:, 1], color='black', linewidth=1.4)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.2, 1.28)
    ax.set_ylim(-0.24, 0.85)
for spine in [s for ax in axes for s in ax.spines.values()]:
    spine.set_visible(False)

# Left: the SAS labels.  The side labels sit just outside each side's
# midpoint; the angle arc sits at vertex 1.
left = axes[0]
label_offsets = [(-14, -4), (7, -5), (2, 8)]
for a, (x, y) in enumerate(vertices):
    left.annotate(f'{a + 1}', (x, y), textcoords='offset points',
                  xytext=label_offsets[a], fontsize=12)
mid12 = 0.5 * (vertices[0] + vertices[1])
mid13 = 0.5 * (vertices[0] + vertices[2])
left.annotate(r'$d_2$', mid12, textcoords='offset points',
              xytext=(0, -16), fontsize=13, color=BLUE)
left.annotate(r'$d_3$', mid13, textcoords='offset points',
              xytext=(-18, 0), fontsize=13, color=BLUE)
angle12 = np.degrees(np.arctan2(*(vertices[1] - vertices[0])[::-1]))
angle13 = np.degrees(np.arctan2(*(vertices[2] - vertices[0])[::-1]))
arc = np.radians(np.linspace(angle12, angle13, 40))
left.plot(vertices[0, 0] + 0.22 * np.cos(arc),
          vertices[0, 1] + 0.22 * np.sin(arc),
          color=ORANGE, linewidth=1.4)
bisector = np.radians(0.5 * (angle12 + angle13))
left.annotate(r'$\phi$',
              (vertices[0, 0] + 0.33 * np.cos(bisector),
               vertices[0, 1] + 0.33 * np.sin(bisector)),
              fontsize=13, color=ORANGE, ha='center', va='center')

# Right: centroid, offsets and projection directions.
right = axes[1]
right.plot([centroid[0]], [centroid[1]], marker='o', color='black',
           markersize=4)
for a, (x, y) in enumerate(vertices):
    # Dashed offset arrow r_a from the vertex to the centroid (the
    # convention of the published kernels; only even powers of the
    # offsets enter them, so the direction never changes a kernel).
    right.annotate('', xy=tuple(centroid), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', color=PURPLE,
                                   linewidth=1.3, linestyle='--'))
    label = 0.55 * (np.array([x, y]) - centroid) + centroid
    right.annotate(rf'$r_{a + 1}$', label, textcoords='offset points',
                   xytext=(6, 4), fontsize=12, color=PURPLE)
    # A short headless stick through the vertex along the offset
    # direction: the axis the shear is rotated onto.
    direction = (np.array([x, y]) - centroid)
    direction = direction / np.hypot(*direction)
    stick = np.outer([-0.11, 0.11], direction) + np.array([x, y])
    right.plot(stick[:, 0], stick[:, 1], color=BLUE, linewidth=2.6,
               solid_capstyle='round')
right.annotate('centroid', centroid, textcoords='offset points',
               xytext=(4, -14), fontsize=11)

for extension in ('pdf', 'png'):
    fig.savefig(os.path.join(HERE, f'triangle_geometry.{extension}'),
                dpi=300)
plt.close(fig)
