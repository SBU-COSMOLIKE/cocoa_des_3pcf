"""Plot frozen first-1000 comparisons without reading any map files.

PME estimates precision. The matrices plotted here are its inverses;
this inversion does not preserve unbiasedness. Direct means the ordinary
sample covariance of the same 1000 vectors, not the true covariance.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Figure style of the whole notebook: the shared plotting functions never change
# rcParams, so these lines set every font, tick and grid default.
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

folder = Path(__file__).resolve().parents[2] / 'chains/gptg_ensemble_15000/preliminary_1000_500'
direct = np.loadtxt(folder / 'covariance_direct.txt')
scale = np.sqrt(np.diag(direct))
normalization = np.outer(scale, scale)
correlation = direct / normalization
records = []
for family, title in [('gaussian', 'Gaussian-moment correction'),
                      ('distinct', 'Distinct-pair correction')]:
    covariances = [np.loadtxt(folder / f'covariance_{family}{order}.txt')
                  for order in (2, 4, 6)]
    differences = [(cov-direct)/normalization for cov in covariances]
    limit = max(np.max(np.abs(value)) for value in differences)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    for column, (order, cov, delta) in enumerate(zip((2, 4, 6), covariances, differences)):
        std = np.sqrt(np.diag(cov))
        corr = cov / np.outer(std, std)
        ratio = np.diag(cov)/np.diag(direct)-1
        records.append([family, order, 100*ratio.min(), 100*ratio.max(),
                        np.max(np.abs(corr-correlation)), np.max(np.abs(delta))])
        # Normalizing by direct standard deviations avoids division by tiny
        # off-diagonal covariance entries. Every order uses the same scale.
        im = axes[0, column].imshow(delta, vmin=-limit, vmax=limit, cmap='RdBu_r')
        axes[0, column].set_title(f'Order {order}: normalized covariance difference')
        # Row index increases downward: the lower triangle is the correction.
        split = np.where(np.tril(np.ones((16, 16), dtype=bool)), corr, correlation)
        cm = axes[1, column].imshow(split, vmin=-1, vmax=1, cmap='RdBu_r')
        axes[1, column].set_title('Lower: correction; upper: direct')
        for row in (0, 1):
            axes[row, column].set_xlabel('Moment index')
            axes[row, column].set_ylabel('Moment index')
    fig.colorbar(im, ax=axes[0, :], label=r'$(C_{\rm PME}-C_{\rm direct})_{ij}/(\sigma_i\sigma_j)$', shrink=.8)
    fig.colorbar(cm, ax=axes[1, :], label='Correlation', shrink=.8)
    fig.suptitle(title + ' — same 1,000 maps; direct is an estimated reference', fontsize=15)
    for extension in ('png', 'pdf'):
        fig.savefig(folder / f'comparison_{family}.{extension}', dpi=170)
    plt.close(fig)
with (folder / 'covariance_differences.txt').open('w') as stream:
    stream.write('# family order min_variance_change_percent max_variance_change_percent max_absolute_correlation_change max_normalized_covariance_change\n')
    for record in records:
        line = ' '.join(str(value) for value in record)
        stream.write(line+'\n')
        print(line)
