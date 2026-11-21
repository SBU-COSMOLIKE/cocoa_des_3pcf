# DES Y3 shear maps at NSIDE 512

**Thanks to Rafael Gomes for supplying these CosmoGrid shear maps.** They
support the covariance work associated with [Gomes et al.](https://arxiv.org/abs/2503.03964)
and [his analysis code](https://github.com/rchgomes/3pcf_integrator).
The download is incomplete. This guide describes the files checked on
20 November 2026; it does not certify the entire archive.

## Start with the viewing notebook

Open [EXAMPLE_VIEW_MAPS.ipynb](EXAMPLE_VIEW_MAPS.ipynb) with this directory
as the working directory. It reads only two explicitly named map files,
prints their shapes and a small sample, and contains the footprint and
shear-component plots. Plotting requires `indices.npy`, the HEALPix ordering
from the producer, and `healpy` in the notebook environment. At the time of
writing, `indices.npy` had not arrived locally and `healpy` was not installed
in the project's Python environment. The notebook therefore reports which
inputs are missing and does not invent coordinates or display a guessed sky.

The numerical map files, including `indices.npy`, are ignored by Git.
Only this README and the viewing notebook are tracked here. Save generated
figures and measured data vectors under the project's `chains/` directory.
Do not commit the simulation download.

## What a file contains

Rafael confirmed that these are **NSIDE 512 CosmoGrid shear maps**. Only
pixels where the DES footprint is nonzero are retained. The companion
`indices.npy` lists those HEALPix pixel indices. Thus the final array axis
is a list of retained pixels, not a rectangular image and not a list of
individual galaxies. Missing sky pixels are unobserved; do not fill them
with zero shear when measuring correlations.

Two inspected files are:

- `ShearMaps_DESY3_run_0000_ncut0_nreal0.npy`
- `ShearMaps_DESY3_run_0000_ncut0_nreal1.npy`

Both contain C-order `float64` arrays of shape `(4, 2, 369190)`, occupying
23,628,288 bytes including the NPY header. The likely axis interpretation is
four source-redshift bins, two shear components, and retained pixels. The
producer should confirm the source-bin order, component order, signs and
coordinate basis. The first 4,096 pixels of each component in these two files
were finite. This is a small inspection, not an integrity check of all pixels
or all files. NPY contains array shape and type, but not these scientific
conventions or a checksum supplied by the producer.

For a one-dimensional common index list, `maps[b, g, j]` belongs to HEALPix
pixel `indices[j]`. Here `b` selects a source bin, `g` a shear component and
`j` the compressed pixel position. **Preserve this pairing.** Sorting indices
without applying the same permutation to the shear values moves the signal
to the wrong sky positions. The notebook requires a common one-dimensional
list; if the downloaded index file instead has separate lists for each cut,
inspect those lists and select the matching cut explicitly.

## Recovering the DES footprint

HEALPix partitions the sphere into equal-area pixels. NSIDE 512 implies
`12 * 512**2 = 3145728` full-sky pixels; the maps retain only a subset.
Each pixel covers about 47.2 square arcminutes, corresponding to a square
about 6.87 arcminutes across. That square is only a useful angular scale:
HEALPix pixels do not form a rectangular image.

HEALPix has two index conventions, RING and NESTED. Both use the same integer
range but assign different positions to those integers. **NSIDE alone does
not specify the ordering.** Set the notebook's `nested` value only after
confirmation: `False` for RING, `True` for NESTED. `healpy.pix2ang` then turns
indices into longitude and latitude. Identify them with RA and Dec only
when the producer confirms that the maps use equatorial coordinates.

The notebook plots the retained-pixel footprint and both components of two
maps in the first source bin. It uses at most 30,000 evenly selected array
positions per panel to keep rendering small; the displayed footprint is
therefore a preview. Both realizations use the same color range. Longitude
is wrapped at 180 degrees to display a footprint crossing zero. This is a
coordinate scatter plot, not an equal-area projection or a science estimator.
A scalar component plot does not rotate the spin-2 shear basis; any actual
correlation measurement must use the producer's shear-basis convention.

## A file is not an independent Universe

The names contain `run`, `ncut` and `nreal`. They suggest a parent simulation,
a footprint cut, and a noise realization, respectively; Rafael's message
confirmed the pixel selection, **not these label meanings**. Confirm them
before counting independent simulations. In particular, if several files
share a density realization and differ only in galaxy shape noise, they
cannot be counted as new independent realizations of cosmic structure.
Keep all descendants of a parent simulation together when assigning training
and validation sets. Do not average different footprints and call that one
DES survey; this changes the measurement whose covariance is being estimated.

Ask the producer for the label meanings, parent simulation IDs and cosmology,
the HEALPix ordering and coordinate frame, source-bin distributions, shape
noise prescription and noiseless-map availability, shear signs/basis,
footprint weights and pixel occupancies, and checksums or expected file sizes.
If different cuts use different pixel lists, obtain that mapping too.
The download can be used for bounded previews while these details arrive;
a survey covariance requires them.

## From maps to a joint covariance

The inference data vector consists of measured two-point correlations and
third-order aperture statistics, not the shear pixels themselves. Measure
both statistics on each selected realization using the same mask, weights,
angular cuts and bin definitions as the analysis. Pair them before estimating
a covariance so that their cross-covariance is retained. Store the measured
vectors, row definitions and parent IDs under `../chains/`.

Rafael's methods paper describes maps at NSIDE 1024, whereas these maps are
512. Its pixel treatment and small-scale cuts therefore need a new check.
Degrading a map changes its small-scale signal, noise and covariance; it is
not corrected by changing a filename or rescaling the survey area. Until
resolution and estimator matching are tested, the maps can teach and test
the covariance workflow, but do not establish a matched published likelihood.

The [PME notebook](../EXAMPLE_COVARIANCE_PME.ipynb) explains how simulations
can correct an analytic precision matrix. The [introductory notes](../notes/covariance_from_simulations/covariance_from_simulations.pdf)
derive the method, explain repeated-noise realizations and describe the
validation needed before using the correction in a forecast. Neither artifact
claims that a DES covariance has already been calibrated from this download.
