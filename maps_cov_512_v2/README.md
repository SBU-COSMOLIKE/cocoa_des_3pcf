# DES Y3 shear maps at NSIDE 512

**Thanks to Rafael Gomes for supplying these CosmoGrid shear maps.** They
support the covariance work associated with [Gomes et al.](https://arxiv.org/abs/2503.03964)
and [his analysis code](https://github.com/rchgomes/3pcf_integrator).
The download is incomplete. This guide describes the files checked on
20–21 November 2026; it does not certify the entire archive.

## Start with the viewing notebook

Open [EXAMPLE_VIEW_MAPS.ipynb](EXAMPLE_VIEW_MAPS.ipynb) with this directory
as the working directory. It reads only two explicitly named map files,
prints their shapes and a small sample, and contains the footprint and
shear-component plots. Plotting uses `indices_512.npy` and `astropy-healpix`. The supplied index file
has 369,190 unique integer entries, matching both inspected maps. Comparing
RING and NESTED interpretations identifies RING geometrically: it recovers
the recognizable DES footprint, while NESTED gives disconnected patterns
across the sky. The notebook displays both interpretations, then uses RING
for two map previews. This does not determine the shear-component sign or
basis convention, which still needs the producer's information.

For the NumPy 1.x Cocoa environment, the coordinate plots were checked using
`astropy-healpix==1.0.3` installed in a temporary review environment. To add
this optional plotting dependency to your active notebook environment:

```bash
python -m pip install --no-deps astropy-healpix==1.0.3
```

The saved notebook outputs include the bounded map inspection and plots;
the viewing notebook itself runs no covariance measurement. A separate
[aperture-moment pilot](../EXAMPLE_COVARIANCE_MAPS.ipynb) now studies a bounded
measurement from these maps.

The numerical map files, including `indices_512.npy`, are ignored by Git.
Only this README and the viewing notebook are tracked here. Save generated
figures and measured data vectors under the project's `chains/` directory.
Do not commit the simulation download.

## What a file contains

Rafael confirmed that these are **NSIDE 512 CosmoGrid shear maps**. Only
pixels where the DES footprint is nonzero are retained. The companion
`indices_512.npy` lists those HEALPix pixel indices. Thus the final array axis
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
to the wrong sky positions. The inspected index file is a common one-dimensional list. Its application
to the other `ncut` files still needs the producer's convention; do not infer
that separate cut labels imply different index files or identical sky rotations.

## Recovering the DES footprint

HEALPix partitions the sphere into equal-area pixels. NSIDE 512 implies
`12 * 512**2 = 3145728` full-sky pixels; the maps retain only a subset.
Each pixel covers about 47.2 square arcminutes, corresponding to a square
about 6.87 arcminutes across. That square is only a useful angular scale:
HEALPix pixels do not form a rectangular image.

HEALPix has two index conventions, RING and NESTED. Both use the same integer
range but assign different positions to those integers. **NSIDE alone does
not specify the ordering.** The notebook compares both interpretations with `astropy_healpix.HEALPix`.
RING gives longitude from -61.17 to 99.40 degrees (wrapped at 180 degrees)
and latitude from -67.84 to 5.45 degrees, with the recognizable DES southern
footprint and equatorial extension. NESTED spans all longitudes and reaches
+39.74 degrees in latitude with a disconnected pixel pattern. We therefore
use `order="ring"` for these previews and interpret coordinates as RA/Dec
from the footprint match; compare Figure 1 of the
[DES Y3 shape-catalog paper](https://academic.oup.com/mnras/article/504/3/4312/6211014).
Record this as geometric evidence rather than
as a producer-confirmed coordinate or shear-basis declaration.

The notebook plots the retained-pixel footprint and both components of two
maps in the first source bin. It uses at most 30,000 evenly selected array
positions per panel to keep rendering small; the displayed footprint is
therefore a preview. Both realizations use the same color range. Longitude
is wrapped at 180 degrees to display a footprint crossing zero. This is a
coordinate scatter plot, not an equal-area projection or a science estimator.
A scalar component plot does not rotate the spin-2 shear basis; any actual
correlation measurement must use the producer's shear-basis convention.

## Reading a filename

For example, `ShearMaps_DESY3_run_0000_ncut0_nreal17.npy` separates into:

| Part | Meaning and present evidence |
|---|---|
| `ShearMaps` | The stored quantities are shear-map values, as Rafael confirmed. |
| `DESY3` | The maps target DES Year 3; the precise source distributions and weights still need their producer record. |
| `run_0000` | Run identifier 0, padded to four digits. We interpret this provisionally as the parent density/sky realization. Confirm its link to the underlying CosmoGrid simulation before treating different runs as independent. |
| `ncut0` | Footprint-cut identifier 0. We interpret different cuts as survey patches obtained from a parent sky; the actual rotations and overlap need the producer's definition. |
| `nreal17` | Realization identifier 17. The working interpretation is a shape-noise draw at fixed run and cut, but the producer has not yet confirmed what is randomized. With labels starting at zero, it is the eighteenth labeled realization, not seventeen new independent skies. |
| `.npy` | NumPy's array file format. It records array shape and numerical type, not the scientific meaning of these labels. |

Thus the intended reading is **run 0, footprint cut 0, realization 17**.
The labels are identifiers, not source-bin indices, shear-component indices,
resolution settings or version numbers. Source bins and components are axes
inside each file; NSIDE 512 comes from Rafael's description and the folder.
The `_v2` in the folder name is a release label, not a noise realization.
We have not been given a change record explaining that release revision.

Two files differing only in `nreal` are expected to share the parent sky and
cut, while changing the realization draw. Two files differing only in `ncut`
are expected to select different patches from the same parent. These are
working interpretations to confirm, not metadata encoded by the NPY format.
Until confirmed, retain all three labels in measured-vector records and
keep the entire `run` group together in training/validation splits. In
particular, never pass the number of files as the independent-sky count.

## A file is not an independent Universe

The filename table gives the working interpretation, which still needs
producer confirmation before counting independent simulations. If several files
share a density realization and differ only in galaxy shape noise, they
cannot be counted as new independent realizations of cosmic structure.
Keep all descendants of a parent simulation together when assigning training
and validation sets. Do not average different footprints and call that one
DES survey; this changes the measurement whose covariance is being estimated.

Ask the producer to confirm the label meanings, parent simulation IDs and
cosmology, the inferred RING/equatorial convention, source-bin distributions, shape
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

## Covariance pilot and storage

The [measurement driver](../scripts/measure_covariance_maps.py) streams one
map at a time, with OpenMP limited to four workers and BLAS to one. It saves
16 coarse aperture moments per map and a small set of primary aperture
values under `../chains/`; it never copies a map or writes a per-map figure.
The first pass used cut zero, draws 0–3, and 48 run labels: 192 files. Its
generated tables occupied less than 1 MB. The notes, notebooks, and driver
are tracked; the downloaded arrays and generated measurements stay ignored.

From Cocoa, measure the bounded sample:

```bash
python projects/des_3pcf/scripts/measure_covariance_maps.py --max-runs 48
```

Measure all complete, settled files currently present:

```bash
python projects/des_3pcf/scripts/measure_covariance_maps.py --all
```

After another download batch, append files not previously measured:

```bash
python projects/des_3pcf/scripts/measure_covariance_maps.py --all --resume
```

Resume is for a completed prior pass with the same estimator and geometry.
It checks the saved input sizes and timestamps and the center table. Files
modified in the preceding two minutes, incomplete NPY payloads, and files
that change during reading are deferred. The notebook analyzes the small
text tables without reopening any maps. A local inventory is not a release
manifest; confirm completion and integrity against the producer's record.

The pilot integrates Gaussian tangential-shear filters at 30 and 60 arcminutes
around 64 fixed centers, separately in all four source-bin slots. Its raw
squared and cubed moments retain noise self-products. They are not the
published catalogue or finite-triangle estimator. The initial 48-run test
gave negative precision eigenvalues for the first- and second-order
corrections. Repeated draws dominated the measured fluctuations, and weak
cross-draw shear correlations do not establish the assumed shared sky.
Confirm whether these files contain signal plus noise or noise only, and
what `nreal` randomizes, before interpreting the covariance split physically.
