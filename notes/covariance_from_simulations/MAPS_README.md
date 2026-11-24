# DES Y3 shear maps at NSIDE 512

**Thanks to Rafael Gomes for supplying these CosmoGrid shear maps.** They
support the covariance work associated with [Gomes et al.](https://arxiv.org/abs/2503.03964)
and [his analysis code](https://github.com/rchgomes/3pcf_integrator).
The local inventory settled at 15,997 map files on 21 November 2026. Every
present map was measured by the coarse diagnostic. Three combinations of the
observed labels are absent: run 168/cut 2/draws 7 and 19, and run 168/cut
3/draw 19. There are 200 distinct run labels in this download. Rafael
confirmed on 23 November 2026 that the paper used 199 parent simulations and
only `nreal0`; the download also contains repeated noise realizations. Its
extra parent label still needs to be mapped to the published selection.

Both simulation folders are on the group's server, pinned by committed
SHA-256 manifests:
[data/rafael_maps_cov_512_v2.sha256](../../data/rafael_maps_cov_512_v2.sha256)
lists 15,999 files (378 GB: the 15,997 maps plus `indices_512.npy` and
`missing.txt`), and
[data/rafael_datavector_2pt3pt.sha256](../../data/rafael_datavector_2pt3pt.sha256)
lists 800 files (21 MB). The manifests pin the redistributed copies; Rafael
supplied no checksums of his own.

**Step :one:**: from `projects/des_3pcf`, download one folder or both into
`chains/rafael/`:

```bash
bash scripts/download_rafael_sims.sh maps_cov_512_v2
```

The first argument `datavector_2pt3pt` selects the measured vectors instead,
and `all` selects both folders. The script keeps a file only after its
SHA-256 matches the manifest, skips files already present with the right
hash, and stops on an existing mismatched file, so an interrupted download
can simply be restarted.

The map archive is at `chains/rafael/maps_cov_512_v2/`, beside
`chains/rafael/datavector_2pt3pt/`. The notebook reads the maps from that
location; no map copies are needed.

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
[aperture-moment pilot](../../EXAMPLE_COVARIANCE_MAPS.ipynb) now studies a bounded
measurement from these maps.

The numerical map files, including `indices_512.npy`, are ignored by Git.
The guide and viewing notebook are tracked under notes/covariance_from_simulations; the downloaded arrays live under chains/rafael/maps_cov_512_v2/. Save generated
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
coordinate basis. The initial inspection checked the first 4,096 pixels of each component
in these two files. The later all-map diagnostic checked finite values
throughout every present map while measuring it. These numerical checks do
not establish archive completeness or replace producer checksums. NPY contains array shape and type, but not these scientific
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
Rafael confirmed RING ordering on 23 November 2026. The coordinate and
shear-basis conventions remain distinct from the pixel-order declaration.

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
| `run_0000` | Parent CosmoGrid lightcone simulation 0, padded to four digits; confirmed by Rafael. |
| `ncut0` | DES cutout 0 of four per parent full sky. Rafael describes these four cutouts as independent survey realizations. |
| `nreal17` | Noise realization 17 on the same sky and cutout, confirmed by Rafael. The paper used only nreal0; extra noise draws are not additional independent skies. |
| `.npy` | NumPy's array file format. It records array shape and numerical type, not the scientific meaning of these labels. |

Thus the intended reading is **run 0, footprint cut 0, realization 17**.
The labels are identifiers, not source-bin indices, shear-component indices,
resolution settings or version numbers. Source bins and components are axes
inside each file; NSIDE 512 comes from Rafael's description and the folder.
The `_v2` in the folder name is a release label, not a noise realization.
We have not been given a change record explaining that release revision.

Two files differing only in `nreal` share the parent sky and cutout while
changing the noise draw. Two files differing only in `ncut` select different
cutouts of the same parent. Rafael confirmed these label meanings. Retain
all three labels in measured-vector records and count each cutout (one
`run`, `ncut` pair) once: following Rafael's construction, the four cutouts
of a parent sky are independent survey realizations, while repeated `nreal`
draws of one cutout are not. In particular, never pass the number of files
as the realization count.

## A file is not an independent Universe

The filename table records Rafael's confirmed interpretation. Files that
share a cutout and differ only in `nreal` (galaxy shape noise on the same
sky) are not new independent realizations of cosmic structure; the paper
used `nreal0` alone. The four `ncut` cutouts of one parent sky do count as
independent realizations: Rafael built the released covariance that way
(796 mocks = 199 parents x 4 cutouts), and this project follows his choice.
Do not average different footprints and call that one DES survey; this
changes the measurement whose covariance is being estimated.

Rafael confirmed the labels and RING ordering, and noiseless convergence and
shear maps can be generated from CosmoGrid or from GowerSt2 (DES Y6). Still
obtain the exact parent selection, cosmology, coordinate basis, source-bin
distributions, shape-noise prescription, shear signs, footprint weights and
pixel occupancies.
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

The separate `chains/rafael/datavector_2pt3pt/` download contains the
NSIDE 1024 paper-estimator measurements: 400 matched NPY/FITS pairs, the
four independent DES cutouts of each of 100 fiducial CosmoGrid simulations.
Each `.npy` holds one joint vector in the released data-vector row order
(200 xi-plus, 200 xi-minus and 80 Map3 entries), and its `.fits` carries
all bin and scale information; the NPY ordering agrees exactly with the
FITS columns. The paper covariance used 199 simulations (796 mocks =
199 x 4 cutouts); Rafael retains only these first 400 vectors. They are not
the coarse moments measured by the pilot below.

The [PME notebook](../../EXAMPLE_COVARIANCE_PME.ipynb) explains how simulations
can correct an analytic precision matrix. The [introductory notes](../covariance_from_simulations/covariance_from_simulations.pdf)
derive the method, explain repeated-noise realizations and describe the
validation needed before using the correction in a forecast. Neither artifact
claims that a DES covariance has already been calibrated from this download.

## Covariance pilot and storage

The [measurement driver](../../scripts/measure_covariance_maps.py) streams one
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
corrections. With all 200 primary run labels, the second-order precision is
positive and its mean validation weight is closer to one, but that single
average does not establish calibrated parameter errors. The all-map pass
saved about 12 MB of tables, with no temporary map copies. Repeated draws dominated the measured fluctuations, and weak
cross-draw shear correlations do not establish the assumed shared sky.
Rafael now confirms repeated noise realizations on the same skies. The
historical weak cross-draw signal remains a measurement diagnostic to
understand; it does not overturn that provenance.

## Compare with the released covariance

The project includes Rafael's 480-row joint covariance and its paired row
definitions. The [map notebook](../../EXAMPLE_COVARIANCE_MAPS.ipynb) audits these
inputs and reads a bounded xi-plus comparison using source-bin slot four,
cut zero, draw zero, and the released bins from 39.622 to 99.527 arcminutes.
The separate [pair driver](../../scripts/measure_covariance_xip.py) uses exact
angular binning and saves only small text vectors. It needs these optional
packages in addition to the viewing notebook's astropy-healpix dependency:

```bash
python -m pip install --no-deps treecorr==5.1.2 LSSTDESC.Coord==1.3.0
```

From Cocoa, measure 32 primary run labels:

```bash
python projects/des_3pcf/scripts/measure_covariance_xip.py --runs 32
```

The comparison is conditional on equal pixel weights and the specified
shear basis, and it does not remove the 512-versus-1024 resolution difference.
If the released covariance shares these parent skies, it is not independent
validation. Every released covariance used 796 mocks, the four cutouts
of each of 199 parent simulations; the Map3-only matrix is a block of the
joint xi/Map3 one. The COVMAT headers are typos: `NSIM=500` in the joint
`xip_xim_map3_covariance_8ARCMINCUT_25Feb25.fits` and in the observed
`real_map3_desy3_unblind_scalecut3pcf_796simcov_2025Mar11.fits`, and
`NSIM=400` in `sim_map3_emu-NLA-cosmoCosmogrid-24FEBRUARY2025_WITH_COV.fits`.
Preserve the original FITS files and record 796 as the producer-confirmed
count in analysis metadata. Treat the cutouts (ncut) as independent and the
noise realizations (nreal) of one cutout as not independent. The 400 local
vectors of datavector_2pt3pt may serve as the first simulation input of the
covariance tests; their row order is that of the released data vectors.
The Map3 block is identical to the standalone Map3 file, but the standalone
two-point covariance differs from the joint file's two-point block. Use the
joint reference consistently; do not splice the two products together.

For 32 primary run labels, the four measured variances are 1.41, 1.81, 1.58
and 1.05 times the released variances. Leave-one-run-out standard errors are
0.37, 0.42, 0.45 and 0.28. This is a comparison of single-survey fluctuations;
the covariance is not divided by 32. Matching scales encourages further
tests, but these numbers do not establish equal matrices or shared provenance.

The driver also correlates draws zero and one of eight runs. If each pair
shares one sky and has independent noise, their cross-correlation retains
that sky signal. Eight pairs of different run labels provide a control.
The notebook derives this test, plots it against the same-map measurement,
and explains why small shared pixel variance alone cannot classify a map as
noise-only. These measurements help interpret the labels; they do not replace
the producer's account of what was randomized.

The completed eight-pair test finds a same-run cross signal consistent with
zero, while the same-map signal is clearly positive. It challenges the
fixed-sky interpretation under independent, uncorrelated pixel noise;
it does not overturn Rafael's subsequent confirmation of repeated noise
on fixed skies. These historical label tests
remain in the main map notebook.

The [comparison note for Rafael](../covariance_comparison/covariance_comparison.pdf)
and its [notebook](../covariance_comparison/EXAMPLE_COVARIANCE_COMPARISON.ipynb)
now focus only on two matrix comparisons. First, all 15,997 measured
aperture vectors enter both the direct covariance and the
second-, fourth- and sixth-order Eifler estimates, with the starting matrix fixed
and no sample partition.
Second, all 32 available matching xi-plus measurements are compared with
the released four-bin DES covariance. Both comparisons have normalized
difference plots and split-triangle correlation plots. Compact records
reproduce the figures without reopening the maps. The nominal file count
used in the Eifler square correction is stated explicitly; it is not an
established independent-sky count.

The separate [map-generation note](../gptg_simulations/gptg_simulations.pdf)
and [worked notebook](../gptg_simulations/EXAMPLE_GPTG_SIMULATIONS.ipynb)
fit an initial point-transformed Gaussian model to 32 supplied maps and save
64 new four-bin maps with a stated Gaussian noise model under
`chains/gptg_method/maps/`. Their names begin `GPTG_DES_`; they are not
Rafael's original realizations. The note explains the fit approximations,
source-bin cross correlations, noise-corrected cubic moments and current
covariance differences. These generated maps are not yet validated replacements
for the supplied N-body simulations.

New generated-map ensembles use float32 files and float64 calculations. The
64-map check found a maximum relative precision-direction change of 5.3e-8
from storage rounding. The original float64 pilot remains a small reference
for repeating that test; Rafael's downloaded inputs are never converted.
