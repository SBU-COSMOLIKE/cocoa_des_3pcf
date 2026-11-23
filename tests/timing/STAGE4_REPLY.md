# Reply: pixel window and timing tests before stage 4

23 November 2026. Response to the discussion brief at `eb016f6`.
This document specifies the proposed migration; it does not claim that the
replacement timing driver or notebook changes have been implemented.

## Decisions

Retire the private prediction package. Preserve the archived pixel-window
comparison in the existing code-comparison notebook, and preserve repeatable
scaling tests through the normal Cobaya likelihood. The user explicitly
requests **`tests/benchmark_scaling.py`**, with instructions in
[the timing README](README.md). Moving the old package into `tests/` without
removing its private CAMB producer would not satisfy this requirement.

There is no need to add an unused shared Python module or a production YAML
option solely to preserve an archived comparison.

## 1. Which pixel window describes the measurement?

### Catalogue measurements and pixel averages are different observations

A catalogue contains galaxy positions, weights and shear estimates.
TreeCorr can correlate those entries directly. Such a measurement does not
acquire a HEALPix pixel window merely because its covariance was estimated
using pixelized simulations.

The DES Y3 data paper describes the measured third-order aperture statistic
as a calculation from the shear catalogue without convergence-map
reconstruction. It uses TreeCorr multipoles, converts them to SAS bins and
then applies the triangle cuts. Therefore the catalogue likelihood should
not acquire a HEALPix window by default. Map-based mocks must separately be
checked for resolution effects relative to that catalogue estimator.
[DES Y3 data paper, section IV.1](https://arxiv.org/html/2508.14018v2#S4.SS1)

A pixel average is different: replacing the field inside each pixel with an
average removes some angular detail. An isotropic approximation represents
this operation by a harmonic window. For a scalar field it gives

```math
C_\ell^{\mathrm{pixel}} = W_\ell^2 C_\ell.
```

For a bispectrum the corresponding approximation has one factor on each
leg:

```math
B^{\mathrm{pixel}}(\ell_1,\ell_2,\ell_3)
=W_{\ell_1}W_{\ell_2}W_{\ell_3}
 B(\ell_1,\ell_2,\ell_3).
```

These equations require that the window describe the actual map-making
operation. Evaluating a band-limited field at pixel centres does not, by
itself, perform a pixel average. Likewise, smoothing a scalar convergence
map before converting it to shear is not the same construction as averaging
a spin-2 shear field within pixels. The latter requires attention to the
spin basis and polarization response. Unequal catalogue weights and a
mask can introduce additional departures from an isotropic window.
[HEALPix pixel-window definition](https://healpix.sourceforge.io/html/intro_Pixel_window_functions.htm)

Consequently, the scalar `TEMPERATURE` column is defensible for reproducing
a specified scalar-smoothing convention. It is not a general prescription
for every shear map. Do not replace it automatically with the polarization
column either: first identify how the maps were produced.

### Preserve the archived convention exactly, but label its scope

Keep the input FITS file and
[its provenance manifest](../../data/pixel_window_manifest.yaml), the native
optional window and [its tests](../data_vector/test_pixel_window.py).
Place the short loader and its actual use in
[the existing code-comparison notebook](../../EXAMPLE_COMPARE_CODES.ipynb).
That is a real consumer and the appropriate home for a reproduction-only
convention. Remove the private package only after that comparison works.

The notebook should state all four choices explicitly: scalar column,
retained multipoles, linear interpolation and zero extension. In particular,
zeroing the window above `3*nside - 1` is an additional band limit. It is
not the physical definition of the pixel window. The default output length
of `healpy.pixwin` is `3*nside - 1`, whereas the official tables extend
farther. Preserve the old cutoff for that comparison, without promoting it
to a general rule.
[healpy API](https://healpy.readthedocs.io/en/latest/generated/healpy.sphtfunc.pixwin.html)

For any later production map likelihood, establish the map transfer first.
Apply it consistently to the mean and covariance of the measured fields;
a signal power spectrum carries two leg factors, and a bispectrum three.
Noise depends on where and how it was introduced, so it cannot automatically
receive the same factors. Keep the existing refusal of unmatched analytical
covariances. Finite-SAS triangle selection and pixel smoothing are separate
estimator issues; solving one does not solve the other.

Ask Rafael which archived prediction used this scalar window and cutoff,
whether its target was a map or the catalogue data, and whether the supplied
maps already contain that smoothing. This prevents applying it twice.

## 2. Keep the scaling test, under tests/

### The replacement must measure real recomputation

The scaling test is a requested maintenance capability even if no paper
quotes its results. Its driver belongs at `tests/benchmark_scaling.py`.
Use the numbered Cobaya examples and the project's existing likelihood
methods. Do not retain a separate prediction class, CAMB producer or
configuration schema to support the benchmark.

Preserve the useful experiment:

- Save ten cosmology-and-nuisance parameter points and reuse the same points
  for every thread count. Use a separate warm-up point.
- Run NLA and TATT separately, at the requested OpenMP counts, with BLAS
  fixed to one. Initialize each thread configuration in a fresh process.
- Keep the estimator, selected rows, theory choices and accuracy controls
  identical across thread counts. Record those settings with the results.
- Change the physical parameters through ordinary likelihood setters so
  parameter-dependent work is recomputed. Fixed geometry and FFT plans
  should retain their normal reuse.
- Check that vectors agree across thread counts to the established
  numerical tolerance. A faster calculation with different outputs is not
  a successful scaling result.

Do not turn a count of repeated unchanged calls into a performance claim.
Also do not rebuild the entire model for every point if a real chain retains
its geometry: that measures a different workload.

### Nuisance-only timings cannot isolate CAMB

The brief proposes separating CAMB by timing another run in which only
nuisance parameters change. That does not isolate CAMB. A cosmology change
also rebuilds power-dependent tables, the matter bispectrum and other
cosmology-dependent work. A nuisance-only call can reuse those quantities.
Subtracting the two runtimes would attribute their cost to CAMB incorrectly.

Similarly, `cached=False` is not a promise that CAMB lies outside the
interval; the actual component calls and timing boundaries must be checked.

Report clearly separated quantities from the same physical evaluations:

1. **Full Cobaya evaluation:** the complete elapsed evaluation time.
2. **CAMB component:** its measured interval within that evaluation.
3. **Other theory and likelihood work:** includes the supplied bispectrum,
   optional feedback producer, native parameter-dependent preparation and
   projection.

The user's CAMB-excluded timing should retain item 3. Moving BiHalofit from
C into a theory block must not make its cost disappear from the reported
prediction time. Use existing component timing facilities where adequate;
otherwise keep the necessary measurement code local to the test, without
adding a runtime framework.

A second diagnostic may prepare all providers through Cobaya before timing
only the native calculation. Label it **native preparation and projection,
all provider calculations excluded**. It answers a useful OpenMP question,
but is not interchangeable with the CAMB-excluded total. A nuisance-only
benchmark may also be useful as a separately named fast-update measurement.

### LSST control and historical records

Keep the LSST comparison available through LSST's own likelihood and test
conventions. DES should not own a private LSST prediction implementation.
The shared scaling plot can combine the two projects' saved timing records;
this does not require a shared execution framework.

Retain the dated timing records and their implementation hashes as
historical evidence. They describe the old workload, not current production
performance. The replacement driver need not reproduce their exact seconds.
Its README should explain the changed timing boundary before comparing
results. Store generated benchmark products under ignored `chains/`, and
retain only intentional test evidence under `tests/timing/`.

Do not delete the old harness before the replacement has a small functional
check: provider use, changed-parameter recomputation, recorded timing
boundaries and output agreement. Long scaling runs can follow on a quiet
machine. There is no need to retain thousands of lines merely to orchestrate
these checks, or to promise a particular line count before implementing them.

## 3. Other capabilities mentioned in the brief

Track the fixed-response feedback-amplitude experiment and the cross-bin
noise/forecast-row capability explicitly when deleting their old consumers.
Their scientific meaning should survive in the relevant notebook, theory
block or covariance interface if still required. Do not silently describe
either as migrated when no replacement consumer exists, and do not preserve
the private prediction framework just to keep them callable.

The immediate agreement is therefore: notebook-owned archived pixel-window
reproduction, no default catalogue smoothing, and a repeatable scaling
driver in `tests/` using the ordinary Cocoa prediction path.
