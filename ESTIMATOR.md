# Matching the aperture estimator

1. [Separate a measurement from its theory quadrature](#definition)
2. [Side bins, cuts and representative lengths](#geometry)
3. [The archived scalar pixel window](#pixels)
4. [Run the comparison configuration](#run)
5. [Remaining checks](#remaining)

## Separate a measurement from its theory quadrature <a name="definition"></a>

An aperture label such as 14 arcmin does not uniquely define the measured
third-order statistic. The estimator also selects triangles, averages shear
products within bins, converts shear components to a chosen triangle frame,
and contracts those averages with aperture weights. A theory can match the
final labels while integrating different triangles. The packed public Y3
FITS files contain the final 80 values, labels and covariance; they do not
contain the original within-bin triangle weights or mean sides.

The calculation here exposes the conventions recovered from Rafael Gomes's
[archived Y3 code](https://github.com/rchgomes/3pcf_integrator/tree/d6098d326dcb03bc750adcc9627119f52d928944).
[The source-by-source review](tests/studies/estimator_conventions.md)
records the differences between that code, the paper's stated binning and
what the measured files establish. Implementing a named theory convention
makes a comparison possible; it does not resolve missing catalogue details.

## Side bins, cuts and representative lengths <a name="geometry"></a>

The original Y3 radial bank has 20 logarithmic bins from 0.5 to 80 arcmin.
The FFT transform needs that complete regularly spaced bank. A low-side cut
therefore zeros the corresponding aperture weights; it does not generate a
replacement logarithmic grid beginning at the cut. `side_min_arcmin: 8`
retains lower edges greater than or equal to 8 arcmin, namely original
indices 11 through 19. The same selection applies to both side axes and
preserves the reflected-triangle contribution. A notebook can instead pass
an explicit boolean `side_keep` array to `Map3Engine`.

The radial transform and aperture contraction use distinct coordinates.
`theta_lower` contains lower edges, and `dlnt` is their logarithmic width.
The default `representative_rule: geometric` uses
$`t_{\rm rep}=t_{\rm lower}\exp(\Delta/2)`$ in the aperture kernel and
radial measure. `area_weighted` uses the mean for the area measure
$`t\,dt`$:

$$
\frac{t_{\rm rep}}{t_{\rm lower}}
=\frac{2}{3}\frac{e^{3\Delta}-1}{e^{2\Delta}-1}
+\mathrm{representative\_offset}.
$$

The archived Y3 offset is -0.015226, labeled empirical in that code. It is
a change to a dimensionless factor, not an angular displacement. It was
used with the archived radial width and must not be transferred blindly
to narrower bins. Representatives outside their bins are refused. The Y6
code's analytic default uses zero offset; these are separate conventions.

The example uses the active archived 100 angular centers
$`\phi_j=(j+1/2)\pi/100`$. The methods paper describes 63 angular bins;
that difference remains explicit. `angular_average: point` retains point
samples. The optional `x_modes` averages Fourier modes before the centroid
rotation; it is not a catalogue-weighted average after that rotation.

## The archived scalar pixel window <a name="pixels"></a>

A finite-area pixel averages a field over angle. The archived theory uses
one scalar window per Fourier leg,
$`B^W=W(\ell_1)W(\ell_2)W(\ell_3)B`$. The required input is the
HEALPix table's `TEMPERATURE` column, following `healpy.pixwin` with its
scalar default. Its polarization column is a different choice. Applying a
scalar window to this prediction reproduces a declared archived convention;
it does not establish that the catalogue estimator is exactly that field
average.

The [pinned manifest](data/pixel_window_manifest.json) identifies the official
72-kB FITS by commit, byte count and SHA256. At nside=1024, the default
retains integer multipoles 0 through 3071, although the FITS contains more
rows. The reader uses linear interpolation and zero above 3071. The final
retained window value is about 0.646, so this extension is discontinuous.
Its tiny floating-point excess above one at ell=0 is preserved.

The native implementation stores independent copies in the physical model
and each prepared projection grid. A grid remains usable after deleting its
model. Multipole preparation includes $`W(\ell_3(\mu))`$ inside the
angular integral. The factors for the first two legs are applied after
interpolating the prepared multipoles, which prevents interpolation from
creating nonzero support above their cutoff. Both full-filter callbacks
apply all three factors once. No zero is inserted into a logarithmic
bispectrum interpolation table.

Raw matter, projected-bispectrum queries and ordinary xi predictions retain
their unsmoothed definitions. The pixel option applies to the three-point
angular observables only, matching this particular archived route. A user
who needs smoothing in a two-point estimator must declare and validate that
separately. The current analytical covariance has no matching pixel factors
and refuses a smoothed physical-model owner. A supplied measured covariance
still needs estimator agreement; a successful matrix factorization cannot
establish that agreement.

## Run the comparison configuration <a name="run"></a>

[des_y3_archived_geometry.yaml](configs/des_y3_archived_geometry.yaml)
combines the recovered radial bank, lower-edge selection, empirical side
representatives, 100 angular centers and scalar pixel window. Its physical
inputs and numerical settings remain the declared TATT demonstration. It
uses exact $`\log(160)/20`$ for the radial width rather than the rounded
0.253758 in the archived configuration. It is a theory-convention comparison,
not a published posterior reconstruction.

Run from the activated `Cocoa/` directory with the existing project build
and prepared demonstration sources/tables.

**Step :one:**: retrieve or verify the pinned pixel window.

```bash
python projects/des_3pcf/scripts/fetch_pixel_window.py
```

**Step :two:**: evaluate the explicitly declared comparison vector.

```bash
python projects/des_3pcf/scripts/run_3pcf.py evaluate projects/des_3pcf/configs/des_y3_archived_geometry.yaml
```

The runtime reader performs no download. The input SHA256, retained ell
range, interpolation and upper extension enter the saved provenance.
Replacing the window requires a new model/provider; notebook callers can
construct that owner without restarting the kernel.

## Remaining checks <a name="remaining"></a>

The unit checks establish array ownership, the three-leg factors, exact
first/second-leg cutoff support, the complex aperture contraction, and
restoration after changing a retained plan's physical window. They compare
direct angular sums independently and exercise both C++ adapters. They do
not certify the survey prediction's angular resolution. The third-leg
cutoff introduces a jump inside its angular integral; refine `nshape`,
angular sampling and output multipole grids before interpreting small
residuals. Compare each convention separately at fixed physical inputs.

Exact catalogue agreement still requires the original triangle selection,
within-bin means/weights, angular conversion and mask treatment. The chosen
lower-edge cut is explicit but not proven to be the original measurement
rule. Exact likelihood reproduction also needs the published scale cuts,
source treatment, priors and fixed compression matrix. The independent
mock-vector calibration, shared-sky correlations and covariance sign
provenance remain open. These limitations are separate from the implemented
[compression and finite-simulation choices](COMPRESSION.md).

**We thank Sunao Sugiyama and Rafael Gomes for the original theory and
analysis implementations**, and the HEALPix/healpy authors for the pixel
window tables. Sources and retained licenses are listed in the
[core credits](../../external_modules/code/cosmolike_core/cosmolike/3pcf/CREDITS.md).
