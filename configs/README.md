# Configuration files

1. [Schema and worked flows](#schema)
2. [Which example to use](#examples)
3. [Physical preparation keys](#physics)
4. [Mean estimator keys](#estimators)
5. [Fisher and precision keys](#fisher)

## Schema and worked flows <a name="schema"></a>

The YAML files describe examples, not validated DES forecasts.
The driver loads YAML with its data-only reader; JSON remains readable for
frozen test configurations. Every path inside a configuration is relative to the
configuration file, independently of the shell's directory. The driver
restricts OpenMP to one through six workers, sets every BLAS vendor to
one thread before importing numerical libraries, creates only the
requested output files, and installs no software.

This page defines the configuration keys and the file choices. The
[project instructions](../README.md#evaluate) give the worked
prepare/evaluate flow, the thread settings, the `--validate-config` and
`--overwrite` flags and the demonstration source file; the
[Fisher guide](../../../external_modules/code/cosmolike_core/cosmolike_notebook_utils/README.md#fisher) gives the derivative flow and the
[covariance guide](../covariance/README.md) gives the covariance flow.

## Which example to use <a name="examples"></a>

| File | Observable definition and available inputs |
|---|---|
| `des_y3_nla_demo_fisher.yaml` | Worked 480-row NLA demonstration with its recorded source processing and measured-edge PCHIP reconstruction. |
| `des_y3_tatt_demo_fisher.yaml` | The corresponding TATT demonstration, with its larger IA parameter space and separate Fisher base. |
| `des_y3_supplied_bihalofit.yaml` | The NLA demonstration with an explicit Python-produced baseline bispectrum and CAMB matter-density background. It writes separate physical tables and results. |
| `y3_full_filter_example.yaml` | A joint forecast definition with 400 xi± entries and 80 full-filter aperture moments. Preparation reads the pinned public Y3 midpoint curves. Area and per-bin shape-noise properties are unset. |
| `y3_finite_sas_example.yaml` | All 400 xi± and 80 aperture rows in the pinned public simulated Y3 FITS files, with their 480-by-480 covariance. Apertures integrate a finite side-angle-side shear grid. |
| `y6_full_filter_example.yaml` | A forecast design with 100 aperture moments at 3, 5, 10, 20 and 40 arcmin, plus 400 illustrative xi± entries. Source curves, survey area and noise properties remain unset. |
| `des_y3_tatt_fisher.yaml` | A full 480-row IA Fisher design using CFASTPT two-point TATT and the three-point TATT model, with explicitly recorded covariance/estimator limitations. |
| `des_y3_nla_fisher.yaml` | The same design with quadratic IA fixed and its derivative columns removed; a comparison conditional on the simpler model. |

The YAML demonstrations use an approximate 8–80 arcmin range for both SAS
side parameters; `y3_finite_sas_example.yaml` starts at 0.1 arcmin.
Neither configuration adopts the published xi scale mask, and exact
agreement with the public covariance's bin selection, averaging and
simulation pixel treatment remains unvalidated.

The Y6 xi bins and reference cosmology are design choices, not claims
about a released Y6 likelihood; the example invents no source curve,
measured data vector, covariance or calibration prior. The Y3
covariance's filename also does not justify inferring a number of
independent skies: the driver checks its block labels and dimensions when
it reads the file. [Paper targets](../README.md#estimator) lists the cut,
compression and likelihood choices that separate these examples from the
published analysis.

The full-filter templates also support an analytic covariance. Before
using them, supply a nonnegative source model and their missing
area/noise inputs, then prepare their own physical tables. The original
Y3 file in those templates contains negative samples, and the physical
reader refuses it. The demonstration's processed source is one explicitly
labeled alternative; the driver never inserts it silently.

## Physical preparation keys <a name="physics"></a>

`source.path` names an input in `data/`. `physics.tables` names a generated
file in `runs/`; create it with `run_3pcf.py prepare` before evaluating that
configuration. Output means, covariances and Fisher matrices also go in
`runs/`. The [data guide](../data/README.md) lists the preparation commands
and distinguishes generated results from the frozen comparisons in `tests/`.

`prepare` reads source-redshift midpoint curves from a FITS table and
reconstructs them with the configured source model. The worked
demonstrations use measured-edge PCHIP with an independent source grid;
the design templates use linear midpoint interpolation. Both normalize
the sampled curve and record the original integrals, and the reader
refuses to discard nonzero source support beyond the configured grid.
The [accuracy guide](../../../external_modules/code/cosmolike_core/cosmolike/3pcf/README.md#accuracy-controls) defines the
reconstruction choices and their endpoint policies.

CAMB then supplies power, distance and growth tables. Wavenumbers use
$`h\,\mathrm{Mpc}^{-1}`$, distances use $`\mathrm{Mpc}/h`$ and power uses
$`(\mathrm{Mpc}/h)^3`$. The saved growth is $`D(z)`$ with $`D(0)=1`$; in
a massive-neutrino model this scale-independent effective growth remains
an explicit approximation of the three-point matter/IA model.

The `physics` table grids include their endpoints, so `count` means the
number of stored nodes. The integration grids under `mean` and
`covariance` instead specify fixed Gauss-Legendre rules; a logarithmic
rule includes the change-of-variable factor in its returned weights.
Increasing a count or widening a range creates a different requested
calculation, and no adaptive loop keeps increasing the work until a
result appears stable. Accuracy comparisons belong to the
[accuracy guide](../../../external_modules/code/cosmolike_core/cosmolike/3pcf/README.md#accuracy-controls).

The halo-profile radius table in the full-filter examples uses 512
positive logarithmic nodes from 1e-7 to 0.5 radians, plus an explicit
zero node. `include_zero` is available only for a positive logarithmic
table, and `count` counts the positive nodes, so this example stores 513
entries. The [covariance guide](../covariance/README.md#refinement)
explains the inner-resolution requirement behind this spacing.

**Dense readers.** `physics.source_readers: false` prepares dense uniform
tables once and uses linear arithmetic-index reads during repeated
calculations; `physics.table_boost` controls that dense-grid spacing
separately from the input `k`, `z` and `kernel_nz` preparation grids.
Setting `source_readers: true` requests source cubic reads for an
implementation comparison; it is not the default numerical route.

**Matter-fit anchors.** `physics.fit_nz` selects the coarse redshift
preparation of matter-fit coefficients, independently of the dense
readers above.

**Bispectrum source.** `physics.bispectrum_source` defaults to `internal`.
Selecting `supplied` asks Python to prepare the positive, full BiHalofit
bispectrum before constructing the C model. This route requires
`physics.bispectrum_table` with uniform `z`, `logk`, `logu` and `v` axes,
plus an optional integer `boost` for dense C preparation. For sorted sides
$`a\leq b\leq c`$, these coordinates mean redshift, $`\ln b`$,
$`\ln(a/b)`$ and $`(c-b)/a`$. The logarithmic coordinates already contain
logarithms, so their grid objects use ordinary linear spacing. The
[BiHalofit guide](../python/des3pcf/README.md) explains the fit and its controls.

Run `prepare` for the supplied example before evaluating it: its physical
file must contain CAMB's `omega_m_z` array. An internal-route file lacking
that array is refused; the reader does not infer the missing background.
The baseline table uses fixed bounds and rejects out-of-domain queries.
`accuracy.bispectrum_boost` refines its coarse intervals, multiplied once
by `accuracy.accuracy_boost`. It leaves the dense-table ratio unchanged.
Saved evaluation, covariance and Fisher results record the actual baseline
hashes and fit controls; Fisher records the central model before taking
parameter displacements.

The supplied example uses 49, 193, 97 and 33 nodes on those four axes, with
dense boost 1. In the saved 480-row fiducial NLA comparison, these counts
change the data chi-squared by -0.0831549 relative to the internal fit;
all 400 two-point entries remain byte-identical. The two coarsest tested
tables exceed the group's 0.2 limit. The
[comparison record](../tests/studies/results/bihalofit_joint_y3.json)
identifies every input and both binaries. This one-point check does not
establish derivative accuracy across the forecast parameter space.

**Projection grids.** The prepared projection route under
`aperture_settings` exposes its triangle grids and dense-table boosts;
`projection_mode: "direct"` selects the explicit reference calculation.
Full-filter predictions need no prepared Legendre multipoles, whereas the
finite-SAS transform does; the example settings record that difference.
`nshape: 160` selects the production bounded shape-coordinate grid, and
the logarithmic/linear reference-grid counts apply when `reference_mode`
is true.

**Triplet batches.** `projection_batch_size: 4` limits how many source
triplets keep prepared projection tables together. Each batch shares
matter calculations; the calculation consumes one batch and releases it
before the next.

## Mean estimator keys <a name="estimators"></a>

`full_filter` is the default mean estimator. It integrates the Fourier
aperture filters over the configured ell and opening-angle quadratures,
and it is the observable whose analytic covariance the C calculation
returns. The [covariance guide](../covariance/README.md) defines the six
stored covariance terms, the gravitational-only IA assumption, the xi
`noise_mode` and pair-weight choices, the massive-neutrino options and
the zero-calibration requirement.

`finite_sas` is an explicit alternative for source comparisons. Side
edges use arcminutes, and `phi_bins` divides 0 to $`\pi`$ into uniform
bins. `angular_average: "point"` keeps the historical center-angle rule;
`"x_modes"` averages the X-projected Fourier modes before applying the
centroid rotation at the bin center. Neither choice is an exact integral
of the varying centroid rotation over the bin. The transform can use the
historical untapered FFT or the tapered coefficient window.
`ordered_legs: true` keeps ordered source legs; `false` requests the
historical transpose shortcut, which assumes a symmetric bispectrum.

`mean.two_point.mode: "isolated_nla"` projects the scalar/NLA spectra; it
cannot represent nonzero tidal-torquing, density-weighting or
velocity-shear terms in xi. `"core_tatt"` instead imports the existing
project extension at its explicit module directory and uses its unchanged
CFASTPT E/B calculation. The driver checks for the compiled `.so` rather
than importing the source loader stub when that binary is absent. The
adapter owns that ordinary core's mutable state during the run; mean
calls are serial at the Python level and C owns the OpenMP workers.

## Fisher and precision keys <a name="fisher"></a>

The [Fisher guide](../../../external_modules/code/cosmolike_core/cosmolike_notebook_utils/README.md#fisher) defines the derivative keys
(`fisher.steps`, `fisher.priors`, `fisher.bounds`, `fisher.fom_pairs`),
the saved arrays and the rank diagnostic. `fisher` reads one fiducial
covariance and holds it fixed throughout every parameter displacement.

PME is optional under `fisher.precision`:

```json
{
  "mode": "pme",
  "sample_covariance": {
    "path": "../data/joint_sample_covariance.npz",
    "key": "covariance"
  },
  "independent_count": 300,
  "order": 2,
  "sample_assumption": "independent_gaussian_vectors"
}
```

The count above illustrates the schema; it is not a DES realization
count. An NPZ matrix file needs an exact `row_labels` array beside
`covariance`, and it may be rank deficient. The
[Fisher guide](../../../external_modules/code/cosmolike_core/cosmolike_notebook_utils/README.md#fisher) states the required row agreement,
the independence assumption and the refusal of an indefinite precision.

Output NPZ files contain numerical and string arrays plus a JSON metadata
string; loading them never requires a Python pickle. Lossless ZIP storage
does not reduce the statistical vector: every xi± row, aperture row and
cross block remains present.
