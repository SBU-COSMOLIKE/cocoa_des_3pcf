# Running the three-point pipeline

The JSON and YAML files describe examples, not validated DES forecasts.
They use the same schema; YAML is loaded with its safe data-only reader.
Paths are relative to the configuration file. We assume Cocoa is activated
with `source start_cocoa.sh`, the current directory is
`Cocoa/projects/des_3pcf/`, and the [project thread settings](../README.md#evaluate)
are applied before starting Python.
For the worked Y3 demonstration, first create the processed source FITS
using [the project instructions](../README.md#evaluate). Its explicit
clipping assumption is recorded separately from the original measurement.
The commands below use that demonstration and its supplied joint covariance.

**Step :one:**: inspect the preparation inputs.

```bash
python scripts/run_3pcf.py prepare configs/des_y3_nla_demo_fisher.yaml \
  --validate-config
```

**Step :two:**: prepare the physical tables.

```bash
python scripts/run_3pcf.py prepare configs/des_y3_nla_demo_fisher.yaml
```

**Step :three:**: evaluate the declared mean vector.

```bash
python scripts/run_3pcf.py evaluate configs/des_y3_nla_demo_fisher.yaml
```

**Step :four:**: calculate the requested Fisher derivatives.

```bash
python scripts/run_3pcf.py fisher configs/des_y3_nla_demo_fisher.yaml
```

The separate full-filter templates also support an analytic covariance.
Before using them, supply a nonnegative source model and their missing
area/noise inputs, then prepare their own physical tables. The original
Y3 file named in those templates contains negative samples and is refused
by the physical reader. The demonstration's processed source is one
explicitly labeled alternative; it is not inserted silently. The covariance
command for a fully specified full-filter example is:

```bash
python scripts/run_3pcf.py covariance configs/y3_full_filter_example.json
```

`--validate-config` checks the versioned configuration and reports missing
files or unset survey properties. It does not import a compiled extension,
run CAMB, or evaluate a physical integral. Remove the flag to execute the
chosen command. An existing output is preserved unless `--overwrite` is
supplied. The driver creates only the requested output directories/files;
it installs no software and downloads no data.

`prepare` reads source-redshift midpoint curves from a FITS table and uses
the declared reconstruction. The worked demonstrations use measured-edge
PCHIP with an independent source grid; the older design templates select
linear midpoint interpolation. Both normalize the sampled curve and record
the original integrals. The reader refuses to discard
nonzero source support beyond the configured grid. CAMB then supplies power,
distance and growth tables. Wavenumbers use h/Mpc, distances use Mpc/h and
power uses (Mpc/h)^3. The saved growth is D(z), with D(0)=1. In a massive
neutrino model its scale-independent effective growth remains an explicit
approximation of the three-point matter/IA model.

The `physics` table grids include their endpoints. `count` therefore means
the number of stored nodes. The integration grids under `mean` and
`covariance` instead specify fixed Gauss-Legendre rules; a logarithmic rule
includes the change-of-variable factor in its returned weights. Increasing
a count or widening a range creates a different requested calculation.
There is no adaptive loop that keeps increasing the work until a result
appears stable. Accuracy needs separate comparisons of those choices.

The stored halo-profile radius table uses 512 positive logarithmic nodes
from 1e-7 to 0.5 radians, plus an explicit zero node. Its `include_zero`
option is available only for a positive logarithmic table; `count` counts
the positive nodes, so this example has 513 entries. Resolving the inner
halo profile needs close spacing near zero. A uniform table over that full
range does not provide the same inner resolution. These are explicit
starting grids, whose refinement must be checked for the requested rows.

`physics.source_readers: false` prepares dense uniform tables once and
uses linear arithmetic-index reads during repeated calculations. The
`physics.table_boost` integer controls that dense-grid spacing separately
from the input `k`, `z` and `kernel_nz` preparation grids. Setting
`source_readers: true` requests source cubic reads for an explicit
implementation comparison. It is not the default numerical route.
`physics.fit_nz` separately selects the coarse redshift preparation of
matter-fit coefficients. The prepared projection route under
`aperture_settings` also exposes its triangle grids and dense-table boosts;
`projection_mode: "direct"` selects the explicit reference calculation.
Full-filter predictions do not need prepared Legendre multipoles, whereas
the finite-SAS transform does. The example settings record that difference.
Their `nshape: 160` selects the production bounded shape-coordinate grid;
the named logarithmic/linear reference-grid counts apply when
`reference_mode` is true. `projection_batch_size: 4` limits how many source
triplets retain prepared projection tables together. Each batch shares
matter calculations, is consumed, and is released before the next batch.

## Which example to use

| File | Observable definition and available inputs |
|---|---|
| `des_y3_nla_demo_fisher.yaml` | Worked 480-row NLA demonstration with recorded source processing, measured-edge PCHIP and the current numerical studies. |
| `des_y3_tatt_demo_fisher.yaml` | The corresponding TATT demonstration, retaining its larger IA parameter space and separate Fisher base. |
| `y3_full_filter_example.json` | A joint forecast definition with 400 xi± entries and 80 full-filter aperture moments. The pinned public Y3 midpoint curves are used for preparation. Area and per-bin shape-noise properties are unset. |
| `y3_finite_sas_example.json` | All 400 xi± and 80 aperture rows in the pinned public simulated Y3 FITS files, with their supplied 480-by-480 covariance. Apertures integrate a finite side-angle-side shear grid. |
| `y6_full_filter_example.json` | A forecast design with 100 aperture moments at 3, 5, 10, 20 and 40 arcmin, plus 400 illustrative xi± entries. Source curves, survey area and noise properties must be supplied. |
| `des_y3_tatt_fisher.yaml` | A full 480-row IA Fisher design using CFASTPT two-point TATT and the three-point TATT model, with explicitly recorded covariance/estimator limitations. |
| `des_y3_nla_fisher.yaml` | The same design with quadratic IA fixed and its derivative columns removed; a comparison conditional on the simpler model. |

[EXAMPLE_FISHER_IA.ipynb](../EXAMPLE_FISHER_IA.ipynb) reads matching saved
demonstration means and full Fisher results, or runs them if absent. Its
expensive cells therefore execute real calculations; there is no disabled
production switch. Reuse verifies input and implementation identities.
The [accuracy notebook](../EXAMPLE_FISHER_ACCURACY.ipynb) reads completed
refinement checks and lists missing ones explicitly.
The new designs use an approximate 8–80 arcmin range for both SAS side
parameters, whereas the older finite-SAS JSON example starts at 0.1 arcmin.
Exact agreement with the public covariance's retained bins, averaging and
simulation pixel treatment remains unvalidated. Neither configuration
silently adopts the missing published xi scale mask.

The Y6 xi bins and inherited reference cosmology are design choices, not
claims about a released Y6 likelihood. No source curve, measured data vector,
covariance or calibration prior has been invented. The Y3 covariance's
filename also does not justify inferring a number of independent skies.
Its block labels and dimensions are checked when it is used. The examples
do not apply the original source analysis's scale masks, MOPED compression,
or Percival likelihood. Matching that analysis needs a separate comparison
of source curves, IA conventions, priors and estimator definitions.

`full_filter` is the default mean estimator. It integrates the Fourier
aperture filters over the supplied ell and opening-angle quadratures. The
analytic C covariance uses this observable. Its six outputs are Gaussian,
bispectrum squared, power times one-halo trispectrum, power times
bispectrum, connected one-halo, and selected halo-abundance response. The
last term does not include every growth, dilation or tidal SSC response.
The covariance is gravitational only: varying TATT in the mean does not
add connected nonlinear-IA covariance. These assumptions are stored with
every result.

For xi covariance the full-filter examples choose `noise_mode: 1`,
`large_area: 1` and uniform angular-area pair weights. The pure shape-noise
term then uses the continuous annular-bin overlap from the actual xi lower
and upper edges. This replaces the finite-ell approximation to that white
noise contribution; signal and signal-noise terms retain their explicit
quadrature. `noise_mode: 0` instead declares the finite-band calculation.
Custom pair weights or a finite survey-window calculation require that
finite-band mode until a matching continuous noise treatment is supplied.

The analytic halo covariance uses a single matter fluid and
P_linear(k,0)D(z)^2. It does not use the ordinary core's P_cb and
scale-dependent halo variance treatment for massive neutrinos. Its default
`neutrino_model: "massless"` refuses a nonzero mnu. The two full-filter
examples explicitly select `"single_fluid_effective_growth"` because their
illustrative reference cosmology has nonzero mnu. That selection records a
physical approximation; it does not establish its accuracy for a forecast.
The analytic covariance also requires zero fiducial multiplicative
calibrations m_i, because its lensing weights do not include those factors.
The Fisher calculation can vary m_i in the mean while retaining this one
zero-calibration fiducial covariance. A nonzero fiducial calibration is
refused rather than ignored.

`finite_sas` is an explicit alternative for source comparisons. Side edges
are specified in arcminutes, and `phi_bins` divides 0 to pi into uniform
bins. `angular_average: "point"` retains the historical center-angle rule;
`"x_modes"` averages the X-projected Fourier modes before applying the
centroid rotation at the bin center. Neither choice is an exact integral
of the varying centroid rotation over the bin. The historical untapered
FFT and a corrected coefficient window are selectable. Ordered source legs
use `ordered_legs: true`; setting it false requests the historical
transpose shortcut, which assumes a symmetric bispectrum. An analytic
full-filter covariance is refused for a finite-SAS Fisher calculation.

## Two-point physics and statistical choices

`mean.two_point.mode: "isolated_nla"` projects the supplied scalar/NLA
spectra. It cannot represent nonzero tidal-torquing, density-weighting or
velocity-shear terms in xi. `"core_tatt"` instead imports the named existing
project extension and uses its unchanged CFASTPT E/B calculation. Its
module directory is explicit. The driver checks for the compiled `.so`,
rather than importing the source loader stub when that binary is absent.
The adapter owns that ordinary core's mutable state during the run; mean
calls are serial at the Python level and C owns the OpenMP workers.

`fisher` reads one fiducial covariance and holds it fixed throughout every
parameter displacement. It saves the full mean, exact row labels, parameter
names, derivatives at h and h/2, both Fisher matrices, and their numerical
choices. `priors` maps a varied parameter to a Gaussian standard deviation.
An empty object means no Gaussian prior. Parameter bounds constrain the
central finite-difference stencil; they are not Gaussian priors. No
pseudoinverse is substituted for an unresolved Fisher direction: the
derivatives and information are saved with a named rank diagnostic, and no
marginalized FoM is reported for that case.

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

The count above illustrates the schema; it is not a DES realization count.
The supplied matrix must describe the entire same joint vector. An NPZ
requires an exact `row_labels` array beside `covariance`, and it may be
rank deficient. The analytic matrix must be positive definite and fixed
independently of the sample covariance. Correlated patches do not provide
independent realizations. Subtracting a two-point block and treating the
remaining matrix as Wishart is not this calculation. PME diagnostics and
the resulting precision are saved; a nonpositive precision is refused
before a Fisher contraction.

Output NPZ files contain numerical/string arrays and a JSON metadata string;
they never require loading a Python pickle. Lossless ZIP storage does not
reduce the statistical vector. Every xi± row, aperture row and cross block
remains present.

For an analytic covariance command, `covariance.save_components` defaults
to `false`. The file then contains the final covariance, row definitions,
the covariance model's mean and metadata. Setting it to `true` also saves
the six-term cube and each named family. Both choices call the same C entry
and compute the same six contributions; this option reduces saved
diagnostics and file size, not integral count or C peak memory.

The driver restricts OpenMP to one through four workers and sets BLAS
environment variables to one before importing numerical libraries. The
notebook also uses `threadpoolctl` to limit pools already loaded in its
kernel. Changing an environment variable alone cannot resize such a pool.
