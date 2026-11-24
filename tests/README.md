# Physical and numerical checks

The tests are organized by the quantity being checked.

- [Data-vector checks](data_vector/README.md) cover the physical bispectra,
  angular transforms, interpolation and both aperture routes.
- [Covariance checks](covariance/README.md) cover halo preparation, joint
  cumulant terms, xi spin/noise factors, shared geometry and the
  production covariance driver.
- [Likelihood, forecast and driver checks](#python_cli) cover exact
  parameter keys, survey row ordering, derivatives, Fisher weighting and
  precision expansion.
- <a name="optional_studies"></a>[Optional studies](studies/README.md#optional_studies)
  measure angular orders, interpolation and source choices, derivative
  steps, feedback responses and estimator conventions, with saved records
  in `studies/results/`. Run them separately from the routine tests.
- The [validation record](VALIDATION.md) holds the dated executed totals
  in each build mode, the environment they ran in, and the command flow
  that reproduces them.

1. [Running the checks](#running)
2. [Likelihood, forecast and driver checks](#python_cli)
3. [Build modes and thread ownership](#modes)
4. [What passing establishes](#scope)

## Running the checks <a name="running"></a>

We assume Cocoa is active, the component library has been built using the
[project instructions](../README.md#build), and the current directory is
`Cocoa/`. Most tests create small arrays without DES data products or a
Boltzmann solve. A few build the actual Cobaya model of a numbered
example, so they run CAMB at one point. The BACCO component checks also
use the installed paired emulator and weights; they do not download them
or run a survey forecast.

**Step :one:**: reserve parallel work for OpenMP and keep BLAS serial.

Linux:

```bash
export OMP_NUM_THREADS=4
export OMP_THREAD_LIMIT=4
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
```

macOS:

```bash
export OMP_NUM_THREADS=4
export OMP_THREAD_LIMIT=4
export OMP_PROC_BIND=disabled
export OMP_PLACES=cores
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
```

Set these variables before starting Python. The checks normally request
one OpenMP worker and use at most four for comparisons. BLAS remains at
one thread throughout, including in child processes. Run one test process
at a time; additional test processes would each have their own thread team.

**Step :two:**: run the small physical, numerical and driver checks.

```bash
python -m pytest -q projects/des_3pcf/tests
```

The tests use one worker for ordinary calculations and explicitly compare
one, two, three and four OpenMP workers for output ownership. They report test
results, not performance comparisons. The covariance references are
analytic formulas or independently organized contractions; the aperture
reference includes the hash of its pinned external source.

## Likelihood, forecast and driver checks <a name="python_cli"></a>

The Python files directly in `tests/` check the Cobaya likelihood, the
forecast arithmetic and the project drivers. Most run on small arrays the
tests build; the files named in the likelihood paragraph below evaluate
the actual Cobaya model of a numbered example.

**Input and configuration contracts.** `test_input_contracts.py` stops
invalid scientific inputs with a named diagnostic, and
`test_bacco_configuration.py` refuses invalid BACCO targets before a
forecast or sampler starts. `refusal_worker.py` runs one deliberately
invalid call in a separate Python process, so the required diagnostic and
exit status are observed rather than assumed.

**Likelihood on the Cobaya route.** `test_likelihood_calibration.py`
checks the per-row multiplicative shear calibration and bit-identical
restoration when a parameter point is revisited.
`test_bfmt_suppression.py` pins the likelihood's power-response read
against an independent interpolation and checks the end-to-end wiring of
the power-only feedback key; `test_bacco_cobaya.py` does the same for the
paired BACCO responses. `test_notebook_wrappers.py` checks that the
interface wrappers hold exactly the likelihood's joint vector and follow
parameter changes. `test_nautilus_examples.py` runs the shared Nautilus
driver on an analytic posterior and checks that every prior enters the
posterior exactly once. Theory tables and the background come from one
evaluation of one Cobaya model, so the retired cross-file check that the
supplied bispectrum table and the background share their redshifts and
$`\Omega_{\rm m}(z)`$ has no failure mode left to catch.

**Sources and survey rows.** `test_prepare_demo_sources.py`,
`test_source_grids.py`, `test_survey.py` and `test_estimator_geometry.py`
cover demonstration source preprocessing, source and power-table
sampling, survey labels and radial binning.
`test_core_two_point_sequences.py` compares the restored DES Y3 two-point
bridge with independently initialized processes.

**Forecast arithmetic.** `test_forecast.py`, `test_fisher_comparison.py`,
`test_fixed_dark_energy.py`, `test_derivative_templates.py` and
`test_finite_sign_pair.py` check derivatives, joint information
accounting, fixed dark energy with varied nuisances, covariance-weighted
degeneracy and finite model distances on analytic examples.
`test_compression.py`, `test_inference.py`, `test_precision_components.py`
and `test_precision_notebook.py` cover local Fisher preservation under
compression, statistical selection and the known-component precision
examples. `test_fisher_derived_quantities.py`, `test_fisher_notebook.py`,
`test_fisher_cobaya_notebooks.py` and `test_fisher_accuracy.py` check actual
notebook cells and shared numerical functions, using small analytic inputs
without running physical forecasts.

**Study arithmetic and reuse.** `test_accuracy.py` and
`test_full_smaller_step_study.py` check the shared accuracy estimator,
exact half/quarter/double-step stencil reuse, QR/SVD nuisance projection,
weak directions and joint observable-block substitutions. The scientific
calculations are visible in `EXAMPLE_FISHER_ACCURACY.ipynb`.
`test_coupling_high_orders.py` compares the geometric coupling with
independent quadrature at the larger study orders.

**Timing controls.** `test_benchmark_scaling.py` runs the real timing
driver ([the timing guide](timing/README.md)) on two points at one and
two threads and checks its component accounting, recomputation guards and
cross-thread agreement. `test_scaling_plot.py` checks the timing display
using controlled numbers, never measured runtimes. `test_quadrature.py`
checks fixed quadrature measures, fine rules and oscillating integrals.

## Build modes and thread ownership <a name="modes"></a>

The default build retains strict floating-point arithmetic. Debug mode
uses the project Makefile's diagnostic flags. Rebuild after changing the
mode so that object files do not retain flags from the earlier build.

**Step :one:**: build the debug mode.

```bash
COSMOLIKE_DEBUG_MODE=1 make -C projects/des_3pcf/interface -f MakefileCosmolike -j1
```

**Step :two:**: run the same checks.

```bash
python -m pytest -q projects/des_3pcf/tests
```

**Step :three:**: restore the default build.

```bash
make -C projects/des_3pcf/interface -f MakefileCosmolike -j1
```

The Makefile keeps separate strict and debug object directories and
relinks the mode you select. Thread comparisons retain the same input
arrays and ordered sums.
There is no timing loop in the routine tests; timing runs are the
separate quiet-machine exercise of [the timing guide](timing/README.md).

## What passing establishes <a name="scope"></a>

Passing establishes agreement for the tested component inputs, including
complex signs, repeated field labels, output layout and caller-owned
memory. Independent references include the physical TATT components and
angular transforms as well as the covariance terms. The checks do not
establish a finite-catalogue covariance, Y3/Y6 publication reproduction
or survey Fisher accuracy. Those require the inputs and
refinements of the whole estimator. Whether shared smoothing may replace
the local variance reference is decided by the full-forecast acceptance
record in
[studies/shared_variance_validation.md](studies/shared_variance_validation.md).

## Matter-emulator support <a name="matter_emulator_support"></a>

[check_matter_emulator_support.py](check_matter_emulator_support.py) asks the
candidate Cobaya provider for the two matter fields required by this
project: total matter (`delta_tot`) and cold matter (`delta_nonu`, baryons
plus cold dark matter). It uses fixed neutrino mass 0.06 eV. This small
check precedes any random-cosmology accuracy comparison, because replacing
cold matter by total matter would change the physical model.

**Step :one:**: select the candidate code checkout.

```bash
export EMULATOR_CODE_DIR=/path/to/emulators_code
```

**Step :two:**: select its model-data checkout.

```bash
export EMULATOR_DATA_DIR=/path/to/emulators_data_lcdm
```

**Step :three:**: run from activated Cocoa. This small support check
uses one thread for each numerical library.

```bash
python projects/des_3pcf/tests/check_matter_emulator_support.py
```

The result is written to `chains/matter_emulator_prior/support.txt`.
The check explicitly enables `cold_power: eisenstein_hu`, using the project's
number of massive species. It verifies the linear-field interface at fixed
mass in flat LCDM. It does not establish the approximation's accuracy or
supply nonlinear cold power. Exit status 1 reports a missing required field.

## Cold-matter substitution error

`compare_cold_total_power.py` isolates the error of using total power where
the likelihood requests linear cold power. It uses CAMB in both runs,
retains the actual neutrino density and all other model choices, and keeps
the measurement covariance fixed. Each command starts a fresh process, so a
cached native table cannot conceal a changed spectrum. The three points are
the current first example's fiducial and two changes of Omega_m to 0.25 and
0.40, all at mnu=0.06 eV. Other fiducial parameters stay fixed.

Run from activated Cocoa, first with the true cold spectrum:

```bash
python projects/des_3pcf/tests/compare_cold_total_power.py --mode exact --output projects/des_3pcf/chains/cold_total_mnu006
```

Then replace only cold power with total power:

```bash
python projects/des_3pcf/tests/compare_cold_total_power.py --mode total --output projects/des_3pcf/chains/cold_total_mnu006
```

The saved vectors, spectrum ratios, configuration and native/source revisions
remain under `chains/`. The reported distance is the prediction difference
contracted with the covariance inverse, not the difference between two fits
to the observed data. For a finite-simulation likelihood it remains a
Gaussian covariance-distance diagnostic, not a change in that log-likelihood.

At the 24 November 2026 run, the first example's joint mean changed only
at rounding level: covariance distances were 6.94e-28, 9.76e-29 and 1.01e-25
at the fiducial and the two shifted matter densities. This is the NLA,
fixed-covariance example, with its native mask and MOPED transformation.
The cold table is handed to the core, but the active shear/Map3 mean does
not depend on the halo variance that consumes it. These numbers do not
validate substituting total power in a halo covariance or in another
observable that actively uses the cold field.

Applying the emulator's three-species Eisenstein-Hu ratio to these saved
CAMB total spectra gives maximum fractional cold-power residuals of
0.000300, 0.000324 and 0.000249 at the same three cosmologies, over the
saved k=0.0001--5 1/Mpc and z=0,0.5,1,2 samples. This checks the field
conversion even though this example's mean barely responds to that field.

## Official hybrid-emulator test

The E5 test follows LSST Y1: fifty random cosmologies drawn from the
numbered example, restricted to 0.2 < Omega_m < 0.5, fixed mnu=0.06 eV,
and 10^9 A_s <= 3.5. Both routes use plain mead2020 and CAMB background
quantities. The candidate supplies total power and approximate linear cold
power through the explicit Eisenstein-Hu option. It uses the installed
emulator code and data; environment overrides are the same as LSST Y1.

```bash
python -m pytest -s projects/des_3pcf/tests/data_vector/test_emul2.py
```

Each invocation creates a separate `chains/hybrid_emulator_tests/` directory.
The statistic is the direct vector difference contracted with the fixed
precision, with no data-residual linear term. The Fisher calculation uses
CAMB derivatives at two step sizes and marginalizes the sampled nuisance
parameters with their Gaussian prior curvature and hard bounds. Its figure
uses the approved error thresholds 0.2, 0.5 and 1, the project title and the
3.5 amplitude cap. These contours describe a centered local forecast,
not a fit to the observed posterior. The shared test also draws 25 local
points from the bounded Fisher sample, evaluates CAMB and the emulator at
those points, and overlays them as diamonds beside the broad-prior circles.
Their separate predictions and errors are saved under `local/` within the
same run directory. `--local-points` supplies a different named parameter
table when an explicit study calls for one. Accuracy exceedances are recorded;
incomplete numerical comparisons or a failed Fisher calculation fail E5.

Example 1 supplies the observed joint xi/Map3 vector, its scale cuts,
MOPED compression and simulation covariance. The native selection is
applied after evaluating and re-scoring the full prediction; the precision
is the inverse of that selected covariance. The Fisher overlay uses the
Gaussian fixed-covariance approximation, not the curvature of the Percival
finite-simulation likelihood. BiHalofit consumes the selected provider power;
CAMB still supplies the background and sigma8(z) used for growth.
