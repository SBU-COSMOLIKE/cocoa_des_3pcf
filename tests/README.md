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
Exit status 1 means a required field is unavailable; it does not describe
an accuracy failure. The reviewed candidate supplies finite total-matter
power but explicitly rejects cold-matter power. Therefore the ordinary
projects' 50-point CAMB comparison cannot yet be applied to this likelihood
unchanged. This check supplies no substitute spectrum and runs no maps,
TreeCorr measurements or forecasts.
