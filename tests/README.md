# Physical and numerical checks

The tests are organized by the quantity being checked.

- [Data-vector checks](data_vector/README.md) cover the physical bispectra,
  angular transforms, interpolation and both aperture routes.
- [Covariance checks](covariance/README.md) cover halo preparation, joint
  cumulant terms, xi spin/noise factors and shared geometry.
- [Python and CLI checks](#python_cli) cover exact parameter keys, survey
  row ordering, derivatives, Fisher weighting and precision expansion.
- <a name="optional_studies"></a>[Optional studies](studies/README.md#optional_studies)
  measure angular orders, interpolation and source choices, derivative
  steps, feedback responses and estimator conventions, with saved records
  in `studies/results/`. Run them separately from the routine tests.
- The [validation record](VALIDATION.md) holds the dated executed totals
  in each build mode, the environment they ran in, and the command flow
  that reproduces them.

1. [Running the checks](#running)
2. [Python and CLI checks](#python_cli)
3. [Build modes and thread ownership](#modes)
4. [What passing establishes](#scope)

## Running the checks <a name="running"></a>

We assume Cocoa is active, the component library has been built using the
[project instructions](../README.md#build), and the current directory is
`Cocoa/`. Most tests create small arrays without DES data products or a
Boltzmann solve. The BACCO component checks also use the installed paired
emulator and weights; they do not download them or run a survey forecast.

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

## Python and CLI checks <a name="python_cli"></a>

The Python files directly in `tests/` check the likelihood, forecast and
command-line layers. Most run on small arrays the tests build; a few run
physical C predictions (`test_pipeline.py`) or a bounded CAMB solve
(`test_camb_tables.py`).

**Input and configuration contracts.** `test_input_contracts.py` stops
invalid scientific inputs with a named diagnostic, and
`test_bacco_configuration.py` refuses invalid BACCO targets before a
forecast or sampler starts. `refusal_worker.py` runs one deliberately
invalid call in a separate Python process, so the required diagnostic and
exit status are observed rather than assumed.

**CAMB preparation.** `test_camb_tables.py` checks bounded CAMB
preparation against a fresh final-amplitude solve and the opposite unit
API. `test_camb_setting_identity.py` checks CAMB settings and saved
identities without running CAMB or native C.

**Pipeline, sources and CLI.** `test_pipeline.py` checks the assembled
mean pipeline with physical C calculations, and `test_pipeline_cli.py`
exercises the command-line workflow on small physical tables the tests build.
`test_pipeline_setting_sequences.py` and
`test_core_two_point_sequences.py` compare prediction caches and the
restored DES Y3 two-point bridge with independently initialized
processes. `test_prepare_demo_sources.py`, `test_source_grids.py`,
`test_survey.py` and `test_estimator_geometry.py` cover demonstration
source preprocessing, source and power-table sampling, survey labels and
radial binning. `test_prediction_provenance.py` checks exact-file
provenance of saved predictions.

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
calculations are visible in `EXAMPLE_FISHER_ACCURACY.ipynb`; the retired
Fisher workflow/archive wrappers have no replacement layer.
`test_coupling_high_orders.py` compares the geometric coupling with
independent quadrature at the larger study orders, and
`test_code_comparison.py` checks source identity and isolated execution
for the external-code comparison.

**Timing controls without measurement.** `test_scaling.py`,
`test_scaling_plot.py` and `test_scaling_threepcf.py` check benchmark
control, the timing display and workload preparation using controlled
numbers, never measured runtimes. `test_quadrature.py` checks fixed
quadrature measures, fine rules and oscillating integrals.

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
There is no timing loop. Separate cold-start and changed-state performance
measurements belong to a later quiet-machine exercise.

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
