# Physical and numerical checks

The tests are organized by the quantity being checked.

- [Data-vector checks](data_vector/README.md) cover the physical bispectra,
  angular transforms, interpolation and both aperture routes.
- [Covariance checks](covariance/README.md) cover halo preparation, joint
  cumulant terms, xi spin/noise factors and shared geometry.
- Python and CLI checks cover exact parameter keys, survey row ordering,
  derivatives, Fisher weighting and precision expansion.
- [Validation record](VALIDATION.md) lists the checks actually executed.
- [BACCO response study](studies/bacco_validation.md) compares direct emulator
  values, native interpolation and joint-forecast continuation choices;
  the [example commands](../BACCO.md#running) run its notebook and off-grid check.

1. [Running the checks](#running)
2. [Build modes and thread ownership](#modes)
3. [What passing establishes](#scope)

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
relinks the selected mode. Thread comparisons retain the same input
arrays and ordered sums.
There is no timing loop. Separate cold-start and changed-state performance
measurements belong to a later quiet-machine exercise.

## What passing establishes <a name="scope"></a>

Passing establishes agreement for the tested component inputs, including
complex signs, repeated field labels, output layout and caller-owned
memory. Independent references include the physical TATT components and
angular transforms as well as the covariance terms. The checks do not
establish a finite-catalogue covariance, Y3/Y6 publication reproduction
or survey Fisher accuracy. Those require the declared inputs and
refinements of the whole estimator.
