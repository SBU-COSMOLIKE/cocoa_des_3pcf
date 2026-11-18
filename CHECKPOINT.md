# Work in progress: three-point implementation

The performance work is checkpointed at core `517eb15` on `lowrank_fftlog`.
Project work is on `main`. The core changes stay in the isolated 3pcf
implementation and its documentation. Other teams' core files remain outside
these commits. The project checkpoint includes tests of retained interpolation
geometry, changed physical inputs, worker counts and covariance row selections.
Its full strict-build suite passes 394 tests. The debug run passed 393
checks and found one test-only NumPy keyword mismatch; the repaired source
and CLI group passes all thirty checks. No C sanitizer error was reported.
The strict extension is restored. Subsequent Fisher-notebook and quadrature
work remains in progress and has its own numerical checks.

The original CLI comparison is recorded in [BENCHMARK.md](BENCHMARK.md). It uses
LSST Y1 as a control and times NLA and TATT separately, excluding CAMB.
All 120 measured predictions and twelve warmups preserve float64 bits across
thread counts. The [shared-work comparison](benchmarks/2026-11-18/README.md)
gives eight-thread medians of 1.467 seconds for NLA and 1.956 seconds for
TATT, with all predictions bitwise equal to the original. The
declared grid and input policy remain part of those results. The dedicated
Y3 timing configuration records the authorized clipping of negative source
samples; scientific forecast inputs retain their separate source policy.

The physical producer follows LSST's bounded CAMB solve and separate
extrapolating power interpolation. It reuses matter transfers for sigma8
normalization. Both wavenumber limits enter the provider and outer mean cache
keys. Forty-six control/state/configuration checks and thirteen CAMB checks
pass, including an independent final-amplitude solve and opposite unit API.

The [validation record](tests/VALIDATION.md) gives the executed checks and
limits. The preceding 281-check suite passed in strict and debug builds, including physical
SIMD lane invariance, cache/state sequences and joint Fisher algebra. The
large-area covariance work reduction preserves all 16 saved comparison
arrays bitwise. The debug run reports no sanitizer diagnostics; it uses
undefined-behavior and floating-point-division checks, not AddressSanitizer.
The strict extension is restored. BLAS is serial and OpenMP uses at most four
cores for tests.

The review also corrected squeezed-triangle cancellation, checked workspace
products before multiplication, and rejected fractional or overflowing
notebook source-bin labels before integer conversion. The two-point adapter
owns immutable settings and resets/restores its declared ordinary-core state
before each evaluation. Global and stage-specific 3pcf accuracy controls are
explained in [ACCURACY.md](ACCURACY.md).

The shared plotting module has 13 passing array checks and 30 inspected render
modes. The [original-code comparison notebook](EXAMPLE_COMPARE_CODES.ipynb)
and [IA/Fisher notebook](EXAMPLE_FISHER_IA.ipynb) contain executed examples
with 20 completed code cells and eight figures. Both use the strict checkpoint.
Their small calculations do not constitute an executed DES survey forecast.
Public Y3 files are verified inputs; exact publication cuts, estimator agreement
and likelihood choices remain separate questions in [PAPER_TARGETS.md](PAPER_TARGETS.md).

Further reviews check physical measures and conventions, ownership and bounds,
repeated work and parallel layout. Performance trials use ten changed physical
points, with a separate warmup and CAMB excluded. The covariance trials use a
coarse gravity-only full-filter calculation with a large-area approximation;
their runtime cannot be identified with a converged finite-triangle TATT
covariance. A separate refinement study examines that distinction.
The demonstration forecast may use explicitly recorded source clipping and
renormalization. That input choice does not establish source-calibration or
estimator accuracy. No dependency upgrade, remote push or change to existing
data-vector C files is included.

The active calculation builds default-accuracy NLA/TATT Fisher bases with
free w0 and wa. It reuses the shared core's fourth-difference noise estimator
to screen accuracy settings with selected columns, followed by full
derivative checks. Joint YAML/Nautilus examples and a precision-expansion
teaching notebook are being prepared. The public repository also contains
observed Y3 aperture measurements; their matched data and analysis choices
are being checked separately from the simulated input selection.
Downloaded inputs and numerical outputs are excluded by `.gitignore`; the
small teaching outputs embedded in the notebooks are retained.
