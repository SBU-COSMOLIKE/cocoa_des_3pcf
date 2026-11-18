# Work in progress: three-point implementation

The reviewed core is checkpointed at `98824eb`, following `d832c5f` and
`8c58ace`, on the existing core branch `lowrank_fftlog`. These commits contain
only the isolated 3pcf implementation, shared plotting folder and associated
documentation. Other teams' core changes remain outside them. The project
has its own `codex/3pcf-port` branch, starting with `0e7d9af` and `e2fc60a`.

The [validation record](tests/VALIDATION.md) gives the executed checks and
limits. All 281 project checks pass in the strict build, including physical
SIMD lane invariance, cache/state sequences and joint Fisher algebra. The
large-area covariance work reduction preserves all 16 saved comparison
arrays bitwise. Earlier debug checks passed; debug validation of the final
physical-tail and covariance changes is still pending. BLAS is serial and
OpenMP uses at most four cores for tests.

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
repeated work and parallel layout. Performance claims require a later quiet
machine measurement. No timing study, survey-scale calculation, dependency
upgrade, remote push or change to existing data-vector C files is included.
Downloaded inputs and numerical outputs are excluded by `.gitignore`; the
small teaching outputs embedded in the notebooks are retained.
