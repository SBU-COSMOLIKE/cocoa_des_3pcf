# Work in progress: three-point implementation

The reviewed core is checkpointed at `d832c5f`, after the initial port at
`8c58ace`, on the existing core branch `lowrank_fftlog`. Only the isolated
3pcf code and its documentation were included. Other teams' core changes
remain outside these commits. The project has its own `codex/3pcf-port`
branch, starting with `0e7d9af`.

The [validation record](tests/VALIDATION.md) gives the executed checks and
limits. The full 232 checks passed in strict and debug builds. A further
review found scalar/vector rounding differences in multipole preparation
and reads. Eight new output-extension checks reproduced them; the correction
passed 100 strict data-vector/component checks and 38 debug physical/state
checks. The strict extension is restored. BLAS remains serial and OpenMP
uses at most four cores for tests.

The review also corrected squeezed-triangle cancellation, checked workspace
products before multiplication, and rejected fractional or overflowing
notebook source-bin labels before integer conversion. The two-point adapter
owns immutable settings and resets/restores its declared ordinary-core state
before each evaluation. Global and stage-specific 3pcf accuracy controls are
explained in [ACCURACY.md](ACCURACY.md).

Active work includes the shared three-point plotting module, live original-code
comparison notebook, and YAML-driven IA Fisher notebook. Their small examples
and tests are being executed separately. They do not constitute an executed
DES survey forecast. Public Y3 files are verified inputs; exact publication
cuts, estimator agreement and likelihood choices remain separate acceptance
questions in [PAPER_TARGETS.md](PAPER_TARGETS.md).

Further reviews check physical measures and conventions, ownership and bounds,
repeated work and parallel layout. Performance claims require a later quiet
machine measurement. No timing study, survey-scale calculation, dependency
upgrade, remote push or change to existing data-vector C files is included.
Downloaded inputs and numerical outputs are excluded by `.gitignore`.
