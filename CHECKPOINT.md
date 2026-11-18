# Work in progress: three-point implementation

This checkpoint preserves the implemented port and its current checks. It is
not a release or a claim of DES publication reproduction. The ordinary
two-point C files are unchanged by this work. The shared numerical module
lives in `cosmolike_core/cosmolike/3pcf/`; this repository owns its build,
survey preparation, Python prediction/Fisher workflow and tests.

The corresponding core checkpoint is commit `8c58ace` on its existing
`lowrank_fftlog` branch. It contains only the 3pcf module and associated
documentation; other in-progress core work was excluded.

## Checks completed before this checkpoint

| Check | Executed result |
|---|---|
| Physical model/projection setting sequences | 12 passed; 150 independent fresh-process references across both adapters |
| Transform/aperture setting sequences | 6 passed; 66 independent fresh-process references across both adapters |
| Physical covariance setting sequences | 8 passed; 82 independent fresh-process references across both adapters |
| Ordinary two-point core restoration | 4 passed against the existing DES Y3 extension; no ordinary-core rebuild |
| Accuracy controls, Python prediction/cache sequences, forecast, survey and CLI checks | 61 passed |
| Physical covariance equations and boundary checks | 26 passed on the current strict extension |
| Transform equations and physical checks | 29 passed on the strict extension |

The state comparisons require identical float64 bit patterns after dimensions,
same-size values, physics choices, ownership and worker settings change in a
running interpreter. They include retained old objects and return to earlier
settings. Earlier independent references and their definitions are retained
in the tests. The [validation record](tests/VALIDATION.md) still describes the
initial foundation milestone; it will be replaced with the full strict/debug
record after the active numerical revisions below are tested.

## Revisions in progress

The independent core review identified squared-side cancellation in squeezed
triangle F2/IA geometry, an inconsistent multiplication order in a covariance
SIMD scalar tail, and covariance dimension products calculated before their
overflow guards. Their corrections and independent tests are pending at this
checkpoint. Thus passing state tests must not be interpreted as proof that
all physical inputs are already accurate.

The ordinary two-point adapter resets and restores its declared configuration
before evaluation, including after a partially configured raw notebook call.
Its next refinement will protect its declared settings from partial edits,
matching the prepared three-point engines. Global and stage-specific 3pcf
accuracy controls are implemented and tested; their guide is [ACCURACY.md](ACCURACY.md).

The following work remains authorized and active:

- Complete the independent core revision passes and scientific tests, then
  run the full strict and debug checks and restore the strict extension.
- Add LSST Y1-style YAML and executable example notebooks, especially a live
  comparison with pinned original fastnc and Rafael Gomes routines.
- Build a DES Y3 full 2pcf + 3pcf Fisher notebook focused on supported IA and
  baryonic-feedback questions, with paper selections and priors identified.
- Separate formula checks, numerical refinement and paper-level reproduction;
  do not substitute guessed survey inputs or claim an unsupported joint
  baryonic/velocity-shear two-point model.
- Keep the core skill implementation references synchronized with the code,
  tests, notebook workflow and remaining physical assumptions.

No timing study, survey-scale calculation or performance claim is included.
Downloaded public data and compiled build products are excluded by `.gitignore`.
