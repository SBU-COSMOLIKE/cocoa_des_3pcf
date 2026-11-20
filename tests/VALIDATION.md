# Validation record

This page records the executed totals of the project tests in each build
mode, the environment they ran in, and the command flow that reproduces
them. What each check measures is in the sector guides: the
[test overview](README.md), the [data-vector guide](data_vector/README.md)
and the [covariance guide](covariance/README.md). Optional-study results
live with their saved records in the [studies guide](studies/README.md);
they are not counted here.

1. [Executed totals](#totals)
2. [Reproducing the totals](#commands)
3. [What the totals do not establish](#limits)
4. [Native interfaces and Cobaya provider validation](#native_provider)

## Executed totals <a name="totals"></a>

The earlier strict collection ran on 2026-11-18 using Darwin arm64, Clang 19.1.7,
Python 3.11.12 and NumPy 1.26.3, at project commit `b821e2d` with core
commit `3b7ec85`. Compilation used one worker. BLAS used one thread
throughout. Ordinary calculations used one OpenMP worker; thread
comparisons used at most four. One test process ran at a time.

| Build mode | Most recent full collection |
|---|---|
| Strict (`-O3`, native instruction selection, the core's strict floating-point flags) | 883 passed, exit 0 (2026-11-19; details below) |
| Debug (`-O0`, undefined-behavior and floating-point-division sanitizers) | 502 passed, no sanitizer diagnostic (core checkpoint `6a7eae7`) |

The Python checks added between that debug checkpoint and the November 18
strict run also ran with the debug extension: the 44 source, survey,
CLI and provenance checks, and the 48 checks affected by the BACCO
neutrino and continuation change. The November 19 native migration has
not had a new sanitizer run. There is no AddressSanitizer result in
this record. Linux and other processor architectures remain untested.

## Reproducing the totals <a name="commands"></a>

We assume Cocoa is active, the component library is built with the
[project instructions](../README.md#build), the shell is Bash, the
current folder is `cocoa/Cocoa`, and the thread variables of the
[test environment](README.md#running) are set: serial BLAS and at most
four OpenMP workers. The
[build modes section](README.md#modes) explains the strict and debug
extensions and their separate object directories.

**Step :one:**: run the strict collection.

```bash
python -m pytest -q projects/des_3pcf/tests
```

**Step :two:**: build the debug extension with one compilation worker.

```bash
COSMOLIKE_DEBUG_MODE=1 make -C projects/des_3pcf/interface -f MakefileCosmolike -j1
```

**Step :three:**: run the same collection with the debug extension.

```bash
python -m pytest -q projects/des_3pcf/tests
```

**Step :four:**: restore the strict extension.

```bash
make -C projects/des_3pcf/interface -f MakefileCosmolike -j1
```

## What the totals do not establish <a name="limits"></a>

The totals establish the component checks described in the sector guides
at their controlled inputs; the
[test overview](README.md#scope) says what passing establishes. These
runs select no DES integration bounds, triangle grids, scale cuts or
nuisance priors, and no Y3/Y6 posterior reproduction or runtime speedup
is claimed by this record. The
[paper-target guide](../README.md#estimator) separates those survey
questions from agreement between shared numerical kernels.

## Input layout and shared Nautilus driver, 19 November 2026

Generated theory tables, artificial feedback responses and trained compression
matrices use `chains/`. Released measurements, mock inputs, sources and covariances
use `data/`; frozen numerical comparisons use `tests/`. All 25 released files
match their recorded source hashes. The processed source matches its YAML hash.
The 26 relocated timing records, arrays and figures retain their exact bytes.

The four Nautilus examples call `cocoa_hybrid_sampling.run` as LSST Y1 does,
with `hybrid=False` for physical theory. Their Cobaya likelihood uses the
existing joint provider. A real two-parameter analytic Nautilus run checked
that independent Gaussian priors were not counted twice, external correlated
priors entered once, and a repeated output prefix was refused without changing
its files. The default hybrid loader still rejects physical-only configurations.

Checks used serial BLAS and `OMP_THREAD_LIMIT=6`, one numerical job at a time:

| Check | Result |
|---|---|
| Source processing, original-code configuration, public covariance, pixel window and CLI configuration tests | 40 passed; live upstream execution deselected |
| Shared Nautilus examples, likelihood priors/selection and scaling control/plot tests | 64 passed |
| Accuracy-result identities, Fisher screening and step-plot readers | 26 passed |
| Real `EXAMPLE_NAUTILUS1.py` through `4.py`, each with `--check` | All four exited successfully |

The four fiducial log likelihoods were respectively 7124.056851, 7124.056851,
6398.736620 and 1810.569075. These are normalized likelihood densities, not
best-fit statistics or posterior-reproduction results. The observed example
exercised a fresh native evaluation: fixed numeric selectors from Cobaya are
converted to integers only after checking exact integrality.

All nine notebooks validate and retain their saved outputs and execution
counts. No notebook, refreeze, survey posterior or timing study was rerun.
The edited Markdown pages render with tables; their local links and anchors
resolve. Shared algorithms are documented in the core READMEs, while the
project top directory contains only its README and linked appendices.

## Native interfaces and Cobaya provider validation <a name="native_provider"></a>

On 19 November 2026 the full collection passed **883 tests in 309.71 s**
with the strict extension built against core `0b476c1`. BLAS used one
worker, OpenMP used at most six, and numerical jobs ran sequentially.
The project now uses one extension for xi and Map3, native numerical
preparation and IA decomposition, and the ordinary Cobaya likelihood
method sequence. CAMB and the matter bispectrum are YAML theory components.

The previous collection had three failures: two fine Gauss-Legendre
rules exceeded the existing constant-integral tolerance, and a preflight
test still expected no native import. The native rule now uses GSL's
fixed Legendre workspace; the 4096- and 8192-node checks pass without
loosening their tolerances. Preflight may import numerical initializers,
but still runs neither CAMB nor physical model integration.

The [real Cobaya check](studies/cobaya_joint_check.py) then loaded
`EXAMPLE_NAUTILUS1.yaml` and `EXAMPLE_NAUTILUS2.yaml` at their production
sampling, without starting a sampler. The covariance-weighted difference
from each fixed synthetic input was:

| Model | Reference residual Δχ² |
|---|---|
| NLA | 5.89046386112555104e-08 |
| TATT | 5.53610663043381469e-08 |

In both models, separate changes of A1 by 0.01, the first source shift
by 0.001 and Ωm by 0.003 changed the vector and retained finite posterior
values. Every return to the fiducial restored the prediction bitwise.
These are regression and state-restoration checks against synthetic data,
not accuracy bounds or a reproduction of the measured DES posterior.

After correcting remaining CLI provenance references to the removed
two-point adapter, **26 CLI tests passed in 36.25 s**. This includes one
new test collected after the full run: a joint TATT mean and calibration
Fisher calculation through the real CLI. It checks finite results, row
ordering and that both probes record the same compiled extension. Thus
884 distinct tests passed across the full run and this targeted rerun;
the complete 884-test collection was not rerun as a single job.

Static checks parse 29 project Python files and 72 notebook code cells
after IPython's normal input transformation. Notebook structures validate;
nbformat warns about existing cells without IDs. Markdown rendering and
local links also pass. Notebook outputs are retained from their earlier
runs. No notebook, refreeze, sampler or timing study ran during this check.

## LSST Y1 consistency review and matched sampler examples, 19 November 2026

The likelihood was compared directly with LSST Y1's base and probe subclass,
and all four launchers with `EXAMPLE_EMUL2_NAUTILUS1.py`. The follow-up removed
the Python compression object from likelihood initialization, a repeated
native reset and separate native-function aliases. Native calls now use
`ci` explicitly. Survey rows retain ordinary owned arrays without a frozen
dataclass or reflective assignment loop. Numerical physics code was unchanged.

`EXAMPLE_EVALUATE1–4.yaml` and `EXAMPLE_MCMC1–4.yaml` match the corresponding
Nautilus case's likelihood, theory, parameters and priors. They use the same
simulation covariance, including the compressed Percival choice in case 4.
The MCMC controls follow LSST Y1, with speed measurement enabled instead of
copying LSST's parameter blocking.

The focused survey, geometry, sampler, compression and native-likelihood
collection passed **55 tests in 11.87 s**. The actual NLA and TATT provider
checks also passed: reference residual Δχ² was respectively
`5.89046386007120215e-08` and `5.53751267227229527e-08`, with bitwise
restoration after each cosmology, IA and source-shift displacement.
The MCMC3 and MCMC4 science configurations were loaded through Cobaya and
evaluated once, without starting MCMC: their fiducial log likelihoods were
`6398.3498025582367` and `1810.5690753417277`.

These are normalized densities and regression checks, not posterior results.
No DES chain, notebook or refreeze was run. BLAS stayed at one worker and
OpenMP at most six; numerical jobs ran sequentially.

## Notebook prose and rendering review, 19 November 2026

All nine notebooks received a second reading of their explanations, notation,
input descriptions and instructions. Six needed small prose corrections;
their structure and scientific results were retained. Corrections include
the methods-paper reference for compression, the current dataset-based
compression loading, the Python matter/native IA distinction, the Y3 IA
fiducial label and the smooth fourth-derivative contribution to the noise
diagnostic.

The nine notebooks exported to HTML without executing cells. Headless Chrome
rendered all 84 Markdown cells and 532 mathematical expressions without a
MathJax error. Page screenshots were inspected at 1440 pixels wide; the
table and displayed-equation width checks reported no overflow. The local
Markdown links resolved. Code-cell hashes, including outputs and execution
counts, remained identical across all nine notebooks.

The HTML, screenshots and browser diagnostics are generated review artifacts
under ignored `chains/notebook_prose_review/`. Saved notebook results still
come from their earlier numerical runs. Neither notebook execution nor
refreezes resumed during this prose review.
