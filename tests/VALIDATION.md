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

## Executed totals <a name="totals"></a>

The strict collection ran on 2026-11-18 using Darwin arm64, Clang 19.1.7,
Python 3.11.12 and NumPy 1.26.3, at project commit `b821e2d` with core
commit `3b7ec85`. Compilation used one worker. BLAS used one thread
throughout. Ordinary calculations used one OpenMP worker; thread
comparisons used at most four. One test process ran at a time.

| Build mode | Most recent full collection |
|---|---|
| Strict (`-O3`, native instruction selection, the core's strict floating-point flags) | 684 passed, exit 0 (2026-11-18) |
| Debug (`-O0`, undefined-behavior and floating-point-division sanitizers) | 502 passed, no sanitizer diagnostic (core checkpoint `6a7eae7`) |

The checks added after that debug checkpoint are Python-only; their
affected groups also ran with the debug extension: the 44 source, survey,
CLI and provenance checks, and the 48 checks affected by the BACCO
neutrino and continuation change. There is no AddressSanitizer result in
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
matrices use `runs/`. Released measurements, mock inputs, sources and covariances
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
