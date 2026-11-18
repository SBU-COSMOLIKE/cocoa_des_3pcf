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
[paper-target guide](../PAPER_TARGETS.md) separates those survey
questions from agreement between shared numerical kernels.
