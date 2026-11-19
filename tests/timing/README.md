# OpenMP scaling at ten random parameter points

1. [The two workloads](#workloads)
2. [Run on a quiet machine](#run)
3. [What enters the measured interval](#interval)
4. [Read the figures and records](#results)
5. [Saved measurements](#measurements)

## The two workloads <a name="workloads"></a>

This command-line study evaluates ten different parameter points with one,
four and eight OpenMP threads. Each thread count receives the same saved
points and physical tables. A separate point warms each process before the
ten measurements begin. The study uses the production interfaces and does
not execute a notebook. NLA and TATT are separate cases for each data vector.

| Workload | Observable |
|---|---|
| LSST Y1 3x2pt, NLA and TATT separately | All 1560 entries of the existing production vector, with the all-ones mask, 26 angular bins and multipole limit 65000 |
| DES Y3 three-point, NLA and TATT separately | All 80 aperture rows: twenty source triplets and four filter radii, using the finite-triangle geometry in `configs/des_y3_nla_scaling.yaml` and `configs/des_y3_tatt_scaling.yaml` |

The LSST points come from its existing `nmodels_points` test sampler with
seed 137. The study keeps its prior boxes, amplitude rejection and
alternating held parameters. The DES points follow the same seed-plus-point-index convention
with explicit cosmological and nuisance-parameter intervals. DES uses
$`\sigma_8`$ while the LSST sampler uses $`A_s`$; their point banks are separate.
This compares each workload's scaling relative to its own one-thread result.
The two data vectors do not implement the same observable. Within each data
vector, NLA and TATT use the same cosmological points, source shifts and grids.
Quadratic IA amplitudes are inactive in NLA and active in TATT. The three-point
NLA calculation omits physical residual-grid preparation but keeps its
multipole and natural-component transform loops.

The public Y3 source distribution has twelve negative samples in its first
bin, with integrated magnitude about 0.041% of that bin's signed
normalization. The timing-only configuration clips negative samples to
zero, normalizes the distributions, and records the original file
identity, the removed negative area and the normalization; its redshift
grid extends to 3.1 so the positive source shifts stay inside it. The
[Fisher guide](../../../../external_modules/code/cosmolike_core/cosmolike_notebook_utils/README.md#fisher) explains the matching demonstration-source
assumption. This input policy serves timing only and changes neither the
original FITS file nor the forecast configurations.

The three-point inputs use a CAMB solve cutoff of 5 and interpolation support
through 1000, both in $`\mathrm{Mpc}^{-1}`$; the saved table spans
$`10^{-5}`$ to $`1000\,h\,\mathrm{Mpc}^{-1}`$. The
[accuracy guide](../../../../external_modules/code/cosmolike_core/cosmolike/3pcf/README.md#accuracy-controls) explains these two separate
limits, their units and the convergence question attached to the
extrapolated tail.

## Run on a quiet machine <a name="run"></a>

We assume Cocoa is activated with `source start_cocoa.sh`, the current
directory is `Cocoa/`, and both project extensions are built. The driver
sets all BLAS vendors to one thread before numerical imports. Its children
set OpenMP to the requested count with a thread limit of eight. The study
design measures one, four and eight workers; the project's tests and
notebooks keep their four-worker cap.

**Step :one:**: inspect the input files and parameter banks without a calculation.

```bash
python projects/des_3pcf/scripts/benchmark_scaling.py preflight
```

Preflight hashes inputs, checks the observable geometry and source policy,
and generates parameter dictionaries. It does not call CAMB, initialize a
physical C model or produce timing measurements. An unresolved input blocks
the run before any physical preparation starts.

**Step :two:**: after making the machine quiet, run the study.

```bash
python projects/des_3pcf/scripts/benchmark_scaling.py run
```

The command starts with LSST Y1 NLA, then LSST Y1 TATT, three-point NLA and
three-point TATT. It prepares eleven physical table archives per case with
one worker, then runs that case's three CLI children one at a time.
TATT reuses its NLA case's CAMB arrays only after exact physical-input checks,
and the run records the source/configuration identities and the reused file
hash.
The default block order is four, one, then eight threads, with a fresh
process for every block. Each block evaluates its separate warmup and then
the ten saved random points.
The output directory must not already exist; select another path in
`configs/openmp_scaling.yaml` for a separate study.

The optional `--reuse-from` argument accepts an earlier two-workload study
directory containing `tables/lsst/`. It reuses verified physical tables only;
all timed predictions are rerun. This avoids repeating CAMB when expanding a
completed LSST NLA test to the four-case comparison.

**Step :three:**: regenerate the figure from completed measurements if needed.

```bash
python projects/des_3pcf/scripts/benchmark_scaling.py plot
```

This reads saved timings and the output-consistency record. It performs no
scientific calculation and cannot fill in a missing measurement.

## What enters the measured interval <a name="interval"></a>

The three-point interval constructs the point's physical C model and computes
all twenty triplets through projection, natural-component transforms and
aperture integration. It keeps the fixed angular geometry and FFT plans.
The LSST interval installs the point's cosmology and nuisances and
evaluates the masked production data vector through the existing entry
point. Successive points change physical inputs through the ordinary
setters. The driver edits no internal nonce and uses no run-mode toggle
to manufacture a cache miss.

CAMB, table-file reads, extension imports, global initialization, the first
geometry evaluation and output checks are outside those intervals. The run
records physical table preparation times, including CAMB and source
processing, separately. The figure therefore measures the
data-vector calculation, not the full wall time of a sampler or Fisher run.
The driver loads the LSST covariance only to initialize the existing
production entry point; the benchmark performs no covariance calculation.

Each child reads its OpenMP runtime's maximum thread count and verifies that
every loaded BLAS pool has one thread. The run saves copies and hashes of the
two extension binaries. These binary identities matter while another team is
developing the core: a Git branch name alone does not identify a compiled file.

The same parameter point must return identical float64 bits at all three
thread counts. Changed parameter points must also change the vector. Any
failure stops the summary and preserves the individual results and child
logs for inspection. The summary does not accept a timing merely because
it completed.

## Read the figures and records <a name="results"></a>

The final directory contains 120 measured predictions and twelve separate
warmups, with `threepcf_scaling.pdf`, `threepcf_scaling.png`,
`openmp_scaling.pdf`, `openmp_scaling.png`,
`timings.csv`, full JSON records, physical tables, output vectors and child
logs. The left panel shows every elapsed time, its median and the observed
minimum-to-maximum range. The right panel shows speedup relative to each
workload's median one-thread time and an ideal linear-scaling reference.

The ten dots represent different parameter points. Their spread includes
parameter dependence and machine variation; it is not a confidence interval
or ten repeated measurements at one cosmology. Point-by-point times remain
available in the CSV and JSON files. Accuracy settings, observable rows and
thread count must accompany any performance claim. These timings do not
establish survey-level convergence of these grids or a published forecast.

## Saved measurements <a name="measurements"></a>

Dated records under `benchmarks/` keep the measured times, figures,
binary identities and saved vectors; this guide does not repeat their
numbers. The [repeated-work comparison](../../tests/timing/2026-11-18/README.md)
measures the implementation this project builds, for both the mean
prediction and the covariance workload. The
[Apple M2 Pro record](../../tests/timing/2026-11-17/README.md) measures the
original implementation on the same ten points and machine, with LSST Y1
as the control calculation. Each record names its resolution and the work
inside its intervals; neither times the accuracy-tested Fisher settings.
