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

| Workload | Declared observable |
|---|---|
| LSST Y1 3x2pt, NLA and TATT separately | All 1560 entries of the existing production vector, with the all-ones mask, 26 angular bins and the declared multipole limit 65000 |
| DES Y3 three-point, NLA and TATT separately | All 80 aperture rows: twenty source triplets and four filter radii, using the finite-triangle geometry in `configs/des_y3_nla_scaling.yaml` and `configs/des_y3_tatt_scaling.yaml` |

The LSST points come from its existing `nmodels_points` test sampler with
seed 137. Its prior boxes, amplitude rejection and alternating held parameters
are retained. The DES points follow the same seed-plus-point-index convention
with explicit cosmological and nuisance-parameter intervals. DES uses
$`\sigma_8`$ while the LSST sampler uses $`A_s`$; their point banks are separate.
This compares each workload's scaling relative to its own one-thread result.
The two data vectors do not implement the same observable. Within each data
vector, NLA and TATT use the same cosmological points, source shifts and grids.
Quadratic IA amplitudes are inactive in NLA and active in TATT. The three-point
NLA calculation omits physical residual-grid preparation, but retains its
declared multipole and natural-component transform loops.

The supplied Y3 source distribution has twelve negative samples in its first
bin. Their integrated magnitude is about 0.041% of that bin's signed
normalization. The timing-only configuration explicitly clips negative
samples to zero and normalizes the distributions. It records the original
file identity, removed negative area and normalization. Its redshift grid
extends to 3.1 to retain the declared positive source shifts. This input policy
does not change the original FITS file or the forecast configurations and is
not an accepted source-calibration prescription for a scientific result.

The three-point inputs use a CAMB solve cutoff of 5 and interpolation support
through 1000, both in $`\mathrm{Mpc}^{-1}`$. The saved table still spans
$`10^{-5}`$ to $`1000\,h\,\mathrm{Mpc}^{-1}`$. These are separate controls:
the high-wavenumber table reads use CAMB's extrapolating power interpolator.
Amplitude normalization reuses the computed matter transfers. The
[accuracy guide](ACCURACY.md#camb-limits) explains the units and convergence
question attached to the extrapolated tail.

## Run on a quiet machine <a name="run"></a>

We assume Cocoa is activated with `source start_cocoa.sh`, the current
directory is `Cocoa/`, and both project extensions are built. The driver
sets all BLAS vendors to one thread before numerical imports. Its children
set OpenMP to the requested count with a thread limit of eight. This study
has explicit authorization to use eight threads; routine development tests
retain their four-thread cap.

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
TATT reuses its NLA case's CAMB arrays only after exact physical-input checks;
source/configuration identities and the reused file hash are retained.
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
aperture integration. Its fixed angular geometry and FFT plans are retained.
The LSST interval installs the point's cosmology and nuisances and calls
`compute_data_vector_masked`. Successive points change physical inputs through
the ordinary setters. No internal nonce is edited and no run-mode toggle is
used to manufacture a cache miss.

CAMB, table-file reads, extension imports, global initialization, the first
geometry evaluation and output checks are outside those intervals. Physical
table preparation times, including CAMB and source processing, are recorded
separately. The figure therefore measures the
data-vector calculation, not the full wall time of a sampler or Fisher run.
In particular, the LSST covariance is loaded only to initialize its existing
production entry point; no covariance calculation is part of the benchmark.

Each child reads its OpenMP runtime's maximum thread count and verifies that
every loaded BLAS pool has one thread. The run saves copies and hashes of the
two extension binaries. These binary identities matter while another team is
developing the core: a Git branch name alone does not identify a compiled file.

The same parameter point must return identical float64 bits at all three
thread counts. Changed parameter points must also change the vector. Any
failure stops the summary and preserves the individual results and child
logs for inspection. A timing is not accepted merely because it completed.

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
establish survey-level convergence of the chosen grids or a published forecast.

## Saved measurements <a name="measurements"></a>

The [Apple M2 Pro record](benchmarks/2026-11-17/README.md) contains the completed
NLA and TATT comparison, individual measurements, binary identities and saved
vectors. Eight-thread median times are 2.285 seconds for the 80-entry NLA
vector and 3.966 seconds for TATT, each about five times faster than its
one-thread result. All thread-count comparisons preserve float64 bits.
LSST Y1 is retained as the control calculation. CAMB is excluded from these
durations; the linked record defines the other included and excluded work.
