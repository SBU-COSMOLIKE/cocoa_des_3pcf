# OpenMP scaling of the Cobaya evaluation

1. [What is measured](#interval)
2. [Run on a quiet machine](#run)
3. [Outputs and checks](#results)
4. [Dated records and the changed boundary](#measurements)

## What is measured <a name="interval"></a>

The driver [tests/benchmark_scaling.py](../benchmark_scaling.py) times
`model.logposterior` of a numbered example at several OpenMP thread
counts. Each thread count runs in a fresh child process with every BLAS
library at one thread. The child builds the Cobaya model once, evaluates
one untimed warm-up point (the YAML's fiducial), then the same saved
parameter points. The points are drawn once from the prior with a fixed
seed and saved to `points.npz`; a saved file written with another YAML,
seed or count raises instead of being reused.

Every point changes cosmology and nuisance parameters, so CAMB, the
bispectrum block and the likelihood all recompute; fixed geometry and FFT
plans keep their ordinary reuse. Three quantities come from the same
evaluations:

- `full`: the elapsed time of `model.logposterior` at the point.
- `camb`: CAMB's own intervals, read from Cobaya's component timers.
  CAMB is two components, `camb.transfers` (the Boltzmann solve) and
  `camb` (the power spectra), and both are counted.
- `other`: `full` minus `camb`; the bispectrum block, any feedback block
  and the likelihood's native preparation and projection.

## Run on a quiet machine <a name="run"></a>

We assume Cocoa is activated with `source start_cocoa.sh`, the current
directory is `Cocoa/`, and the project extension is built. The driver
sets the thread counts itself; other running programs change the times,
so use a quiet machine.

**Step :one:**: time the NLA case.

```bash
python projects/des_3pcf/tests/benchmark_scaling.py \
  --yaml projects/des_3pcf/EXAMPLE_EVALUATE3.yaml \
  --output projects/des_3pcf/chains/benchmark_scaling/nla \
  --threads 1 4 8
```

**Step :two:**: time the TATT case into its own directory.

```bash
python projects/des_3pcf/tests/benchmark_scaling.py \
  --yaml projects/des_3pcf/EXAMPLE_EVALUATE4.yaml \
  --output projects/des_3pcf/chains/benchmark_scaling/tatt \
  --threads 1 4 8
```

Command options, one per flag:

- `--yaml`: the numbered example to time.
- `--output`: the result directory, under the ignored `chains/`.
- `--threads`: the OpenMP team sizes, one fresh process each.
- `--npoints`, `--seed`: how many prior points are drawn, and with which
  seed.
- `--tolerance`: the permitted relative disagreement between thread
  counts.

**Step :three:**: run the functional check, which exercises the real
driver on two points at one and two threads.

```bash
python -m pytest -q projects/des_3pcf/tests/test_benchmark_scaling.py
```

## Outputs and checks <a name="results"></a>

The output directory holds `points.npz` (the saved points),
`threads_<n>.npz` (one child's times, component intervals and data
vectors) and `summary.json` (settings, the measured boundaries in words,
medians and the cross-thread agreement). The driver prints each thread
count's median `full`, `camb` and `other` seconds.

The run fails, and writes no summary, if a component does not recompute
exactly once at a point, if two consecutive points give the same vector,
or if the thread counts disagree beyond the tolerance. Plot completed
records in a notebook with the shared thread-scaling plot of
`cosmolike_core/cosmolike_notebook_utils/`; the driver performs no
plotting.

The ten dots of such a plot are different parameter points. Their spread
includes parameter dependence and machine variation; it is not a
confidence interval. Accuracy settings, observable rows and thread count
must accompany any performance claim, and these timings do not establish
survey-level convergence of the grids.

## Dated records and the changed boundary <a name="measurements"></a>

The dated records in this folder measured the retired private route, and
their intervals have a different boundary from the current driver's.
Compare them with a current run only after stating both boundaries:

| Interval | Dated records (retired route) | Current driver |
|---|---|---|
| Timed call | The bare three-point C prediction, constructed from prepared physical tables | `model.logposterior` of a numbered example |
| CAMB | Outside the interval; tables were prepared beforehand | Inside `full`, separated into `camb` by component timers |
| File reads, imports, initialization | Outside the interval | Outside the timed points (the model is built once, before the warm-up) |
| Likelihood | Not evaluated | Inside `other` |

A median from the old records is therefore not the old value of `full` or
of `other`; the two workloads time different code. The records stay as
evidence of the implementation they measured:

- The [Apple M2 Pro record](2026-11-17/README.md) measured the retired
  route's per-point rebuild on ten points, with LSST Y1 as the control
  calculation. Its producing harness was retired with the private route,
  so that workload can no longer be rerun.
- The [repeated-work comparison](2026-11-18/README.md) measured the
  per-point rebuild against the prepared-geometry reuse of the core, for
  the mean prediction and the covariance workload, at the same old
  boundary.
