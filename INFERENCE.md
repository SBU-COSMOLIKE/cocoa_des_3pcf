# Joint xi± and Map3 inference with Nautilus

These examples call the same `JointPrediction` used by the Fisher driver.
Each inference YAML additionally specifies data, independent prior densities,
an explicit row selection, and a sampling budget. The primary examples
generate a noiseless synthetic data vector from their own fiducial mean.

Contents:

1. [Prepare and inspect](#prepare)
2. [Sample and inspect the output](#sample)
3. [Why the likelihood uses the joint covariance](#likelihood)
4. [Public observed measurements](#observed)

| Example | IA model | Fitted data |
|---|---|---|
| [EXAMPLE_NAUTILUS1.yaml](EXAMPLE_NAUTILUS1.yaml) | NLA | Synthetic fiducial mean |
| [EXAMPLE_NAUTILUS2.yaml](EXAMPLE_NAUTILUS2.yaml) | TATT | Synthetic fiducial mean |
| [EXAMPLE_NAUTILUS3.yaml](EXAMPLE_NAUTILUS3.yaml) | NLA | Public observed unblinded Y3 vector, as a demonstration |

The observed example does not reproduce an accepted DES likelihood. Its
recorded source processing, finite-triangle estimator, full 480-row selection
and fixed Gaussian likelihood differ from parts of the published analysis.
Its fitted measurements are public, but agreement of those modeling choices
still needs validation. See the [reproduction plan](PAPER_REPRODUCTION_PLAN.md)
before interpreting a posterior.

<a name="prepare"></a>
## Prepare and inspect

**Step :one:**: activate Cocoa and select the CPU allocation. Run from the
`Cocoa/` directory. This example uses one C worker and one BLAS worker.

Linux:

```bash
source start_cocoa.sh
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
```

macOS:

```bash
source start_cocoa.sh
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=disabled
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

**Step :two:**: inspect the declared input files without importing a compiled
model or starting CAMB.

```bash
python projects/des_3pcf/EXAMPLE_NAUTILUS1.py --validate-config
```

**Step :three:**: inspect the TATT configuration.

```bash
python projects/des_3pcf/EXAMPLE_NAUTILUS2.py --validate-config
```

The prediction YAML points to the demonstration source file and physical
tables used in `EXAMPLE_FISHER_ACCURACY.ipynb`. Prepare those inputs through
that notebook's named preparation step when inspection reports them missing.
The inference script never substitutes a different source file or runs an
unrequested preparation command. It uses the installed Cocoa Nautilus
package; dependency installation follows Cocoa's pinned package mechanism.

**Step :four:**: evaluate an actual changed IA parameter and return to the
fiducial before starting a sampler.

```bash
python projects/des_3pcf/EXAMPLE_NAUTILUS1.py --check
```

**Step :five:**: repeat the physical check for TATT.

```bash
python projects/des_3pcf/EXAMPLE_NAUTILUS2.py --check
```

`--check` reports the fiducial likelihood and prior separately, calculates
the configured A1 displacement, and requires a changed selected mean and
bitwise equality after restoring the fiducial. This checks model reach and
state restoration. It does not establish angular-quadrature convergence,
scientific model accuracy or posterior convergence.

<a name="sample"></a>
## Sample and inspect the output

**Step :one:**: from the activated `Cocoa/` directory, select the CPU
allocation for sampling.

Linux:

```bash
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
```

macOS:

```bash
export OMP_NUM_THREADS=1
export OMP_PROC_BIND=disabled
export OMP_PLACES=cores
export OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

**Step :two:**: launch the NLA example with a declared evaluation budget.

```bash
python projects/des_3pcf/EXAMPLE_NAUTILUS1.py --nlive 1000 --neff 10000 --maxfeval 100000
```

`--nlive`: choose the number of live points.

`--neff`: choose the target effective posterior sample count.

`--maxfeval`: stop after the stated likelihood budget even if convergence
has not been reached.

`--outroot`: choose another output prefix; existing outputs are never
silently overwritten or resumed.

`--resume`: continue an existing checkpoint only after the saved YAML,
resolved prediction, prior order, physical inputs, compiled model and
selected-data hashes match. Use CLI budget overrides to extend a run;
editing its YAML requires a different output prefix. Live-point count,
network count and seed remain fixed during a resumed run.

The YAML can require many physical predictions. No full sampling run is
implied by installing these examples or by completing `--check`.

An MPI launch follows the DES Y3 arrangement: one master and separate
worker-owned models. With `threads:1`, `mpirun -n 4 --bind-to none` uses
three likelihood workers and one master. The demonstration refuses an
allocation whose ranks times C workers exceeds four. Do not combine a
multi-rank launch with an unrelated numerical run on the same allocation.

The output prefix contains:

| Suffix | Contents |
|---|---|
| `_data.npz` | Selected data, covariance, row labels and retained full-vector indices |
| `_checkpoint.hdf5` | Nautilus's sampling state |
| `_run.json` | Target identity written before sampling and every explicit continuation |
| `.1.txt` | Weighted GetDist rows when posterior samples exist |
| `.paramnames` | Names and plot labels in chain-column order |
| `.json` | Input hashes, model assumptions, check results, sampling budget and convergence flag |

A budget-limited run is recorded with `converged:false`. If no posterior
samples survive, the checkpoint and JSON remain; no chain is fabricated.
The script preserves the Gaussian normalization in its log evidence, but
that evidence still belongs to the declared approximation and priors.

<a name="likelihood"></a>
## Why the likelihood uses the joint covariance

For residual vector $`r=\mu(p)-d`$, the likelihood is

```math
\log L=-\frac12 r^T C^{-1}r
        -\frac12\log\det C-\frac{n}{2}\log(2\pi).
```

The covariance is fixed during sampling. Its off-diagonal xi/Map3 entries
remain present. A Cholesky factor is calculated once, and each residual is
scored by a triangular solve. Selecting fewer data rows selects both axes
of C before factorization; selecting a submatrix of the full precision
would answer a different, conditional question.

The small numerical units of shear correlations can make the determinant
term large and positive. A positive log likelihood is allowed: a continuous
probability density can exceed one. Compare likelihood differences using
the same data rows and units. The `--check` chi-squared is evaluated at the
configured fiducial before fitting; it is not a best-fit goodness-of-fit
statistic or a diagnosis of which model assumption causes a residual.

Nautilus's `Prior` receives normalized uniform or Gaussian distributions.
Finite Gaussian bounds in the YAML explicitly truncate and renormalize the
density. The likelihood adds no prior penalty. Fisher prior entries in the
referenced prediction file are not used by these sampling examples, so
they cannot be counted a second time.

The examples fix w0,wa and neutrino mass, and vary five cosmological
parameters plus IA, photo-z and shear calibration. Uniform sigma8 is an
explicit demonstration prior; it is not uniform S8. The covariance uses no
Hartlap, Percival or PME modification. These choices are recorded in the
JSON alongside the exact sampled order and normalized prior specifications.

The three inference YAMLs explicitly set
`projection_overrides: {signed_baseline: 1}`. Intrinsic alignment changes the
effective source kernel of the NLA baseline. A product of three such kernels
can be zero or negative within the declared IA priors; taking its logarithm
would therefore exclude physical predictions for numerical reasons. This
mode prepares the logarithm of the positive gravity-only baseline $`B_{GGG}`$
and the signed ratio $`B_{\rm NLA}/B_{GGG}`$ on the same triangle coordinates.
The interpolated prediction multiplies those two factors, retaining zeros
and negative values without an absolute value or clipping. The separate
TATT residual is unchanged, and the two interpolation tables still require
resolution checks. The override is included in inspection and resume
metadata; the referenced Fisher YAML is unchanged. Its tests establish the
representation's domain and state handling, not the accuracy of the full
TATT prediction across the sampling priors.

<a name="observed"></a>
## Public observed measurements

**Step :one:**: fetch the separately pinned observed files after activation.

```bash
python projects/des_3pcf/scripts/fetch_y3_observed.py
```

**Step :two:**: inspect the observed-data demonstration configuration.

```bash
python projects/des_3pcf/EXAMPLE_NAUTILUS3.py --validate-config
```

The [manifest](data/y3_observed_manifest.json) identifies the pair chosen by
the upstream joint configuration and its 480-row covariance. The original
source columns and angular labels match the simulated reference used by the
demonstration. Their raw negative source-density samples have not been
altered in the downloaded files. The model retains the separately recorded
processed demonstration source file, and therefore does not claim that its
source treatment is the publication's calibrated prescription.

The pinned upstream configuration pairs the observed vectors and covariance.
The inspected packing scripts do not recover the realization-level Map3 sign
processing used to estimate C23. The selected covariance is retained unchanged;
this is unresolved provenance, not evidence of a wrong sign. See Section 6
of the [estimator record](tests/studies/estimator_conventions.md). A successful
observed `--check` establishes parameter response and restored prediction
state; it does not resolve that provenance or reproduce the published fit.
