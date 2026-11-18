# Fiducial angular-order follow-up after the full triangle checks

The next small study should compare five numerical configurations for
both NLA and TATT, starting from the completed `triangle_1p5` Fisher
configuration. Preserve its accuracy overrides, including its triangle
refinement.
It should evaluate the same 480-row mean and retain the same joint covariance.
It should not calculate Fisher derivatives yet. The first two configurations
separate the Legendre and Fourier cutoffs. Two further cases isolate angular
integration and coupling interpolation; the last changes both together. This is a
proposed study protocol, not a report of completed angular-order tests. The
accompanying driver defaults to preflight. Keep this protocol unchanged
during execution: its exact bytes are part of the study identity. Record
completed measurements in a separate results note after verifying them.

## What the present controls actually do

The current TATT demonstration specifies `lmax=mmax=20`, ten logarithmic side
bins from 8 to 80 arcmin, 63 opening-angle centers on `(0, pi)`, point angular
sampling, centroid projection and ordered source legs. Its aligned FFT grid
has 576 nodes. The NLA configuration uses the same estimator geometry.
The paper comparison target lists `Lmax=50, Mmax=100` for the natural-component
figures. Matching those orders alone would not match their estimator or their
physical inputs. The project already records differences in the original
side-bin bank, angular sampling, pixel window and representative side length.
Sources: [TATT configuration](../../configs/des_y3_tatt_demo_fisher.yaml),
[paper target](../../PAPER_REPRODUCTION_PLAN.md),
[estimator comparison](estimator_conventions.md).

The Legendre expansion describes the triangle-angle dependence of the
projected bispectrum. Its coefficients are then coupled to Fourier modes
whose phases reconstruct the opening-angle dependence of the shear
correlation. These two cutoffs are different approximations. In particular,
`Mmax` need not equal `Lmax`: the spin-dependent geometric coupling has a
rational angular phase with harmonics beyond the finite Legendre series.
The source implements that coupling by finite sums for each requested
`(L,M)` and side ratio.
[Coupling derivation](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/coupling_3pcf.c#L18).

For the prepared projection used here, `aperture_settings.nmu=64` is **not**
the active angular integration count. It is used by the alternative direct
projection and is still validated as greater than `lmax`. The prepared route
uses `projection_settings.nmu_log` and `nmu_linear`, currently 32 and 32. It
integrates a piecewise-linear bispectrum against each Legendre polynomial
with analytic interval weights; it is not a 64-node Gauss-Legendre rule.
Raising the maximum degree does not insert more samples of that bispectrum.
Both occurrences of `lmax` must be changed together, because the engine
requires the projection and transform orders to agree.
[Engine validation](../../python/des3pcf/mean.py),
[prepared/direct dispatch](../../python/des3pcf/mean.py),
[angular weights](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/physics_grid_3pcf.c#L563).

The pending `angular_2` screening case does refine the correct prepared
quantities. `Accuracy.projection_settings` applies
`integration_accuracy_boost * angular_boost` to both angular counts, using
`ceil(N * factor)`. Either factor set to two therefore makes 32+32 become
64+64. Setting both to two gives 128+128. The global sampling
`accuracy_boost` does not enter this angular count; it controls different
prepared grids. These changes leave the angular endpoint cutoff `epmu`
unchanged. `integration_accuracy_boost` also changes other integrations,
including the LOS nodes, so it is not an isolated angular test.
[Count rule](../../python/des3pcf/accuracy.py),
[prepared angular mapping](../../python/des3pcf/accuracy.py).

A third angular control, `transform_options.npsi`, concerns the geometric
coupling, rather than the physical bispectrum table. Its value 200 reproduces
historical linear interpolation of the analytic coupling. Setting it to zero
evaluates the analytic finite sum at each actual FFT side ratio and removes
that interpolation. This does not change the SAS estimator, the physical
bispectrum, or the size of the stored coupling table. It also does not make
the floating-point calculation exact. By contrast,
`projection_settings.npsi=80` samples the physical multipole table and must
remain unchanged during this particular test.
[Coupling branches](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/coupling_3pcf.c#L129).

## Five explicit candidates

All entries below are overrides of each model's completed `triangle_1p5`
Fisher configuration. They inherit its existing accuracy overrides.
Use new output paths and record the full effective configuration. Keep the
physical cosmology, IA parameters, processed source archive and its measured
edge reconstruction, all row selections, covariance, side bins, phi centers,
Fourier limits, FFT node request, angular averaging, window, padding and
ordered-leg convention identical. Do not replace these overrides by a global
accuracy multiplier. In particular, keep the inherited `triangle_boost=1.5` identical in the
baseline and all candidates; do not reset it to one.

| Case | `aperture_settings.lmax` | `projection_settings.lmax` | `aperture_settings.mmax` | Prepared mu counts | Transform `npsi` | Comparison |
|---|---:|---:|---:|---|---:|---|
| Inherited triangle_1p5 baseline | 20 | 20 | 20 | 32+32 | 200 | Saved corrected mean, subject to identical provenance |
| `orders_L50_M20` | 50 | 50 | 20 | 32+32 | 200 | Effect of added Legendre degrees at fixed Fourier cutoff |
| `orders_L50_M50` | 50 | 50 | 50 | 32+32 | 200 | Effect of added signed Fourier modes relative to the preceding case |
| `orders_L50_M50_angular` | 50 | 50 | 50 | 64+64 | 200 | Effect of prepared angular sampling at fixed coupling interpolation |
| `orders_L50_M50_coupling` | 50 | 50 | 50 | 32+32 | 0 | Effect of removing coupling interpolation at fixed prepared angular sampling |
| `orders_L50_M50_guard` | 50 | 50 | 50 | 64+64 | 0 | Higher angular sampling and removal of historical coupling interpolation |

For the angular-only and combined cases, either explicitly set
`nmu_log=nmu_linear=64` or apply
`angular_boost=2` to the original 32+32 counts; do not do both. Keep
`integration_accuracy_boost=1`. The direct-mode `nmu=64` passes the existing
`nmu > lmax` validation for all five candidates and can remain unchanged.
No automatic constraint relates `mmax` to the number of phi centers, so this
must be checked by the study rather than expected from an API refusal.
[Transform domain checks](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/transform_3pcf.c#L125).

The last case is a combined guard. The angular-only and coupling-only cases
make cancellation between its two changes visible. Compare each with the
shared L50/M50 case, rather than interpreting adjacent cases as isolated
changes when both settings differ. A useful later diagnostic is to retain
the multipoles for representative ordered triplets, such as (1,1,1) and
(1,3,4), and pass those same arrays to both coupling plans. That isolates
coupling without recomputing the projected bispectrum. Retaining all twenty
dense multipole arrays is unnecessary for that diagnostic.

## Why not jump directly to Mmax=100 on 63 point samples?

The declared centers are

```math
\phi_j=(j+1/2)\pi/63,\qquad j=0,\ldots,62.
```

They obey the exact sampled identity

```math
 e^{i(M+126)\phi_j}=-e^{iM\phi_j}.
```

The signed range `-100,...,100` therefore includes distinct harmonics with
identical sampled vectors up to a sign; for example, +100 and -26. A finite
point estimator is still mathematically defined with these modes: the
analytic prediction can legitimately receive contributions from both. The
problem is interpreting that result as a separately resolved angular
integral or claiming that the 63 samples tested the new harmonics. The
proposed Mmax=50 avoids this particular pairwise alias within its retained
signed range. It does not establish convergence of the full angular
integrand, which also contains angle-dependent aperture weights and centroid
rotations. The reconstruction is not a plain band-limited scalar Fourier
series after those operations.

A later Mmax=100 check has two distinct possible meanings. One can evaluate
the same 63-center point estimator and explicitly report this sampled
high-order contribution, including its aliases. Alternatively, one can
study angular quadrature with, for example, 127 and 255 centers at fixed
Lmax=50 and Mmax=100. Both counts avoid the pairwise mode collision above.
That second exercise changes the current finite quadrature weights and is a
separate estimator-resolution study; it must not be silently bundled into
an order-only comparison. It would not establish agreement with the
publication's catalogue-weighted estimator. Enabling `angular_average=x_modes`
is also a declared convention change: it averages the X-projected modes,
then applies the centroid phase at the center. It is not an exact average of
the centroid-projected shear over a measurement bin.
[Phi construction](../../scripts/run_3pcf.py),
[mode average and phase](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/transform_3pcf.c#L212).

The radial grid must also remain explicit. `aligned_fft_grid` chooses its
spacing and length from the fixed side bins and Fourier request, independently
of L and M. Thus these candidates retain the actual 576-node grid and exact
selected radial positions; no rounding to new nearest bins occurs. Higher
M nevertheless raises the largest Bessel order to M+3. The existing transform
permits that order but does not certify its Fourier-boundary or FFT accuracy.
If the new modes affect the mean, repeat the accepted highest-order case with
an FFT-density refinement at the same physical side bins, and then a separately
recorded Fourier-range check. Avoid calling a successful order-only comparison
a radial convergence result.
[Aligned radial grid](../../python/des3pcf/mean.py),
[Bessel orders](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/transform_3pcf.c#L418).

## Memory before running

These are allocation formulas, not measured process memory. The first table
is the original-grid worked example; the inherited triangle_1p5 adjustment
is given below. Let N=576 be the
FFT length, T=10 the selected side count, P=63 the phi count, B=20 the triplet
batch size and W=4 the maximum OpenMP workers for this study. The prepared
multipole table has dense axes E=127 and Q=159, since its coarse 64 and 80
nodes are each refined with ratio two. Its four planes represent two physical
components and two ordered-side charts, including allocated zero residual
planes in NLA.

| Persistent allocation | Formula in bytes | L20/M20 | L50/M20 | L50/M50 |
|---|---|---:|---:|---:|
| Geometric coupling | `8*(M+1)*(L+1)*(2*N-1)` | 3.87 MiB | 9.40 MiB | 22.84 MiB |
| All prepared multipole charts, B=20 | `8*B*4*(L+1)*E*Q` | 258.82 MiB | 628.57 MiB | 628.57 MiB |
| Plan's refilled physical multipoles | `8*(L+1)*N*N` | 53.16 MiB | 129.09 MiB | 129.09 MiB |
| Selected signed-mode outputs | `8*(2*M+1)*4*T*T` | 0.13 MiB | 0.13 MiB | 0.31 MiB |
| Two complex Mellin tables, no padding | `16*(M+4)*(N+N/2+1)` | 0.32 MiB | 0.32 MiB | 0.71 MiB |

Exact coupling (`npsi=0`) retains precisely the same coupling allocation.
Its temporary Legendre coefficient array at L=50 is only `8*51*51` bytes.
It does not allocate a dense operator mapping every triangle to every
multipole. The FFT scratch is about 40.57 MiB for four workers at the fixed
576 grid; it does not grow with the number of M jobs. The owned query map is
about 13.92 MiB on a machine with eight-byte `size_t`.
[Plan allocations](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/transform_3pcf.c#L151),
[FFT scratch and kernels](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/fftlog_3pcf.c#L245),
[query storage](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/physics_grid_3pcf.c#L1252).

The fixed triangle tables must be included as well. For the present TATT
counts, twenty dense residual tables of shape `(63,319,319)` consume about
978.23 MiB. Twenty signed-baseline pairs of shape `(63,47,47)` consume another
42.47 MiB. Natural-cubic preparation temporarily retains both old and new
arrays for the axis being refined, and there are background/source tables,
wrapper objects, FFTW plans and process libraries beyond these estimates.
The triplets are finalized successively, so twenty copies of that temporary
upsampling peak are not created simultaneously by `finish_grid`.
[Preparation lifecycle](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/physics_grid_3pcf.c#L889),
[batch finalization](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/physics_grid_3pcf.c#L1151).

With the original triangle counts, the named arrays put L50/M50 TATT near
1.8 GiB before those additional allocations. The intended triangle_1p5
baseline instead resolves its coarse counts to nr=np=48, nu=nv=36 and
nshape=240; with dense ratio two, its twenty residual tables have shape
(95,479,479) and occupy about 3.25 GiB. Its two signed-baseline planes
occupy about 146 MiB across twenty triplets. Including the L50 multipole
and transform arrays puts the named persistent arrays near 4.2 GiB, before
upsampling scratch, physical models and process overhead. This still leaves
a reasonable allocation margin on a 32-GiB machine for one sequential case;
it is not an observed peak-memory claim. Use one process at a time, BLAS threads fixed at one and OpenMP at
most four, and release earlier plans/grids between cases. If a later test
refines triangle counts or retains multiple physical models, recalculate
this budget. Reducing `projection_batch_size` is available if needed, but
keep it fixed across this first study so that an execution change does not
complicate the comparison.

## What to save and how to decide the next step

Save the full 480-row vector for each case, the exact effective configuration,
compiled/source/input identities and actual grid sizes. Compare adjacent
cases as well as each one to the current baseline. The 400 xi rows should
be bitwise unchanged, because none of their inputs is altered. For the
remaining differences report absolute row residuals and

```math
\Delta\chi^2=(v_{\rm case}-v_{\rm base})^{\mathsf T}
 C^{-1}(v_{\rm case}-v_{\rm base}),
```

using the full fixed joint covariance, including the cross block. A fractional
row change is supplementary and must not divide through a baseline zero.
Inspect aperture radii and source triplets separately so that cancellation
in an overall statistic does not hide a localized change. Retain real and
imaginary natural-component slices for the representative triplets if they
help identify a phase or high-order problem.

This study tests mean predictions at two fiducial IA models. It does not test
parameter derivatives, their fourth differences, the omitted angular cap,
missing physical contributions, or Fisher-FoM stability. If the new orders
matter, first settle the relevant numerical guard; then select a declared
order baseline and repeat only the derivative/FoM checks needed for that
choice. Do not extrapolate a small fiducial residual into a claim that the
IA self-calibration or dark-energy forecast has converged.


## Running the proposed study

The baseline argument must name the completed `triangle_1p5` Fisher
configuration, whose `outputs.fisher` points to its saved full Fisher
archive. Its original fiducial mean is reused; no baseline mean is recalculated.
Use separate study directories for the NLA and TATT models. The commands
below assume Cocoa is active and the project is the current directory.
The script can also find
its project from its installed location under `tests/studies/`; `--project`
provides an explicit override.

**Step :one:**: Set thread ownership for the current platform.

Linux:

```bash
export OMP_NUM_THREADS=4 OMP_THREAD_LIMIT=4
export OMP_PROC_BIND=close OMP_PLACES=cores OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

macOS:

```bash
export OMP_NUM_THREADS=4 OMP_THREAD_LIMIT=4
export OMP_PROC_BIND=disabled OMP_PLACES=cores OMP_DYNAMIC=false
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

**Step :two:**: Inspect the TATT inputs and proposed changes.

This command constructs no
physical model and leaves the output directory untouched.

```bash
python tests/studies/angular_order_study.py \
  runs/tatt_accuracy/triangle_1p5.json --output runs/tatt_angular_orders
```

**Step :three:**: Request the five sequential TATT mean predictions.

Run one physical study at a time. Complete any earlier physical calculation
before this command.

```bash
python tests/studies/angular_order_study.py \
  runs/tatt_accuracy/triangle_1p5.json --output runs/tatt_angular_orders --execute
```

**Step :four:**: Inspect the NLA baseline with a separate destination.

```bash
python tests/studies/angular_order_study.py \
  runs/nla_accuracy/triangle_1p5.json --output runs/nla_angular_orders
```

The baseline must declare an integer `threads` value from one through four.
BLAS remains at one. The driver calls the ordinary `run_3pcf.py evaluate`
command separately for each candidate and does not run a benchmark. It saves
the complete effective JSON, mean NPZ, log, result receipt, explanatory note
and a final summary. A resumed invocation accepts only a verified prefix of
those results; conflicting files are refused rather than overwritten.

The shared `prediction_identity` records exact input-file and implementation
hashes, not angular orders or numerical settings. Comparing it to the baseline
therefore establishes that the physical files and implementation stayed the
same; it does not establish which angular candidate was evaluated. The driver
checks the exact candidate configuration SHA256 and the retained configuration
dictionary separately. It also compares rows, measurement values and the fixed
joint covariance with the baseline, checks that xi values are bitwise unchanged,
and verifies dependencies before and after execution. Beginning/end disk checks
cannot detect a file changed and restored between checks or inspect native
code already loaded into memory; the sequential fresh child processes should
run while source and binaries are stable.

The generated `summary.json` reports mean differences in the full joint
covariance metric. It does not report new derivatives or a validated Fisher
FoM. A result for the combined sampling/coupling guard must retain that name
when displayed, even if its difference is small.
