# Fiducial angular-order study for NLA and TATT

This study compared five numerical configurations for each
intrinsic-alignment model, NLA and TATT. Every case starts from that
model's completed `triangle_1p5` Fisher configuration, keeps its accuracy
settings, including the triangle boost of 1.5, and evaluates one 480-row
mean at the same physical parameter point. The study evaluates no
derivatives. In every case the 400 two-point rows stay bitwise unchanged,
and the 80 aperture rows keep their ordering, source inputs and fixed
joint covariance.

1. [The three angular controls](#controls)
2. [The five cases](#cases)
3. [Why the Fourier cutoff stops at 50](#alias)
4. [Memory budget](#memory)
5. [Measured distances between predictions](#distances)
6. [Independent check of the geometric coupling](#coupling)
7. [What the study establishes, and remaining checks](#scope)
8. [Saved records and reproduction](#records)

## The three angular controls <a name="controls"></a>

The Legendre degree L describes the triangle-angle dependence of the
projected bispectrum. The Fourier order M describes the opening-angle
dependence of the transformed shear correlation. The geometric coupling
reconstructs each Fourier mode from the Legendre coefficients through a
rational angular phase whose harmonics extend beyond the finite Legendre
series, so M need not equal L. The source evaluates that coupling as a
finite sum for each requested order pair and side ratio.
[Coupling derivation](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/coupling_3pcf.c#L18)

The baseline sets `lmax=mmax=20`, ten logarithmic side bins from 8 to
80 arcmin, 63 opening-angle centers on the interval from 0 to $`\pi`$,
point angular sampling, centroid projection and ordered source legs, with
a 576-node aligned FFT grid
([TATT configuration](../../configs/des_y3_tatt_demo_fisher.yaml)).
The archived Y3 theory configuration uses `Lmax=50` and `Mmax=100`
together with its own estimator geometry and physical inputs, so matching
those orders alone would not match its predictions
([the estimator comparison](estimator_conventions.md)).

Raising either order does not add samples of the input bispectrum, so the
angular samples receive their own control. The projection integrates a
piecewise-linear bispectrum against each Legendre polynomial with analytic
interval weights; its counts are `projection_settings.nmu_log` and
`nmu_linear`, 32+32 in the baseline. The direct-route count
`aperture_settings.nmu=64` does not enter this calculation. The engine
requires the projection and transform `lmax` values to agree, so the study
changes both together.
[Angular weights](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/physics_grid_3pcf.c#L563);
the count rules are in [the accuracy guide](../../ACCURACY.md#stages).

The third control, `transform_options.npsi`, concerns the geometric
coupling rather than the physical bispectrum table. Its value 200
reproduces linear interpolation of the analytic coupling; the value zero
evaluates the analytic finite sum at each actual FFT side ratio, which
removes that interpolation without changing the estimator, the physical
bispectrum or the floating-point roundoff. The separate
`projection_settings.npsi=80` samples the physical multipole table and
stays unchanged throughout this study.
[Coupling branches](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/coupling_3pcf.c#L129)

## The five cases <a name="cases"></a>

Every case overrides its model's `triangle_1p5` Fisher configuration and
inherits the remaining accuracy settings unchanged. The physical
cosmology, IA parameters, processed source archive, row order, covariance,
side bins, angle centers, Fourier limits, FFT node request, angular
averaging, window, padding and ordered-leg convention stay identical,
and `triangle_boost=1.5` keeps its inherited value in the baseline and in
every candidate.

| Case | `aperture_settings.lmax` and `projection_settings.lmax` | `aperture_settings.mmax` | Prepared angular counts | Transform `npsi` | Question |
|---|---:|---:|---|---:|---|
| `triangle_1p5` baseline | 20 | 20 | 32+32 | 200 | Saved corrected mean with identical provenance |
| `orders_L50_M20` | 50 | 20 | 32+32 | 200 | Added Legendre degrees at fixed Fourier cutoff |
| `orders_L50_M50` | 50 | 50 | 32+32 | 200 | Added signed Fourier modes beyond the preceding case |
| `orders_L50_M50_angular` | 50 | 50 | 64+64 | 200 | Prepared angular sampling at fixed coupling interpolation |
| `orders_L50_M50_coupling` | 50 | 50 | 32+32 | 0 | Coupling interpolation removed at fixed angular sampling |
| `orders_L50_M50_guard` | 50 | 50 | 64+64 | 0 | Both changes together |

The last case is a combined guard. The angular-only and coupling-only
cases make cancellation between its two changes visible, so each case is
compared with the shared L=M=50 reference rather than with its neighbor
in the table. For the angular cases the study sets
`nmu_log=nmu_linear=64` directly and keeps `integration_accuracy_boost=1`;
that boost also rescales other integrations, including the line-of-sight
nodes, so it would not give an isolated angular test. No validation in the
engine relates `mmax` to the number of angle centers, which is why the
study measures that relation itself.
[Transform domain checks](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/transform_3pcf.c#L125)

## Why the Fourier cutoff stops at 50 <a name="alias"></a>

The configuration sets the angle centers

```math
\phi_j=(j+1/2)\pi/63,\qquad j=0,\ldots,62,
```

which obey the exact sampled identity

```math
 e^{i(M+126)\phi_j}=-e^{iM\phi_j}.
```

The signed range from -100 to 100 therefore contains distinct harmonics
with identical sampled vectors up to a sign; +100 and -26 form one such
pair. A finite point estimator remains mathematically defined with those
modes, but the 63 samples cannot test the new harmonics separately. The
cutoff M=50 avoids this pairwise alias within its kept signed range; it
does not establish convergence of the full angular integrand, which also
contains angle-dependent aperture weights and centroid rotations.
[Angle construction](../../scripts/run_3pcf.py),
[mode average and phase](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/transform_3pcf.c#L212)

The radial grid stays explicit as well. `aligned_fft_grid` chooses its
spacing and length from the fixed side bins and Fourier request,
independently of L and M, so every case keeps the actual 576-node grid
and the same radial positions. Higher M raises the largest Bessel order
to M+3; the transform permits that order without certifying its
Fourier-boundary or FFT accuracy, which is why an FFT-density check
appears in the remaining-checks list below.
[Aligned radial grid](../../python/des3pcf/mean.py),
[Bessel orders](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/transform_3pcf.c#L418)

## Memory budget <a name="memory"></a>

These allocation formulas sized the study before execution; they are not
measured process memory. The worked example uses the original grid counts;
the `triangle_1p5` adjustment follows below. Let N=576 be the FFT length,
T=10 the kept side count, P=63 the angle count, B=20 the triplet batch
size and W=4 the maximum OpenMP workers. The prepared multipole table has
dense axes E=127 and Q=159, since its coarse 64 and 80 nodes are each
refined with ratio two. Its four planes hold two physical components and
two ordered-side charts, including the allocated zero residual planes in
NLA.

| Persistent allocation | Formula in bytes | Size at L=M=20 | Size at L=50, M=20 | Size at L=M=50 |
|---|---|---:|---:|---:|
| Geometric coupling | `8*(M+1)*(L+1)*(2*N-1)` | 3.87 MiB | 9.40 MiB | 22.84 MiB |
| All prepared multipole charts, B=20 | `8*B*4*(L+1)*E*Q` | 258.82 MiB | 628.57 MiB | 628.57 MiB |
| Plan's refilled physical multipoles | `8*(L+1)*N*N` | 53.16 MiB | 129.09 MiB | 129.09 MiB |
| Kept signed-mode outputs | `8*(2*M+1)*4*T*T` | 0.13 MiB | 0.13 MiB | 0.31 MiB |
| Two complex Mellin tables, no padding | `16*(M+4)*(N+N/2+1)` | 0.32 MiB | 0.32 MiB | 0.71 MiB |

Exact coupling (`npsi=0`) keeps the same coupling allocation; its
temporary Legendre coefficient array at L=50 is `8*51*51` bytes, and it
allocates no dense operator mapping every triangle to every multipole.
The FFT scratch is about 40.57 MiB for four workers at the fixed 576
grid and does not grow with the number of M jobs. The owned query map is
about 13.92 MiB with an eight-byte `size_t`.
[Plan allocations](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/transform_3pcf.c#L151),
[FFT scratch and kernels](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/fftlog_3pcf.c#L245),
[query storage](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/physics_grid_3pcf.c#L1252)

The fixed triangle tables add to this. The `triangle_1p5` baseline
resolves its coarse counts to nr=np=48, nu=nv=36 and nshape=240; with
dense ratio two, its twenty residual tables have shape (95,479,479) and
occupy about 3.25 GiB, and its signed-baseline planes about 146 MiB
across twenty triplets. Including the L=50 multipole and transform arrays
puts the named persistent arrays near 4.2 GiB, before upsampling scratch,
physical models and process overhead; this is an allocation estimate, not
an observed peak. The triplets finalize one after another, so the
temporary upsampling peak of `finish_grid` is not multiplied by twenty.
The study therefore ran one process at a time with serial BLAS and at
most four OpenMP workers, releasing earlier plans and grids between
cases, and kept `projection_batch_size` fixed so an execution change
would not enter the comparison.
[Preparation lifecycle](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/physics_grid_3pcf.c#L889),
[batch finalization](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/physics_grid_3pcf.c#L1151)

## Measured distances between predictions <a name="distances"></a>

All ten calculations completed. For two calculated vectors the study
measures

```math
D^2=(d_a-d_b)^{\mathsf T}C^{-1}(d_a-d_b),
```

using the complete fixed 480-row covariance, including the
two-point–three-point cross block. This is a distance between predictions
at one parameter point, not a fit to observed data. Although the
two-point difference is zero, the full inverse weights the aperture
difference with the aperture covariance conditional on the two-point
data, so replacing C by its aperture block would change the metric. The
phrase "direct coupling" below means evaluation of the analytic finite
sum at the actual Fourier side ratio; it removes coupling interpolation,
not floating-point roundoff.

| Candidate relative to the baseline | $`D^2`$, NLA | $`D^2`$, TATT |
|---|---:|---:|
| L=50, M=20; other settings unchanged | 0.00143060 | 0.00156042 |
| L=50, M=50; other settings unchanged | 0.00143060 | 0.00156042 |
| L=M=50; angular samples 64+64 | 0.00491712 | 0.00527268 |
| L=M=50; direct coupling | 0.000612707 | 0.000691225 |
| L=M=50; angular samples 64+64 and direct coupling | 0.00338021 | 0.00365968 |

The first two rows look equal at the printed precision, but the saved
vectors differ: comparing L=50, M=50 directly with L=50, M=20 gives
$`D^2=5.37\times10^{-18}`$ for NLA and $`5.72\times10^{-18}`$ for TATT.
The added M=21 through M=50 contributions change these aperture means
negligibly at the tested parameter point; displaced parameter vectors
are a separate question.

To isolate angular sampling from coupling interpolation, the next table
compares every row with the same L=M=50, 32+32-sample,
interpolated-coupling reference. Adjacent rows of the first table would
otherwise change two controls at once.

| Change from the shared L=M=50 reference | $`D^2`$, NLA | $`D^2`$, TATT |
|---|---:|---:|
| Angular samples 32+32 to 64+64 only | 0.00118991 | 0.00125743 |
| Remove coupling interpolation only | 0.000206343 | 0.000205111 |
| Make both changes | 0.000675263 | 0.000716676 |

The combined distance is smaller than the angular-only distance: the two
changes partly cancel in the covariance metric, so their squared
distances must not be added as independent error variances. This is why
the study keeps separate angular-only and coupling-only cases.

## Independent check of the geometric coupling <a name="coupling"></a>

The finite Fourier sum was also compared with a direct fixed
Gauss-Legendre integral of its defining angular expression. This tests
the geometric routine separately from any cosmology or aperture mean.
The [component test](../test_coupling_high_orders.py) covers every L from
0 through 50 and every M from 0 through 100 at seven side-ratio angles,
including equal sides and both angular endpoints, with the endpoint
phases in their analytic limiting forms.

Both the 512-node and 1024-node tests pass. Their largest absolute
differences from the native sum are $`8.27\times10^{-13}`$ and
$`1.46\times10^{-12}`$; on the natural coupling normalization $`4\pi^2`$
these are $`2.09\times10^{-14}`$ and $`3.71\times10^{-14}`$. This
absolute metric includes the zero coefficients without dividing by a
vanishing reference. The two quadrature calculations differ from each
other by about $`2.01\times10^{-12}`$, so the slightly larger 1024-node
residual reflects the scale of their numerical agreement rather than
monotonic improvement from more nodes. The numerical record is
`results/high_order_coupling_check.json`, and both OpenMP and BLAS used one
worker.

This component check covers the coupling coefficients at the larger
orders; the subsequent Bessel transform and the full aperture estimator
at M=100 keep their own tests.

## What the study establishes, and remaining checks <a name="scope"></a>

These fiducial distances are small in the fixed covariance metric, and no
acceptance tolerance was imposed after observing them. The full
[triangle Fisher comparison](fisher_accuracy_validation.md) shows why a
mean distance alone cannot bound a marginalized ellipse: derivatives
divide differences by parameter steps, and nuisance marginalization can
leave a much smaller response than the original derivative columns. The
angular study therefore certifies neither the L=M=20 nor the L=M=50
Fisher calculation.

The unchanged 63 point-angle centers avoid pairwise aliasing between the
modes in the signed range from -50 to 50; convergence of the complete
angle-dependent aperture kernel is a separate question. Remaining checks:

1. A derivative check with L=M=50, 64+64 angular samples and direct
   coupling as an explicit named candidate, comparing the main cosmology
   columns first and all nuisance columns after a material change.
2. An M=100 response, either reported as the sampled contribution at the
   63 centers, aliases included, or measured with 127 and 255 centers as
   a separate estimator-resolution study; both counts avoid the pairwise
   collision above. Enabling `angular_average=x_modes` changes the
   estimator convention and keeps its own name.
3. An FFT-density refinement at the accepted highest order and the same
   physical side bins, followed by a separate Fourier-range check.
4. The number of angle centers and the catalogue's finite-bin weights,
   which this study holds fixed.

The independent parameter-step investigation continues on the
triangle-refined configuration, because it asks a different question on
an explicitly identified grid.

## Saved records and reproduction <a name="records"></a>

The driver is [angular_order_study.py](angular_order_study.py). It
records the full configurations, input and implementation identities,
every 480-row vector and a receipt for each completed case. The committed
summaries are `results/nla_angular_orders/summary.json` and
`results/tatt_angular_orders/summary.json`; a run writes its working
copies, including the vectors, under the local `runs/` directory, which
Git ignores. A
resumed invocation accepts only a verified prefix of earlier results and
refuses conflicting files, and no numerical result is relabeled with a
different physical implementation identity.

The shared `prediction_identity` records exact input-file and
implementation hashes, not angular orders, so matching it to the baseline
establishes that the physical files and implementation stayed the same.
The driver separately checks each candidate configuration's SHA256, the
rows, measurement values and fixed joint covariance against the baseline,
verifies that the xi rows are bitwise unchanged, and checks dependencies
before and after execution. The generated `summary.json` reports mean
differences in the full joint covariance metric; the combined
sampling/coupling guard keeps its name in any display, even where its
difference is small.

The commands below run from the project directory with Cocoa active. The
executed study ran one model process at a time with serial BLAS and four
OpenMP workers, and measured no runtime.

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

**Step :two:**: Inspect the NLA inputs and proposed changes without
launching predictions.

```bash
python tests/studies/angular_order_study.py \
  runs/nla_accuracy/triangle_1p5.json --output runs/nla_angular_orders
```

**Step :three:**: Execute or verify the five sequential NLA mean
predictions.

```bash
python tests/studies/angular_order_study.py \
  runs/nla_accuracy/triangle_1p5.json --output runs/nla_angular_orders --execute
```

**Step :four:**: Execute or verify TATT separately, after the NLA
calculation finishes.

```bash
python tests/studies/angular_order_study.py \
  runs/tatt_accuracy/triangle_1p5.json --output runs/tatt_angular_orders --execute
```

The baseline argument names the completed `triangle_1p5` Fisher
configuration, whose `outputs.fisher` points to its saved full Fisher
archive; its original fiducial mean is reused without recalculation. The
baseline declares an integer `threads` value from one through four, and
the driver calls the ordinary `run_3pcf.py evaluate` command separately
for each candidate.
