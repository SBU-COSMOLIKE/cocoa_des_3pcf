# Source reconstruction and photo-z derivative checks

The six-case study and the completed combined refinements support a declared
numerical baseline for the next demonstration Fisher diagnostic: measured-edge
PCHIP sources with 9601 source nodes, 401 kernel nodes and 129 line-of-sight
nodes. Keep `dz1` step 0.001 and its 0.0005 check. This is a working baseline
for testing the full derivative matrix, not evidence of global convergence
or a validation of the source reconstruction as a physical calibration.
The remaining smallest-step behavior, other photo-z columns and further
line-of-sight refinement still need examination.

## What was held fixed

The study evaluated the same 400 two-point and 80 aperture rows, with the
same full joint covariance, for NLA and TATT. It reused the saved background
and matter-power arrays and made no CAMB calculation. All six cases used
the corrected signed baseline and scaled residual interpolation, with
`shape_stretch=10.1062201`, `nshape=160`, and `psi_scale=0.02`. The actual
aligned transform has 576 Fourier nodes. The two-point integration has
4096 Fourier nodes. No scale cut or compression was introduced.

Only `dz1` was displaced. The nine saved points are zero and
`±0.002`, `±0.001`, `±0.0005`, `±0.00025`. They supply all three five-point
stencils with steps $s=h,h/2,h/4$, where $h=0.001$. Each displaced source
curve was reconstructed from the original processed FITS samples and then
normalized; it was not shifted from a previously resampled curve.

For a vector $v$ containing all 480 rows, define

$$
\|v\|_C=\sqrt{v^{\mathsf T}C^{-1}v}.
$$

Here $C$ is the full joint covariance, including its cross block. The
derivative $D_s$ is the fourth-order approximation to the first derivative,

$$
D_s=\frac{d(-2s)-8d(-s)+8d(s)-d(2s)}{12s}.
$$

The percentages below compare these vectors in the covariance metric.
They are not percentages of a final parameter error or FoM: marginalization
can amplify a small error in a derivative column that is nearly degenerate
with other columns.

## Step agreement improves, but is not a grid-convergence test

The first percentage is
$100\|D_{h/2}-D_h\|_C/\|D_{h/2}\|_C$; the second is
$100\|D_{h/4}-D_{h/2}\|_C/\|D_{h/4}\|_C$.

| Source choice | NLA: $h\to h/2$ | NLA: $h/2\to h/4$ | TATT: $h\to h/2$ | TATT: $h/2\to h/4$ |
|---|---:|---:|---:|---:|
| Linear, 1201 nodes | 41.71% | 9.80% | 43.24% | 10.10% |
| PCHIP, 1201 nodes | 0.691% | 0.0835% | 0.695% | 0.0818% |
| PCHIP, 2401 nodes | 0.349% | 0.279% | 0.348% | 0.276% |
| PCHIP, 4801 nodes | 0.0259% | 0.0381% | 0.0238% | 0.0372% |

The linear 1201-node derivative cannot be accepted at these steps. Its
large change does not prove that the linear source convention is physically
wrong. It shows that this finite source grid and these displacement sizes
do not supply a stable derivative for that convention.

PCHIP reduces this problem considerably. Nevertheless, the 4801-node result
does not improve monotonically when the derivative step is halved again.
For TATT, the absolute derivative-change norms are 0.0383 and 0.0599.
Thus the smallest step is not automatically the most reliable estimate.
The source-grid spacing is about 0.00258 at 1201 nodes and 0.000646 at
4801 nodes. Several displacement sizes remain smaller than one source
sampling interval. This comparison concerns the preparation/read grid,
not the original approximately 0.01-wide FITS sampling.

## Separate preparation and integration changes still matter

The following table compares each listed case with PCHIP1201. Mean changes
refer to the zero-shift vector. Derivative changes use $s=h/2$ and are divided
by $\|D_{h/2}^{\mathrm{PCHIP1201}}\|_C$. This named reference is not assumed
to be the correct answer.

| Change from PCHIP1201 | Mean norm: NLA | Mean norm: TATT | Relative derivative: NLA | Relative derivative: TATT |
|---|---:|---:|---:|---:|
| Linear1201 source model | 0.0723 | 0.0716 | 25.83% | 26.07% |
| Source count 2401 | 0.00160 | 0.00236 | 0.892% | 1.068% |
| Source count 4801 | 0.00310 | 0.00458 | 1.134% | 1.347% |
| Kernel count 201→401 | 0.00255 | 0.00255 | 0.531% | 0.533% |
| LOS count 65→129 | 0.1165 | 0.1162 | 0.898% | 0.980% |

The small zero-shift changes would give an incomplete account of numerical
accuracy. At the same time, increasing the source count from 2401 to 4801
changes the $h/2$ derivative by 0.248% for NLA and 0.288% for TATT, relative
to the 4801-node derivative. At $h/4$ those changes are larger: 0.481% and
0.515%. These differences exceed the within-case step changes in the first
table. Agreement between two steps on one grid therefore cannot select the
source grid by itself.

The kernel and LOS refinements were tested only with 1201 source nodes.
They leave the two-point vector unchanged, so their numbers above describe
changes confined to the aperture entries, weighted with the full joint
inverse covariance. In contrast, the source-resolution derivative change
is dominated by the two-point entries. For example, the TATT PCHIP2401
minus PCHIP1201 $h/2$ derivative has full norm 1.738; zeroing the aperture
entries gives 1.726, while zeroing the two-point entries gives 0.112.
These last two norms are not additive because of the cross covariance.

The source and radial-integration errors must consequently be checked
together. The completed combined study does this for PCHIP4801 at the base
radial settings, PCHIP4801 with kernel401 and LOS129, and PCHIP9601 with
kernel401 and LOS129. These results are discussed next. A separate
LOS129→257 comparison remains useful because the combined study does not
measure the error left by 129 LOS nodes.

## Completed combined refinements

All differences in this section use the newly evaluated 4801-node baseline
from `source_combined_convergence`. Comparisons do not mix an old result
with a changed implementation. For each change, the derivative percentage
is divided by the derivative norm of the **new**, more refined case, at
$s=h/2$.

| Change within the combined run | Mean norm: NLA | Mean norm: TATT | Relative derivative: NLA | Relative derivative: TATT |
|---|---:|---:|---:|---:|
| Source4801: kernel201/LOS65 → kernel401/LOS129 | 0.11532 | 0.11499 | 0.838% | 0.926% |
| Kernel401/LOS129: source4801 → source9601 | 0.000932 | 0.001275 | 0.0775% | 0.0882% |

The source change in the second row is appreciably smaller than the earlier
2401→4801 change. At $h$ its derivative percentages are smaller still,
0.0403% for NLA and 0.0524% for TATT. At $h/4$, however, they grow to
0.187% and 0.194%. The test therefore supports using the larger two steps
for the immediate Fisher diagnostic; it does not establish that halving
all photo-z steps would improve it.

At source9601/kernel401/LOS129, the within-case $h\to h/2$ changes are
0.0193% for NLA and 0.0236% for TATT. The next $h/2\to h/4$ changes are
0.0745% and 0.0735%. The finest source spacing is about 0.000323, so the
smallest step 0.00025 remains below one source sampling interval. This is
one reason to retain grid refinement and parameter-step refinement as
separate tests. The sampled minimum shape kernel also improves to about
$-1.88\times10^{-14}$ across these shifts; that observation remains a
sampled-node check rather than a positivity guarantee between nodes.

The combined radial change remains close to one percent of this derivative
column even though its mean-vector change is only about 0.115 in the joint
covariance norm. Refining both radial axes together cannot identify how
much of the remaining error belongs to each axis or exclude cancellation.
The revised baseline includes both refinements; a later LOS-only or
kernel-only check should start from this revised baseline.

The old and new 4801 baseline mean vectors and derivative arrays agree
bit for bit in both NLA and TATT. Their manifest differences are the paths
of the relocated output and source-table files, the resulting configuration
hash, and the study-script hash after adding combined cases. The physical
source-table content hashes, compiled libraries, project Python hashes,
parent configurations, source FITS, covariance and row inputs agree. Thus
the changed configuration hashes do not indicate changed scientific inputs
for this repeated baseline.

## Why a fourth difference cannot decide this by itself

The saved diagnostic is

$$
q_s=\|d(-2s)-4d(-s)+6d(0)-4d(s)+d(2s)\|_C.
$$

For TATT PCHIP1201, $q_{h/4}=1.39\times10^{-5}$, whereas for PCHIP4801 it is
$2.14\times10^{-4}$. Ranking these two source grids only by this number
would favor the coarser grid, despite its approximately 1.4% difference
from the finer-grid $h/4$ derivative in the same covariance metric.
For PCHIP4801, $q_s$ decreases from $2.55\times10^{-4}$ at $h$ to
$6.86\times10^{-5}$ at $h/2$, then increases at $h/4$.

The shared Fisher-noise estimator can convert a fourth difference into a
noise model under its stated assumptions. Those assumptions do not follow
from these data. An odd coherent error can change a first derivative while
vanishing in this even fourth difference. Smooth physical curvature can
also generate a nonzero fourth difference, and a support-edge cusp is
deterministic. In addition, PCHIP is continuously differentiable; it does
not guarantee the higher smoothness needed to infer an $s^4$ decrease of
this diagnostic everywhere. Keep the saved fourth differences, one-sided
slopes and direct grid changes as separate evidence.

## Source support is a modeling choice

The supplied processed FITS has a last midpoint at $z=2.995$ and an outer
upper cell edge at $z=3.0$. The last densities in all four bins are nonzero,
with a spike above the preceding sample. The probabilities represented by
the last histogram cell, if its density is taken as constant across its
0.01 width, are approximately 0.146%, 0.174%, 0.0242% and 0.175% in bins
1–4. Their provenance needs a scientific explanation before adopting a
source model for inference.

Linear midpoint interpolation with zero outside its midpoint range has a
nonzero jump at the upper support endpoint. Products of shifted source
densities can then have a deterministic cusp when their supports begin
or cease to overlap. Refining the grid can resolve that behavior; it cannot
turn the declared source function into a globally smooth function.

The tested PCHIP policy adds a zero at the measured upper edge and a zero
plateau to the background limit 3.1. This avoids inventing a populated
0.105-wide tail from the last midpoint to 3.1. It also changes the source
reconstruction. It preserves the stated outer support, not the probability
of every histogram cell. For example, at PCHIP4801 the four raw integrals
are approximately 1.000368, 1.000437, 1.000062 and 1.000441 before
renormalization. For bin1 the normalized mean redshift is 0.335575,
compared with 0.334755 for linear1201. Part of the latter comparison is
still a finite-grid difference, so it is not a clean measurement of only
the source-model shift.

Source positivity does not guarantee positivity of the later C shape
kernel. At PCHIP1201 with `dz1=+0.002`, the sampled shape kernel reaches
$-4.74\times10^{-10}$ near $z=3.00328$, just above the shifted upper support;
its positive peak is $1.44\times10^{-6}$. At PCHIP4801 the most negative
sample across the tested shifts is about $-2.35\times10^{-12}$. The decrease
is encouraging, but these are minima at the sampled LOS nodes, not a proof
between nodes. No clipping should conceal the remaining spline behavior.

## Declared demonstration baseline and remaining tests

The two demonstration configurations declare measured-edge PCHIP with
`source_nz=9601`, `kernel_nz=401`, and 129 explicit LOS nodes. Their global
and stage accuracy boosts are all one: the refined counts are the recorded
baseline, not hidden multipliers. They retain the corrected projection
controls stated above and a photo-z step of 0.001, with the ordinary Fisher
calculation also evaluating its 0.0005 check. New table and result names
preserve the earlier files. A new full Fisher base is required because the
source reconstruction and numerical definition have changed.

These choices are defensible for a **demonstration Fisher diagnostic**:
the tested `dz1` source refinement at the primary steps is below 0.1% in
both models, and its within-case step agreement is better still. That
statement has a limited scope. It does not bound the marginalized FoM,
validate the other parameter columns, determine the remaining LOS error,
or show that the selected source model matches the survey calibration.
An acceptance threshold for the final Fisher result must be assessed with
its complete derivative matrix and priors.

Retain the linear native convention as a separately labeled reference,
and establish the meaning of the final FITS cell before interpreting the
linear/PCHIP difference as an uncertainty in the physical source model.
Smoother derivatives are useful evidence about the numerical calculation;
they are not sufficient grounds to replace a physical source convention.

For subsequent numerical checks, first examine `dz2`, `dz3` and `dz4` with
the same two primary steps. This study supplies no evidence that those
columns have already converged. A LOS129→257 check at source9601/kernel401
would isolate the largest unresolved radial-integration direction. If the
smallest-step photo-z check is needed, refine the source sampling again
before relying on $h/4$; do not assume that a smaller parameter step alone
reduces its error. Full Fisher and accuracy-screen results should retain
these qualifications until those independent directions have been tested.

## Saved evidence

The numerical statements above use the local six-case results in
`runs/source_policy_convergence/` and the completed follow-up in
`runs/source_combined_convergence/`, relative to the project directory,
especially each case's `summary.json`, `diagnostics.npz`, `point_*.npz`,
`manifest.json` and `resolved_numerics.json`. Additional numbers were obtained
only by reading those saved arrays and applying the fixed joint covariance;
no physical prediction or CAMB calculation was launched for this review.
The [study script](source_policy_convergence.py) records the exact resolved
configuration and dependencies in each manifest. The original input
configuration is retained in project commit `97e5010`; later demonstration
defaults must not be substituted for that archived definition. Each run
used at most four OpenMP workers and one BLAS worker.
