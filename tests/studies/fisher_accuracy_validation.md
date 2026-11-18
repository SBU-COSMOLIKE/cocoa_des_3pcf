# Fisher accuracy after the interpolation and source corrections

The corrected NLA and TATT calculations have complete default-resolution
Fisher bases. Neither is yet accepted as numerically converged. This record
separates those complete calculations from the less expensive tests that
reuse their marginalization weights. The executable presentation is
[EXAMPLE_FISHER_ACCURACY.ipynb](../../EXAMPLE_FISHER_ACCURACY.ipynb).

## What is held fixed

The configurations retain all 400 xi and 80 Map3 rows, their supplied joint
covariance including cross correlations, and the stated calibration and
photo-z Gaussian priors. NLA varies 17 parameters; TATT varies 20.
Both vary w0 and wa without Gaussian priors. The covariance is fixed across
all displacements. No scale cut, compression or finite-mock precision
correction is introduced by a numerical refinement.

Sources use the declared clipping followed by measured-edge PCHIP, with
9601 density nodes, 401 kernel nodes and 129 line-of-sight nodes. These
choices follow the [source study](source_policy_validation.md).
Signed component ratios and the stretched triangle/mixed-angle tables
follow the [interpolation study](interpolation_validation.md).
Their numerical settings define a starting calculation. They do not
settle the remaining source-model and measured-estimator questions.

## Complete default-resolution calculations

Each parameter has a five-point derivative at displacement h and h/2.
Those stencils share points, giving six distinct displaced means per
parameter and one common central mean. Thus the full NLA base requires
103 distinct means, and the full TATT base requires 121.

| Model | w0–wa FoM at h | w0–wa FoM at h/2 | h/2 relative to h |
|---|---:|---:|---:|
| NLA | 0.1800586574 | 0.1911739287 | +6.173% |
| TATT | 0.1363035239 | 0.1440473499 | +5.681% |

Both joint and xi-only matrices pass the parameter-rank check. The
likelihood information satisfies

```math
F_{\rm joint}^{\rm like}
=F_\xi^{\rm like}+F_{{\rm Map3}\mid\xi}^{\rm like}.
```

The conditional term contains the aperture response that cannot be
predicted from xi through their cross-covariance. The relative matrix
residual is 5.67e-16 for NLA and 2.36e-16 for TATT. Priors are added once
after this likelihood decomposition. The equality checks information
accounting; it does not establish derivative accuracy.
The discarded initial TATT value 12.93 belongs to an interpolation failure
and is not one of these bases.

For a derivative column $`D`$, compare the two steps using

```math
\frac{\lVert D_h-D_{h/2}\rVert_C}{\lVert D_{h/2}\rVert_C},
\qquad \lVert v\rVert_C^2=v^{\mathsf T}C^{-1}v.
```

The subscript $`C`$ specifies the full joint covariance metric. It weights
xi and Map3 with their uncertainties and cross correlations, despite their
different raw amplitudes. The Hubble-h column changes by 1.751% for
NLA and 1.793% for TATT; the Omega_m column changes by 0.426% and 0.431%.
Photo-z changes are smaller than 0.06% in both models at these two steps.
That last observation is a step comparison at one source resolution;
it is not a source-grid refinement of every photo-z column.

## The selected-column experiment

The cost reduction comes from keeping the full default Fisher matrix as
the marginalization reference. A numerical candidate supplies only the
columns selected for inspection. The saved default arrays also supply
the parameter-covariance weights in the shared noise formula. Increasing
one integration count therefore does not initially require rebuilding
every nuisance derivative.

| Calculation | IA model | Distinct mean evaluations |
|---|---|---:|
| Full default base, with two derivative spacings | NLA | 103 |
| Full default base, with two derivative spacings | TATT | 121 |
| One candidate, four selected columns | NLA | 17 |
| One candidate, four selected columns | TATT | 17 |
| Two-column smaller/larger-step follow-up, reusing default means | NLA | 8 |
| Two-column smaller/larger-step follow-up, reusing default means | TATT | 8 |

These are model-evaluation counts, not timing measurements. Reusing
the default matrix makes candidate selection less expensive; it does
not turn the noise formula into a convergence bound. The triangle
example below explains why a full follow-up is still needed.

The shared core functions estimate diagonal false information from the
fourth difference of the retained mean vectors and weight it by the
default Fisher covariance. Both models select w0, wa, Hubble h and
Omega_m. Each candidate therefore needs 17 distinct means. The remaining
columns stay at their default values for this screening experiment.
The selected columns cover 99.913% of the measured NLA contribution and
99.942% for TATT. With every baseline column included, the assumed
diagonal-noise model gives FoM inflation ratios 1.009727 and 1.008720.
These roughly one-percent estimates do not explain away the larger actual
step changes above. Their assumptions must be tested against the derivatives.

Two ratios answer different questions. The noise ratio compares the
shared diagonal-jitter estimate at the candidate and default settings,
using the same selected columns. The derivative-replacement ratio instead
replaces those four actual derivative columns in the full base before
recalculating the FoM. It includes their cross terms with every retained
column. It still is not a complete finer-grid Fisher calculation.

The first completed NLA candidates give:

| Refined setting | Noise-ratio change | Selected-derivative FoM change | Central-mean joint delta chi-squared |
|---|---:|---:|---:|
| Line-of-sight intervals x2 | -0.000334% | +0.3311% | 0.00028645 |
| Triangle table counts x1.5 | +0.001741% | -6.7211% | 0.0191191 |
| Ordinary two-point accuracy x1.5 | -0.0000937% | -0.03424% | 0.0000107435 |
| Xi Fourier quadrature x2 | -0.0000749% | -0.09373% | 1.841e-10 |

The central-mean distance uses the complete joint covariance. It measures
the change of the prediction at one fixed parameter point, with no fit to
data. The triangle change illustrates why a small mean distance and an
unchanged fourth difference are insufficient acceptance tests. Its four
weighted derivative changes are only 0.14–0.25%, yet they alter a weak
marginalized parameter combination enough to change the ellipse area.

An independent saved-array check confirms that matrix inversion is not
causing this amplification. The parameter-scaled Fisher condition number
is about 3.74e4 at default and 4.32e4 after the triangle substitution.
QR projection of nuisance directions, singular-value decomposition of
the whitened derivative matrix, and the existing Fisher inverse agree
in FoM to about 1e-12. Substituting only the refined Omega_m column gives
a ratio 0.91267; substituting only Hubble h gives 1.00588, and substituting
w0 and wa together gives 1.01558. These substitutions are diagnostic
changes to a shared matrix, not independent likelihood experiments.

The reason is easier to see in the whitened data space. First express
each derivative in measurement-error units and add rows representing the
Gaussian priors. Then remove from each dark-energy derivative every part
that another varied parameter can imitate. Only 5.75% of the original w0
column norm and 8.56% of the wa norm survive this projection in the NLA
base. The triangle refinement changes those small remaining responses
by 9.32% and 5.77%. A small change compared with the original large
columns can therefore be substantial compared with the information that
survives marginalization.

Thus Hubble h dominates the measured fourth-difference noise ranking,
while Omega_m drives this coherent triangle sensitivity. The cheap test
has located a useful full follow-up without replacing that follow-up.

## Full derivatives and compensating nuisance changes

The completed NLA and TATT triangle refinements change every derivative
column. In each model, the 4 x 5 x 480 selected stencil values agree bit
for bit with the earlier four-column scan. Reconstructing those four
derivatives in the same arithmetic also gives identical bits. The
differences below therefore
come from updating the remaining columns, rather than from different
predictions at the shared parameter points.

| NLA derivative matrix at h/2 | w0–wa FoM | Ratio to the full default h/2 FoM |
|---|---:|---:|
| All columns at default triangle resolution | 0.1911739287 | 1.000000 |
| Only the four selected columns refined | 0.1783249932 | 0.932789 |
| Only the other thirteen columns refined | 0.2049386912 | 1.072001 |
| All seventeen columns refined | 0.1914995559 | 1.001703 |

| TATT derivative matrix at h/2 | w0–wa FoM | Ratio to the full default h/2 FoM |
|---|---:|---:|
| All columns at default triangle resolution | 0.1440473499 | 1.000000 |
| Only the four selected columns refined | 0.1364189361 | 0.947042 |
| Only the other sixteen columns refined | 0.1523851638 | 1.057882 |
| All twenty columns refined | 0.1439674353 | 0.999445 |

Both partial substitutions mix two numerical grids. They change the
relations between parameter responses, which determine what survives
nuisance marginalization. Refining all columns restores compensating
changes that either mixed matrix misses. The FoM is a nonlinear function
of the derivative matrix, so the two partial shifts do not add. Their
large shifts correctly request a full check but do not predict its answer.

At the larger derivative step, the full NLA FoM changes from 0.1800586574
to 0.1805178664, a 0.255% change. Thus the full triangle-grid changes are
smaller than the original h-versus-h/2 disagreement. This result does not
settle the derivative-step choice, higher angular orders or other
integration grids.

The corresponding TATT change at the larger step is from 0.1363035239
to 0.1362851406, or -0.0135%. At h/2 it is -0.0555%. Both models
therefore have small full-Fisher changes for this particular triangle
refinement, despite the large mixed-grid substitutions. Their refined
h/2 values still differ from their refined h values by 6.083% for NLA
and 5.637% for TATT. A stable triangle comparison has not removed the
step sensitivity.

The paper-related quantity also needs its own check. With dark energy
fixed and all other parameters marginalized, the NLA joint/xi
matter-density–S8 FoM ratio changes from 2.138926 to 2.156096 at h/2,
an increase of 0.803%. The joint S8 error changes from 0.01546466 to
0.01542283. These are local demonstration constraints under the stated
uncut vector and priors; they are not the paper's posterior comparison.

For TATT, the same fixed-dark-energy FoM ratio changes from 2.920618
to 2.944914, an increase of 0.832%. Its joint S8 error changes from
0.01550684 to 0.01548462. The two-point derivative rows remain bitwise
unchanged for both models and both step lengths. Thus these comparisons
isolate the stated three-point triangle refinement, while retaining
all two-point information and the cross-covariance. The complete array
record is `runs/fisher_array_diagnostics/full_triangle.json`; the
reporting command below verifies the shared stencils, reconstructs the
Fisher matrices and checks the configuration and physical identities.

## Step follow-up and remaining checks

The targeted step study adds eight means per model.
Here h labels the original derivative spacing for each column; the
physical Hubble parameter is a separate use of that letter. The absolute
original spacings are 0.003 for Omega_m and 0.005 for Hubble h.
For each of Omega_m and Hubble h, the stored vectors already contain the
center and all points needed except +/-h/4 and +/-4h. Adding those points
provides stencils at h/4 and 2h, alongside the saved h/2 and h stencils.
The original data are reused as explicit arrays; no predictions are
inserted into an internal model cache. Four independent cubic-polynomial
tests check the node mapping, exact four/eight call counts, matching
shared points and rejection of displacements beyond the stated bounds.

The table substitutes only those two columns into the full h/2 Fisher:

| Selected-column spacing | NLA FoM / base | TATT FoM / base | NLA diagonal-noise inflation | TATT diagonal-noise inflation |
|---|---:|---:|---:|---:|
| h/4 | 1.003987 | 1.002267 | 1.000137 | 1.000207 |
| h/2 | 1.000000 | 1.000000 | 1.009591 | 1.008356 |
| h | 0.944901 | 0.953369 | 1.022409 | 1.023189 |
| 2h | 0.985502 | 0.985674 | 1.007357 | 1.006456 |

These noise-inflation columns concern the same two tested parameters;
they differ from the all-parameter values above. Their denominators are
the fixed full h/2 Fisher, while each addition uses its own absolute
step. The h/2 derivative-replacement ratio differs from one only at
about 3e-14, measuring the accumulation-order roundoff of the shared
derivative helper on the same saved vectors.

The smaller steps give closer agreement for the quantity tested here:
0.40% for NLA and 0.23% for TATT. Larger steps are not monotonically
better. This is evidence about two columns at the default grid, rather
than a complete finer-grid or all-column step-convergence result.

The same eight-mean experiment is now complete on each full
triangle-refined grid. Here the denominator and every unchanged
derivative come from that model's full `triangle_1p5` h/2 archive:

| Selected-column spacing on triangle-refined grid | NLA FoM / refined base | TATT FoM / refined base | NLA diagonal-noise inflation | TATT diagonal-noise inflation |
|---|---:|---:|---:|---:|
| h/4 | 1.004076 | 1.002550 | 1.000138 | 1.000212 |
| h/2 | 1.000000 | 1.000000 | 1.009152 | 1.008189 |
| h | 0.946568 | 0.954660 | 1.021609 | 1.022916 |
| 2h | 0.987116 | 0.987006 | 1.007038 | 1.006343 |

The smaller-step changes remain close to the default-grid comparison.
The stored evidence is in `runs/nla_triangle_step_study/` and
`runs/tatt_triangle_step_study/`, separately from the default-grid files
used by the notebook's current step plots. The h/2 substitution differs
from one by about 2e-12 or less because the helper accumulates the
identical retained vectors in a different arithmetic order. This is
distinct from the physical h/4 changes of 0.408% and 0.255%.

The noise interpretation also deserves a numerical check. At a fixed
variance of independent evaluation jitter, halving the parameter step
would increase its expected false-information coefficient by four,
because that coefficient contains an inverse step squared. Here the
measured fourth-difference coefficient instead decreases: its h/4-to-h/2
ratio on the refined grid is 0.0737/0.0765 for Omega_m in NLA/TATT and
0.00704/0.01057 for Hubble h. A smooth fourth-derivative contribution
would scale differently again: its fourth difference is proportional
to the fourth power of the step, giving a coefficient proportional to
the sixth power. Neither these two spacings nor one measured fourth
difference identifies the error process. The observations reinforce
why the diagnostic must accompany actual derivative and grid changes,
instead of being interpreted as a measured independent-noise variance.

The completed full triangle refinements evaluate every derivative at
both step lengths. Their xi rows contribute most of the baseline step discrepancy;
increasing only the three-point grids cannot settle it. The targeted
step comparison is now checked at that refined grid. A smaller-step
calculation of every column remains a separate test.

The current candidate plan leaves the angular truncations Lmax=Mmax=20
fixed. Its multipole control refines interpolation tables, not those
orders. External CAMB tables and integration bounds also need their own
tests. The historical xi quadrature figure used the earlier linear
source reconstruction and remains motivation for the selected node
count, rather than a current-source accuracy certificate.

Physical assumptions remain separate: the finite-SAS weights have not
been matched to the catalogue estimator; reusing the supplied fixed
covariance does not validate its Gaussian or connected TATT contributions;
small xi scales have no validated baryonic cut;
and a bispectrum-only feedback parameter cannot supply a complete joint
feedback forecast. The notebook records numerical evidence for its
declared demonstration, rather than adopting published improvements as
targets that the calculation must reach.

## Reproduce the saved-array and step studies

Start in `Cocoa/projects/des_3pcf` with Cocoa active.

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

**Step :two:**: Inspect nuisance projection with independent matrix algebra.

The saved-array reports use serial BLAS, write under
`runs/fisher_array_diagnostics`, and evaluate no physical model.

```bash
python tests/studies/analyze_des3pcf_fisher_amplification.py
```

**Step :three:**: Separate xi and aperture contributions to step sensitivity.

```bash
python tests/studies/analyze_des3pcf_fisher_step_blocks.py
```

**Step :four:**: Check that both full triangle archives are available.

```bash
python tests/studies/full_triangle_report.py
```

**Step :five:**: Compare their full derivatives and parameter constraints.

```bash
python tests/studies/full_triangle_report.py --execute
```

Here `--execute` requests array analysis only. This report verifies the
selected and full stencils, contrasts partial and full substitutions,
and compares IA errors and the matter-density/amplitude information gain
with dark energy fixed or varied. Missing full archives are reported;
they are never calculated by this command.

**Step :six:**: List the missing points for a parameter-step follow-up.

The following command checks the original baseline and lists the eight
new physical points. It does not evaluate them. Use a new output directory
if a prior executed study already occupies the chosen path.

```bash
python tests/studies/cosmology_step_study.py \
  configs/des_y3_nla_demo_fisher.yaml \
  --output runs/nla_step_followup --larger-step
```

**Step :seven:**: Calculate the eight additional physical means.

For this separate script, `--execute` requests predictions with the
YAML's four-worker OpenMP setting. Run one physical study at a time.

```bash
python tests/studies/cosmology_step_study.py \
  configs/des_y3_nla_demo_fisher.yaml \
  --output runs/nla_step_followup --larger-step --execute
```

The script verifies the saved baseline, row order, covariance and physical
dependencies before work and again before saving. Its metadata also hashes
the baseline archive, configuration, study script and shared diagnostic.
It refuses an occupied output directory and retains every assembled stencil.
