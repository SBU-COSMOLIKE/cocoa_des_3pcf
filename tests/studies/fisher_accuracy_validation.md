# Fisher accuracy after the interpolation and source corrections

The corrected NLA and TATT calculations have complete Fisher bases at the
default resolution and on the triangle-refined grid, with full derivative
matrices at three step lengths on that refined grid. Neither model is yet
accepted as numerically converged. This page separates the complete
calculations from the less expensive screening tests that reuse their
marginalization weights. The executable presentation is
[EXAMPLE_FISHER_ACCURACY.ipynb](../../EXAMPLE_FISHER_ACCURACY.ipynb).

1. [What is held fixed](#fixed)
2. [Complete default-resolution bases](#bases)
3. [Column screening on the default base](#screening)
4. [Full triangle-refined derivatives](#triangle)
5. [Two-column step studies](#steps)
6. [Full smaller-step derivatives on the refined grid](#h4)
7. [Remaining comparisons](#remaining)
8. [Reproduce the saved-array and step studies](#reproduce)

## What is held fixed <a name="fixed"></a>

The configurations keep all 400 xi and 80 Map3 rows, their joint
covariance including cross correlations, and the stated calibration and
photo-z Gaussian priors. NLA varies 17 parameters; TATT varies 20. Both
vary $`w_0`$ and $`w_a`$ without Gaussian priors. The covariance stays
fixed across all displacements, and no numerical refinement introduces a
scale cut, compression or finite-mock precision correction.

Sources use the stated clipping followed by measured-edge PCHIP, with
9601 density nodes, 401 kernel nodes and 129 line-of-sight nodes,
following [the source study](source_policy_validation.md). Signed
component ratios and the stretched triangle and mixed-angle tables follow
[the interpolation study](interpolation_validation.md). These settings
define a starting calculation; the remaining source-model and
measured-estimator questions stay open.

## Complete default-resolution bases <a name="bases"></a>

Each parameter has a five-point derivative at displacement $`h`$ and
$`h/2`$. The stencils share points, giving six distinct displaced means
per parameter and one common central mean, so the full NLA base requires
103 distinct means and the full TATT base 121.

| Model | $`w_0`$–$`w_a`$ FoM at $`h`$ | $`w_0`$–$`w_a`$ FoM at $`h/2`$ | FoM change, $`h`$ to $`h/2`$ |
|---|---:|---:|---:|
| NLA | 0.1800586574 | 0.1911739287 | +6.173% |
| TATT | 0.1363035239 | 0.1440473499 | +5.681% |

Both joint and xi-only matrices pass the parameter-rank check. The
likelihood information satisfies

```math
F_{\rm joint}^{\rm like}
=F_\xi^{\rm like}+F_{{\rm Map3}\mid\xi}^{\rm like},
```

where the conditional term contains the aperture response that the
cross-covariance cannot predict from xi. The relative matrix residual of
this equality is $`5.67\times10^{-16}`$ for NLA and
$`2.36\times10^{-16}`$ for TATT, and priors enter once after the
decomposition. The equality checks information accounting, not derivative
accuracy. The discarded initial TATT figure of merit 12.93 belongs to the
interpolation failure recorded in
[the interpolation study](interpolation_validation.md) and is not one of
these bases.

For a derivative column $`D`$, the two steps are compared through

```math
\frac{\lVert D_h-D_{h/2}\rVert_C}{\lVert D_{h/2}\rVert_C},
\qquad \lVert v\rVert_C^2=v^{\mathsf T}C^{-1}v.
```

The subscript $`C`$ names the full joint covariance metric, which weights
xi and Map3 with their uncertainties and cross correlations despite their
different raw amplitudes. The Hubble-parameter column changes by 1.751%
for NLA and 1.793% for TATT; the $`\Omega_m`$ column changes by 0.426%
and 0.431%. Photo-z changes stay below 0.06% in both models at these two
steps, which is a step comparison at one source resolution rather than a
source-grid refinement of every photo-z column.

## Column screening on the default base <a name="screening"></a>

The cost reduction comes from keeping the full default Fisher matrix as
the marginalization reference: a numerical candidate supplies only the
columns under inspection, and the saved default arrays supply the
parameter-covariance weights in the shared noise formula. Increasing one
integration count therefore does not require rebuilding every nuisance
derivative at screening time.

| Calculation | IA model | Distinct mean evaluations |
|---|---|---:|
| Full default base, two derivative spacings | NLA | 103 |
| Full default base, two derivative spacings | TATT | 121 |
| One candidate, four screening columns | NLA | 17 |
| One candidate, four screening columns | TATT | 17 |
| Two-column step follow-up, reusing default means | NLA | 8 |
| Two-column step follow-up, reusing default means | TATT | 8 |

These are model-evaluation counts, not timing measurements. Reusing the
default matrix makes candidate screening less expensive; it does not turn
the noise formula into a convergence bound, as the triangle example below
shows.

The shared core functions estimate diagonal false information from the
fourth difference of the kept mean vectors and weight it by the default
Fisher covariance. Both models screen $`w_0`$, $`w_a`$, the Hubble
parameter $`h`$ and $`\Omega_m`$, so each candidate needs 17 distinct
means while the remaining columns stay at their default values. The four
screening columns cover 99.913% of the measured NLA noise contribution
and 99.942% for TATT. With every baseline column included, the assumed
diagonal-noise model gives FoM inflation ratios 1.009727 and 1.008720;
these roughly one-percent estimates do not explain the larger actual step
changes above, so their assumptions are tested against the derivatives.

Two ratios answer different questions. The noise ratio compares the
shared diagonal-jitter estimate at the candidate and default settings,
using the same screening columns. The derivative-replacement ratio
instead replaces those four actual derivative columns in the full base
before recalculating the FoM, which includes their cross terms with every
kept column; it still is not a complete finer-grid Fisher calculation.

The completed NLA candidates give:

| Refined setting | Noise-ratio change | FoM change from replacing the four screening columns | Central-mean joint $`\Delta\chi^2`$ |
|---|---:|---:|---:|
| Line-of-sight intervals x2 | -0.000334% | +0.3311% | 0.00028645 |
| Triangle table counts x1.5 | +0.001741% | -6.7211% | 0.0191191 |
| Ordinary two-point accuracy x1.5 | -0.0000937% | -0.03424% | 0.0000107435 |
| Xi Fourier quadrature x2 | -0.0000749% | -0.09373% | $`1.841\times10^{-10}`$ |

The central-mean distance uses the complete joint covariance and measures
the change of the prediction at one fixed parameter point, with no fit to
data. The triangle row shows why a small mean distance and an unchanged
fourth difference are insufficient acceptance tests: its four weighted
derivative changes are only 0.14% to 0.25%, yet they alter a weak
marginalized parameter combination enough to change the ellipse area.

An independent saved-array check rules out matrix inversion as the cause
of this amplification. The parameter-scaled Fisher condition number is
about $`3.74\times10^{4}`$ at default and $`4.32\times10^{4}`$ after the
triangle substitution. QR projection of nuisance directions,
singular-value decomposition of the whitened derivative matrix and the
existing Fisher inverse agree in FoM to about $`10^{-12}`$. Substituting
only the refined $`\Omega_m`$ column gives a FoM ratio 0.91267,
substituting only the Hubble parameter gives 1.00588, and substituting
$`w_0`$ and $`w_a`$ together gives 1.01558; these are diagnostic changes
to a shared matrix, not independent likelihood experiments.

The whitened data space makes the reason visible. Express each derivative
in measurement-error units, add rows representing the Gaussian priors,
and remove from each dark-energy derivative every part that another
varied parameter can imitate. Only 5.75% of the original $`w_0`$ column
norm and 8.56% of the $`w_a`$ norm survive this projection in the NLA
base, and the triangle refinement changes those small remaining responses
by 9.32% and 5.77%. A change that is small compared with the original
columns can therefore be substantial compared with the information that
survives marginalization. The Hubble parameter dominates the measured
fourth-difference noise ranking, while $`\Omega_m`$ drives this coherent
triangle sensitivity; the cheap test located the full follow-up without
replacing it.

## Full triangle-refined derivatives <a name="triangle"></a>

The completed NLA and TATT triangle refinements change every derivative
column. In each model, the 4 x 5 x 480 screening stencil values agree bit
for bit with the earlier four-column scan, and reconstructing those four
derivatives in the same arithmetic gives identical bits. The differences
below therefore come from updating the remaining columns, not from
different predictions at the shared parameter points.

| NLA derivative matrix at $`h/2`$ | $`w_0`$–$`w_a`$ FoM | Ratio to the full default $`h/2`$ FoM |
|---|---:|---:|
| All columns at default triangle resolution | 0.1911739287 | 1.000000 |
| Only the four screening columns refined | 0.1783249932 | 0.932789 |
| Only the other thirteen columns refined | 0.2049386912 | 1.072001 |
| All seventeen columns refined | 0.1914995559 | 1.001703 |

| TATT derivative matrix at $`h/2`$ | $`w_0`$–$`w_a`$ FoM | Ratio to the full default $`h/2`$ FoM |
|---|---:|---:|
| All columns at default triangle resolution | 0.1440473499 | 1.000000 |
| Only the four screening columns refined | 0.1364189361 | 0.947042 |
| Only the other sixteen columns refined | 0.1523851638 | 1.057882 |
| All twenty columns refined | 0.1439674353 | 0.999445 |

Both partial substitutions mix two numerical grids, which changes the
relations between parameter responses that determine what survives
nuisance marginalization. Refining all columns restores compensating
changes that either mixed matrix misses; the FoM is a nonlinear function
of the derivative matrix, so the two partial shifts do not add. Their
large shifts correctly request a full check without predicting its
answer.

At the larger derivative step, the full NLA FoM changes from 0.1800586574
to 0.1805178664, a 0.255% change, so the full triangle-grid changes are
smaller than the original $`h`$-versus-$`h/2`$ disagreement. The
corresponding TATT change at the larger step is from 0.1363035239 to
0.1362851406, or -0.0135%; at $`h/2`$ it is -0.0555%. Both models have
small full-Fisher changes for this triangle refinement, while their
refined $`h/2`$ values still differ from their refined $`h`$ values by
6.083% for NLA and 5.637% for TATT: a stable triangle comparison has not
removed the step sensitivity.

The paper-related quantity has its own numbers. With dark energy fixed
and all other parameters marginalized, the NLA joint-to-xi
$`\Omega_m`$–$`S_8`$ FoM ratio changes from 2.138926 to 2.156096 at
$`h/2`$, an increase of 0.803%, and the joint $`\sigma(S_8)`$ changes
from 0.01546466 to 0.01542283. For TATT the same ratio changes from
2.920618 to 2.944914, an increase of 0.832%, and its joint
$`\sigma(S_8)`$ from 0.01550684 to 0.01548462. These are local
demonstration constraints under the stated uncut vector and priors, not
the paper's posterior comparison. The two-point derivative rows stay
bitwise unchanged for both models and both step lengths, so these
comparisons isolate the three-point triangle refinement while keeping all
two-point information and the cross-covariance. The complete array record
is `results/fisher_array_diagnostics/full_triangle.json`; the reporting command
below verifies the shared stencils, reconstructs the Fisher matrices and
checks the configuration and physical identities.

## Two-column step studies <a name="steps"></a>

The two-column step study adds eight means per model. Here $`h`$ labels
the original derivative spacing for each column; the physical Hubble
parameter is a separate use of that letter. The absolute original
spacings are 0.003 for $`\Omega_m`$ and 0.005 for the Hubble parameter.
For each of the two, the stored vectors already contain the center and
all points needed except $`\pm h/4`$ and $`\pm 4h`$; adding those points
provides stencils at $`h/4`$ and $`2h`$ alongside the saved $`h/2`$ and
$`h`$ stencils. The original data enter as explicit arrays, with no
predictions inserted into an internal model cache, and four independent
cubic-polynomial tests check the node mapping, the exact four or eight
call counts, the matching shared points and the rejection of
displacements beyond the stated bounds.

The table substitutes only those two columns into the full $`h/2`$
Fisher:

| Spacing of the two substituted columns | NLA FoM / base | TATT FoM / base | NLA diagonal-noise inflation | TATT diagonal-noise inflation |
|---|---:|---:|---:|---:|
| $`h/4`$ | 1.003987 | 1.002267 | 1.000137 | 1.000207 |
| $`h/2`$ | 1.000000 | 1.000000 | 1.009591 | 1.008356 |
| $`h`$ | 0.944901 | 0.953369 | 1.022409 | 1.023189 |
| $`2h`$ | 0.985502 | 0.985674 | 1.007357 | 1.006456 |

These noise-inflation columns concern the same two tested parameters, so
they differ from the all-parameter values above. The denominators are the
fixed full $`h/2`$ Fisher, while each addition uses its own absolute
step. The $`h/2`$ derivative-replacement ratio differs from one only at
about $`3\times10^{-14}`$, which measures the accumulation-order roundoff
of the shared derivative helper on the same saved vectors.

The smaller steps give closer agreement for the quantity tested here:
0.40% for NLA and 0.23% for TATT, and larger steps are not monotonically
better. This is evidence about two columns at the default grid, not a
complete finer-grid or all-column step-convergence result.

The same eight-mean experiment completed on each full triangle-refined
grid, with the denominator and every unchanged derivative from that
model's full `triangle_1p5` $`h/2`$ archive:

| Spacing of the two substituted columns, refined grid | NLA FoM / refined base | TATT FoM / refined base | NLA diagonal-noise inflation | TATT diagonal-noise inflation |
|---|---:|---:|---:|---:|
| $`h/4`$ | 1.004076 | 1.002550 | 1.000138 | 1.000212 |
| $`h/2`$ | 1.000000 | 1.000000 | 1.009152 | 1.008189 |
| $`h`$ | 0.946568 | 0.954660 | 1.021609 | 1.022916 |
| $`2h`$ | 0.987116 | 0.987006 | 1.007038 | 1.006343 |

The smaller-step changes stay close to the default-grid comparison. The
stored evidence is in `results/nla_triangle_step_study/`
and `results/tatt_triangle_step_study/`, separately from
the default-grid files behind the notebook's step plots. The $`h/2`$
substitution differs from one by about $`2\times10^{-12}`$ or less
because the helper accumulates the identical kept vectors in a different
arithmetic order; this is distinct from the physical $`h/4`$ changes of
0.408% and 0.255%.

The noise interpretation also received a numerical check. At a fixed
variance of independent evaluation jitter, halving the parameter step
would increase the expected false-information coefficient by four,
because that coefficient contains an inverse step squared. The measured
fourth-difference coefficient instead decreases: its $`h/4`$-to-$`h/2`$
ratio on the refined grid is 0.0737/0.0765 for $`\Omega_m`$ in NLA/TATT
and 0.00704/0.01057 for the Hubble parameter. A smooth fourth-derivative
contribution would scale differently again, with a fourth difference
proportional to the fourth power of the step and a coefficient
proportional to the sixth power. Neither these two spacings nor one
measured fourth difference identifies the error process, so the
diagnostic accompanies actual derivative and grid changes instead of
standing for a measured independent-noise variance.

## Full smaller-step derivatives on the refined grid <a name="h4"></a>

The full smaller-step calculation changes the step for every varied
parameter. A fourth-order central derivative at $`h/4`$ is

```math
D_{h/4}=\frac{d(-h/2)-8d(-h/4)+8d(h/4)-d(h/2)}{3h}.
```

The complete triangle archive already contains the outer points and the
center, so only $`\pm h/4`$ are missing: the full new derivative matrix
requires two new means per parameter, 34 for NLA and 40 for TATT. The
completed two-column study supplies four of those points, reducing the
actual new evaluations to 30 and 36 without changing the set of
derivatives in the Fisher matrix; its four additional large-step points
are not needed here.

Both calculations completed with exact parameter coordinates, unchanged
physical and numerical settings, all 480 rows, the same covariance and
priors. The saved-vector lookup stays separate from the physical engine's
cache. The existing derivative routine first reconstructs the original
$`D_{h/2}`$ bit for bit, then forms $`D_{h/4}`$; rebuilt original Fisher
and prior matrices also agree with their archive, and the new results
have their own format with an explicit step divisor, so no old result is
relabeled.

| Model on triangle-refined grid | Full FoM at $`h`$ | Full FoM at $`h/2`$ | Full FoM at $`h/4`$ | FoM change, $`h/2`$ to $`h/4`$ |
|---|---:|---:|---:|---:|
| NLA | 0.1805178664 | 0.1914995559 | 0.1898830823 | -0.8441% |
| TATT | 0.1362851406 | 0.1439674353 | 0.1430277060 | -0.6527% |

Every two-point and joint parameter matrix stays rank resolved. The
smaller-step disagreements are substantially below the previous 6.08%
and 5.64% comparisons, and they differ in sign from the partial
two-column changes: the remaining cosmological and nuisance derivatives
matter. These numbers establish the measured change between two complete
derivative matrices at this integration and interpolation setting; they
imply no one-percent acceptance tolerance.

The corresponding fixed-dark-energy quantities are:

| Model | Joint/xi $`\Omega_m`$–$`S_8`$ FoM gain at $`h/2`$ | Gain at $`h/4`$ | Joint $`\sigma(S_8)`$ at $`h/2`$ | Joint $`\sigma(S_8)`$ at $`h/4`$ |
|---|---:|---:|---:|---:|
| NLA | 2.156096 | 2.163192 | 0.01542283 | 0.01533137 |
| TATT | 2.944914 | 2.942680 | 0.01548462 | 0.01544720 |

The gain changes by +0.329% for NLA and -0.0759% for TATT. Dark energy is
fixed at $`w_0=-1`$, $`w_a=0`$ in this table, with every other parameter
marginalized and each Gaussian prior included once. The IA notebook's
original figures continue to use the separately stated default grid, so
these follow-up numbers do not silently replace its saved forecast.

The step test is not a Richardson extrapolation to an assumed exact
answer. A smooth quintic test verifies the derivative routine's formal
factor-of-sixteen error reduction when its spacing is halved; the
physical calculation also contains interpolation and internal numerical
integrals, whose smoothness the polynomial example does not establish.
The xi rows contribute most of the original step discrepancy, so
increasing only three-point grids cannot settle every remaining
derivative question.

The quarter-step calculation is now written directly in
[EXAMPLE_FISHER_ACCURACY.ipynb](../../EXAMPLE_FISHER_ACCURACY.ipynb).
Its full-refinement cell retains the existing stencil vectors. The
quarter-step loop reuses their center and inner samples and evaluates
only the missing pair for each parameter. Polynomial tests execute those
notebook cells to check the derivative formula, coordinates and exact
reuse without a physical calculation.

## Remaining comparisons <a name="remaining"></a>

The current candidate plan leaves the angular truncations L=M=20 fixed;
its multipole control refines interpolation tables, not those orders, and
[the angular-order study](angular_order_results.md) measures the fiducial
effect of raising them. External CAMB tables and integration bounds need
their own tests. The historical xi quadrature figure used the earlier
linear source reconstruction, so it motivates the chosen node count
without serving as a current-source accuracy certificate.

Physical assumptions stay separate: the finite-SAS weights have not been
matched to the catalogue estimator, reusing the fixed covariance does not
validate its Gaussian or connected TATT contributions, small xi scales
have no validated baryonic cut, and a bispectrum-only feedback parameter
cannot supply a complete joint feedback forecast. The notebook records
numerical evidence for its stated demonstration rather than adopting
published improvements as targets the calculation must reach.

## Reproduce the comparisons <a name="reproduce"></a>

Open [EXAMPLE_FISHER_ACCURACY.ipynb](../../EXAMPLE_FISHER_ACCURACY.ipynb)
in activated Cocoa. Select the numbered NLA or TATT example near the top;
its covariance, fiducial parameters and priors remain explicit. The
notebook sets BLAS to one thread and limits OpenMP before imports.

First compute the complete default derivatives and noise weights. The
selected-column screen identifies promising settings cheaply. Then choose
a setting and compute its complete derivatives at both spacings; a screen
alone is not a convergence result. Full and selected-column comparisons
use the joint covariance, including its cross block, and add each prior
once. They also report constraints with dark energy varied and fixed.

The quarter-step section reuses the retained full-refinement stencil and
adds only the missing near-center pair per column. The wider-step section
adds the missing outer pair for matter density and Hubble h. Its selected
column substitutions diagnose those parameters without claiming that all
columns have converged at the wider spacing.

The following array-only sections expose nuisance QR projection, independent
SVD and inverse checks of the dark-energy ellipse, weak parameter
directions, and separate two-point/aperture row substitutions. They require
no new physical means. The final angular-order loop does require new
fiducial predictions and records its settings separately from the derivative
step tests. Numerical output belongs under `chains/fisher_accuracy/`.
