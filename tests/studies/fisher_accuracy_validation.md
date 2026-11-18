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
likelihood identity F_joint = F_xi + F_Map3_given_xi holds to a relative
matrix norm of 5.67e-16 for NLA and 2.36e-16 for TATT. This checks the
cross-covariance accounting; it does not establish derivative accuracy.
The discarded initial TATT value 12.93 belongs to an interpolation failure
and is not one of these bases.

For a derivative column D, the useful step comparison is
norm(D_h-D_h2)/norm(D_h2), where norm(v)^2 = v^T C^-1 v. It weights xi and
Map3 with their actual uncertainties and cross correlations, despite their
very different raw amplitudes. The Hubble-h column changes by 1.751% for
NLA and 1.793% for TATT; the Omega_m column changes by 0.426% and 0.431%.
Photo-z changes are smaller than 0.06% in both models at these two steps.
That last observation is a step comparison at one source resolution;
it is not a source-grid refinement of every photo-z column.

## The selected-column experiment

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

## Step follow-up and remaining checks

The targeted step study has now calculated eight new means per model.
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

Full triangle refinements must evaluate every derivative at both step
lengths. Their xi rows contribute most of the baseline step discrepancy;
increasing only the three-point grids cannot settle it. The targeted
step comparison must also be checked at the selected refined grid.

The current candidate plan leaves the angular truncations Lmax=Mmax=20
fixed. Its multipole control refines interpolation tables, not those
orders. External CAMB tables and integration bounds also need their own
tests. The historical xi quadrature figure used the earlier linear
source reconstruction and remains motivation for the selected node
count, rather than a current-source accuracy certificate.

Physical assumptions remain separate: the finite-SAS weights have not
been matched to the catalogue estimator; the fixed covariance omits
connected TATT changes; small xi scales have no validated baryonic cut;
and a bispectrum-only feedback parameter cannot supply a complete joint
feedback forecast. The notebook records numerical evidence for its
declared demonstration, rather than adopting published improvements as
targets that the calculation must reach.

## Reproduce the saved-array and step studies

Start in `Cocoa/projects/des_3pcf` with Cocoa active and serial BLAS.
These two commands inspect saved arrays and write reports under
`runs/fisher_array_diagnostics`; they evaluate no physical model.

```bash
python tests/studies/analyze_des3pcf_fisher_amplification.py
```

```bash
python tests/studies/analyze_des3pcf_fisher_step_blocks.py
```

The following command checks the original baseline and lists the eight
new physical points. It does not evaluate them. Use a new output directory
if a prior executed study already occupies the chosen path.

```bash
python tests/studies/cosmology_step_study.py \
  configs/des_y3_nla_demo_fisher.yaml \
  --output runs/nla_step_followup --larger-step
```

Adding `--execute` requests those means with the YAML's four-worker
OpenMP setting. This is an explicit physical calculation:

```bash
python tests/studies/cosmology_step_study.py \
  configs/des_y3_nla_demo_fisher.yaml \
  --output runs/nla_step_followup --larger-step --execute
```

The script verifies the saved baseline, row order, covariance and physical
dependencies before work and again before saving. Its metadata also hashes
the baseline archive, configuration, study script and shared diagnostic.
It refuses an occupied output directory and retains every assembled stencil.
