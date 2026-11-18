# Paper comparisons, public measurements and inference examples

The immediate scientific target is an observable-level comparison: change
the IA parameters, calculate both xi and Map3, and then ask which differences
remain informative after the cross-covariance and nuisance parameters are
included. A change in an equilateral matter bispectrum alone does not answer
that question. The projection and angular filters can change its sign and
relative importance in a measured statistic.

## Targets that have a computational route

| Publication target | Calculation with this project | Acceptance condition |
|---|---|---|
| 2601.09133, Figure 9 | Compare the two stated IA amplitude pairs in xi+ and Map3 | Common source curves, cosmology and no-IA denominator; refined angular transforms |
| 2601.09133, Figures 3–4 | Scan IA amplitudes for the two illustrated source triplets at 14 arcmin | Figure 3 uses eta1=eta2=1; distinguish Ct from Ct/C1 |
| 2601.09133, Figures 1–2 | Compare the real natural component Gamma0 at a 60-degree opening angle | Ordered source legs and side labels; paper Lmax=50, Mmax=100 followed by refinement |
| 2601.09133, Figures 5–6 | Separate operator contributions and numbers of intrinsic-shape fields | Verify component sum closure; projected field-placement diagnostics require an additional explicit calculation |
| 2503.03964, Figures 2, 4–5 | Compare the downloaded simulated mean and display the covariance blocks | Preserve labels; plotting a supplied covariance is not an independent covariance estimate |
| 2503.03964, Figure 7 | Evaluate the specified baryonic matter-bispectrum transfer | Match redshift, triangle shape, amplitude definition and suppression-only choice |
| 2503.03964, Figure 10 | Inject a matched power/bispectrum contaminant and examine inference bias | Supply both responses and the exact retained estimator before interpreting robustness |

These figure specifications come from the
[TATT paper, Sections II.6–III.2](https://arxiv.org/html/2601.09133v1#S3)
and the [methods paper, Sections III–V](https://arxiv.org/html/2503.03964v2#S3).
The target lists are specifications; they do not record completed figure
reproductions. The notebooks and saved numerical checks must provide that
evidence separately.

For the first row, use `(A1,A2)=(-0.2,0.4)` and `(0.2,-0.4)` and retain
the other Table 1 parameters. The current joint implementation can evaluate
these E/B two-point and E-mode three-point responses. The TATT paper motivates
self-calibration but supplies no joint Fisher gain to use as an acceptance
number. A Fisher result from these examples is an additional calculation.
[TATT paper, Table 1 and Figure 9](https://arxiv.org/html/2601.09133v1#S3)

The existing low-level IA component API is useful for checking signs and
normalizations. It does not make the total-only aperture API a projected
GGI/GII/III decomposition. Such curves must be constructed with the intended
field placements and checked against their sum. Likewise, the three-point
velocity-shear route does not supply a velocity-shear two-point model.

## What the public observed files establish

The separate [observed manifest](data/y3_observed_manifest.json) records
commit `d6098d326dcb03bc750adcc9627119f52d928944` of
[3pcf_integrator](https://github.com/rchgomes/3pcf_integrator/tree/d6098d326dcb03bc750adcc9627119f52d928944).
Every downloaded file is checked against its Git blob hash and byte count.
The upstream
[joint unblinded configuration](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/unblind-metacal-796sims-february-Percival-2ptfix-3pcfsc/unblind-metacal-shear-2pt-map3-NLA-all.ini)
explicitly associates these three files:

| Input under `data/upstream_y3_observed/` | Verified role |
|---|---|
| `data/dv/real_2pt_desy3_unblind_g2flip_matchElisa_2025Mar10.fits` | Observed xi+:200 rows and xi−:200 rows, with source distributions |
| `data/dv/real_map3_desy3_unblind_scalecut3pcf_796simcov_2025Mar11.fits` | Observed Map3:80 rows |
| `data/covariance/xip_xim_map3_covariance_8ARCMINCUT_25Feb25.fits` | Joint 480-by-480 covariance, block offsets 0, 200, 400 |

Header and array inspection confirms xi-bin edges from 2.5 to 250 arcmin and
equal Map3 radii 7, 14, 25, 40 arcmin. The observed xi source labels, angular-bin
edges and every `nz_source` column exactly match the selected simulated
reference. The source table contains twelve negative midpoint densities
in its first bin. The demonstration's recorded clipping/normalization is
therefore an explicit source-model approximation for either data selection;
the downloaded original bytes remain unchanged.

Use the separately labeled joint covariance. The two-point file's own
400-by-400 covariance retains extra gammat/wtheta header entries from its
source format; it must not be used to infer a joint Map3 covariance.
Measured xi and Map3 must be selected together with both covariance axes.

The observational companion,
[2508.14018v2, Sections IV–VI](https://arxiv.org/html/2508.14018v2#S4),
is distinct from the simulated methods paper. Its existence and the public
FITS files settle availability; they do not settle exact agreement with our
source reconstruction, triangle estimator, scale cuts or likelihood.

## Conditions still separating a forecast from a paper reproduction

The pinned run uses an NLA mean, an 8-arcmin-cut trained emulator, MOPED
compression and a finite-simulation likelihood prescription. Its included
configuration also references external CosmoSIS scale-cut and prior files.
Those external files and dependency versions are not in either twelve-file
manifest. The trained emulator and compression matrix are also outside
these selections. The methods-paper 83% posterior FoM improvement cannot be
treated as an exact benchmark for a full 480-row Gaussian TATT Fisher matrix.
[Methods paper, Sections IV–V](https://arxiv.org/html/2503.03964v2#S4)

The configured direct reference also names a pixel window at nside 1024.
The project mean uses explicit flat-sky bin transforms and a stated finite
SAS integral; it does not infer a pixel window from the input filename.
Compare pixel smoothing, full-sky versus flat-sky xi, pair weights, finite
triangle boundaries and side-bin averaging one at a time before combining
their residuals. Matching the number80 is insufficient to establish that
two aperture estimators are the same measurement.

For xi, increasing the ell-node count alone does not test the angular-bin
integral. The Bessel phase changes as ell times angle, so large-angle bins
at high ell can remain unresolved by a fixed 32-node angular rule. Reuse
the same physical power spectra and compare both individual refinements
and the joint ell/theta refinement. Assess the mean residual and the
parameter-derivative residual in covariance units; ratios near zero do
not give a stable convergence criterion.

The pinned TATT source's redshift-zero nonlinear power times growth squared
is `power_mode=1`; full nonlinear power at each redshift is `power_mode=2`.
They are different physical prescriptions. A numerical-resolution change
cannot establish agreement between them. Parameter normalization, independent
eta2 evolution and ordered source permutations also need explicit comparison.

## IA information and baryonic feedback

The inference and Fisher examples use one joint mean and retain C23. They
can ask whether additional Map3 information tightens a marginalized IA
combination rather than only changing a plotted curve. The useful diagnostic
is the joint result alongside the selected two-point result, with the same
prior information added once. Conditional three-point information separates
the effect of correlations from treating the measurements as independent.

The current bispectrum feedback amplitude `fb` has no matched power-spectrum
response in the joint wrapper. Both joint routes refuse to interpret that
amplitude as a shared two-/three-point baryon parameter. A baryonic constraint
requires an explicit physical family giving both P(k,z) and B(k1,k2,k3,z),
including their IA convention. The existing matter-transfer plot and a
specified contamination-bias calculation can precede that extension; neither
is a measurement of a feedback parameter.

## Example responsibilities

| Entry | Scientific role |
|---|---|
| `EXAMPLE_FISHER_ACCURACY.ipynb` | Numerical-refinement and Fisher interpretation through the shared prediction |
| `EXAMPLE_NAUTILUS1.yaml` | NLA sampling of a noiseless mean-generated synthetic data vector |
| `EXAMPLE_NAUTILUS2.yaml` | TATT sampling of a noiseless mean-generated synthetic data vector |
| `EXAMPLE_NAUTILUS3.yaml` | Observed NLA demonstration with explicitly unvalidated estimator/source/cut assumptions |
| `scripts/fetch_y3_observed.py` | Verify and download the pinned observed inputs; no model execution |

The [inference guide](INFERENCE.md) states how to inspect these choices and
check a changed physical parameter before sampling. Executed checks and
posterior convergence are separate records. A sampling budget is not a
claim that its posterior has converged.
