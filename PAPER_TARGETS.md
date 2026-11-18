# Paper targets and joint DES forecasts

The first useful scientific notebook should reproduce the different IA
responses of two- and three-point shear, then ask whether those differences
survive covariance weighting and nuisance marginalization. A visibly
different curve does not by itself imply a tighter parameter constraint:
the difference may lie in a noisy combination of measurements, or resemble
the effect of changing cosmology. The notebook should show those steps
separately so the reader can understand where any improvement comes from.

This document defines acceptance targets. It does not report a completed
survey forecast or a reproduction of either paper's numerical results.
Executed component and state tests belong in [the validation record](tests/VALIDATION.md).

## Four calculations with different scientific meanings

| Calculation | Question it answers | Required comparison |
|---|---|---|
| Y3 methods-paper reproduction | Does the port recover the specified NLA analysis? | Identical estimator, source distributions, cuts, covariance, compression, priors and likelihood |
| TATT-paper reproduction | Does the port recover the physical IA contributions and their angular dependence? | The same parameter values and ordered field components, before any Fisher calculation |
| Y3 joint Fisher forecast | How much additional local information does Map3 supply about cosmology and IA? | Full selected xi versus full selected xi plus Map3, with common priors and the cross-covariance retained |
| Y6 joint Fisher forecast | How does the answer change for the Y6 survey assumptions? | Explicit Y6 inputs; copied Y3 assumptions must be labeled as forecast assumptions |

Here Map3 denotes $`\langle M_{\rm ap}^3\rangle`$, the third moment of
a compensated, angularly filtered shear field. The two-point vector
contains $`\xi_+`$ and $`\xi_-`$ for every retained tomographic pair and
separation bin. “Full selected xi” means that none of these retained rows
has been replaced by a small illustrative subset. Compression is an
additional comparison, not a condition for using the joint vector.

## Primary references and pinned inputs

The publication targets are [Gomes et al., 2503.03964v2](https://arxiv.org/abs/2503.03964v2)
and [Gomes et al., 2601.09133v1](https://arxiv.org/abs/2601.09133v1).
The numerical source record is:

| Source | Pin and role |
|---|---|
| [Sunao Sugiyama's fastnc TATT branch](https://github.com/git-sunao/fastnc/tree/7b2336ce37a909bed51f07f32a2a4f096e71b6ce) | `7b2336ce37a909bed51f07f32a2a4f096e71b6ce`; tensor conventions and original TATT numerical comparison |
| [Rafael Gomes's 3pcf_integrator](https://github.com/rchgomes/3pcf_integrator/tree/d6098d326dcb03bc750adcc9627119f52d928944) | `d6098d326dcb03bc750adcc9627119f52d928944`; selected Y3 vectors, covariance and analysis configuration |
| [Rafael Gomes's Y6 pipeline](https://github.com/rchgomes/desy6-3pcf/tree/fd59fe4fee694d359453d5f573d6a4a55a47a812) | `fd59fe4fee694d359453d5f573d6a4a55a47a812`; Y6 workflow and explicitly provisional settings |

The [Y3 manifest](data/y3_reference_manifest.json) records the twelve
selected upstream files and their Git blob hashes. Downloaded copies live
under `data/upstream_y3/`. That selection includes simulated two-point and
Map3 vectors, source distributions, and the labeled joint covariance. It
does not include every external file referenced by the upstream pipeline.
In particular, a reference in an INI file is not evidence that its target
has been downloaded or that the dependency version has been pinned.

Observed unblinded Y3 inputs are separately recorded in the
[observed manifest](data/y3_observed_manifest.json) and downloaded by
[fetch_y3_observed.py](scripts/fetch_y3_observed.py). Its exact upstream
joint configuration pairs the March 10 two-point vector with the March 11
Map3 vector and the February 25 joint covariance. These are measurements,
not the synthetic vectors in the twelve-file reference selection above.
The corresponding observational paper is
[Gomes et al., 2508.14018v2](https://arxiv.org/abs/2508.14018v2).
The [reproduction plan](PAPER_REPRODUCTION_PLAN.md) distinguishes available
measurements from a verified match to their estimator and likelihood.

## The Y3 methods-paper target

The NLA reference uses 796 mock patches from 199 parent skies, an 8 arcmin
low cut on both SAS triangle side parameters, and equal aperture radii
7, 14, 25 and 40 arcmin. Vector
dimensions before cuts, after cuts, and after partial compression are
480, 307 and 96; the last combines 16 compressed xi values with 80 Map3
values. Figures 2–6 address mean, covariance and compression; Figures 7
and 10 address baryons; Figures 8–9 and Table 4 give the
$`(\Omega_m,S_8)`$ FoM change, 930.8 to 1700.8, or 83%.
[Methods paper, Sections III–V](https://arxiv.org/html/2503.03964v2#S3)

Figure 12 reports only about 3% tightening of the NLA alignment-amplitude
posterior. A new TATT Fisher improvement must therefore be measured under
its own declared nuisance model; it is not an already established result
of that NLA analysis.
[Methods paper, Section V](https://arxiv.org/html/2503.03964v2#S5)

Table 3 supplies these Gaussian calibration priors; each ordered tuple
refers to source bins 1–4. $`\sigma`$ is the standard deviation.
[Methods paper, Table 3](https://arxiv.org/html/2503.03964v2#S4.T3)

| Quantity | Mean | $`\sigma`$ |
|---|---|---|
| Photo-z shift $`\Delta z_i`$ | (0,0,0,0) | (0.018,0.015,0.011,0.017) |
| Shear calibration $`m_i`$ | (−0.0063,−0.0198,−0.0241,−0.0369) | (0.0091,0.0078,0.0076,0.0076) |

The published baryon exercise injects OWLS-like power suppression and a
rescaled TNG bispectrum response to validate scale cuts. It does not fit
a feedback parameter. Its likelihood includes finite-simulation
uncertainty through the Percival prescription.
[Methods paper, Sections IV.1 and IV.3](https://arxiv.org/html/2503.03964v2#S4)

The selected [sampling-values file](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/test-cosmogridc-simdv-joint-2pt-moped-run-796sims-february-Percival/des-y3-shear-values.ini)
gives the following uniform ranges and starting values. The middle entry
in an upstream range is a sampler start; it is not automatically the
cosmology or nuisance value used to generate the synthetic data.

| Parameter | Range | Start |
|---|---|---|
| $`\Omega_m`$ | 0.1–0.5 | 0.290 |
| $`S_8`$ | 0.7–0.84 | 0.759 |
| $`h`$ | 0.55–0.91 | 0.69 |
| $`\Omega_b`$ | 0.03–0.07 | 0.048 |
| $`n_s`$ | 0.87–1.07 | 0.97 |
| $`\sum m_\nu`$ in eV | 0.06–0.6 | 0.0773 |
| $`A_1,\eta_1`$ | each −5–5 | 0.7, −1.7 |

This file fixes $`w=-1`$ and the IA pivot at 0.62. Its finite bounds on
$`m_i`$ and $`\Delta z_i`$ do not replace the Gaussian priors loaded
separately. The [selected run](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/test-cosmogridc-simdv-joint-2pt-moped-run-796sims-february-Percival/des-y3-shear-2pt-map3-NLA-all.ini)
references the standard-library `examples/des-y3-priors.ini`; its included
[two-point configuration](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/cosmic-shear.ini)
references `examples/des-y3-scale-cuts.ini`. Neither external file is in
the selected manifest. The Gaussian numbers above can be recovered from
the publication, while the exact pair-by-pair cut file and dependency
version still need to be recorded.

A notebook should first print all row labels, retained indices and counts,
and apply the same index selection to both axes of the joint covariance.
Matching a count alone is insufficient: two selections can have the same
length while retaining different physical scales. The selected covariance
must accompany the finite-triangle mean estimator that generated its row
definitions. The analytic full-aperture covariance is a different
calculation; assigning its matrix to the same 80 labels does not establish
estimator equivalence.

There are three additional requirements before an exact posterior
comparison. The selected [MOPED configuration](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/moped.ini)
references a compression matrix absent from this twelve-file selection.
The selected run uses a trained emulator with an 8 arcmin cut, also absent
from the selection. Finally, the upstream xi prediction is full-sky and
bin-averaged with `ell_max=40000`, whereas the present `XiEngine` uses
flat-sky $`J_0/J_4`$ transforms. Obtain the external products or rebuild
their specified operations, and measure the flat-sky difference with the
same spectra and bin weights. A Fisher forecast can proceed with declared
approximations, but it should not use the 83% posterior gain as an exact
pass/fail number.

## The TATT-paper target

The selected numerical target retains the all-E projected bispectrum.
Its low-level IA engine also exposes other components, but inserting them
into a scalar E-mode transform would not produce their correct spin phases.
The present joint forecast therefore combines two-point TATT E/B spectra
with this stated three-point E-mode model. A finite-triangle estimator
also needs an explicit E/B leakage check after its cuts and bin averages.

Table 1 sets
$`(\Omega_m,S_8,h,n_s,\Omega_b)=(0.289,0.772,0.722,0.959,0.0463)`$
and $`(A_1,\eta_1,A_2,\eta_2,b_{\rm TA})=(-0.33,2.81,0.65,1.69,0.91)`$.
Figures 1–2 show $`\operatorname{Re}\Gamma_0`$ at a 60-degree opening
angle. Figures 3–4 vary TATT/velocity shear at 14 arcmin for bins
(1,1,1) and (1,3,4); Figures 5–6 separate operators and field
placements; Figures 7–8 compare total responses. Figure 9 contrasts
$`(A_1,A_2)=(-0.2,0.4)`$ with (0.2,−0.4).
[TATT paper, Table 1 and Figures 1–9](https://arxiv.org/html/2601.09133v1#S3)

The paper describes a nonlinear power replacement in Section II.3 and
BiHalofit restoration of the linear-alignment bispectrum contribution.
Its illustrative velocity-shear choice is $`C_t=2.5C_1`$ for Figures 5
and 7. Section III.2 leaves quantitative self-calibration to a likelihood
analysis; it does not supply an IA Fisher improvement to reproduce.
[TATT paper, Sections II.3 and III.2](https://arxiv.org/html/2601.09133v1#S2.SS3)

The first proposed notebook comparisons should be numerical tables beneath
those plots: absolute values, signs, each component's contribution to the
sum, and discrepancies under increased numerical resolution. A ratio near
a zero of the denominator is a poor accuracy measure. Keep those points
and show the absolute residual alongside the ratio. Ordered mixed-bin
contributions require particular care because exchanging a side without
its source label changes the observable. Separating the contributions
before summation can reveal an error hidden by cancellation in the total.

Two source-versus-paper distinctions must be visible in the configuration.
The pinned [IA power reader](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L106)
multiplies a supplied nonlinear spectrum by $`D(z)^2`$; the pinned
[integrator](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/fastnc_interface.py#L159)
supplies its redshift-zero row. In this port that convention is
`power_mode=1`. Full nonlinear redshift evolution is `power_mode=2`;
`power_mode=0` uses the linear spectrum. Record which result is an exact
pinned-source comparison and which tests the physical redshift evolution.
A numerical boost cannot turn one of these models into another.

The [pinned CAMB configuration](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/linear-power.ini)
selects Takahashi Halofit. Our CAMB producer exposes that choice explicitly
and records its version, sampled grids and growth definition. Match the
power arrays as well as the cosmological parameter labels before attributing
a discrepancy to the three-point transform. With massive neutrinos, a
growth ratio inferred from the total-matter variance is a specified
effective convention; its equality to a different upstream growth table
must be checked, not assumed.

The same integrator's TATT parameter block assigns `alphaIA_2` from `a2`.
The new interface deliberately has an independent `eta2`; copying that
assignment would change the physical redshift law. Likewise, the pinned
kernel interprets its `Ct` input as the ratio $`C_t/C_1`$. Use
`ct_mode=0` for that comparison, with `ct=2.5` for the stated example;
`ct_mode=1` describes a separately supplied physical coefficient.
Document the chosen pivot, $`\bar C_1\rho_{\rm crit}`$ normalization,
growth function and neutrino prescription rather than inferring them from
the parameter names.

The ordinary-core two-point adapter translates the IA normalization and
pivot, and provides TATT E/B power for the common cosmology and source
tables. It currently rejects nonzero `ct`: ordinary TATT power does not
supply the velocity-shear extension. A Map3 velocity-shear plot is supported;
a joint TATT+VS forecast requires the corresponding two-point model.
Keeping two-point VS identically zero would define a different restricted
forecast and must not be described as a shared physical TATT+VS model.

## Turning the response comparison into a Fisher question

For a fixed covariance $`C`$, let $`D_{ia}=\partial\mu_i/\partial p_a`$
be the derivative of observable $`i`$ with respect to parameter $`a`$.
The local forecast is

$$
F=D^{\mathsf T}C^{-1}D+F_{\rm prior}.
$$

Use the same physical parameters, calibration priors and xi selection in
the xi-only and joint calculations. Marginalized uncertainties come from
$`F^{-1}`$, not from $`1/\sqrt{F_{aa}}`$, which holds other parameters
fixed. A uniform prior has no local curvature in its interior; replacing
its width by a Gaussian precision would add information absent from that
forecast definition. Near prior boundaries, a local Fisher ellipse does
not replace posterior sampling.

A useful diagnostic is the new information left after conditioning on xi.
Write the joint covariance in blocks $`C_{22},C_{23},C_{32},C_{33}`$.
For parameter-independent covariance, define

$$
C_{3|2}=C_{33}-C_{32}C_{22}^{-1}C_{23},\qquad
D_{3|2}=D_3-C_{32}C_{22}^{-1}D_2.
$$

Then the joint likelihood Fisher matrix is the sum of the xi information
and $`D_{3|2}^{\mathsf T}C_{3|2}^{-1}D_{3|2}`$. This directly tests
whether the extra IA response supplies a measurable direction after its
correlation with xi is accounted for. Show the marginalized errors and
parameter correlations, then repeat at several fiducial IA choices and
with halved/doubled derivative steps. Any claimed improvement should
survive numerical refinement and the stated prior choices.

The IA notebook also fixes w0=-1 and wa=0 at the saved fiducial while
retaining every other varied cosmological, IA, photo-z and shear-calibration
parameter. It selects the retained Fisher rows and columns before inversion;
selecting the covariance after inversion would still marginalize dark energy.
The saved Gaussian priors enter once. It reports sigma(S8), the
(Omega_m,S8) FoM ratio and IA errors for each experiment. This changes one
assumption toward the published LambdaCDM analysis; it does not reproduce
its scale cuts, likelihood or posterior gain.

The primary Y3 forecast should vary all five TATT parameters alongside the
chosen cosmological and calibration parameters. Fixing the redshift slopes
or $`b_{\rm TA}`$ can be a separate diagnostic, clearly identified as
such. Use the [Fisher guide](FISHER.md) for exact project parameter keys.
The supplied examples select `signed_baseline=1`, interpolating a positive
gravity baseline and a signed NLA/gravity ratio. If another prepared
logarithmic projection rejects a negative baseline, the
notebook must report the affected parameter range and use an appropriate
supported evaluation path; it must not silently omit derivative points.
Refine the original power/source/redshift sampling separately from the
fine interpolation tables. Fine-grid convergence only establishes the
accuracy of reading the prepared interpolant.

For an analytic-covariance forecast, disclose that nonlinear IA is present
in the mean while the implemented connected covariance is pure lensing.
Its halo-abundance response is not a complete SSC model. The
massive-neutrino option uses a named single-fluid effective-growth
approximation, not the ordinary core's separate cold-matter halo variance.
These choices can be tested against supplied joint covariances without
calling a finite-triangle covariance a full-filter reference. A live
upstream analytic-covariance agreement remains a separate acceptance item:
independent published-equation tests and saved fixtures do not constitute
a comparison to an unexecuted external backend.

## What a baryon-constraint notebook would require

The present bispectrum parameter has the explicit meaning

$$
B(f_b)=B_{\rm gravity}\,[1+f_b(R_B-1)],
$$

where $`R_B`$ is the implemented TNG-calibrated bispectrum transfer.
This follows the actual [C provider](../../external_modules/code/cosmolike_core/cosmolike/3pcf/physics_3pcf.c)
and its [documented conventions](../../external_modules/code/cosmolike_core/cosmolike/3pcf/README.md).
A constraint on $`f_b`$ is therefore a constraint on this specified
response amplitude. It is not automatically a constraint on an AGN heating
temperature, a gas-ejection radius, or an arbitrary feedback model.

A physical joint feedback forecast needs a common family
$`R_P(k,z;f_b)`$ and $`R_B(k_1,k_2,k_3,z;f_b)`$, including how it affects
the IA power and bispectrum terms. The current ordinary-core adapter
rejects nonzero `fb` because this relation is not supplied. Setting `fb`
only in Map3 and hiding it from xi would manufacture a zero two-point
derivative. The resulting apparent gain would answer a restricted
template-amplitude question, not the requested comparison of feedback
constraints from two and three points.

The proposed sequence is to reproduce the injected-contamination curves,
declare a common power/bispectrum feedback family, then calculate the
joint derivatives and their degeneracies with IA. Repeat with more than
one physically motivated family and with different retained scales. The
small-scale relation between powers of a single halo profile does not
uniquely specify the full projected response over all triangle shapes.
Until this family is provided, the joint TATT forecast is the supported
first scientific target; the baryon notebook should expose the missing
physical input explicitly.

## Y6 inputs and final acceptance record

The pinned [Y6 data guide](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/data/README.md)
states that measurement and emulator files are external. Its
[prior file](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/config/priors/des-y6-priors.ini)
explicitly labels its calibration widths as placeholders. Its
[Map3 configuration](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/config/common/map3.ini)
uses proposed radii 3, 5, 10, 20 and 40 arcmin, no current Map3 scale cut,
and `baryon_model=none`. These files specify a workflow, not an approved
Y6 likelihood or a feedback model.

The [Y6 backend](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/desy6_3pcf/fastnc_backend.py#L166)
also offers Takahashi and BACCO matter-bispectrum options, but explicitly
rejects combining them with its TATT bispectrum mode. Their presence is
not evidence that a joint IA-plus-feedback inference has already been
defined. They are concrete candidates to inspect when choosing the common
feedback family described above.

Measured sky data are unnecessary for a forecast centered on a theoretical
mean, but source distributions, noise levels, area or mask, selection,
and calibration assumptions are still required. Enter them as identified
Y6 inputs or explicitly chosen forecasts. Save the resolved configuration,
source-file hashes, row labels, covariance prescription, physical
conventions, derivative steps and refinement comparisons with each result.
The final acceptance record should separate component accuracy, estimator
agreement, figure reproduction and parameter inference. Passing an earlier
step supplies evidence for the next; it does not replace it.
