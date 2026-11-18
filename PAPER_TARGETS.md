# Paper targets and joint DES forecasts

1. [Four calculations with different scientific meanings](#calculations)
2. [Primary references and pinned inputs](#pins)
3. [The Y3 methods-paper target](#y3)
4. [Public observed measurements](#observed)
5. [The TATT-paper target](#tatt)
6. [Figure targets](#figures)
7. [Fisher comparisons built on these targets](#fisher)
8. [Baryonic feedback targets](#baryons)
9. [Y6 inputs](#y6)
10. [Remaining reproduction gaps](#gaps)
11. [Example responsibilities](#examples)

This page records the publication targets of the joint xi±/Map3 port:
the pinned sources, the public measurements, the figure targets and the
gaps that still separate a forecast from a reproduction. It states
acceptance conditions; it does not report completed figure reproductions
or survey forecasts. Executed component and state tests belong in
[the validation record](tests/VALIDATION.md).

## Four calculations with different scientific meanings <a name="calculations"></a>

| Calculation | Question it answers | Required comparison |
|---|---|---|
| Y3 methods-paper reproduction | Does the port recover the specified NLA analysis? | Identical estimator, source distributions, cuts, covariance, compression, priors and likelihood |
| TATT-paper reproduction | Does the port recover the physical IA contributions and their angular dependence? | The same parameter values and ordered field components, before any Fisher calculation |
| Y3 joint Fisher forecast | How much additional local information does Map3 supply about cosmology and IA? | Full xi versus full xi plus Map3, with common priors and the cross-covariance kept |
| Y6 joint Fisher forecast | How does the answer change for the Y6 survey assumptions? | Explicit Y6 inputs; copied Y3 assumptions must be labeled as forecast assumptions |

Here Map3 denotes $`\langle M_{\rm ap}^3\rangle`$, the third moment of
a compensated, angularly filtered shear field. The two-point vector
contains $`\xi_+`$ and $`\xi_-`$ for every kept tomographic pair and
separation bin. "Full xi" means that no kept row has been replaced by a
small illustrative subset. Compression is an additional comparison, not
a condition for using the joint vector.

## Primary references and pinned inputs <a name="pins"></a>

The publication targets are [Gomes et al., 2503.03964v2](https://arxiv.org/abs/2503.03964v2),
the methods paper, and [Gomes et al., 2601.09133v1](https://arxiv.org/abs/2601.09133v1),
the TATT paper. The observational companion is
[Gomes et al., 2508.14018v2](https://arxiv.org/abs/2508.14018v2).
The numerical source record is:

| Source | Pin and role |
|---|---|
| [Sunao Sugiyama's fastnc TATT branch](https://github.com/git-sunao/fastnc/tree/7b2336ce37a909bed51f07f32a2a4f096e71b6ce) | `7b2336ce37a909bed51f07f32a2a4f096e71b6ce`; tensor conventions and original TATT numerical comparison |
| [Rafael Gomes's 3pcf_integrator](https://github.com/rchgomes/3pcf_integrator/tree/d6098d326dcb03bc750adcc9627119f52d928944) | `d6098d326dcb03bc750adcc9627119f52d928944`; Y3 vectors, covariance and analysis configuration |
| [Rafael Gomes's Y6 pipeline](https://github.com/rchgomes/desy6-3pcf/tree/fd59fe4fee694d359453d5f573d6a4a55a47a812) | `fd59fe4fee694d359453d5f573d6a4a55a47a812`; Y6 workflow and explicitly provisional settings |

The [Y3 manifest](data/y3_reference_manifest.json) records the twelve
chosen upstream files and their Git blob hashes. Downloaded copies live
under `data/upstream_y3/`. That selection includes simulated two-point and
Map3 vectors, source distributions, and the labeled joint covariance. It
does not include every external file referenced by the upstream pipeline:
a reference in an INI file is not evidence that its target has been
downloaded or that the dependency version has been pinned.

The [observed manifest](data/y3_observed_manifest.json) separately records
the observed unblinded Y3 inputs, and
[fetch_y3_observed.py](scripts/fetch_y3_observed.py) downloads them.
These are measurements, not the synthetic vectors in the twelve-file
reference selection. The [observed section](#observed) lists the files
and their verified roles.

## The Y3 methods-paper target <a name="y3"></a>

The NLA reference uses 796 mock patches from 199 parent skies, an 8 arcmin
low cut on both SAS triangle side parameters, and equal aperture radii
7, 14, 25 and 40 arcmin. Vector dimensions before cuts, after cuts, and
after partial compression are 480, 307 and 96; the last combines 16
compressed xi values with 80 Map3 values. Figures 2–6 address mean,
covariance and compression; Figures 7 and 10 address baryons; Figures 8–9
and Table 4 give the $`(\Omega_m,S_8)`$ FoM change, 930.8 to 1700.8,
or 83%.
[Methods paper, Sections III–V](https://arxiv.org/html/2503.03964v2#S3)

Figure 12 reports only about 3% tightening of the NLA alignment-amplitude
posterior. A new TATT Fisher improvement must therefore be measured under
its own stated nuisance model; it is not an already established result
of that NLA analysis.
[Methods paper, Section V](https://arxiv.org/html/2503.03964v2#S5)

Table 3 supplies these Gaussian calibration priors; each ordered tuple
refers to source bins 1–4. $`\sigma`$ is the standard deviation.
[Methods paper, Table 3](https://arxiv.org/html/2503.03964v2#S4.T3)

| Quantity | Mean | $`\sigma`$ |
|---|---|---|
| Photo-z shift $`\Delta z_i`$ | (0,0,0,0) | (0.018,0.015,0.011,0.017) |
| Shear calibration $`m_i`$ | (−0.0063,−0.0198,−0.0241,−0.0369) | (0.0091,0.0078,0.0076,0.0076) |

The pinned [sampling-values file](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/test-cosmogridc-simdv-joint-2pt-moped-run-796sims-february-Percival/des-y3-shear-values.ini)
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
separately. The [pinned run](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/test-cosmogridc-simdv-joint-2pt-moped-run-796sims-february-Percival/des-y3-shear-2pt-map3-NLA-all.ini)
references the standard-library `examples/des-y3-priors.ini`; its included
[two-point configuration](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/cosmic-shear.ini)
references `examples/des-y3-scale-cuts.ini`. Neither external file is in
the manifest; the [gaps table](#gaps) tracks them.

A comparison first prints all row labels, kept indices and counts, and
applies the same index selection to both axes of the joint covariance.
Matching a count alone is insufficient: two selections can have the same
length while keeping different physical scales. The chosen covariance
must accompany the finite-triangle mean estimator that generated its row
definitions. The analytic full-aperture covariance is a different
calculation; assigning its matrix to the same 80 labels does not establish
estimator equivalence.

## Public observed measurements <a name="observed"></a>

The observed manifest records commit
`d6098d326dcb03bc750adcc9627119f52d928944` of
[3pcf_integrator](https://github.com/rchgomes/3pcf_integrator/tree/d6098d326dcb03bc750adcc9627119f52d928944),
and the fetch script checks every downloaded file against its Git blob
hash and byte count. The upstream
[joint unblinded configuration](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/unblind-metacal-796sims-february-Percival-2ptfix-3pcfsc/unblind-metacal-shear-2pt-map3-NLA-all.ini)
explicitly associates these three files:

| Input under `data/upstream_y3_observed/` | Verified role |
|---|---|
| `data/dv/real_2pt_desy3_unblind_g2flip_matchElisa_2025Mar10.fits` | Observed xi+:200 rows and xi−:200 rows, with source distributions |
| `data/dv/real_map3_desy3_unblind_scalecut3pcf_796simcov_2025Mar11.fits` | Observed Map3:80 rows |
| `data/covariance/xip_xim_map3_covariance_8ARCMINCUT_25Feb25.fits` | Joint 480-by-480 covariance, block offsets 0, 200, 400 |

Header and array inspection confirms xi-bin edges from 2.5 to 250 arcmin
and equal Map3 radii 7, 14, 25, 40 arcmin. The observed xi source labels,
angular-bin edges and every `nz_source` column exactly match the simulated
reference. The source table contains twelve negative midpoint densities
in its first bin; the downloaded original bytes stay unchanged, and the
demonstration's recorded clipping, normalization and PCHIP reconstruction
are an explicit source-model approximation for either data selection.
The [Fisher guide](FISHER.md#parameters) records that construction and
the [source study](tests/studies/source_policy_validation.md) explains
why the last source feature matters for derivatives.

Use the separately labeled joint covariance. The two-point file's own
400-by-400 covariance keeps extra gammat/wtheta header entries from its
source format; it must not be used to infer a joint Map3 covariance.
Select measured xi and Map3 together, and apply the selection to both
covariance axes.

The upstream packing scripts that we inspected do not recover the
realization-level Map3 sign processing that upstream used when estimating
the cross-covariance block $`C_{23}`$ between xi and Map3. The project
keeps the covariance unchanged: this is unresolved provenance, not
evidence of a wrong sign. Section 6 of the
[estimator record](tests/studies/estimator_conventions.md) documents the
inspected scripts. The other guides link here instead of restating this
caveat.

The observational companion, [2508.14018v2, Sections IV–VI](https://arxiv.org/html/2508.14018v2#S4),
is distinct from the simulated methods paper. Its existence and the public
FITS files settle availability; they do not settle exact agreement with our
source reconstruction, triangle estimator, scale cuts or likelihood.

## The TATT-paper target <a name="tatt"></a>

The numerical target keeps the all-E projected bispectrum.
Its low-level IA engine also exposes other components, but inserting them
into a scalar E-mode transform would not produce their correct spin phases.
The present joint forecast therefore combines two-point TATT E/B spectra
with this stated three-point E-mode model. A finite-triangle estimator
also needs an explicit E/B leakage check after its cuts and bin averages.
The [finite-window E/B study](tests/studies/finite_window_eb.md) explains
why single-B bispectra are the leading-order question and specifies an
ideal-aperture null test followed by the identical finite-SAS contraction.
No nonzero leakage correction has been measured for the present weights.

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

Two source-versus-paper distinctions must stay visible in the
configuration. The pinned [IA power reader](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/ia_bispectra.py#L106)
multiplies an input nonlinear spectrum by $`D(z)^2`$, where $`D(z)`$ is
the linear growth factor; the pinned
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
growth ratio inferred from the total-matter variance is a specific
effective convention; its equality to a different upstream growth table
must be checked, not assumed.

The same integrator's TATT parameter block assigns `alphaIA_2` from `a2`.
This port has an independent `eta2`; copying that assignment would change
the physical redshift law. Likewise, the pinned kernel interprets its
`Ct` input as the ratio $`C_t/C_1`$. Use `ct_mode=0` for that comparison,
with `ct=2.5` for the stated example; `ct_mode=1` describes a separately
specified physical coefficient. Document the chosen pivot,
$`\bar C_1\rho_{\rm crit}`$ normalization, growth function and neutrino
prescription rather than inferring them from the parameter names.

The ordinary-core two-point adapter translates the IA normalization and
pivot, and provides TATT E/B power for the common cosmology and source
tables. It rejects nonzero `ct`: ordinary TATT power does not supply the
velocity-shear extension. A Map3 velocity-shear plot is supported; a
joint TATT+VS forecast requires the corresponding two-point model.
Keeping two-point VS identically zero would define a different restricted
forecast and must not be described as a shared physical TATT+VS model.

## Figure targets <a name="figures"></a>

| Publication target | Calculation with this project | Acceptance condition |
|---|---|---|
| 2601.09133, Figure 9 response mechanism | Compare both NLA and TATT sign choices in xi+ and Map3 | One no-IA reference; stated power, source and estimator choices; numerical refinement |
| 2601.09133, Figures 3–4 | Scan IA amplitudes for the two illustrated source triplets at 14 arcmin | Figure 3 uses eta1=eta2=1; distinguish Ct from Ct/C1 |
| 2601.09133, Figures 1–2 | Compare the real natural component Gamma0 at a 60-degree opening angle | Ordered source legs and side labels; paper Lmax=50, Mmax=100 followed by refinement |
| 2601.09133, Figures 5–6 | Separate operator contributions and numbers of intrinsic-shape fields | Verify component sum closure; projected field-placement diagnostics require an additional explicit calculation |
| 2503.03964, Figures 2, 4–5 | Compare the downloaded simulated mean and display the covariance blocks | Preserve labels; plotting an input covariance is not an independent covariance estimate |
| 2503.03964, Figure 7 | Evaluate the specified baryonic matter-bispectrum transfer | Match redshift, triangle shape, amplitude definition and suppression-only choice |
| 2503.03964, Figure 10 | Inject a matched power/bispectrum contaminant and examine inference bias | Supply both responses and the exact estimator selection before interpreting robustness |

These figure specifications come from the
[TATT paper, Sections II.6–III.2](https://arxiv.org/html/2601.09133v1#S3)
and the [methods paper, Sections III–V](https://arxiv.org/html/2503.03964v2#S3).
The target lists are specifications; they do not record completed figure
reproductions. The notebooks and saved numerical checks must provide that
evidence separately.

For the Figure 9 target, use `(A1,A2)=(-0.2,0.4)` and `(0.2,-0.4)` and
keep the other Table 1 parameters. The current joint implementation can
evaluate these E/B two-point and E-mode three-point responses. The TATT
paper motivates self-calibration but supplies no joint Fisher gain to use
as an acceptance number; a Fisher result from these examples is an
additional calculation.
[TATT paper, Table 1 and Figure 9](https://arxiv.org/html/2601.09133v1#S3)

For that comparison the Fisher driver prepares five means: a common no-IA
reference, two NLA sign choices and two TATT sign choices. The IA notebook
shows each model against that reference and evaluates the finite sign-pair
distance with all 480 rows. It subtracts the Map3 mean difference
predicted from the xi mean difference through $`C_{32}C_{22}^{-1}`$, then
weights the remaining difference by the conditional Map3 covariance. It
holds every other parameter fixed; it is not a marginalized significance.
The figure uses a maximum-reference normalization within each bin, while
the paper's pointwise denominator is a separate plotting convention.

[EXAMPLE_IA_COMPONENTS.ipynb](EXAMPLE_IA_COMPONENTS.ipynb) demonstrates
the projected operator/placement API. It carries the physical
density-weighting, torque and velocity-shear vertices, and all ordered
GGI/GII/III placements, through the same finite aperture calculation,
and checks their sum against the ordinary total. The torque split keeps
the $`-2C_2/3`$ part of the tensor density coefficient. Matching the
paper's Figures 5–6 still requires matching its estimator and numerical
inputs. The active velocity-shear vertex differs from the literal printed
equations 62 and 67 of the TATT paper; the notebook records that
unresolved issue. A three-point velocity-shear diagnostic also does not
supply a joint velocity-shear two-point model.

## Fisher comparisons built on these targets <a name="fisher"></a>

The [Fisher guide](FISHER.md#information) defines the local forecast, the
marginalization convention and the conditional Map3 information used to
compare xi-only and joint experiments. Use the same physical parameters,
calibration priors and xi selection in both experiments, and repeat at
several fiducial IA choices and with halved and doubled derivative steps.
A claimed improvement should survive numerical refinement and the stated
prior choices. Near prior boundaries, a local Fisher ellipse does not
replace posterior sampling.

The IA notebook also fixes $`w_0=-1`$ and $`w_a=0`$ at the saved fiducial
while keeping every other varied cosmological, IA, photo-z and
shear-calibration parameter. It selects the kept Fisher rows and columns
before inversion; selecting the covariance after inversion would still
marginalize dark energy. The saved Gaussian priors enter once. It reports
$`\sigma(S_8)`$, the $`(\Omega_m,S_8)`$ FoM ratio and IA errors for each
experiment. This changes one assumption toward the published LambdaCDM
analysis; it does not reproduce its scale cuts, likelihood or posterior
gain.

The primary Y3 forecast varies all five TATT parameters alongside the
chosen cosmological and calibration parameters. Fixing the redshift
slopes or $`b_{\rm TA}`$ can be a separate diagnostic, clearly identified
as such. Use the [Fisher guide](FISHER.md) for exact project parameter
keys. The examples select `signed_baseline=1`, interpolating a positive
gravity baseline and a signed NLA/gravity ratio; if another prepared
logarithmic projection rejects a negative baseline, report the affected
parameter range and use a supported evaluation path rather than silently
omitting derivative points. Refine the original power/source/redshift
sampling separately from the fine interpolation tables, because fine-grid
convergence only establishes the accuracy of reading the prepared
interpolant.

For an analytic-covariance forecast, disclose that nonlinear IA is present
in the mean while the implemented connected covariance is pure lensing.
Its halo-abundance response is not a complete SSC model, and the
massive-neutrino option uses a single-fluid effective-growth
approximation rather than the ordinary core's separate cold-matter halo
variance. These choices can be tested against external joint covariances
without calling a finite-triangle covariance a full-filter reference. A
live upstream analytic-covariance agreement remains a separate acceptance
item: independent published-equation tests and saved fixtures do not
constitute a comparison to an unexecuted external backend.

## Baryonic feedback targets <a name="baryons"></a>

The present bispectrum parameter has the explicit meaning

```math
B(f_b)=B_{\rm gravity}\,[1+f_b(R_B-1)],
```

where $`R_B`$ is the implemented TNG-calibrated bispectrum transfer.
This follows the C provider and its documented conventions in
`external_modules/code/cosmolike_core/cosmolike/3pcf/physics_3pcf.c` and
`external_modules/code/cosmolike_core/cosmolike/3pcf/README.md`.
A constraint on $`f_b`$ is therefore a constraint on this specified
response amplitude. It is not automatically a constraint on an AGN heating
temperature, a gas-ejection radius, or an arbitrary feedback model.

The published baryon exercise injects OWLS-like power suppression and a
rescaled TNG bispectrum response to validate scale cuts. It does not fit
a feedback parameter. Its likelihood includes finite-simulation
uncertainty through the Percival prescription.
[Methods paper, Sections IV.1 and IV.3](https://arxiv.org/html/2503.03964v2#S4)

A physical joint feedback forecast needs a common family
$`R_P(k,z;f_b)`$ and $`R_B(k_1,k_2,k_3,z;f_b)`$, including how it affects
the IA power and bispectrum terms. The current ordinary-core adapter
rejects nonzero `fb` because this relation is not part of the model.
Setting `fb` only in Map3 and hiding it from xi would manufacture a zero
two-point derivative, answering a restricted template-amplitude question
instead of comparing feedback constraints from two and three points.

The [paired-response guide](FEEDBACK.md) supplies a shared amplitude for
independently specified matter-power and matter-bispectrum responses,
with the quadratic IA terms and the covariance kept at DMO. The
[BACCO guide](BACCO.md) connects the installed Burger2025 emulators to
the joint mean with five physical feedback coordinates. The older
bispectrum-only `fb` remains unsuitable as a joint feedback parameter.

## Y6 inputs <a name="y6"></a>

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
feedback family described in the [baryon section](#baryons).

Measured sky data are unnecessary for a forecast centered on a theoretical
mean, but source distributions, noise levels, area or mask, selection,
and calibration assumptions are still required. Enter them as identified
Y6 inputs or explicitly chosen forecasts. Save the resolved configuration,
source-file hashes, row labels, covariance prescription, physical
conventions, derivative steps and refinement comparisons with each result.
The final acceptance record separates component accuracy, estimator
agreement, figure reproduction and parameter inference; passing an earlier
step supplies evidence for the next, not a replacement for it.

## Remaining reproduction gaps <a name="gaps"></a>

The pinned Y3 run uses an NLA mean, an emulator trained with the 8 arcmin
cut, MOPED compression and the Percival finite-simulation likelihood.
The following items still separate this project's forecasts from an exact
reproduction of that run:

| Item | What is missing or different | Where it is recorded |
|---|---|---|
| Compression matrix | The pinned [MOPED configuration](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/moped.ini) references a matrix absent from the twelve-file manifest | [Compression guide](COMPRESSION.md) |
| Trained emulator | The pinned run uses an emulator with the 8 arcmin cut; its files are absent from the manifest | [Pinned run](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/test-cosmogridc-simdv-joint-2pt-moped-run-796sims-february-Percival/des-y3-shear-2pt-map3-NLA-all.ini) |
| Scale-cut file | `examples/des-y3-scale-cuts.ini` is referenced but not in the manifest | [Y3 target](#y3) |
| Priors file | `examples/des-y3-priors.ini` is referenced but not in the manifest; the Gaussian widths above come from the publication | [Y3 target](#y3) |
| Dependency versions | Upstream CosmoSIS dependency versions are unrecorded | [Pins](#pins) |
| xi prediction geometry | Upstream xi is full-sky and bin-averaged with `ell_max=40000`; the present `XiEngine` uses flat-sky $`J_0/J_4`$ transforms | [Two-point configuration](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/cosmic-shear.ini) |
| Pixel window | The pinned direct reference names a window at nside 1024; this project applies a window only through its explicit option | [Estimator guide](ESTIMATOR.md#pixels) |
| Finite-triangle estimator | The packed FITS products lack the original within-bin triangle weights | [Estimator guide](ESTIMATOR.md) |
| Source reconstruction | The demonstration clipping and PCHIP construction is not the publication's calibrated treatment | [Fisher guide](FISHER.md#parameters) |
| Map3 sign processing | The realization-level processing behind the covariance cross block is not recovered | [Observed section](#observed) |
| Posterior FoM benchmark | The 83% posterior FoM improvement belongs to the MOPED-plus-Percival analysis; it is not an exact pass/fail number for a 480-row Gaussian TATT Fisher matrix | [Methods paper, Sections IV–V](https://arxiv.org/html/2503.03964v2#S4) |

Obtain the external products or rebuild their specified operations, and
measure the flat-sky difference with the same spectra and bin weights.
Compare pixel smoothing, full-sky versus flat-sky xi, pair weights,
finite triangle boundaries and side-bin averaging one at a time before
combining their residuals. Matching the count of 80 aperture rows is
insufficient to establish that two aperture estimators are the same
measurement.

For xi, increasing the ell-node count alone does not test the angular-bin
integral. The Bessel phase changes as multipole times angle, so
large-angle bins at high multipole can remain unresolved by a fixed
32-node angular rule. Reuse the same physical power spectra and compare
both individual refinements and the joint ell/theta refinement. Assess
the mean residual and the parameter-derivative residual in covariance
units; ratios near zero do not give a stable convergence criterion.

## Example responsibilities <a name="examples"></a>

| Entry | Scientific role |
|---|---|
| `EXAMPLE_FISHER_IA.ipynb` | IA sign responses, finite model distances and two-point/joint local information with varied or fixed dark energy |
| `EXAMPLE_FISHER_ACCURACY.ipynb` | Numerical-refinement and Fisher interpretation through the shared prediction |
| `EXAMPLE_NAUTILUS1.yaml` | NLA sampling of a noiseless mean-generated synthetic data vector |
| `EXAMPLE_NAUTILUS2.yaml` | TATT sampling of a noiseless mean-generated synthetic data vector |
| `EXAMPLE_NAUTILUS3.yaml` | Observed NLA demonstration with explicitly unvalidated estimator/source/cut assumptions |
| `scripts/fetch_y3_observed.py` | Verify and download the pinned observed inputs; no model execution |

The [inference guide](INFERENCE.md) states how to inspect these choices and
check a changed physical parameter before sampling. Executed checks and
posterior convergence are separate records. A sampling budget is not a
claim that its posterior has converged.
