# Recorded implementation and numerical comparisons

These sections retain the measurements and assumptions behind the project
examples. They describe the versions that produced the recorded results;
commands and file locations in a measurement history are not a replacement
for the current [project instructions](../../README.md). No study was rerun
as part of the documentation reorganization. Saved numerical arrays remain
in [results/](results).

## Accuracy measurements <a name="accuracy"></a>

## Executed numerical studies <a name="accuracy-studies"></a>

| Study | What it measures |
|---|---|
| [Interpolation validation](interpolation_validation.md) | Projection-table comparisons against direct projected values and the full 80-row aperture-mean refinement |
| [Source and photo-z policy](source_policy_validation.md) | Source-model choices separated from source, kernel and radial resolution |
| [Fisher accuracy record](fisher_accuracy_validation.md) | Full NLA/TATT derivative bases, individually refined derivative columns, and coherent changes that the fourth-difference noise estimate misses |
| [Finite-window E/B study](finite_window_eb.md) | Why refinement of the E-only mean cannot measure leakage from omitted single-B bispectra after triangle cuts |
| [Fiducial angular-order study](angular_order_results.md) | Legendre and Fourier truncation orders separated from angular sampling and coupling interpolation; ten NLA/TATT fiducial mean comparisons and their distinction from derivative convergence |
| [BACCO response study](bacco_validation.md) | Direct emulator values, native interpolation, response-grid refinement and finite-difference steps, separated from neutrino and continuation choices |


## bihalofit python <a name="bihalofit_python"></a>

### What is established, and what must be measured

The focused equation tests compare all four analytical components with the
installed pre-change C model, at three fit/read resolutions and in both
production and source-reference modes. They include flattened, unequal and
strongly squeezed triangles, permutation symmetry, the tree-level
equilateral limit and retained-input ownership. The first six comparisons
had a largest relative difference of $`3.8\times10^{-14}`$.
Their `2e-11` test allowance covers implementation
arithmetic in this controlled comparison; it is not an allowed survey
prediction error.

The separate study
[bihalofit_refinement.py](bihalofit_refinement.py) writes and
reloads every input before evaluating a four-row joint example containing
Map3, xi+, xi- and Map2. It fixes the full summed covariance, including the
two/three-point cross terms, and measures

```math
\Delta\chi^2=\Delta m^{\mathsf T}C^{-1}\Delta m.
```

It refines the full-bispectrum grid at fixed fit settings, then investigates
the fit quadrature and root separately. Every native input table, synthetic
data vector, full covariance and mask has a saved hash. The study refuses a
non-positive-definite covariance rather than adding a ridge or dropping
modes. Its small source/power fixture tests numerical integration of the
route; it does not establish DES convergence. A survey run must repeat the
test with its actual binning, scale cuts, covariance and parameter points,
then check derivatives and Fisher-information stability.

The [saved measurement](results/bihalofit_refinement.json)
uses 41 matched fit/background redshifts, fixed coefficient-reader boost
8, and full-table dense boost 2. The measured table errors are:

| Coarse nodes in z, ln r, ln u, v | $`\Delta\chi^2`$, full joint covariance |
| --- | --- |
| 5, 17, 9, 5 | 0.0797234 |
| 9, 33, 17, 9 | 0.00622678 |
| 17, 65, 33, 17 | 0.0000162025 |

Every number in the last column is the squared residual norm defined
above. The two-point rows remain byte-identical. The permanent
[joint test](../data_vector/test_bihalofit_joint.py) requires the fine
grid to improve on the medium grid and to stay below 0.00005, about three
times the measured fine-grid error. Its
[committed fixture manifest](../data_vector/reference_bihalofit/manifest.json)
checks compressed inputs, synthetic data, full covariance and mask by
SHA-256 before the test reads them. The permanent test does not rebuild its
metric from mutable helper fixtures. This is a tighter test for this fixture
than the group's 0.2 ceiling; it does not select production grids for DES.

The fit's own integrations were varied separately. Relative to 4001
variance nodes, 8001 slope nodes and a radius tolerance of one part in
10 billion, the original 1000/2000-node rules and radius tolerance of one
part in 100,000 changed the queried full bispectrum by at most 0.00001545.
The intermediate 2001/4001-node rule with a radius tolerance of one part in
100 million was within 0.00000001491 of the tight result. These are relative
bispectrum changes, not covariance-weighted observable errors. Tightening
the root is therefore a distinct refinement from increasing the new
four-dimensional table.

The [squeezed stress test](../data_vector/test_bihalofit_squeezed.py)
also samples short/middle ratios of one part in 10 billion and one part in
100 billion, with nonzero gaps between the longer sides. In this regime
the tree formula combines large terms with opposite signs. Small rounding
differences in the two almost equal power values are amplified before the
terms cancel. The measured production full-bispectrum difference reached
0.000000621, and the source-cubic comparison reached 0.00000280; the tree
diagnostic reached a relative difference of 0.0000657. The one-halo piece
still agreed near floating-point precision.

This test keeps the ordinary tight equation comparison unchanged. It
reconstructs the tree kernel independently at 80 decimal digits, using
each route's actual power values, and checks an error bound that includes
the propagated power rounding and the sizes of the unsubtracted terms.
The decimal arithmetic belongs only to validation. Full-table positivity
and a final observable comparison remain necessary: the stress test is
not a reason to ignore errors merely because a triangle is squeezed.

### A 480-row DES Y3 mean comparison

The [joint DES Y3 record](results/bihalofit_joint_y3.json)
uses 400 two-point shear-correlation entries and 80 third-order aperture
entries, with the full fixed public simulation covariance. Every row is
retained by an all-one mask, including the two/three-point cross-covariance.
The simulated data vector, source reconstruction, power, Hubble rate,
growth and current finite-triangle assumptions are fixed across the
comparison. In particular, the triangle model samples two side lengths and
their included angle, called SAS: 10 side bins from 8 to 80 arcmin and 63
angle bins. Its exact correspondence to the published bin averaging and
measurement edges remains a separate estimator question.

The baseline uses uniform ranges from 0 to 3.1 in redshift, from -18 to 12
in the logarithm of the middle wavenumber, from -24 to 0 in the logarithm
of the short/middle ratio, and from 0 to 1 in the gap/short ratio. The dense
boost is one in every case. The internal-model data chi-squared is
25.2934888162 at this single NLA fiducial point. The observed convergence
is:

| Nodes in z, ln r, ln u, v | $`\Delta\chi^2_{\mathrm{data}}`$ | $`\delta^{\mathsf T}C^{-1}\delta`$ |
| --- | --- | --- |
| 9, 65, 33, 9 | -0.5713294 | 0.04115738 |
| 17, 97, 49, 17 | -0.3131430 | 0.01176530 |
| 33, 129, 65, 25 | -0.1573676 | 0.003453150 |
| 49, 193, 97, 33 | -0.08315490 | 0.0007998647 |

The columns measure different things. Let $`\delta`$ be the supplied mean
minus the internal mean, $`m_{\mathrm{ref}}`$ the internal mean, and $`d`$ the
fixed data vector. Then

```math
\Delta\chi^2_{\mathrm{data}}
 = 2(m_{\mathrm{ref}}-d)^{\mathsf T}C^{-1}\delta
   + \delta^{\mathsf T}C^{-1}\delta .
```

The first term depends on the existing residual relative to this particular
data realization, and can be negative. The squared vector difference in
the last column is nonnegative. Both decrease in magnitude with grid
refinement here. All 400 two-point entries remain byte-identical; the
changes enter the aperture part of the vector.

The background change was measured independently. The supplied Hubble rate
and CAMB agree to a maximum relative difference of $`2.22\times10^{-16}`$. Replacing
CAMB's matter-density history by the older radiation-free history on the
same final grid changes the data chi-squared by $`-2.95\times10^{-6}`$ relative
to the CAMB-background result. The squared vector difference for that
replacement is $`1.39\times10^{-10}`$. This control shows that the density
history accounts for only a small part of the measured route difference
at this point; it does not establish general neutrino or dark-energy
accuracy.

The retired optional configuration (`configs/des_y3_supplied_bihalofit.yaml`,
kept in Git history) used the 49/193/97/33 counts tested here. Its absolute data chi-squared shift is below
0.2 at this fiducial point. There has been no Fisher-derivative convergence
test for this baseline table. At the time of that comparison no default switch was made. Parameter-space
checks and further refinement are required before interpreting forecasts
with these particular counts. The fixed NLA covariance and the declared
source and finite-triangle assumptions also remain part of that forecast.

Finally, implementation agreement does not extend the physical calibration
of the fit. Takahashi et al. calibrated dark-matter-only simulations,
including constant-$`w`$ cosmologies. This port does not establish an
accurate massive-neutrino or arbitrary-$`w_0w_a`$ bispectrum. Its use of
$`P_{\mathrm{L}}(k,0)D(z)^2`$ imposes scale-independent growth; supplying a more
complete background does not remove that approximation. Baryonic feedback
remains a separately supplied response. Matching the measurement, finite
triangle cuts, IA prescription and covariance assumptions of Rafael's
analyses remains necessary for reproducing their inferences.

### Shared variance inputs

Both the direct project producer and the optional Cobaya theory block can
consume Gaussian and top-hat variances from the ordinary CosmoLike FFTLog
engine. Select `variance_source: shared` in the theory block, or
`physics.variance_source: shared` in the CLI configuration. The default
remains `local` until full joint mean and Fisher acceptance is established.
The shared route supplies owned z/rnl/neff/sigma8 arrays to the fit and
retains the amplitude of the supplied power spectrum. A separate sigma8
scalar no longer renormalizes that power.

The default `variance_growth: factorized` uses P(k,0)D(z)^2. The optional
`evolving` choice uses the supplied full P(k,z). Those prescriptions can
differ with massive neutrinos; neither changes the existing growth choice
on the perturbative tree and IA power legs. For the Cobaya adapter, D is
still the provider's sigma8(z)/sigma8(0), and Omega_m(z) remains the sum of
its baryon, cold-dark-matter and massive-neutrino fractions. All h-unit
conversions occur at the Python boundary.

The theory's `variance_accuracy` mapping accepts the shared controls
described in [Accuracy](../../../../external_modules/code/cosmolike_core/cosmolike/3pcf/README.md#shared-smoothing). Table counts derive
from an unchanged baseline. The cached bispectrum includes every supplied
variance array in its input identity, and old outputs retain their own
copies after replacement. The physical-model and CLI tests compare C and
Python equations with identical anchors, exercise both covariance adapters,
and check unchanged power when only the redundant sigma8 scalar changes.
These tests establish the interface contract; they do not establish survey
forecast convergence or a new neutrino halo prescription.


## comparison <a name="comparison"></a>

### Executed example <a name="comparison-executed"></a>

The committed notebook outputs use one OpenMP worker, one thread for
every BLAS library, and this transform configuration:

    nfft: 32            # Fourier grid points
    lmax: 3             # Legendre truncation
    mmax: 5             # angular Fourier truncation
    opening angles: 7

The full-array diagnostics below belong to that configuration; the
16-point values above belong to the smaller automated test. Each value is
a dimensionless maximum scaled difference, not a percentage or a forecast
error budget.

| Comparison | Maximum scaled difference | Interpretation |
| --- | ---: | --- |
| TATT untapered two-Bessel transform | 1.36e-14 | Matching finite operator |
| Main tapered two-Bessel transform | 1.47e-14 | Matching corrected operator |
| TATT angular-coupling table | 2.44e-4 | Demonstrated source rectangle bias |
| Natural components, matching 200-node interpolation | 4.34e-4 | Source integration bias remains |
| Natural components, exact C coupling | 3.37e-3 | Different interpolation rule |
| Equal-filter aperture integral | 1.16e-15 | Matching finite operator |
| Unequal-filter aperture integral | 7.58e-1 | Expected original branch error |
| Frozen matter reference | 6.43e-6 | Includes the source fit's root error |
| Frozen IA, linear-power prescription | 4.25e-15 | Stored reference, all 19 terms |
| Frozen IA, nonlinear-power prescription | 1.69e-15 | Stored reference, all 19 terms |

[EXAMPLE_COMPARE_CODES.ipynb](../../EXAMPLE_COMPARE_CODES.ipynb) keeps these
cells with their outputs, including all five figures. The measurements
above were produced by live execution of the original codes; the current
notebook compares against saved JSON references under
`tests/data_vector/` that record the source revisions, and the project
no longer imports or runs those codes.

### Credits and checks <a name="comparison-credits"></a>

The fastnc implementations are credited to Sunao Sugiyama and Rafael
Heringer Gomes. Rafael's `fast_map3.py` credits Mike Jarvis and TreeCorr
and carries its redistribution notice. The recorded workflow imported
those files from the user's checkouts; it did not distribute modified
copies.

The retired `tests/test_code_comparison.py` (kept in Git history) checked
source-hash refusal, missing-source reporting, thread settings and real
execution of the available original files, with equal-filter agreement
and a required unequal-filter difference. It retired with the
live-execution route; the fixture comparison lives in the notebook.


## paper targets <a name="paper_targets"></a>

### Figure targets <a name="paper_targets-figures"></a>

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

[EXAMPLE_IA_COMPONENTS.ipynb](../../EXAMPLE_IA_COMPONENTS.ipynb) demonstrates
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

### Fisher comparisons built on these targets <a name="paper_targets-fisher"></a>

The [Fisher guide](../../../../external_modules/code/cosmolike_core/cosmolike_notebook_utils/README.md#fisher) defines the local forecast, the
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
as such. Use the [Fisher guide](../../../../external_modules/code/cosmolike_core/cosmolike_notebook_utils/README.md#fisher) for exact project parameter
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

### Baryonic feedback targets <a name="paper_targets-baryons"></a>

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

The [paired-response guide](../../../../external_modules/code/cosmolike_core/cosmolike/3pcf/README.md#paired-feedback) supplies a shared amplitude for
independently specified matter-power and matter-bispectrum responses,
with the quadratic IA terms and the covariance kept at DMO. The
[BACCO guide](../../../../external_modules/code/cosmolike_core/theory/bacco_bispectrum/README.md#bacco) connects the installed Burger2025 emulators to
the joint mean with five physical feedback coordinates. The older
bispectrum-only `fb` remains unsuitable as a joint feedback parameter.

### Y6 inputs <a name="paper_targets-y6"></a>

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
feedback family described in the [baryon section](#paper_targets-baryons).

Measured sky data are unnecessary for a forecast centered on a theoretical
mean, but source distributions, noise levels, area or mask, selection,
and calibration assumptions are still required. Enter them as identified
Y6 inputs or explicitly chosen forecasts. Save the resolved configuration,
source-file hashes, row labels, covariance prescription, physical
conventions, derivative steps and refinement comparisons with each result.
The final acceptance record separates component accuracy, estimator
agreement, figure reproduction and parameter inference; passing an earlier
step supplies evidence for the next, not a replacement for it.

### Remaining reproduction gaps <a name="paper_targets-gaps"></a>

The pinned Y3 run uses an NLA mean, an emulator trained with the 8 arcmin
cut, MOPED compression and the Percival finite-simulation likelihood.
The following items still separate this project's forecasts from an exact
reproduction of that run:

| Item | What is missing or different | Where it is recorded |
|---|---|---|
| Compression matrix | The pinned [MOPED configuration](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/moped.ini) references a matrix absent from the twelve-file manifest | [Compression guide](../../README.md#compression) |
| Trained emulator | The pinned run uses an emulator with the 8 arcmin cut; its files are absent from the manifest | [Pinned run](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/test-cosmogridc-simdv-joint-2pt-moped-run-796sims-february-Percival/des-y3-shear-2pt-map3-NLA-all.ini) |
| Scale-cut file | `examples/des-y3-scale-cuts.ini` is referenced but not in the manifest | [Y3 target](../../README.md#estimator) |
| Priors file | `examples/des-y3-priors.ini` is referenced but not in the manifest; the Gaussian widths above come from the publication | [Y3 target](../../README.md#estimator) |
| Dependency versions | Upstream CosmoSIS dependency versions are unrecorded | [Pins](../../README.md#estimator) |
| xi prediction geometry | Upstream xi is full-sky and bin-averaged with `ell_max=40000`; the present `XiEngine` uses flat-sky $`J_0/J_4`$ transforms | [Two-point configuration](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/cosmic-shear.ini) |
| Pixel window | The pinned direct reference names a window at nside 1024; this project applies a window only through its explicit option | [Estimator guide](../../README.md#estimator) |
| Finite-triangle estimator | The packed FITS products lack the original within-bin triangle weights | [Estimator guide](../../README.md#estimator) |
| Source reconstruction | The demonstration clipping and PCHIP construction is not the publication's calibrated treatment | [Fisher guide](../../../../external_modules/code/cosmolike_core/cosmolike_notebook_utils/README.md#fisher) |
| Map3 sign processing | The realization-level processing behind the covariance cross block is not recovered | [Observed section](../../README.md#estimator) |
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

### Example responsibilities <a name="paper_targets-examples"></a>

| Entry | Scientific role |
|---|---|
| `notes/intrinsic_alignments/EXAMPLE_FISHER_IA.ipynb` | IA sign responses, finite model distances and two-point/joint local information with varied or fixed dark energy |
| `EXAMPLE_FISHER_ACCURACY.ipynb` | Numerical-refinement and Fisher interpretation through the shared prediction |
| `EXAMPLE_NAUTILUS1.yaml` | NLA sampling of a noiseless mean-generated synthetic data vector |
| `EXAMPLE_NAUTILUS2.yaml` | TATT sampling of a noiseless mean-generated synthetic data vector |
| `EXAMPLE_NAUTILUS3.yaml` | Observed NLA demonstration with explicitly unvalidated estimator/source/cut assumptions |
| `data/upstream_y3_observed/` | Versioned observed inputs with their source manifest; no download step |

The [inference guide](../../README.md#sampling) states how to inspect these choices and
check a changed physical parameter before sampling. Executed checks and
posterior convergence are separate records. A sampling budget is not a
claim that its posterior has converged.
