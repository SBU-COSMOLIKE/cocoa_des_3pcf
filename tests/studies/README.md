# Optional studies

## :interrobang: FAQ: What are optional studies? <a name="optional_studies"></a>

Optional studies explore how numerical settings and modeling choices
affect predictions; they can be expensive, so they run separately from
the [routine tests](../README.md#running). Each study below has a
tracked page with its measurements and limits, a driver script where one
exists, and saved records under [`results/`](results).
[`results/MANIFEST.json`](results/MANIFEST.json) pins every record with
its SHA-256 hash, the study it belongs to, and what it holds; the BACCO
records carry their own hashes in
[results/bacco/execution.json](results/bacco/execution.json).
`results/` also holds two execution records: the archived-geometry CLI
record `estimator_execution.json`, described under
[Estimator conventions](#estimator), and the feedback-scenario notebook
record
[results/feedback_scenarios_validation.json](results/feedback_scenarios_validation.json),
which holds the input hashes and scope of the executed
[EXAMPLE_FEEDBACK_SCENARIOS.ipynb](../../EXAMPLE_FEEDBACK_SCENARIOS.ipynb).

Every command below runs from `Cocoa/projects/des_3pcf` with the
[test environment](../README.md#running) set: serial BLAS, at most four
OpenMP workers, and one study process at a time. Regeneration commands
write under the local `chains/` directory, which Git ignores; the files in
`results/` are the versioned evidence. A deliberate update replaces the
record in `results/` and its manifest entry together, with the reason in
the study page; do not overwrite a saved record to make a changed number
pass.

1. [Angular orders](#angular)
2. [Fisher accuracy](#fisher)
3. [Interpolation for the Y3 demonstration](#interpolation)
4. [Source reconstruction and photo-z derivatives](#sources)
5. [BACCO feedback responses](#bacco)
6. [Estimator conventions](#estimator)
7. [E and B after finite triangle cuts](#finite_eb)
8. [Joint IA conventions](#ia_conventions)
9. [Stretched-edge conditioning](#stretched_edge)
10. [Shared-variance acceptance](#shared_variance)
11. [Cobaya provider and state restoration](#cobaya_state)

## Cobaya provider and state restoration <a name="cobaya_state"></a>

The [joint check](cobaya_joint_check.py) loads the NLA and TATT example
YAMLs with their CAMB and bispectrum theory components. It compares the
result with the fixed synthetic input, then changes the IA amplitude,
source redshift shift and matter density separately. Each change must
affect the prediction, and restoring the parameters must restore its
bytes. This checks provider wiring and stale state; it does not establish
numerical convergence or agreement with observed DES measurements.

```bash
python tests/studies/cobaya_joint_check.py --threads 4
```

The run starts no sampler and writes `chains/cobaya_joint_check.txt`.
Completed validation is recorded in [VALIDATION.md](../VALIDATION.md).

## Angular orders <a name="angular"></a>

This study measures the covariance-metric distance
$`D^2=(d_a-d_b)^{\mathsf T}C^{-1}(d_a-d_b)`$ between 480-row predictions
when the Legendre degree $`L`$, the Fourier order $`M`$, the prepared
angular samples and the coupling interpolation change from the inherited
baseline, for NLA and TATT. An independent Gauss-Legendre integral also
checks the geometric coupling at every $`L`$ through 50 and $`M`$
through 100. The five cases, the measurements and their limits are in
[the angular-order study](angular_order_results.md).

Open [EXAMPLE_FISHER_ACCURACY.ipynb](../../EXAMPLE_FISHER_ACCURACY.ipynb)
and choose the numbered NLA or TATT example. Its final angular-order
section changes the truncations, quadrature and coupling interpolation
explicitly, evaluates the fiducial prediction through Cobaya, and checks
that the two-point rows remain unchanged. Configurations and numerical
arrays are written below `chains/fisher_accuracy/`.

Saved records:
[results/nla_angular_orders/summary.json](results/nla_angular_orders/summary.json),
[results/tatt_angular_orders/summary.json](results/tatt_angular_orders/summary.json)
and
[results/high_order_coupling_check.json](results/high_order_coupling_check.json).

## Fisher accuracy <a name="fisher"></a>

This study measures how the demonstration NLA and TATT Fisher forecasts
respond to the derivative step (spacings $`h/4`$ through $`2h`$) and to
triangle-grid refinement: full derivative matrices, screening-column
substitutions and the diagonal-noise screening, with every distance
taken in the full joint covariance metric. The measurements, their
nonmonotonic behavior and the remaining open comparisons are in
[the accuracy record](fisher_accuracy_validation.md); the executable
presentation is
[EXAMPLE_FISHER_ACCURACY.ipynb](../../EXAMPLE_FISHER_ACCURACY.ipynb).

The notebook contains the full-derivative comparison, nuisance projection,
row-block substitutions, and smaller- and wider-step studies as ordinary
cells. Array-only cells reuse derivatives already in memory. Quarter-step
and wider-step cells identify their additional physical evaluations before
the loops. There is no separate Fisher study launcher.

Saved records: the array diagnostics in
[results/fisher_array_diagnostics/](results/fisher_array_diagnostics)
(`full_triangle.json`, `amplification.json`, `step_blocks.json`), the
two-column step studies in
[results/nla_triangle_step_study/](results/nla_triangle_step_study) and
[results/tatt_triangle_step_study/](results/tatt_triangle_step_study),
the full quarter-step results in
[results/nla_full_h4/](results/nla_full_h4) and
[results/tatt_full_h4/](results/tatt_full_h4), and the baseline
equality record
[results/baseline_refresh_comparison.json](results/baseline_refresh_comparison.json),
which records that the refreshed NLA and TATT baseline runs reproduce
all 31 numerical arrays of their preserved prior calculations bit for
bit.

## Interpolation for the Y3 demonstration <a name="interpolation"></a>

This study measures the interpolation errors of the projected-bispectrum
and multipole tables and the resulting changes of all 80 aperture rows,
after the signed-residual and shape-chart corrections. It also records
why the first TATT demonstration Fisher was rejected. The measurements
are in [the interpolation record](interpolation_validation.md).

Two drivers reproduce its tables without a CAMB run:

```bash
python tests/studies/multipole_stage2_diagnostic.py --output chains/multipole_stage2_diagnostic
```

```bash
python tests/studies/projection_mean_convergence.py --output chains/projection_mean_convergence
```

Saved records:
[results/projection_stretched_diagnostic.npz](results/projection_stretched_diagnostic.npz),
[results/residual_mean_diagnostic.npz](results/residual_mean_diagnostic.npz),
[results/multipole_mixed_diagnostic.npz](results/multipole_mixed_diagnostic.npz)
and the aperture-row means in
[results/projection_mean_convergence/](results/projection_mean_convergence).

## Source reconstruction and photo-z derivatives <a name="sources"></a>

This study measures how the source reconstruction (linear against
measured-edge PCHIP), the source, kernel and line-of-sight node counts,
and the photo-z displacement size change the 480-row mean and its `dz1`
derivative in the joint covariance metric. It supplies the
source baseline that the demonstration Fisher configurations declare. The
measurements are in [the source record](source_policy_validation.md).

```bash
python tests/studies/source_policy_convergence.py --project . \
  --output chains/source_policy_convergence
```

Saved records: the six-case comparison in
[results/source_policy_convergence/](results/source_policy_convergence)
and the combined follow-up in
[results/source_combined_convergence/](results/source_combined_convergence),
with each case's `summary.json`, `manifest.json`,
`resolved_numerics.json`, `diagnostics.npz` and `point_*.npz`, plus each
model's cross-case `case_comparisons.json`.

## BACCO feedback responses <a name="bacco"></a>

This study measures a five-parameter joint Fisher forecast of two-point
shear with and without Map3, using independently emulated power and
bispectrum feedback responses, and compares the neutrino, continuation
and interpolation prescriptions. The measurements are in
[the BACCO record](bacco_validation.md); the
[example commands](../../README.md#feedback) run its notebook and off-grid
interpolation check.

Saved records in [results/bacco/](results/bacco): `execution.json`,
`fisher_summary.json`, `interpolation_summary.json`,
`interpolation_samples.npz` and the exported figures.

## Estimator conventions <a name="estimator"></a>

This review separates what the publication specifies, what the pinned
theory code computes, and what the downloaded measurement files can
recover: radial binning, angular sampling, pixel window and
representative side lengths. It inspects pinned sources and runs no
physical model; the comparisons are in
[the conventions page](estimator_conventions.md).

The related execution record
[results/estimator_execution.json](results/estimator_execution.json)
holds the configuration, window and result hashes of the
archived-geometry CLI evaluation of all 480 rows. It is an execution
record, not an angular-convergence claim.

## E and B after finite triangle cuts <a name="finite_eb"></a>

This note derives why truncating the aperture weights in a finite
side-angle-side integral can remove the exact E/B separation of the
ideal third moment, and specifies the response test for the omitted
single-B bispectra. The derivation is
[the E/B page](finite_window_eb.md). It is analytic: there is no driver
and no saved record, and no nonzero correction has been established for
the project's weights.

## Joint IA conventions <a name="ia_conventions"></a>

This review checks that the Python and C routes assign the same physical
meaning to the intrinsic-alignment coefficients $`a_1`$, $`a_2`$,
$`\eta_1`$, $`\eta_2`$ and $`b_{\rm TA}`$, including normalization,
pivot, stored signs and the factor of five. The inspected definitions
and file lines are in [the conventions page](joint_ia_conventions.md).
It is a source review: there is no driver and no saved record, and the
two routes' numerical kernels still need their own convergence checks.

## Stretched-edge conditioning <a name="stretched_edge"></a>

This diagnostic measures prepared-against-direct residual ratios of the
stretched PUV shape chart at six nearly flattened triangles, where a
one-ulp change of a side ratio moves the transformed coordinate by a few
times $`10^{-9}`$. The measured ratios and the resulting test policy are
in [the conditioning page](stretched_edge_conditioning.md). It reuses
the controlled spectra of
[../data_vector/test_physics_model.py](../data_vector/test_physics_model.py);
there is no separate driver and the measured values live in the page
itself.

## Shared-variance acceptance <a name="shared_variance"></a>

This record answers whether the shared CosmoLike smoothing route may
replace the local reference variance calculation in the full NLA and
TATT forecasts. It keeps the acceptance criteria, the first full
comparison that failed them, the diagnosis of that failure, and the
passing comparisons on refined CAMB inputs as one chronology in
[the shared-variance record](shared_variance_validation.md). There is no
separate driver and no saved record here; the measured limits live in
the page itself, and the current acceptance statement is in
[shared smoothing controls](../../../../external_modules/code/cosmolike_core/cosmolike/3pcf/README.md#shared-smoothing).
