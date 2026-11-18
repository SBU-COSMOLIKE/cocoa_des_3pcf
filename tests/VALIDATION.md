# Validation record

This record distinguishes equation checks, state-change checks, numerical
refinement and survey reproduction. Passing the first two does not establish
the latter two. The C implementation is isolated in `cosmolike/3pcf/`; the
ordinary two-point source files were not changed or rebuilt for these checks.
The existing DES Y3 extension is used to check the Python two-point bridge.

1. [Executed checks](#executed)
2. [What the references measure](#references)
3. [Notebook state and parallel arithmetic](#state)
4. [Scientific limits](#limits)

## Executed checks <a name="executed"></a>

The current complete strict collection passes **561 tests** (91.42 seconds,
exit 0). This includes finite sign-pair, full-check provenance, saved-node
step-study, historical step-plot and integer-selector regression tests.
The complete collection also includes four analytic fixed-dark-energy
tests and eight retained-stencil rejection cases. The former check retained
nuisance marginalization, the prior counted once, unresolved directions
and input immutability. The step study also passed preflight against
both actual Fisher archives and then evaluated eight new means per model.
The rows, covariance and input/code identities stayed fixed during those
calculations. The build and earlier check history below identify which
native implementation these Python checks exercise.

The saved-step reader's focused group passes **20 tests** (6.24 seconds),
including refusal of missing or malformed retained arrays, changed shared
nodes, changed physical coordinates, duplicate new points and a different
FoM pair. This reading check remains historical: it compares archived
records, without requiring today's physical source or noise helper. A
separate array-only rescore of the preserved NLA/TATT step records reproduces
their selected derivatives and every recorded ratio exactly; it verifies
those records' arithmetic without relabeling their physical provenance.

The three real Nautilus `--check` examples also pass with the corrected
likelihood scalar handling and 480 rows. Their A1 displacement is 0.01;
all three restore the exact fiducial prediction. Synthetic NLA and TATT
start at zero chi-squared and change it by 0.00940607 and 0.02446217.
The observed NLA example changes chi-squared from 1450.71721 to 1450.09425.
This observed-data execution test does not reproduce the published fit:
the example retains all 400 xi rows and uses the declared approximate
aperture estimator and source reconstruction. These checks exercise real
native predictions; a separate two-dimensional Gaussian test exercises
Nautilus sampling and resume. No DES posterior chain is claimed.

A cached first mean originally hid a scalar-type error: fixed integer
selectors were converted to floats before an IA displacement called the
native projection. The likelihood now retains integer types, and independent
regressions require actual integer-index semantics for both Python and
NumPy integer selectors. The physical mean equations did not change.

The IA/Fisher notebook executed twelve code cells with thirteen figures;
the original-code notebook executed nine cells with five figures. All
figures were inspected. The new full NLA/TATT baseline runs reproduce all
31 numerical arrays of their preserved prior calculations bit for bit.
Their original metadata stays separate from the fresh implementation
identities; the equality record is `runs/baseline_refresh_comparison.json`.
Four selected-column scans and eight additional step-study means per
model are refreshed. The full triangle checks then evaluate every
derivative at both spacings: 103 means for NLA and 121 for TATT. Every
shared selected stencil and reconstructed derivative agrees bit for bit
with its scan. Updating all nuisance columns compensates the larger
mixed-grid FoM shifts. At h/2 the full triangle/default FoM ratios are
1.001703 for NLA and 0.999445 for TATT; their internal step sensitivity
remains 6.08% and 5.64%. The
[accuracy record](studies/fisher_accuracy_validation.md) separates these
comparisons from a claim of complete convergence.

The Fisher-accuracy notebook also executed twelve code cells and produced
nine figures from completed saved arrays. Inspection identified a clipped
axis label in the step panels. The shortened label was rerendered and
all affected figures were inspected; all nine final figures are readable.

Two additional component tests compare the finite Fourier coupling with
independent 512- and 1024-node angular integrals at all L<=50 and M<=100,
for seven side-ratio angles. Both pass with serial OpenMP and BLAS.
The largest discrepancy divided by 4*pi² is 3.71e-14. These tests
validate the sampled geometric coefficients, not a complete transform
or a Fisher truncation choice. The
[angular results](studies/angular_order_results.md) record the ten
separate NLA/TATT fiducial comparisons and their remaining limits.

The native interpolation/source-grid checkpoint is `6a7eae7` on
`lowrank_fftlog`. Its full project collection passed **502 tests in strict
and debug builds**, with no sanitizer diagnostics. The subsequent
Python-only measured-source-edge and prediction-provenance additions pass
the full **512-test strict collection**. Their affected source, survey,
CLI and provenance group passes **44 tests in debug**. Both debug runs use
undefined-behavior and floating-point-division checks, not AddressSanitizer.
The strict extension is restored. Compilation uses one worker; numerical
tests use at most four OpenMP workers and serial BLAS.

With the demonstrated source defaults and saved-result checks, the full
strict collection previously passed **524 tests** (89.14 seconds, exit 0). This includes
11 file-replacement/reuse checks and the unit-invariance check for
covariance-weighted derivative differences. No C implementation changed
after the strict/debug native checks above. The first run found five
test-fixture failures: two fixtures requested cosmological derivatives
from fixed tables, and three assumed an absent flag that the revised
example explicitly sets. Their corrections preserve the model validation
and test that inference overrides leave the prediction file unchanged.

The additional finite sign-pair tests and existing derived-quantity tests
pass **13 checks** (0.22 seconds, exit 0), without CAMB or native model
calls. The independent construction uses four unit Gaussian variables:
its known joint squared distance is 15, split into two-point distance 5
and conditional three-point distance 10. Tests change row order and Map3
units, remove the independent Map3 contribution, reverse a consistent sign
convention, and reject mismatched row identities. This command ran from
the same activated Cocoa directory with serial BLAS:

```bash
.local/bin/python -m pytest \
  projects/des_3pcf/tests/test_finite_sign_pair.py \
  projects/des_3pcf/tests/test_fisher_demo_wrappers.py -q
```

The later combined saved-result, full-check provenance, finite-distance
and derived-quantity group passes **28 tests** (1.59 seconds, exit 0).
Four new tests compare archived baseline/candidate physical identities,
retain explicitly historical reading after a current source-file change,
and refuse that change before an executable full-check launch. Identical
row labels and covariance are insufficient if input or implementation
bytes changed between the two calculations.

```bash
.local/bin/python -m pytest \
  projects/des_3pcf/tests/test_accuracy_full_identity.py \
  projects/des_3pcf/tests/test_fisher_result_identity.py \
  projects/des_3pcf/tests/test_finite_sign_pair.py \
  projects/des_3pcf/tests/test_fisher_demo_wrappers.py -q
```

The changes check independent source axes, signed NLA and normalized TATT
residuals, stretched triangle coordinates, mixed-angle multipole tables,
both C++ adapters, retained plans, changed triplet batches, and setting
restoration. The two-node angular test deliberately keeps physical endpoints
identical while changing interpolation coordinates; it verifies that the
retained-query key includes the coordinate definition. Extreme PUV edge
conditioning is recorded as a separate limitation, not hidden by relaxing
the interior reconstruction tests.

The following commands ran from `Cocoa/` after activating its environment
and setting serial BLAS and the four-worker OpenMP cap.

**Step :one:**: run the strict project collection.

```bash
.local/bin/python -m pytest projects/des_3pcf/tests -q
```

**Step :two:**: select the debug extension with one compilation worker.

```bash
COSMOLIKE_DEBUG_MODE=1 make -C projects/des_3pcf/interface -f MakefileCosmolike -j1
```

**Step :three:**: run the project collection with that extension. This full
run contains the 502 checks preceding the Python-only additions.

```bash
.local/bin/python -m pytest projects/des_3pcf/tests -q
```

**Step :four:**: check the added Python source/provenance paths in debug.

```bash
.local/bin/python -m pytest \
  projects/des_3pcf/tests/test_source_grids.py \
  projects/des_3pcf/tests/test_pipeline_cli.py \
  projects/des_3pcf/tests/test_survey.py \
  projects/des_3pcf/tests/test_prediction_provenance.py -q
```

**Step :five:**: restore the strict extension.

```bash
make -C projects/des_3pcf/interface -f MakefileCosmolike -j1
```

The [interpolation study](studies/interpolation_validation.md) reports
separate direct-projection, multipole and full-aperture comparisons. The
first TATT demonstration Fisher result failed these accuracy checks and
is not accepted. Model/source choices and full derivative convergence
remain separate from the successful implementation tests.
The [source study](studies/source_policy_validation.md) records 162 full
480-row evaluations: 108 in the original six-case comparison and 54 in
the combined source/kernel/radial follow-up, with no CAMB calls. It supplies
the declared starting source grids for the new Fisher diagnostic.

The earlier core checkpoint was `98824eb`, following `d832c5f` and `8c58ace`. Checks ran on
2026-11-17 using Darwin arm64, Clang 19.1.7, Python 3.11.12 and NumPy 1.26.3.
Compilation used one worker. BLAS used one thread. OpenMP comparisons used
at most four workers, with one worker for ordinary calculations. No timing
experiment or survey-scale integration was run.

| Calculation | Result |
|---|---|
| Full strict checks before the final multipole rounding correction | 232 passed, exit 0 |
| Full debug checks before the final multipole rounding correction | 232 passed, exit 0; no sanitizer diagnostic |
| Multipole output-extension checks against the preceding strict build | Eight failed, reproducing the scalar/vector rounding discrepancy |
| All data-vector checks and covariance component checks after the correction, strict | 100 passed, exit 0 |
| Physical-model, multipole output-extension and physical-state checks after the correction, debug | 38 passed, exit 0; no sanitizer diagnostic |
| Restore strict extension after debug | Exit 0 |
| Physical projection output-extension checks before the second rounding correction | Eight failed and fourteen passed, reproducing the physical batch discrepancy |
| Full strict checks with both rounding corrections and the covariance work reduction | 281 passed, exit 0 |
| Full debug checks with both rounding corrections and the covariance work reduction | 281 passed, exit 0; no sanitizer diagnostic |
| Saved covariance comparison before and after omitting unused large-area transforms | All 16 arrays match bitwise; both adapters, joint/xi-only and large-area/finite-window |
| Large-area profile and halo-offset independence checks | Six passed; included in the 281 checks |
| Shared plotting array checks | 13 passed; separate from the project checks |
| Shared plotting render review | 30 modes inspected; four teaching figures retained in the core README |
| Original-code notebook executed with the strict checkpoint | Nine code cells completed, five figures, no cell errors |
| IA/Fisher notebook executed with the strict checkpoint | Eleven code cells completed, three figures, no cell errors; survey calculations disabled |
| CLI timing control, source policy and plot checks | 26 passed; one OpenMP worker and serial BLAS; no physical predictions |
| LSST timing binding inspection | All 30 methods and used keywords match the installed extension; import only |
| CLI timing input preflight | Ready; ten random points per workload; no CAMB or measured evaluations |
| Four-case timing, CAMB control and cache checks | 46 passed with at most four OpenMP workers and serial BLAS |
| Bounded CAMB solve, extrapolation and transfer reuse | 13 passed; independent final-As solve and opposite unit API |
| CLI NLA/TATT timing study | 120 measured predictions plus twelve warmups; every thread-count comparison bitwise equal |

The strict build uses `-O3` with native instruction selection and the core's
strict floating-point flags. Debug uses `-O0`, undefined-behavior and
floating-point-division sanitizers. There is no AddressSanitizer result in
this record. Linux and other processor architectures remain untested.

Build and environment commands are in the [test guide](README.md). From the
project directory, after that environment setup, the exact test selections
were:

**Step :one:**: run all checks in the selected build mode.

```bash
../../.local/bin/python -B -m pytest -q -x tests
```

**Step :two:**: check the corrected multipole arithmetic and its callers.

```bash
../../.local/bin/python -B -m pytest -q -x \
  tests/data_vector tests/covariance/test_components.py
```

**Step :three:**: check that correction with the debug build.

```bash
../../.local/bin/python -B -m pytest -q -x \
  tests/data_vector/test_multipole_lane_invariance.py \
  tests/data_vector/test_physics_model.py \
  tests/data_vector/test_physics_setting_sequences.py
```

The 281 project checks include the YAML/CLI path, conditional Fisher
decomposition, configuration-identity refusals and both sets of SIMD lane
checks. Notebook execution used Jupyter kernels, with every loaded BLAS pool
limited to one thread. Plotting checks are a separate array-only collection.
The final strict and debug runs both use checkpoint `98824eb`. The strict
extension is restored after debug verification. The subsequent core commit
`34c62d1` changes only the plotting README's environment instructions.

## What the references measure <a name="references"></a>

| Reference | Scientific question |
|---|---|
| Pinned fastnc matter and nineteen IA components | Does the port preserve the declared upstream formulas and conventions? |
| Eighty-digit squeezed-triangle equations | Does stable double-precision geometry retain terms lost by squared-side cancellation? |
| Cartesian Gaussian shear moments | Are the natural-component signs, phases and ordered Fourier legs correct independently of fastnc? |
| Analytic and independently evaluated Mellin transforms | Are FFTLog normalization, Bessel order, taper and radial-bin averaging correct? |
| Pinned aperture kernels and direct complex contractions | Are unequal radii, mirrored terms, bin measures and component order correct? |
| White-noise filtered-pair integral and fifteen Gaussian pairings | Are Fourier measures and Gaussian contraction multiplicities correct? |
| Direct NFW profiles, halo variance and abundance derivatives | Are mass definitions, halo normalization and the selected response correct? |
| Separately assembled PB, BB, PT and connected terms | Do shared geometry and source factors preserve the declared covariance equations? |
| Analytic annular noise overlap | Do xi spin factors, E/B noise and exact angular edges give the expected noise covariance? |
| Full joint Schur decomposition and derivative examples | Does the forecast preserve cross covariance and parameter priors? |

Source agreement is not an independent proof of a source formula. The
squeezed-triangle checks deliberately use unrearranged equations at eighty
digits, with the exact binary inputs supplied to C. They cover side ratios
through $`10^{-12}`$, several opening angles, cyclic relabelings and common
power-of-two rescalings. The matter fit retains its separate source-root
accuracy allowance; the geometry tests do not relax that comparison.

One pinned IA entry describes an intended 3:4:5 triangle and vanishes
analytically. The reference and port can give different signs to its tiny
floating-point remainder. That entry is checked against its unsubtracted
vertex amplitude times a roundoff bound. Every other pinned IA entry keeps
its original tolerance. This is different from increasing a whole-array
absolute tolerance until a discrepancy disappears.

## Notebook state and parallel arithmetic <a name="state"></a>

A reused object is compared with a separate process configured directly at
the requested state. The physical-model checks contain 150 such references,
the transform/aperture checks 66, and the physical-covariance checks 82.
They grow and shrink dimensions, change coordinates without changing sizes,
interleave retained models, restore earlier settings and mutate caller or
returned arrays. Identical operations require identical float64 bits.

The project checks also change IA parameters, source shifts, calibration,
physical-table settings and global/stage accuracy controls behind the outer
Fisher cache. Changing a count must affect the intended stage before its
invalidation test can be considered informative. The ordinary two-point
bridge restores its declared state after raw notebook calls and refuses
partial edits to its prepared settings.

The review identified a compiler distinction that strict flags alone do not
remove: a scalar multiply-add can round once while explicit SIMD products
and sums round separately. Local rounding boundaries are used in the affected
covariance and multipole tails. Covariance tests add zero-measure nodes;
multipole tests append an unused query or an extra retained Legendre order.
These operations move an existing output between scalar and vector paths
without changing its equation. The corrected outputs retain identical bits.

The physical projection checks apply the same criterion to source-triplet,
direct-multipole, NLA power and aperture-filter batches. Eight of the 22
cases failed against the earlier strict extension. All pass after the local
rounding boundaries and the unpaired NLA reader were made consistent with
their two-lane operations. Both adapters and both power-reader conventions
are exercised; quadrature weights and nodes remain fixed in each comparison.

Workspace dimensions enter checked `size_t` multiplication before allocation.
Notebook integer source labels are checked before conversion to int32, so
fractional values cannot be truncated into apparently valid bin labels.
Malformed-input tests require the intended diagnostic and exit status in a
child process; a loader failure or unrelated traceback cannot pass them.

## Scientific limits <a name="limits"></a>

The covariance describes the declared full-filter map estimator and selected
one-halo/halo-abundance responses. It is gravitational, with Gaussian shape
noise; it does not supply connected IA covariance, a galaxy catalogue mask,
or every contribution to super-sample covariance. Finite side-angle-side
means need a matching covariance or a demonstrated estimator approximation.
The [paper-target guide](../PAPER_TARGETS.md) separates those questions from
agreement between shared numerical kernels.

Small controlled tests do not select DES integration bounds, triangle grids,
scale cuts or nuisance priors. Survey refinement must compare the full mean
and its derivatives, preferably using a matched covariance through
$`\Delta m^T C^{-1}\Delta m`$. The current baryon bispectrum response alone
does not define a joint power/bispectrum feedback family. No Y3/Y6 posterior
reproduction or runtime speedup is claimed by this record.
