# Work in progress: three-point implementation

Optional fixed compression, normalized Percival inference and recovered
aperture conventions are implemented. EXAMPLE_COMPRESSION executes the
three-row noise-tracer example and the historical TATT derivative algebra.
Conditional xi compression retains 100 of 480 rows with maximum normalized
Fisher difference 1.54e-15; ordinary xi-only compression loses joint
information. The notebook saves labeled matrices with fiducial and input
identity, and supplies an independent diagonal teaching pilot for the
finite-simulation example. Its figure was inspected. COMPRESSION.md and
EXAMPLE_NAUTILUS4.yaml distinguish the statistical assumptions from a
published posterior reproduction.

The physical Percival check used the TATT mean, 100 retained rows, one
OpenMP worker and one BLAS worker. The A1 displacement gave chi squared
0.005483008289552343 and restoring the fiducial was bitwise exact. This
checks the executable likelihood and state restoration, not coverage or
posterior convergence.

ESTIMATOR.md and des_y3_archived_geometry.yaml expose the original radial
bank, a lower-edge cut retaining indices 11..19, area-weighted sides with
the archived empirical offset, 100 angle centers and scalar HEALPix pixel
smoothing. The pinned 72-kB input is verified by SHA256. Native multipole
preparation includes W3 inside the angular integral; W1W2 multiply after
interpolation. Both full-filter callbacks apply all three factors once.
Raw B and xi remain unsmoothed, and the unmatched analytic covariance is
refused. The notebook adapters and prepared grids own their arrays.

The strict full collection passed 630 tests before the final window tests
and refusal were added. The final affected collection passes 133 checks
in strict mode. Debug passed 132 affected checks, then all 14 window checks
including the additional independent prepared angular oracle. The strict
build is restored. The documentation render check reports 39 Markdown
files, 78 tables, 434 local links and no errors. Independent scientific
and didactic review checked the likelihood formula, response conditioning,
unit-stable rank, complex contraction sign, window placement and retained
query lifetime. Shared two-point/team files were not edited.

The archived-geometry CLI evaluated all 480 rows with finite values using
four OpenMP workers and one BLAS worker. Its configuration, window and
result hashes are in tests/studies/estimator_execution.json. This is an
execution record, not an angular-convergence claim.

Catalogue triangle weighting, the exact published cut/compression/prior
choices, angular convergence at the sharp W3 cutoff, shared-sky mock
correlations, and physical simulation calibration remain open. These checks
are not replaced by agreement of local Fisher matrices or by a positive
sample covariance.

The paired baryonic-response component is implemented and reviewed. Both
matter ratios are independently supplied; their shared log-response amplitude
changes only the nonlinear-alignment baseline. Quadratic IA stays DMO,
ratios stay fixed across cosmology, and the covariance stays fixed. The
executed EXAMPLE_JOINT_FEEDBACK notebook evaluates all 480 rows, restores
DMO exactly and retains the same physical model. Its three figures were
inspected, with export margins corrected for the two-point axis labels.
FEEDBACK.md defines the arrays, domain extension and calibration limits.

The affected 48 checks pass in strict and undefined-behavior debug builds;
strict is restored. The separate didactic review traced ratio definitions,
ordered versus sorted triangle sides, owned arrays, serial spline
preparation, linear readers and complete amplitude keys. Independent
review found squeezed-triangle median and axis-validation issues, corrected
before these checks. No hydrodynamically calibrated response or feedback
posterior is claimed by the artificial demonstration.


The projected IA decomposition is implemented in the native projector,
prepared grids and both array adapters. The new EXAMPLE_IA_COMPONENTS
notebook evaluates Y3 triples 111 and 134 and compares all signed pieces
with the ordinary total: maximum closure error divided by the maximum
absolute total is 3.44e-15. Both plots were inspected. The unchanged
ordinary direct projection matches a saved pre-change result bit for bit.
The five new physical checks include the independent equilateral TATT
identity, zero-amplitude limits, leg permutations, both residual scalings,
transform/aperture closure and restoration at four OpenMP workers.

The affected 25 checks pass in both strict and undefined-behavior debug
builds; strict mode is restored. The README render check reports 77 tables,
386 local links and no errors.

The didactic review traced the operator-to-tensor mapping, the field-bit
convention, the distinct baseline/residual source grids and the shared
signed interpolation denominator. The notebook defines each grouping
before plotting it. The active upstream VS vertex and literal manuscript
Eqs.62/67 disagree; the diagnostic retains the implemented vertex and
does not claim that this VS comparison resolves the discrepancy.

The older full-suite and Fisher records below retain their original
implementation identities; they have not been relabeled after this change.


The complete strict collection passes **581 tests** (90.90 seconds).
All three physical Nautilus checks pass with 480 rows and exact restoration
after an IA displacement. The IA/Fisher notebook executed 12 code cells and
produced 13 figures; the original-code comparison executed nine cells and
produced five figures. All eighteen figures were inspected. The covariance
figure now identifies the supplied Y3 covariance explicitly.

Fresh full NLA and TATT bases contain 103 and 121 distinct mean evaluations.
All 31 saved numerical arrays in each match the previous archived bases
bit for bit. This verifies unchanged predictions after repairing integer
model-selector handling in the likelihood. Older records remain under
`runs/pre_likelihood_scalar_fix/` with their original configurations and
identities; none were relabeled. The saved-array comparison is recorded in
`runs/baseline_refresh_comparison.json`.

The four selected-column scans and eight-mean step checks are refreshed
for both models. Both full triangle-grid refinements are also complete.
At h/2, refining all columns changes the w0–wa FoM by +0.170% for NLA
and -0.055% for TATT. The corresponding four-column substitutions change
it by -6.72% and -5.30%. All shared stencil predictions and reconstructed
selected derivatives agree bit for bit: the other nuisance responses
compensate when refined together. Their original h/h2 results differ by
6.08% and 5.64% between parameter steps. Complete h/4 derivatives are
now available as well, requiring only 30 new NLA means and 36 new TATT
means after exact saved-point reuse. The refined h/2-to-h/4 FoM changes
are -0.844% and -0.653%. Every reconstructed original D_h2 matches
bit for bit. Both completed studies pass receipt/identity preflight
with zero additional means required. These are improved step comparisons,
not a complete convergence claim. A small fourth-difference noise score
alone does not establish convergence. The
[Fisher accuracy record](tests/studies/fisher_accuracy_validation.md)
contains the complete comparisons and the saved-array reporting command.

The [ten angular checks](tests/studies/angular_order_results.md) are also
complete. Raising M from 20 to 50 at fixed L=50 has negligible effect on
the tested means. Changing L, angular sampling and coupling interpolation
gives small but nonzero distances; their separate changes partly cancel.
Every two-point row stays bitwise unchanged. These are fiducial checks,
not angular convergence tests of Fisher derivatives.

**The first TATT demonstration Fisher result is not accepted.** Its
prepared projection undersampled squeezed shapes and large Fourier
angles, and cubic interpolation of the unscaled residual produced artificial
high-multipole tails. These errors substantially altered the aperture mean.
The saved initial w0–wa FoM of 12.93 is a diagnostic of that calculation,
not a scientific forecast. The corrected interpolation and source preparation
now have new full NLA and TATT Fisher bases and the completed derivative
refinements above. The earlier speed comparisons establish thread consistency for
their declared coarse calculation; they do not establish the cost of a
converged forecast.

The executed precision-expansion notebook and the fixed-rule quadrature
checks are committed in project `356af5c`. The joint Nautilus examples,
observed Y3 file manifest and signed-domain tests are committed in
`34cecdb`. The core signed-NLA representation is committed in `76c1132`.
Native interpolation and independent source-grid changes are committed in
core `6a7eae7`. The full project collection passes 502 tests in both strict
and debug builds. The later Python source-edge and provenance additions
pass 512 strict checks and 44 affected debug checks; strict is restored.
The current Python saved-result checks and example settings pass the
full 524-test strict collection. Shared plotting changes are committed
in core `9e48fed`; 13 array checks pass, with 30 existing plot modes and
four Fisher display examples rendered and inspected.
A subsequent collection passed 543 strict tests, including the finite
model-distance, full-check provenance and saved-node step-study additions.
The [interpolation study](tests/studies/interpolation_validation.md) reports
the separate table checks and full 80-row mean refinement. Corrected
starting means differ from the finite finer-grid reference by joint
delta chi-squared about 0.024 for both NLA and TATT, with unchanged xi.
These are fiducial mean checks, not derivative convergence.

The [source reconstruction comparison](tests/studies/source_policy_validation.md)
is complete for `dz1`, with 162 full joint means. The final Y3 midpoint
feature is missed by the old 125-point grid; cubic padding to background
z=3.1 would extend it beyond the supplied upper source edge 3.0. The
measured-edge policy states its support and retains native edges. The
demonstration selects PCHIP9601, kernel401 and LOS129, preserving the
source-model qualification. The final source refinement changes the main
derivative by 0.0775% for NLA and 0.0882% for TATT; this does not bound a
marginalized FoM. Distinct table/result paths preserve the rejected files.
Mean and Fisher archives record exact physical-input/code identities at
the beginning and end of a run; accuracy candidates repeat this check
before saving, and executable notebook reuse refuses changed dependencies
or unverified legacy results.

The new bases contain 103 NLA and 121 TATT distinct mean evaluations.
Their w0–wa FoMs are respectively 0.18006 and 0.13630 at step h, and
0.19117 and 0.14405 at h/2. These 6.17% and 5.68% changes prevent a
convergence claim. Both joint and two-point parameter matrices pass the
rank check; the conditional-information identities agree at about
6e-16 or better. The next tests vary numerical settings while keeping
the physical model, covariance, parameters and priors fixed. The shared
noise estimator chooses which derivative columns receive those tests;
complete derivative recalculations remain necessary.
No accepted TATT information gain or completed DES posterior is claimed.

The implementation remains on core `lowrank_fftlog` and project `main`.
Only owned files are committed; the other team's ordinary-core changes
remain outside this work. No remote push is performed.

The earlier performance checkpoint is core `517eb15`. Its ten-point
NLA/TATT CLI comparisons exclude CAMB, preserve bits across worker counts,
and are recorded in [BENCHMARK.md](BENCHMARK.md) and the
[optimization study](benchmarks/2026-11-18/README.md). Those measurements
precede the interpolation corrections. The covariance timing uses a
coarse gravity-only, full-filter, large-area calculation; it is not the
cost of a converged finite-triangle TATT covariance. No further timing
study is running.

The executed precision-expansion notebook teaches separate full-Wishart
and known-component covariance corrections with controlled simulated
vectors. It does not substitute those synthetic vectors for DES mocks.
The installed Nautilus sampling/resume API also passed a separate
normalized two-dimensional Gaussian test. All three physical joint examples pass `--check`: synthetic NLA, synthetic
TATT and observed NLA. Each has 480 rows, responds to an A1 displacement,
and restores the fiducial prediction bit for bit. The observed example has
fiducial chi-squared 1450.7172; it is an execution check with declared analysis
differences, not a reproduced published fit. No full DES posterior has been
sampled.

Observed Y3 aperture data and their matched joint covariance are public
in the pinned upstream repository. The measured-data manifest separates
them from the simulated reference files. Exact published cuts, estimator
weights, compression and finite-mock likelihood choices remain listed in
[PAPER_TARGETS.md](PAPER_TARGETS.md) and
[PAPER_REPRODUCTION_PLAN.md](PAPER_REPRODUCTION_PLAN.md). The demonstration
retains all 400 xi and 80 aperture rows; this is not the published cut or
compressed likelihood. The baryon bispectrum response alone is not a
complete joint power/bispectrum feedback model.

The expanded IA notebook, original-code comparison and Fisher-accuracy
notebook are executed. The accuracy notebook includes the cheap diagnostic,
step comparisons and completed full triangle checks, plus a dated table
recording the additional full h/4 results. Its twelve code cells and nine
figures were executed and inspected; the added table changes prose only.
Fixed-dark-energy
summaries select the posterior Fisher
submatrix before inversion while retaining all other nuisance parameters. The accuracy workflow uses
the shared core estimator and a single full default Fisher base per
declared physical model. Candidate settings evaluate selected derivative
columns; final checks still require full derivatives and both step sizes.
The initial invalid TATT base will not be reused under a new identity.

Tests and physical checks use at most four OpenMP workers and serial
BLAS. Inputs and large numerical outputs remain under ignored local
folders; small teaching outputs are retained in executed notebooks.
[The validation record](tests/VALIDATION.md) distinguishes tests already
run from remaining survey-level accuracy requirements.


The [joint IA source review](tests/studies/joint_ia_conventions.md) finds no
coefficient-convention mismatch. It identifies separate growth/source
interpolation tests and one inaccurate sentence in the `CoreTwoPoint.spectra`
docstring: the ordinary IA loops use Plinear(k,0) with D^4, although the
bridge accepts a full redshift-dependent table. The executable calculation
already has that behavior. The wording correction is recorded while physical
Python files remain frozen for the active Fisher runs.

The IA notebook includes four derivative-shape figures following the
LSST Y1 x DESI FoG note: NLA A1 versus sigma8 and TATT A1 versus A2, for
source auto-bins 1 and 4. Derivatives are multiplied by stated parameter
intervals and divided by diagonal data errors. The response fits retain
all rows and the full covariance; xi-fit coefficients are carried to the
Map3 rows without refitting. The two-parameter overlaps and absolute
residual norms are distinguished from marginalized parameter errors.
The notebook records that a Y6 fiducial requires a separately selected
joint point; its current derivatives remain the declared Y3 calculation.

Validation: the headless IA notebook run completed 13 code cells with
17 figures and no cell errors. All four derivative figures were inspected
after correcting the aperture tick labels. Saved predictions passed their
existing identities; no new physical means or CAMB evaluations were needed.
The subsequent Y6 explanation changed Markdown only and retained every
executed code cell and output unchanged. The command
`python -B -m pytest -q tests/test_derivative_templates.py tests/test_fisher_demo_wrappers.py tests/test_fisher_comparison.py`
passed 13 tests. BLAS and OpenMP each used one thread. The Markdown check
rendered 36 files and verified 384 local links without an error.
