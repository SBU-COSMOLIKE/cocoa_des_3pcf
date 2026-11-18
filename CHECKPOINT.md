# Work in progress: three-point implementation

**The first TATT demonstration Fisher result is not accepted.** Its
prepared projection undersampled squeezed shapes and large Fourier
angles, and cubic interpolation of the unscaled residual produced artificial
high-multipole tails. These errors substantially altered the aperture mean.
The saved initial w0–wa FoM of 12.93 is a diagnostic of that calculation,
not a scientific forecast. The corrected interpolation and source preparation
now have new full NLA and TATT Fisher bases. Further derivative refinement
is underway. The earlier speed comparisons establish thread consistency for
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
normalized two-dimensional Gaussian test. Physical joint example
preflight checks remain to be rerun after the source choice is settled;
no full DES posterior has been sampled.

Observed Y3 aperture data and their matched joint covariance are public
in the pinned upstream repository. The measured-data manifest separates
them from the simulated reference files. Exact published cuts, estimator
weights, compression and finite-mock likelihood choices remain listed in
[PAPER_TARGETS.md](PAPER_TARGETS.md) and
[PAPER_REPRODUCTION_PLAN.md](PAPER_REPRODUCTION_PLAN.md). The demonstration
retains all 400 xi and 80 aperture rows; this is not the published cut or
compressed likelihood. The baryon bispectrum response alone is not a
complete joint power/bispectrum feedback model.

The expanded IA and Fisher-accuracy notebooks are source drafts awaiting
new accepted numerical inputs and execution. The accuracy workflow uses
the shared core estimator and a single full default Fisher base per
declared physical model. Candidate settings evaluate selected derivative
columns; final checks still require full derivatives and both step sizes.
The initial invalid TATT base will not be reused under a new identity.

Tests and physical checks use at most four OpenMP workers and serial
BLAS. Inputs and large numerical outputs remain under ignored local
folders; small teaching outputs are retained in executed notebooks.
[The validation record](tests/VALIDATION.md) distinguishes tests already
run from remaining survey-level accuracy requirements.
