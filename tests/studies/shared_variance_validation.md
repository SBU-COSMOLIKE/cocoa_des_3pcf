# Shared variance integration checks

The optional shared route supplies copied smoothing results to the C fit,
the Python BiHalofit producer and analytic covariance. The local reference
remains selected by default. No stored survey reference was regenerated.

The strict private extension and its debug/UBSan counterpart each passed
123 targeted checks: shared snapshot ownership and accuracy controls,
Python/C BiHalofit equations, Cobaya provider integration and CLI mean,
Fisher and covariance calculations. The latter include local/shared
variance and internal/supplied bispectrum combinations. A later
configuration-only check also verifies that evolving growth cannot be
mislabelled as single_fluid_effective_growth. The private extensions were
built from the shared sampling implementation; subsequent core changes in
this stage changed comments and documentation only.

The spectrum in the component tests is controlled, not a survey model.
They verify exact cache restoration, retained old outputs, correct units,
explicit factorized/evolving input selection and preservation of the
supplied power amplitude. The equation comparison retains the existing
2e-11 relative allowance for C/Python arithmetic. A separate sigma8 scalar
change with unchanged P and normalized D leaves the shared result fixed.
Covariance checks preserve the declared mass/redshift axes and smoothing
density, and saved files retain separate fit and halo input records.

These checks do not authorize replacing the local prescription. The
acceptance comparison must retain the full joint covariance and all rows,
score the mean change at delta chi-squared <= 0.01, require every whitened
derivative relative change <= 1e-3, and bound marginalized-error and
parameter-pair FoM changes by 0.5%. Repeat with finer shared sampling and
both derivative steps. A failed criterion must be explained before the
local duplicates are retired. Evolving versus factorized power is a
separate physical comparison, not an accuracy refinement.

The first full NLA comparison fails these acceptance criteria. It retains
400 two-point and 80 aperture rows, 17 parameters, the same covariance,
and two derivative spacings. With all shared sampling controls doubled,
the mean difference is 2.1594e-4 in delta chi-squared; all two-point rows
at all 103 evaluated points remain bitwise identical. At the larger step,
the maximum whitened derivative change is 0.2822%, marginalized-error
change 4.0321%, and pair-FoM change 5.2781%. At the smaller step these are
0.2510%, 2.0651%, and 2.3896%. Refining shared sampling from baseline to
twice baseline changes a pair FoM by up to 0.5953%, also above the 0.5%
criterion. The shared route therefore remains optional. A later
investigation resolved this discrepancy; the dated continuation below
records it.

Two effects require separate investigation. The legacy calculation
renormalizes the input power spectrum to its scalar sigma8; its fiducial
power multiplier is about 0.999686. Shared smoothing preserves the input
power amplitude. The old calculation also uses a finite root-solving
tolerance, and its Fisher result changes when the derivative step halves.
Archived CAMB arrays permit testing finer smoothing and root tolerances
without changing the input cosmology or repeating the Boltzmann solves.
These are hypotheses to test, not a justification for relaxing acceptance.

## Diagnosis and current standing (2026-11-19) <a name="diagnosis"></a>

Both hypotheses were tested, and the discrepancy traces to the
normalization difference together with the CAMB transfer sampling behind
it. As a diagnostic, the comparison retained the old sigma8
renormalization as an explicit multiplier on the shared route's linear
power; the factor serves only this test and never enters the production
adapter. At four-times shared sampling that multiplier brought the NLA
comparison within 0.014% per derivative and 0.172% per parameter-pair
FoM of the refined local reference, at both derivative steps. Without
the multiplier, the FoM discrepancy persisted.

Refining the input spectrum removes the multiplier instead of adopting
it. CAMB AccuracyBoost 2 with k_per_logint 50, the transfer samples per
logarithmic wavenumber interval, and 1025 uniform log-k output samples
bring the multiplier within 1.5e-7 of unity. No multiplier therefore
enters production; the shared route keeps the supplied power amplitude.

On these refined inputs the acceptance comparisons pass. The full NLA
comparison passes at four and eight times shared sampling. The full TATT
comparison at four-times shared sampling gives maximum derivative,
marginalized-error and pair-FoM differences of 0.00826%, 0.0541% and
0.0581%, with a mean delta chi-squared of 1.55056e-8 and bitwise
identical two-point rows. There is no eight-times TATT refinement result
yet. This page narrates the investigation; the authoritative current
numbers are the acceptance statement in
[shared smoothing controls](../../../../external_modules/code/cosmolike_core/cosmolike/3pcf/README.md#shared-smoothing).

These passes establish method agreement at the tested inputs and
resolutions, not a forecast guarantee. Full paired forecasts remain
necessary before any default change, and the local route remains the
default.
