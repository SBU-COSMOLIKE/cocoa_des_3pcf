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
