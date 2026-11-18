# Interpolation and source checks for the Y3 demonstration

This record separates failures in the initial demonstration from tests of
their proposed corrections. A stable thread comparison establishes that
workers compute the same declared approximation. It does not establish
the approximation's physical accuracy. The earlier timing results remain
measurements of their recorded coarse calculation; they are not timings
of an accepted, converged forecast.

## Initial Fisher result

The first full TATT demonstration gave a marginalized w0–wa FoM of
12.929976, compared with 0.185420 for NLA. **The TATT result is rejected as
a scientific forecast.** The raw residual interpolation generated large
artificial aperture contributions. Its Omega-m derivative also changed
by about 38.65% between the two finite-difference steps in a
covariance-weighted norm. A norm mixing parameters with different units
hid this problem. Neither the apparently large FoM nor its ratio can be
used to infer additional cosmological information.

The following tests explain the failure before another default Fisher is
computed. They do not yet establish derivative or physical-model
convergence for either IA model.

## Projected bispectrum

Two approximations failed independently. First, cubic interpolation of the
raw signed residual along log perimeter generated a high-multipole tail.
At an equilateral triangle with each ell equal to 100000, the prepared
residual was about 488 times its direct line-of-sight value. Dividing by a
positive equilateral gravitational amplitude before interpolation reduced
this ratio to about 1.0025 in the same diagnostic. The residual keeps its
sign; there is no clipping or change to the physical IA coefficients.

Second, the declared squeezed-shape support was not resolved. For bounds
ell=0.001 to 300000, the smallest fraction was about 8.33e-10, while the first
interior fraction with 80 shape nodes and stretch 3 was about 2.03e-4. This
large first interval cannot resolve arbitrary queries merely because its
endpoint exists. Increasing the concentration to stretch 10.1062201 and
testing 160 then 320 nodes removed the large aperture contamination in the
tested source triplet. The raw and corrected first-bin radius 7 predictions
were about 3.79e-8 and 5.23e-10, respectively; the NLA value was about 5.20e-10.

The shape chart still has two limits. Its rectangular table uses an
off-triangle continuation, which can influence nearby physical queries.
Also, nearly collapsed or flattened triangles can amplify floating-point
uncertainty when physical sides are converted back to chart coordinates.
The [extreme-edge diagnostic](stretched_edge_conditioning.md) retains that
measured limitation. Interior knot tests do not settle it. Direct physical
queries, endpoint-cutoff changes and actual estimator refinement are needed.

## Multipole interpolation

[The stage diagnostic](multipole_stage2_diagnostic.py) holds the projected
bispectrum and angular integration samples bitwise fixed. It integrates
each linear angular segment against its Legendre polynomial with an
independent Gauss rule exact for that product. The comparison therefore
tests interpolation of the prepared multipoles, rather than the physical
bispectrum or the angular endpoint cutoff.

The source triplet is (0, 1, 0). Test scales are 100, 1000, 10000, 100000, with
small angles 0.17, 0.43, 0.68 radians in both ordered-leg charts. The reported
error is divided by one half the integral of the absolute piecewise-linear
bispectrum. It is not divided by a potentially vanishing multipole.

| Angular representation and refinement | Maximum baseline error | Maximum residual error |
|---|---:|---:|
| Uniform log angle, 80 nodes | 1.12927 | 1.38064 |
| Mixed coordinate, scale 0.02, 80 nodes | 0.07564 | 0.11900 |
| Mixed coordinate, radial count 127 | 0.01781 | 0.03300 |
| Mixed coordinate, angular count 159 | 0.07562 | 0.11908 |
| Mixed coordinate, dense factor 4 | 0.01551 | 0.03410 |

The mixed coordinate is log(psi)+psi/scale. It allocates more samples to
large physical angles without introducing a binary search into a repeated
read. Once the angular gaps are removed, scale interpolation and its dense
reader limit these selected comparisons. The table does not establish
accuracy at every Fourier triangle.

## All 80 finite-SAS aperture rows

[The mean study](projection_mean_convergence.py) evaluates every aperture
row for the NLA and TATT fiducial points. Sources retain their original
125-point preparation in this test. The finite-triangle geometry, FFT,
angular integration and public joint covariance are fixed.
Only projection and multipole interpolation change; no CAMB run is needed.

The corrected starting setting uses signed baseline and scaled residual,
shape stretch 10.1062201, 160 shape nodes and mixed-angle scale 0.02. The
projection refinement changes (nr, nu, nv, np, nshape) from
(32, 24, 24, 32, 160) to (63, 47, 47, 63, 319). The multipole refinement changes
(nell, npsi, dense factor) from (64, 80, 2) to (127, 159, 4). Differences below are
against the setting with both refinements applied.

| Changed interpolation stage | NLA, aperture alone | NLA, joint vector | TATT, aperture alone | TATT, joint vector |
|---|---:|---:|---:|---:|
| Corrected starting setting | 0.008357 | 0.023693 | 0.008315 | 0.023696 |
| Multipoles refined | 0.011900 | 0.030557 | 0.011809 | 0.030307 |
| Projection refined | 0.005327 | 0.016699 | 0.005341 | 0.016764 |

Every entry is a delta chi-squared. The aperture-only columns use the
marginal 80-by-80 covariance block. The joint columns insert the same aperture
error into the full 480-row vector, with zero error in its xi entries, and
use the complete covariance inverse. Cross covariance makes that joint
norm more restrictive; it cannot be replaced by the inverse of the
aperture block. The joint columns were calculated from the already saved
means and required no additional physical evaluations.

These differences are small in the selected covariance norms at this
fiducial point. Their nonmonotonic ordering also shows that two errors
can partly cancel. The finest case is a comparison reference, not an exact
solution. Derivatives, further refinement, Fourier limits, angular caps,
source sampling and exact measurement-estimator agreement remain separate
checks before accepting a Fisher result.

## Measured source support

The supplied source FITS ends at a high-redshift bin with upper edge 3.0.
Every source bin has an enhanced final midpoint at 2.995. Its density times
the supplied 0.01 bin width is approximately 0.001461, 0.001738, 0.000242 and
0.001748 for the four source distributions. The old uniform 125-point grid
misses this final feature: replacing its density with the preceding sample
leaves every old sampled value bitwise unchanged.

A smooth reconstruction must not silently spread this final value from
2.995 to the background limit 3.1. Background integration support and measured
source support are different quantities. The native midpoint samples and
bin edges must remain identifiable, including the zero observer row whose
stored lower edge is negative. Physical source support intersects the
supplied edges with nonnegative redshift.

An independent fine linear source grid preserves the declared original
midpoint-curve convention. A cubic reconstruction with explicitly stated
support is a separate model assumption; it does not automatically preserve
histogram-bin masses. Both require source and prepared-kernel refinement,
and checks of the photo-z derivatives. The demonstration FITS already
contains the authorized clipping and renormalization; retaining its native
samples does not mean that those samples are the untouched upstream file.

## Execution record

The intermediate arrays and JSON records are under the ignored local
`runs/` directory: `projection_stretched_diagnostic.npz`,
`residual_mean_diagnostic.npz`, `multipole_mixed_diagnostic.npz`, and
`projection_mean_convergence/`. Tests and models use no more than four
OpenMP workers and serial BLAS. No new timing trial was performed.
The corrected native code is being checked in strict and debug builds;
the final test totals belong in the project validation record.
