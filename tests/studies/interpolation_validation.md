# Interpolation and source checks for the Y3 demonstration

This page records the corrected interpolation settings of the projected
bispectrum and multipole tables, the measured changes they produce on the
full aperture vector, and the measured support of the input sources. A
stable thread comparison establishes that workers compute the same stated
approximation, not that the approximation is physically accurate. The
earlier timing results measure their recorded coarse calculation, not a
converged forecast.

1. [The rejected first Fisher result](#rejected)
2. [Projected bispectrum](#bispectrum)
3. [Multipole interpolation](#multipole)
4. [All 80 finite-SAS aperture rows](#rows)
5. [Measured source support](#source)
6. [Execution record](#record)

## The rejected first Fisher result <a name="rejected"></a>

The first full TATT demonstration gave a marginalized $`w_0`$–$`w_a`$
figure of merit of 12.929976, compared with 0.185420 for NLA. **That TATT
result is rejected as a scientific forecast.** The raw residual
interpolation generated large artificial aperture contributions, and the
$`\Omega_m`$ derivative changed by about 38.65% between the two
finite-difference steps in a covariance-weighted norm; a norm mixing
parameters with different units hid this problem. Neither the apparently
large figure of merit nor its ratio measures additional cosmological
information.

This failure explains the design of the corrected tables below: the
projected residual is interpolated as a signed, amplitude-scaled quantity
on a concentrated shape chart, never as the raw signed residual. The
corrected Fisher bases built on these settings are in
[the Fisher accuracy record](fisher_accuracy_validation.md).

## Projected bispectrum <a name="bispectrum"></a>

Two approximations failed independently. First, cubic interpolation of
the raw signed residual along log perimeter generated a high-multipole
tail: at an equilateral triangle with each $`\ell`$ equal to 100000, the
prepared residual was about 488 times its direct line-of-sight value.
Dividing by a positive equilateral gravitational amplitude before
interpolation reduced this ratio to about 1.0025 in the same diagnostic.
The residual keeps its sign; the physical IA coefficients stay unchanged.

Second, the squeezed-shape support was not resolved. For Fourier bounds
$`\ell`$ from 0.001 to 300000, the smallest side fraction was about
$`8.33\times10^{-10}`$, while the first interior fraction with 80 shape
nodes and stretch 3 was about $`2.03\times10^{-4}`$. This large first
interval cannot resolve arbitrary queries merely because its endpoint
exists. Increasing the concentration to stretch 10.1062201 and testing
160 then 320 nodes removed the large aperture contamination in the tested
source triplet: at aperture radius 7 arcmin in the first bin, the raw and
corrected predictions were about $`3.79\times10^{-8}`$ and
$`5.23\times10^{-10}`$, and the NLA value was about
$`5.20\times10^{-10}`$.

The shape chart keeps two limits. Its rectangular table uses an
off-triangle continuation, which can influence nearby physical queries,
and nearly collapsed or flattened triangles can amplify floating-point
uncertainty when physical sides convert back to chart coordinates. The
[extreme-edge diagnostic](stretched_edge_conditioning.md) measures that
limitation.

## Multipole interpolation <a name="multipole"></a>

The stage diagnostic (the retired `multipole_stage2_diagnostic.py`,
kept in Git history) holds the
projected bispectrum and angular integration samples bitwise fixed and
integrates each linear angular segment against its Legendre polynomial
with an independent Gauss rule exact for that product. The comparison
therefore tests interpolation of the prepared multipoles alone.

The source triplet is (0, 1, 0). The test Fourier magnitudes
$`\sqrt{\ell_1^2+\ell_2^2}`$ are 100, 1000, 10000 and 100000, with angles
0.17, 0.43 and 0.68 radians in both ordered-leg
charts. The reported error is divided by one half the integral of the
absolute piecewise-linear bispectrum, not by a potentially vanishing
multipole.

| Angular representation and refinement | Largest relative error, baseline part | Largest relative error, residual part |
|---|---:|---:|
| Uniform log angle, 80 nodes | 1.12927 | 1.38064 |
| Mixed coordinate, scale 0.02, 80 nodes | 0.07564 | 0.11900 |
| Mixed coordinate, radial count 127 | 0.01781 | 0.03300 |
| Mixed coordinate, angular count 159 | 0.07562 | 0.11908 |
| Mixed coordinate, dense factor 4 | 0.01551 | 0.03410 |

The mixed coordinate is $`\log\psi+\psi/\mathrm{scale}`$. It allocates
more samples to large physical angles without introducing a binary search
into a repeated read. Once the angular gaps are removed, scale
interpolation and its dense reader limit these comparisons; the
table does not establish accuracy at every Fourier triangle.

## All 80 finite-SAS aperture rows <a name="rows"></a>

The mean study (the retired `projection_mean_convergence.py`, kept in
Git history) evaluates every aperture
row for the NLA and TATT fiducial points. Sources keep their original
125-point preparation in this test, and the finite-triangle geometry,
FFT, angular integration and public joint covariance stay fixed. Only
projection and multipole interpolation change; no CAMB run is needed.

The corrected starting setting uses the signed baseline and scaled
residual, shape stretch 10.1062201, 160 shape nodes and mixed-angle scale
0.02. The projection refinement changes (nr, nu, nv, np, nshape) from
(32, 24, 24, 32, 160) to (63, 47, 47, 63, 319). The multipole refinement
changes (nell, npsi, dense factor) from (64, 80, 2) to (127, 159, 4).
Differences below compare each row's setting against the setting with
both refinements applied.

| Changed interpolation stage | $`\Delta\chi^2`$, NLA aperture block | $`\Delta\chi^2`$, NLA joint vector | $`\Delta\chi^2`$, TATT aperture block | $`\Delta\chi^2`$, TATT joint vector |
|---|---:|---:|---:|---:|
| Corrected starting setting | 0.008357 | 0.023693 | 0.008315 | 0.023696 |
| Multipoles refined | 0.011900 | 0.030557 | 0.011809 | 0.030307 |
| Projection refined | 0.005327 | 0.016699 | 0.005341 | 0.016764 |

The aperture-block columns use the marginal 80-by-80 covariance block.
The joint columns insert the same aperture error into the full 480-row
vector, with zero error in its xi entries, and use the complete
covariance inverse; the cross covariance makes that joint norm more
restrictive, so the inverse of the aperture block cannot replace it. The
joint columns come from the already saved means and required no
additional physical evaluations.

These differences are small in the covariance norms at this fiducial
point, and their nonmonotonic ordering shows that two errors can partly
cancel. The finest case is a comparison reference, not an exact solution;
derivatives, Fourier limits, angular caps, source sampling and
measurement-estimator agreement keep their own checks.

## Measured source support <a name="source"></a>

The input source FITS ends at a high-redshift bin with upper edge 3.0.
Every source bin has an enhanced final midpoint at 2.995, whose density
times the 0.01 bin width is approximately 0.001461, 0.001738, 0.000242
and 0.001748 for the four source distributions. The old uniform 125-point
grid misses this final feature: replacing its density with the preceding
sample leaves every old sampled value bitwise unchanged.

A smooth reconstruction must not silently spread this final value from
2.995 to the background limit 3.1: background integration support and
measured source support are different quantities. The native midpoint
samples and bin edges stay identifiable, including the zero observer row
whose stored lower edge is negative, and physical source support
intersects the input edges with nonnegative redshift.

An independent fine linear source grid preserves the original
midpoint-curve convention. A cubic reconstruction with explicitly stated
support is a separate model assumption and does not automatically
preserve histogram-bin masses. Both choices feed the source and
prepared-kernel refinements and the photo-z derivative checks in
[the source study](source_policy_validation.md). The demonstration FITS
already contains the authorized clipping and renormalization, so its
native samples are not the untouched upstream file.

## Execution record <a name="record"></a>

The versioned records are under [`results/`](results):
`projection_stretched_diagnostic.npz`, `residual_mean_diagnostic.npz`,
`multipole_mixed_diagnostic.npz` and `projection_mean_convergence/`.
Both producing drivers were retired with the private prediction route,
so these tables can no longer be regenerated; the saved records are the
preserved evidence.
Tests and models used no more than four OpenMP workers and serial BLAS,
and no new timing trial ran. The corrected native code passed the strict
and debug collections; [the validation record](../VALIDATION.md) keeps
the test totals.
