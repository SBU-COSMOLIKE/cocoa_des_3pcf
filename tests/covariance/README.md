# Physical covariance and contraction checks

1. [Physical references](#reference)
2. [Running covariance checks](#running)

## Physical references <a name="reference"></a>

For constant angular noise and Gaussian aperture filters, the filtered
pair integral has an analytic polynomial times a Gaussian. Comparing
against it checks the angular Fourier normalization, the radial measure,
unequal filter radii, zero separation and the sign change at larger
separation. The test uses 192 Gauss–Legendre nodes on a stated finite
multipole interval whose omitted filter tail is negligible for its inputs.

The Gaussian cubic-map covariance is compared with an independent
enumeration of all fifteen pairings of six positions. Repeated source
labels remain distinct positions so their multiplicities are preserved.
The one-field identity $`6K(r)^3+9K(0)^2K(r)`$ supplies another check.
These tests concern raw filtered maps; they do not derive the noise
contractions for an estimator that excludes repeated galaxies.

The connected contraction is compared with direct matrix multiplication.
A deliberately dependent response bank tests a rank-deficient matrix
with negative cross entries and nonnegative variances. This separates
positive semidefiniteness from invertibility and prevents treating an
isolated connected term as a complete likelihood covariance.

`test_physical_covariance.py` checks the physical model around those
contractions. Its references include a direct radial NFW transform,
halo variance and abundance/bias derivatives, separately assembled
Gaussian E/B spin sums, PB and BB integrals, and unfactored PT and
connected one-halo calculations. Equal aperture radii and repeated xi
bins test the exact sharing of geometry across different source labels.
Requests containing only xi rows are compared with the corresponding
submatrix of a joint request.

The xi white-noise check counts independent galaxy pairs in continuous
annuli, including overlapping and disjoint bins, tomographic noise
products and the cancellation between xi-plus and xi-minus. It tests
both adapters and checks that the analytic NN treatment retains the
signal-signal and signal-noise terms. The finite-band and analytic-bin
modes have explicit refusal checks for incompatible settings.

An independent direct-Bessel halo-shear calculation checks the two-stage
profile interpolation. Coarse-node and dense-read refinements are varied
separately from the fixed physical quadrature. The
[physical guide](../../../../external_modules/code/cosmolike_core/cosmolike/3pcf/COVARIANCE.md)
records the controlled refinement results and their limits.

## Running covariance checks <a name="running"></a>

**Step :one:**: run from an activated `Cocoa/` directory.

```bash
python -m pytest -q projects/des_3pcf/tests/covariance
```

The tests run both adapters, repeated calls and several worker counts.
They use no halo catalogue, survey mask or supplied covariance file.
Agreement validates these calculations for the stated controlled inputs.
A survey covariance still requires matched estimators, its catalogue
inputs and refinement of every integral contributing to the final matrix.
