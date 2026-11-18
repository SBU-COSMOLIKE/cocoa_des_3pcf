# Physical covariance and contraction checks

These tests check covariance construction separately from the mean,
transform and aperture tests in `../data_vector/`. A covariance describes
the joint scatter of measurements; getting each predicted mean right does
not by itself validate their covariance.

1. [What the references measure](#measures)
2. [The physical references](#reference)
3. [The test files](#files)
4. [Running covariance checks](#running)

## What the references measure <a name="measures"></a>

Every independent reference in this folder answers one scientific
question about the C implementation:

| Reference | Scientific question |
|---|---|
| White-noise filtered-pair integral and fifteen Gaussian pairings | Are Fourier measures and Gaussian contraction multiplicities correct? |
| Direct NFW profiles, halo variance and abundance derivatives | Are mass definitions, halo normalization and the choice of response terms correct? |
| Separately assembled PB, BB, PT and connected terms | Do shared geometry and source factors preserve the covariance equations of the model? |
| Analytic annular noise overlap | Do xi spin factors, E/B noise and exact angular edges give the expected noise covariance? |
| Full joint Schur decomposition and derivative examples | Does the forecast preserve cross covariance and parameter priors? |

## The physical references <a name="reference"></a>

For constant angular noise and Gaussian aperture filters, the filtered
pair integral has an analytic polynomial times a Gaussian. Comparing
against it checks the angular Fourier normalization, the radial measure,
unequal filter radii, zero separation and the sign change at larger
separation. The test uses 192 Gauss–Legendre nodes on a finite multipole
interval it states explicitly; the omitted filter tail is negligible
for its inputs.

The cubic-map test compares the Gaussian covariance with an independent
enumeration of all fifteen pairings of six positions. Repeated source
labels remain distinct positions so their multiplicities are preserved.
The one-field identity $`6K(r)^3+9K(0)^2K(r)`$ supplies another check.
These tests concern raw filtered maps; they do not derive the noise
contractions for an estimator that excludes repeated galaxies.

The connected-contraction test compares its result with direct matrix
multiplication.
A deliberately dependent response bank tests a rank-deficient matrix
with negative cross entries and nonnegative variances. This separates
positive semidefiniteness from invertibility and prevents treating an
isolated connected term as a complete likelihood covariance.

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

## The test files <a name="files"></a>

| Test file | Calculation checked |
|---|---|
| `test_components.py` | Filtered Gaussian pairs against the analytic polynomial-times-Gaussian integral, and all fifteen Wick pairings of six positions. |
| `test_physical_covariance.py` | The physical model around those contractions: direct radial NFW transform, halo variance and abundance/bias derivatives, assembled Gaussian E/B spin sums, PB and BB integrals, unfactored PT and connected one-halo terms, the xi white-noise rules, and xi-only requests against the matching joint submatrix. |
| `test_pt_contraction_reuse.py` | The nine P-field choices shared by one-halo P-times-T covariance terms. |
| `test_large_area_reuse.py` | Which prepared arrays actually enter the large-area formulas: an unused shear-profile grid or changed halo-center offset nodes must leave every term unchanged, bit for bit in both adapters. |
| `test_setting_sequences.py` | Covariance setting changes in one Python process, compared bit for bit with separately initialized interpreters. |

Equal aperture radii and repeated xi bins test the exact sharing of
geometry across different source labels. For a quadratic-only request,
the halo response is a direct Fourier integral, so halo-center offset
nodes cannot enter it; cubic rows still need their radial profiles, and
the tests do not remove that physical dependence. The long-mode
survey-window integral remains present in the large-area formulas.

## Running covariance checks <a name="running"></a>

Use the [test environment](../README.md#running): BLAS has one thread,
OpenMP has at most four, and only one test process runs at a time. Set the
environment before Python starts; child processes inherit it.

**Step :one:**: run from an activated `Cocoa/` directory.

```bash
python -m pytest -q projects/des_3pcf/tests/covariance
```

The tests run both adapters, repeated calls and several worker counts.
They use no halo catalogue, survey mask or external covariance file.
Agreement validates these calculations for their controlled inputs.
A survey covariance still requires matched estimators, its catalogue
inputs and refinement of every integral contributing to the final matrix.
