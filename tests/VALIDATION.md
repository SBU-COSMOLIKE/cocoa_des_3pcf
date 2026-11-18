# Initial component validation record

This is the historical foundation-only record. The current implementation
also contains physical projection, transforms and joint covariance. Its
executed interim checks and outstanding revisions are listed in
[CHECKPOINT.md](../CHECKPOINT.md); the full strict/debug record is pending.

The initial component milestone was checked on 2026-11-17. This record
concerns the C files in `cosmolike/3pcf/`, both C++ adapters and this
project's isolated build. It does not certify a complete cosmological
prediction or survey covariance.

| Environment | Value |
|---|---|
| Platform | Darwin, arm64 |
| C/C++ compiler | Cocoa environment Clang 19.1.7 |
| Python | 3.11.12 |
| NumPy | 1.26.3 |
| Parallel checks | 1, 2, 4 and 8 OpenMP workers; OpenBLAS fixed to one |
| Strict build | `-O3`, native instruction selection, strict floating-point flags |
| Debug build | `-O0`, undefined-behavior and floating-point-division sanitizers; strict arithmetic and OpenMP retained |

## Commands and outcomes

The selected Cocoa environment supplied `CONDA_PREFIX`, `C_COMPILER`,
`CXX_COMPILER` and the Python executable. Each build used one compilation
worker. From `projects/des_3pcf/interface/`, the build commands were:

```bash
make -f MakefileCosmolike -j1
COSMOLIKE_DEBUG_MODE=1 make -f MakefileCosmolike -j1
```

The test command below was run from the directory containing `Cocoa/`,
after each selected build. No survey calculation or benchmark was run.

```bash
Cocoa/.local/bin/python -B -m pytest -q -p no:cacheprovider Cocoa/projects/des_3pcf/tests
```

| Check | Outcome |
|---|---|
| Strict component build | Exit 0 |
| Strict component and input-contract tests | 37 passed, exit 0 |
| Debug component build | Exit 0 |
| Debug component and input-contract tests | 37 passed, exit 0; no sanitizer diagnostic |
| Default build restored | Strict extension retained in `interface/` |

The first build exposed a missing NumPy header include for CARMA. The
Makefile now obtains that include directory from the selected Python's
installed NumPy. No dependency was installed or upgraded.

## What the numerical comparisons cover

The saved aperture weights use a verified external fastnc source hash,
with equal-radius and unequal-radius branches. The C code's cyclic
kernel agrees within relative tolerance $`2\times10^{-12}`$ and absolute
floor $`2\times10^{-15}`$ on the stored geometry. The contraction checks
use one through four filter outputs, five independent input samples,
real and imaginary components, and direct complex matrix multiplication.

The filtered-pair comparison uses the analytic white-noise Gaussian-filter
integral, including zero separation, unequal radii and signed large-radius
correlations. Gaussian covariance is checked against independent
enumeration of the fifteen pairings, plus the one-field expression.
The response contraction agrees with direct matrix multiplication and
retains an intentionally rank-deficient positive-semidefinite limit.

Both adapters return the documented axis order. Tests check that notebook
copies accept strided, read-only and unaligned views without changing
them, while production rejects incompatible dtype or alignment. Stored
results survive subsequent calls. Repeated calculations of every C
component agree bit for bit across the tested worker counts.

Malformed-input checks run in child processes and require the intended
diagnostic and exit status one. They cover degenerate and unrepresentable
aperture geometry, invalid widths, nonfinite inputs and outputs, negative
Gram weights, invalid field indices, mismatched triangle dimensions,
asymmetric Gaussian pair functions and unaligned borrowed buffers. A
loader failure or unrelated Python traceback cannot count as a successful
rejection test.

## Independent scientific and didactic review

Separate reviewers read the aperture algebra, the C++ ownership and
layout conversions, and the covariance contractions. The review found
and resolved three concrete boundary issues: centroid underflow before
division, unaligned but contiguous NumPy buffers, and the symmetry
assumption required before reflecting Gaussian covariance entries.
The rejection tests exercise those conditions.

The didactic review checked the complete component for an advanced
undergraduate reader. Function headers and loop comments define the
physical quantity, measures, units, array axes and numerical steps.
SIMD comments explain what each lane computes and why worker writes are
independent. The README distinguishes E/B leakage, finite quadrature,
raw-map Gaussian covariance and the still-missing catalogue estimator.
Documentation rendering, local links, anchors and prohibited vocabulary
are checked separately from the numerical results.

The documentation pass rendered 60 Markdown files, including the original
study, its portable core copy and the updated guides. It checked 568 local
links and 28 anchors; the study and new guides rendered 80 tables. The
prohibited-word and expression scan found no matches in the new prose,
comments or docstrings. The final source-format check found no tabs,
trailing whitespace or lines beyond the applicable C/Python limits.

## Remaining limits

The geometry reference covers moderate side and radius ratios. The
guards reject examples outside double-precision representability; they
do not establish near-collinear accuracy for every finite input.
There is no AddressSanitizer run in this record; the debug build uses the
sanitizers listed above. Linux compilation and processor comparisons
remain untested.

The existing two-point data-vector C files were not edited or rebuilt.
Their large likelihood regressions were not run for this isolated
component build. No source distribution, bispectrum provider, FFTLog
natural-component transform, TATT projection, survey mask, mixed
non-Gaussian covariance, full two-point cross-block or Fisher analysis
is certified by these checks. There are no timing or scaling claims.
