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

The core checkpoint is `98824eb`, following `d832c5f` and `8c58ace`. Checks ran on
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
| Saved covariance comparison before and after omitting unused large-area transforms | All 16 arrays match bitwise; both adapters, joint/xi-only and large-area/finite-window |
| Large-area profile and halo-offset independence checks | Six passed; included in the 281 checks |
| Shared plotting array checks | 13 passed; separate from the project checks |
| Shared plotting render review | 30 modes inspected; four teaching figures retained in the core README |
| Original-code notebook executed with the strict checkpoint | Nine code cells completed, five figures, no cell errors |
| IA/Fisher notebook executed with the strict checkpoint | Eleven code cells completed, three figures, no cell errors; survey calculations disabled |

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
The debug results above precede the final physical-tail and covariance changes;
they do not yet establish debug validation of checkpoint `98824eb`.

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
