# Comparing with the original codes

1. [Inputs and source identity](#inputs-and-source-identity)
2. [Which operations are compared](#which-operations-are-compared)
3. [Process ownership and outputs](#process-ownership-outputs-and-interpretation)
4. [Executed example](#executed-example)
5. [Credits and checks](#credits-and-checks)

`EXAMPLE_COMPARE_CODES.ipynb` runs small numerical comparisons with the
original Python files. The YAML beside it names the source locations and
numerical controls. The notebook calls
`interface/cosmolike_des_3pcf_notebook_wrappers.py`; the implementation is in
`python/des3pcf/comparison.py`. Plotting comes from the shared
`cosmolike_notebook_utils/3pcf/` package. Notebook cells contain setup and
calls to these functions, not copies of numerical algorithms.

The central distinction is between a live execution and a stored reference.
The transform and aperture cases execute original source files during the
run. The matter and intrinsic-alignment cases execute the C port and read
previously saved original outputs. The result table labels the latter
`frozen_reference`. Neither category establishes a complete publication
reproduction: catalogue measurement, the original CosmoSIS chain, survey
cuts, final covariance choices and numerical convergence remain separate.

## Inputs and source identity

Start from the built `des_3pcf` extension in an activated Cocoa environment.
Run the notebook with this project as its working directory. The example
YAML points to the local read-only snapshots used during the source study.
Another installation should replace those three paths with its own
checkouts. Paths are resolved relative to the YAML file. Setting a path to
`null` reports that source as unavailable; no source is downloaded or
installed by the runner.

| Source | Pinned commit | Executed code |
| --- | --- | --- |
| [Sunao Sugiyama and Rafael Heringer Gomes, fastnc TATT](https://github.com/git-sunao/fastnc/tree/7b2336ce37a909bed51f07f32a2a4f096e71b6ce) | `7b2336ce37a909bed51f07f32a2a4f096e71b6ce` | Public `FastNaturalComponents`, angular coupling, and two-Bessel transform |
| [fastnc main](https://github.com/git-sunao/fastnc/tree/e59eab5a07401d2f484bfe00dd48827f2d2162c7) | `e59eab5a07401d2f484bfe00dd48827f2d2162c7` | Standalone `fastnc/hankel/twobessel.py` |
| [Rafael Gomes, 3pcf_integrator](https://github.com/rchgomes/3pcf_integrator/tree/d6098d326dcb03bc750adcc9627119f52d928944) | `d6098d326dcb03bc750adcc9627119f52d928944` | Standalone `python/fast_map3.py` |

The manifest in `data/code_comparison_manifest.json` records SHA256 hashes
for all 15 imported TATT package files and the two standalone files. A
commit label alone does not establish that a local checkout is unchanged.
The runner checks the bytes before executing any source. Missing or changed
files produce an explicit unavailable result. Original files are not
rewritten, and Python bytecode output is disabled in the child processes.

The full TATT import needs NumPy, SciPy, Astropy, Pandas and mpi4py. The
standalone main transform needs NumPy and SciPy. Rafael's aperture file
needs NumPy. The preflight report names these dependencies and also reports
CosmoSIS and TreeCorr availability. The latter two are not needed for these
standalone comparisons and are not replaced with fake modules. An import
or runtime failure is recorded with its stderr rather than hidden by a
different calculation.

## Which operations are compared

**Two-Bessel transforms.** Both implementations receive identical Fourier
samples, Bessel orders, Mellin biases, logarithmic bin widths and zero
padding. The TATT implementation calculates a Fourier-coefficient taper
but returns the untapered array. The pinned main implementation returns
the tapered array. The C call therefore uses `window=False` for TATT and
`window=True` for main. These are two named comparisons. Calling them both
with the same taper flag would compare different finite operators.

**Natural shear components.** An explicit smooth, symmetric set of
bispectrum multipoles is supplied through the original public
`set_bispectrum` interface. Its radial factor is a Gaussian in
`ELL=sqrt(ell1^2+ell2^2)`, with a symmetric angular factor in `sin(2 PSI)`.
This controlled input isolates the angular coupling, FFTLog, radial bin
average, angular-mode sum and rotation to centroid projection. It is not a
physical matter or IA model. The source runs with `cache=False`,
`auto=False`, fixed Fourier endpoints and no automatic theta-grid tuning.

The original branch obtains negative angular modes by transposing positive
modes. The controlled input obeys the symmetry required by that operation.
The port uses `ordered_legs=False` solely to match this source comparison.
An ordered tomographic TATT bispectrum need not obey this symmetry; its
production calculation uses the independent signed-mode route. This
notebook therefore does not validate an asymmetric physical input through
the historical shortcut.

The primary natural-component comparison matches the source's actual
200-node linear coupling table (`npsi=200`). Source table values are
numerically integrated, whereas C table values are analytic. A separate
`natural_exact_coupling` case removes that interpolation (`npsi=0`); it is
explicitly a different finite discretization. The source metadata says
201 nodes but its actual array contains 200, so the comparison reads the
array length instead of trusting that metadata.

There is also a measurable integration bias in the original coupling
table. Its `aint` function repeatedly refines a left-endpoint rectangle
rule. The default loop stops at 4096 samples, and the coupling class
discards the returned convergence flag. For `L=M=0` and `psi=0`, the
integrand is `4*pi*cos(delta)`, whose integral from zero to pi is zero.
The left-endpoint rule instead gives exactly `4*pi^2/N`, or about 0.00964
at N=4096. The live `coupling_tatt_rectangle_bias` case checks this result
and records a short ladder of 256, 1024, 4096 and 16384 samples using the
integration function's public `max_itern` argument. No source code or
class behavior is replaced. Tightening its tolerance alone does not remove
the default sample cap.

For the 16-point test configuration, the full natural-component difference
is 4.21e-4 when both codes use the 200-node interpolation, and 7.10e-3 when
the C calculation instead uses exact coupling. These are observed
differences between the declared finite operators, not DES accuracy
limits. The 1e-3 source-comparison check allows the demonstrated rectangle
bias in that small configuration. It does not replace the independent C
tests against the defining angular integral or the Gaussian-moment test
of general ordered legs. The example notebook records its own 32-point
results rather than assuming these 16-point values apply.

**Finite aperture integral.** Both routines receive the same complex shear
array on a Cartesian product of two logarithmic side grids and a uniform
opening-angle grid. Angles and filter radii are radians; side widths are
natural-log widths, and the opening-angle width is radians. The input
components are already in the convention expected by the aperture
integral. No extra component swap or overall minus sign is applied to
this controlled input.

The equal-radius result is a source-agreement check. The unequal-radius
result exposes a known original-source discrepancy. In Rafael's adapted
file, `k2.all() == 1` and `k3.all() == 1` check whether all ratios are
nonzero. Positive unequal ratios therefore enter the equal-filter branch.
The unchanged original code and the general C kernel should disagree for
the declared unequal filters. That case is marked `expected_discrepancy`.
It is not used to justify changing the C answer to reproduce the error.
Separate saved references and tests cover the general unequal-filter
expression, its mirror contribution and complex contraction.

**Frozen physical model.** The original matter and 19-component IA values
come from `tests/data_vector/reference_physics_fastnc.json`. Its generator
and source hashes identify their origin. The notebook repeats the fixture's
smooth input spectra and background in the C model, including the named
source-reader mode. The output records the fixture hash. The established
matter tolerance includes the original nonlinear-radius root's stopping
error; the tree and IA tests use tighter checks. A right-triangle ddB zero
has a separate high-precision, vertex-scaled roundoff check. Relative error
on that vanishing value is not a measure of physical accuracy.

## Process ownership, outputs and interpretation

Original-code runs occur in separate serial child processes. The TATT
package's mutable class dictionaries therefore die with the process and
cannot affect a later notebook cell. All BLAS thread environment variables
are set to one before a child imports NumPy. OpenMP uses the YAML value,
restricted to 1 through 4; the example uses 1. No timing result or speedup
claim is produced.

The notebook also keeps a public `threadpoolctl` one-thread controller
alive and checks every already-loaded BLAS pool. Environment variables
alone cannot resize a pool imported by an earlier cell in a reused kernel.
Its plotting setup uses the Cocoa STIX fonts and retina display settings.
Component titles sit above the axes so they do not cover curve extrema.

The runner writes full arrays, plotting slices, stdout, stderr and
`comparison_report.json` to the configured output directory. The report
retains resolved settings, source pins and hashes, execution status and
the distinction between live and frozen references. Source inputs remain
read-only. Unavailable or failed cases remain in the report; selecting one
for plotting raises an explicit error.

For an array `x`, the summary reports
`max(abs(port-reference))/max(abs(reference))`. This global scaled error
is finite when individual entries cross zero. Plot panels can instead use
absolute residuals, retaining the original units and signs. Natural shear
components include both real and imaginary parts. A plotted slice does not
replace the full-array diagnostic stored in the report.

The supplied 32-point Fourier grid and small angular truncations are
controlled examples. Increase resolutions only through a declared
convergence study. Small-grid agreement checks conventions and
implementation; it does not set the numerical error budget for DES Y3 or
Y6.

## Executed example

The default notebook was executed with `nfft=32`, `lmax=3`, `mmax=5`, seven
opening angles, one OpenMP worker and one thread for every BLAS library.
All nine code cells completed. The full-array diagnostics below correspond
to that configuration; the 16-point values above belong to the smaller
automated test. Each value is a dimensionless maximum scaled difference,
not a percentage or a forecast error budget.

| Comparison | Maximum scaled difference | Interpretation |
| --- | ---: | --- |
| TATT untapered two-Bessel transform | 1.36e-14 | Matching finite operator |
| Main tapered two-Bessel transform | 1.47e-14 | Matching corrected operator |
| TATT angular-coupling table | 2.44e-4 | Demonstrated source rectangle bias |
| Natural components, matching 200-node interpolation | 4.34e-4 | Source integration bias remains |
| Natural components, exact C coupling | 3.37e-3 | Different interpolation rule |
| Equal-filter aperture integral | 1.16e-15 | Matching finite operator |
| Unequal-filter aperture integral | 7.58e-1 | Expected original branch error |
| Frozen matter reference | 6.43e-6 | Includes the source fit's root error |
| Frozen IA, linear-power prescription | 4.25e-15 | Stored reference, all 19 terms |
| Frozen IA, nonlinear-power prescription | 1.69e-15 | Stored reference, all 19 terms |

`EXAMPLE_COMPARE_CODES.ipynb` includes the executed outputs from these
cells, including all five figures. A matching execution copy and the
individual PNG files are also written under `outputs/code_comparison/`.
The figures were inspected for labels, signs, residual units and the
visible expected unequal-filter discrepancy.

## Credits and checks

The fastnc implementations are credited to Sunao Sugiyama and Rafael
Heringer Gomes. Rafael's `fast_map3.py` credits Mike Jarvis and TreeCorr
and carries its redistribution notice. This workflow imports those files
from the user's checkouts; it does not distribute modified copies.

`tests/test_code_comparison.py` checks source-hash refusal, missing-source
reporting, thread settings and real execution of the available original
files. It checks equal-filter agreement and requires the unequal-filter
case to expose a difference. Source-dependent tests skip with a named
reason when the explicit checkouts or required packages are absent.
