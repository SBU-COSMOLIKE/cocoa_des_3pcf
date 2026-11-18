# DES two-point and three-point calculations

1. [Purpose and current scope](#scope)
2. [Build the compiled library](#build)
3. [Prepare and evaluate a declared data vector](#evaluate)
4. [Notebook examples](#notebooks)
5. [Covariances, Fisher calculations and tests](#tests)
6. [Survey inputs and credits](#targets)

## Purpose and current scope <a name="scope"></a>

This project builds and tests the three-point code in
`cosmolike_core/cosmolike/3pcf/`. It follows the LSST Y1 project layout:
the core owns numerical C routines and the production/notebook C++
adapters, while this project owns the compiled-module registration and
tests. Existing LSST Y1 files are not used as editable dependencies.

The compiled calculation predicts BiHalofit matter bispectra, the selected
TATT and velocity-shear intrinsic-shape terms, angular shear correlations,
and third-order aperture moments. It also constructs the declared analytic
joint covariance. Python prepares physical tables, reads survey files and
forms Fisher derivatives. The full selected two-point vector is retained.
The
[core module guide](../../external_modules/code/cosmolike_core/cosmolike/3pcf/README.md)
explains the equations and estimator limits; the
[numerical guide](../../external_modules/code/cosmolike_core/cosmolike/3pcf/NUMERICS.md)
explains coarse calculations, dense linear reads, fixed quadrature and
shared work. The existing two-point source files are unchanged.

```mermaid
flowchart TB
  A["YAML or JSON configuration and source distributions"] --> B["prepare: CAMB physical tables"]
  B --> C["evaluate: xi± and aperture mean"]
  B --> D["covariance: separate analytic terms"]
  C --> E["fisher: parameter derivatives and covariance weighting"]
  D --> E
  F["Labeled measured joint covariance"] --> E
```

The graph describes an executable calculation. Publication reproduction
also requires the same estimator, cuts, priors and likelihood assumptions.
The supplied examples are explicit starting configurations for refinement;
they are not accepted Y3 or Y6 likelihood settings.

## Build the compiled library <a name="build"></a>

We assume an existing Cocoa installation, its usual environment, and a
Bash shell in `Cocoa/`. This local development project needs no download
and installs no Python package. Its Makefile uses the existing compilers,
SIMDe, GSL, FFTW, Armadillo, OpenBLAS and pybind11. It compiles only the
three-point files and the project registration file.

**Step :one:**: activate the Cocoa environment.

```bash
source start_cocoa.sh
```

**Step :two:**: build with one compilation worker.

```bash
make -C projects/des_3pcf/interface -f MakefileCosmolike -j1
```

The extension is written into `interface/`. It exposes `production` and
`notebook` adapters to the same C calculations. The first borrows correctly
typed contiguous arrays; the second copies through Armadillo arrays for
interactive inspection. Both return arrays owned by Python.

> [!NOTE]
> This project is a local development build. It is not registered as a
> downloadable Cocoa project with a remote URL or release pin. The main
> setup scripts and other projects' compile settings are unchanged.

## Prepare and evaluate a declared data vector <a name="evaluate"></a>

We assume the build above, Cocoa activated, and a shell in `Cocoa/`.
The example uses the public Y3 source distributions. Numerical outputs
go to the paths named in the JSON file. Every configuration path is
relative to that JSON file, independently of the shell's directory.

**Step :one:**: retrieve or verify the selected pinned Y3 inputs.

```bash
python projects/des_3pcf/scripts/fetch_y3_reference.py
```

This reads twelve files named in `data/y3_reference_manifest.json`, checks
their lengths and Git blob hashes, and stores them in `data/upstream_y3/`.
It does not download the whole simulation archive or execute upstream code.

**Step :two:**: select one worker for an initial calculation.

Linux:

```bash
export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=4
export OMP_PROC_BIND=close
export OMP_PLACES=cores
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
```

macOS:

```bash
export OMP_NUM_THREADS=1
export OMP_THREAD_LIMIT=4
export OMP_PROC_BIND=disabled
export OMP_PLACES=cores
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
```

Set the JSON `threads` field to the intended worker count as well. The
driver applies that count after importing its compiled module.

**Step :three:**: check the configuration before preparing physical tables.

```bash
python projects/des_3pcf/scripts/run_3pcf.py prepare \
  projects/des_3pcf/configs/y3_full_filter_example.json --validate-config
```

`--validate-config`: report missing inputs without importing a compiled
extension, running CAMB or evaluating a physical integral. The check
does not establish numerical convergence.

**Step :four:**: prepare the physical tables, then evaluate the mean.

```bash
python projects/des_3pcf/scripts/run_3pcf.py prepare \
  projects/des_3pcf/configs/y3_full_filter_example.json
python projects/des_3pcf/scripts/run_3pcf.py evaluate \
  projects/des_3pcf/configs/y3_full_filter_example.json
```

The first command calibrates the linear amplitude to the requested
$`\sigma_8`$ and stores power, growth and distance arrays. The second
uses those arrays to calculate every declared observable row. Existing
output files are refused unless `--overwrite` is supplied explicitly.

> [!TIP]
> For configuration choices and derivative checks, see the
> [Fisher guide](FISHER.md). For analytic covariance inputs, see the
> [covariance guide](covariance/README.md).

## Notebook examples <a name="notebooks"></a>

The notebooks explain the quantities before calling project wrappers and
the shared plotting functions. Their bounded examples use one OpenMP worker
and serial BLAS. Full survey calculations are explicit opt-in cells and
require the numerical and estimator checks described beside them.

| Notebook | What its executed examples establish |
|---|---|
| [EXAMPLE_COMPARE_CODES.ipynb](EXAMPLE_COMPARE_CODES.ipynb) | Comparisons with pinned original transforms and aperture kernels, with source discretization differences and stored physical references labeled separately |
| [EXAMPLE_FISHER_IA.ipynb](EXAMPLE_FISHER_IA.ipynb) | Public Y3 input/row inspection and a controlled IA response; the full survey Fisher calculation is configured separately |

We assume the compiled project and the thread environment above. Start in
`Cocoa/` and use the selected Cocoa Python kernel.

**Step :one:**: open the project notebooks.

```bash
jupyter lab projects/des_3pcf --no-browser
```

The [comparison guide](COMPARISON.md) explains original-source identities
and what agreement means. [PAPER_TARGETS.md](PAPER_TARGETS.md) distinguishes
the published claims from the proposed Fisher experiments. The
[shared plotting guide](../../external_modules/code/cosmolike_core/cosmolike_notebook_utils/3pcf/README.md)
teaches the triangle maps, aperture panels, signed components and covariance
figures. Its functions take arrays and do not choose a physical model.

## Covariances, Fisher calculations and tests <a name="tests"></a>

The [covariance guide](covariance/README.md) defines the full-filter
estimator, noise and center-window inputs, term outputs and physical
limitations. The [Fisher guide](FISHER.md) explains exact parameter keys,
derivative step checks, priors and the precision-matrix expansion.

The [test guide](tests/README.md) gives the commands for data-vector and
covariance calculations, strict/debug builds and thread comparisons. The
[validation record](tests/VALIDATION.md) identifies the executed checks.
These small checks exercise formulas, array layout and ownership without
running a survey-size calculation or measuring speed.

The core's [developer guide](../../external_modules/code/cosmolike_core/.claude/skills/cosmolike-dev/references/threepcf.md)
contains the implementation map and the transferred design study. It is
the entry point for continuing the port in another checkout.

## Survey inputs and credits <a name="targets"></a>

The intended survey applications are separate DES Y3 and DES Y6 Fisher
forecasts, retaining the full selected two-point vector as the primary
joint option. Reproduction targets are
[the joint two-/three-point analysis](https://arxiv.org/abs/2503.03964)
and [the TATT/velocity-shear analysis](https://arxiv.org/abs/2601.09133).
The Y3 repository contains source distributions, simulated aperture
vectors and joint covariance products. The selected reference has
200 $`\xi_+`$, 200 $`\xi_-`$ and 80 aperture rows; its labeled covariance
has dimension 480. The downloader records which specific files it selects.
The finite-SAS example reads those rows and preserves their ordering.
It does not infer that its finite triangle settings or likelihood are the
final publication settings merely from the filenames.

The inspected Y6 repository keeps its measurement files external. Its
example therefore leaves the source path unset and makes the proposed
forecast radii explicit. The analytic examples also leave catalogue area
and per-bin number density/shape dispersion unset. Gaussian nuisance
priors are empty. These are missing survey inputs, rather than numerical
values to infer from another release.

| Configuration | Observable and covariance choice |
|---|---|
| `configs/y3_full_filter_example.json` | Unrestricted aperture filters, all declared xi bins and the analytic covariance model |
| `configs/y3_finite_sas_example.json` | Pinned Y3 measured-row definitions, finite-triangle aperture calculation and labeled external joint covariance |
| `configs/y6_full_filter_example.json` | Y6 forecast geometry with explicitly missing external source/catalogue inputs |

The work builds on [Sunao Sugiyama's fastnc](https://github.com/git-sunao/fastnc/tree/tatt),
[Rafael Gomes's 3pcf_integrator](https://github.com/rchgomes/3pcf_integrator)
and [his Y6 pipeline](https://github.com/rchgomes/desy6-3pcf), with the
aperture formulas and TreeCorr contributions credited in the
[source and paper record](../../external_modules/code/cosmolike_core/cosmolike/3pcf/CREDITS.md).
