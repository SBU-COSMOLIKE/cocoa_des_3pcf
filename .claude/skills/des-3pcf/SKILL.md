---
name: des-3pcf
description: >-
  Maintain the DES joint two-point and three-point Cocoa project: likelihoods,
  YAML examples, Nautilus launchers, notebooks, inputs, tests and documentation.
  Use for changes to projects/des_3pcf and its shared Python dependencies.
---

# DES 3PCF project conventions

## Required Cocoa likelihood pattern

The user explicitly requires the LSST Y1 implementation and coding style,
not a new interpretation of its architecture. Before editing a likelihood,
read `projects/lsst_y1/likelihood/_cosmolike_prototype_base.py` in the Cocoa
checkout and the corresponding probe subclass. Follow their class methods
and call order directly. Developers already know this pattern; changing it
makes the code harder to maintain even when the new code computes the same
quantity.

- Put native initialization calls directly in the base class `initialize`.
  Do not extract `init_two_point`, `check_two_point`, `two_point_spectra`,
  adapters, dispatchers or equivalent helper layers.
- Follow `get_requirements`, `set_cosmo_related`, `set_source_related`,
  `get_datavector`, `internal_get_datavector`, `compute_logp` and `logp`
  where those operations are needed. Do not add empty methods just to make
  the names match. The provider and native calls must actually follow the
  established sequence.
- Numerical controls and their validation belong in the C++ initialization
  functions, following `generic_interface.cpp`. The native interface owns
  the data, covariance and mask singleton and the joint chi-squared.
  Do not implement a parallel Python covariance/likelihood-state framework.
- Use explicit assignments and named native arguments. Do not introduce
  field-name registries such as `ARRAY_NAMES`, `SCALAR_NAMES` or optional
  field schemas to generate calls. Put each named argument on its own line.
- Keep necessary theory-input checks short and inside the likelihood class;
  do not create input/bispectrum container classes or validation modules.
- Production YAMLs and notebooks use CosmoLike's IA normalization and pivot
  directly: `c1rho = 0.01389` and `zpivot = 0.62`. A conversion from an
  upstream convention belongs only to the explicit code-comparison notebook.
- Repeated code is acceptable. Do not move an unwanted abstraction to
  another file and call that consistency. Implement the established pattern.
- Keep native calls visibly under `ci`, as in LSST Y1. Load optional
  compression matrices and check their input row labels directly in
  `initialize`; native initialization validates and applies the matrix.
  Compression-training utilities are not a runtime likelihood layer.
- Numbered evaluate, MCMC and Nautilus examples must select identical
  likelihoods, theory, parameters and priors. Only sampler settings and
  output prefixes differ. Cases 1–4 currently use simulation covariances;
  do not replace them with analytical covariances as a sampler change.
- Teach optional theory choices inside the numbered examples as commented
  YAML blocks, following LSST Y1's FAST-PT/bfmt examples. Include the matching
  commented parameters and likelihood switch. Do not create named variants
  such as EXAMPLE_EVALUATE_BACCO.yaml for these choices.

## Consistency comes first

Consistency with LSST Y1 and Roman is a non-negotiable requirement. Ordinary
prediction and inference run through Cobaya and the project's likelihood
class; do not create a parallel run_3pcf driver, provider framework or
scripts/des3pcf package. For ordinary Fisher calculations follow LSST Y1: write the orchestration
in EXAMPLE_EVALUATE notebooks and call the shared core Fisher functions.
Only an explicitly identified expensive Fisher calculation justifies an
interface/cosmolike_des_3pcf_fisher.py module, following LSST Y1 x DESI
interface/cosmolike_lssty1xdesi_fisher.py. Do not add a general Fisher CLI
or a top-level python package as an automatic exception. Native numerical operations and their
validation belong in the existing generic 3pcf or project C++ interface.
Do not disguise a second run framework as a renamed helper module.


**Do not overengineer.** Do not add abstractions, wrappers or protective
layers when straightforward code already serves the purpose. Use the same
programming model as the existing projects. In particular, do not introduce
attribute sealing or a notebook protection framework. Keep ordinary copies
of caller-supplied arrays where needed, and the existing numerical and cache
checks. Repeated clear code is preferable to a new framework.

Before adding a file, directory, launcher, YAML option or notebook workflow,
inspect the corresponding implementation in LSST Y1 and DES Y3. Follow its
layout, naming, calling sequence and documentation style. Do not invent a
parallel project framework. Repeating straightforward code is acceptable;
abstraction is not a reason to make this project harder to recognize.

For Nautilus, use LSST Y1's EXAMPLE_EMUL2_NAUTILUS1.py as the reference.
For Cobaya, use the other projects' likelihood directories and evaluate
YAMLs. Explain any scientific difference that prevents an exact match.
Check the actual files before claiming consistency; similar names alone
are insufficient.

## File ownership

- `likelihood/` contains the survey's Cobaya likelihood and YAML defaults.
  Cocoa's normal project-link setup exposes it as `des_3pcf.joint_shear`.
  Users must be able to name it in an ordinary evaluate or sampler YAML.
- `interface/` contains the project bindings and notebook wrappers.
- `scripts/` contains DES-specific preparation, comparison and run scripts.
  Do not create a top-level `python/` package.
- There is no `cosmolike_3pcf/` Python package. Numerical preparation and
  evaluation belong in the existing C++ interfaces under `cosmolike/3pcf/`.
  This includes fixed quadrature, aperture row grouping and IA decomposition.
  Extend existing interfaces; do not add Python owner or adapter layers.
- Survey/FITS/source/pixel readers belong in `scripts/des3pcf/`.
  Reusable plots and statistical algebra belong in `cosmolike_notebook_utils/`.
- General Fisher algebra belongs in the existing
  `cosmolike_notebook_utils/fisher.py`, not in a 3PCF-specific duplicate.
  DES saved-file handling stays in project scripts.
- Matter-bispectrum prescriptions belong in `cosmolike_core/theory/`.
  BiHalofit supplies the `matter_bispectrum` provider result. BACCO response
  machinery is separate under `theory/bacco_bispectrum/`; its Cobaya
  component supplies `baryon_response` through the same producer as offline
  studies. Use it alongside an absolute matter-bispectrum producer; set
  `external_baryon_response: true` in the likelihood. It reads solved CAMB
  products, including cold sigma8, and never runs private CAMB. Do not fold BACCO into the existing `bfmt`
  block. Follow Cocoa's installation keys and start/stop symlink conventions.
- Supplied theory arrays use ordinary dictionaries. Keep basic shape, grid
  and background checks in the project likelihood base; do not introduce
  a separate bispectrum container or input-validation framework.
- Numerical initialization follows LSST Y1: YAML options reach the C++
  interface through the likelihood. Theory-provider grids remain owned by
  their theory components; do not build a separate Python accuracy framework.
- `data/` contains actual inputs: measurements, sources, covariances,
  intentional mock inputs and their provenance. Bundle required public
  inputs; do not make a user run a custom fetch script.
- `tests/` contains frozen predictions and numerical validation evidence.
- `chains/` contains generated theory tables, predictions, Fisher products,
  compression matrices, plots, logs and sampler results. Ignore it in Git.
  Do not introduce `outputs/`, `plots/` or `runs/` for these products.
- `maps_cov_512_v2/` is a local simulation download. Ignore its contents
  except its README and viewing notebook; never stage the map arrays.
  Rafael confirmed NSIDE 512 and footprint pixel IDs in `indices_512.npy`.
  The footprint comparison strongly identifies RING/equatorial coordinates;
  record this as geometric evidence. Confirm shear signs and parent/cut/noise
  labels before measurement. The preview uses astropy-healpix, not healpy.
  File counts are not independent-sky counts. Keep previews bounded and
  store generated measurements in `chains/`. The covariance introduction
  lives in `notes/covariance_from_simulations/` with its LaTeX source and PDF.
- `scripts/measure_covariance_maps.py` is a bounded map diagnostic, not the
  published xi/Map3 estimator. It streams maps with OpenMP <=4 and BLAS=1;
  `--all --resume` appends arrivals after a completed pass without map copies.
  Its 16 raw aperture moments keep noise self-products. The map notebook's
  original Gaussian reference used a separate calibration group and is not a
  noiseless CosmoLike theory covariance. The current comparison keeps that
  starting matrix fixed and uses all 15,997 measured vectors in both methods;
  do not refit it or restore the former three-group split in Rafael's note. Keep these limits and unconfirmed shear/parent
  conventions explicit. Never treat a failed or indefinite PME as a forecast.
- `scripts/measure_covariance_xip.py` compares four actual released xi-plus
  bins with the joint 480-row covariance. It uses exact TreeCorr binning,
  a direct-pair subset check, and bounded same-run/different-run cross tests.
  All-map moments remain a different estimator. Released covariance and
  downloaded maps may share parents: their agreement is a consistency test,
  not independent validation. Record the FITS NSIM=500 versus paired-config
  796 discrepancy; do not silently change the reference or simulation count.
  The separate note and executable notebook live together under
  `notes/covariance_comparison/`. Compact measured records live under
  `tests/covariance_map_comparison/`; the notebook's visible measurement
  switch writes fresh bounded results under `chains/`. Rafael's note contains
  only the complete-sample direct-versus-Eifler comparison and the separate
  matched-bin DES comparison (currently 32 measured primary runs). Show
  normalized differences and split-triangle correlations for both. State
  the nominal degrees of freedom used in the noisy-square formula without
  identifying the file count as an established independent count. Earlier
  split-sample and same-run cross tests remain in the main map notebook as
  history; keep the actual meaning of nreal unresolved until the generating
  procedure is supplied.
- Prefer YAML or plain text to JSON outside frozen test evidence. Keep
  binary scientific arrays where their format is useful; do not convert
  FITS or NPZ arrays to text merely to avoid a file extension.

Keep notebook settings visible in the notebook unless an existing project
pattern calls for a YAML. Do not add a same-name YAML just to hide a few
settings. Examples must not depend on another project's review notes,
private scratch directories or absolute paths from this laptop.

## Scientific boundaries

The likelihood joins two-point shear and third-order aperture statistics.
Its dataset declares rows, covariance, model settings and whether the data
are observed or synthetic. Priors belong in the Cobaya parameter block and
must enter the posterior exactly once.

The native projection consumes a supplied matter bispectrum. BiHalofit is
one theory producer, not a definition of the project. The evaluation and Nautilus YAMLs configure CAMB and a separate
bispectrum theory block. The likelihood reads both through `self.provider`;
never call CAMB or a model-specific producer inside the likelihood. A matching file layout does not establish scientific agreement
with a paper's estimator or likelihood.

## Writing and validation

Keep one top-level README with a table of contents and appendices. Explain
how to run an example and what its result establishes. General numerical
controls and Fisher derivations belong in core documentation; model-specific
equations belong beside the theory code. Give prominent bold thanks to Sunao
Sugiyama and Rafael Gomes, with their original repositories and papers.

Second- through sixth-order covariance research is taught in EXAMPLE_COVARIANCE_PME.ipynb
and notes/covariance_from_simulations/. The exact Wick-contraction script
verifies the Wishart matrix-power coefficients. The distinct-pair estimator
uses fixed independent pairs and descending-degree updates to avoid reusing
a random factor in a product; it assumes a fixed reference and independent
vectors, but does not need Gaussian moments or a calibration/validation split.
Unbiasedness concerns the truncated precision, not its inverse covariance
or the exact infinite-order precision. Keep Gaussian/lognormal derivations
and tests in those teaching notes. Rafael's separate comparison note contains
second/fourth/sixth-order results from the same full sample, explicit nominal-count
assumptions, and the separate matched-bin DES comparison. Explain jackknife
deletions, pseudovalues and the prefactor with a four-number worked example.

For undergraduate teaching notes, use the LSST Y1 × DESI Fisher note's
level of explanation. Put matrix definitions and algebraic operations in
separate numbered equation environments, with prose explaining each step.
Do not hide a derivation in inline math or jump directly from expectations
to a corrected estimator. Work through numerical matrices and distinguish
an observed estimate, its ensemble expectation and the unknown truth.
Depth takes precedence over a short page count when the user asks for it.

Use plain explanations for a physics student. Explain units, array shapes,
assumptions and the reason for a non-obvious operation. Avoid unnecessary
frameworks, compressed control flow and implicit behavior. Put separate
shell commands in separate fenced blocks.

Do not use these words in new prose: commendable, comprehensive, crucial,
crucially, delve, delves, delving, dwelve, innovative, intricate,
meticulous, multifaceted, notable, notably, pivotal, thereby,
transformative, versatile, showcase, showcases, showcasing, underscore,
underscores, underscoring, unveil, unveils, unveiling. Exact external titles
and identifiers retain their spelling. Avoid decorative claims, bot closers,
and phrases such as "it is worth noting" or "provides valuable insights".

After moving code, update all imports, YAML component names, notebook source
cells and file hashes used to reject obsolete saved results. Test a real
Cobaya load and evaluation, not only a Python import. Preserve saved notebook
outputs unless a rerun is requested; do not present old outputs as new runs.
Validate notebook structure and README links. Keep BLAS at one worker and
use the user's current OpenMP limit; run numerical checks serially.

Commit explicit paths in the correct repository. DES uses `main`; core uses
`lowrank_fftlog`. Preserve unrelated working-tree changes. Never push or
amend a commit without a new user instruction.

## Point-transformed Gaussian map pilot

`notes/gptg_simulations/` contains a separate undergraduate note, executable
notebook and plain research scripts for Zhong et al. arXiv:2411.04759.
The initial fit uses 32 supplied maps and produces 64 four-bin NSIDE512 maps
with multipoles through 1024, saved as float64 under ignored
`chains/gptg_method/maps/`. BLAS=1; honor the current user limit for ducc0 native threads. The pilot used
four; the user authorized eight for the large run on 21 November 2026. This is our
implementation, not an author-provided calibration. Compact measurements,
parameters and test results live under `tests/gptg_pilot/`.

Its fit uses a masked/noisy/smoothed convergence proxy and approximate
(E-minus-B)/fsky spectra, with explicit positive-semidefinite adjustments.
Noise is constant independent Gaussian per pixel/source/component, estimated
from high-ell masked B power. These approximations and the unresolved producer
conventions prevent claiming a validated DES covariance. Source-bin-two uses
the paper's b=0 family because the first unrestricted fit was nonmonotone.
Never normalize individual maps by their measured mean or variance. Preserve
map fluctuations and record finite harmonic projection and noise assumptions.

The scripts measure 16 same-source raw moments for the existing all-map
comparison, and 60 raw/noise-corrected tomographic moments at the two radii.
These are fixed-center coarse diagnostics, not the published finite-triangle
catalogue estimator. Shape-noise subtraction of aperture products is distinct
from correcting powers of a noisy sample covariance in PME.

The sixth-order empirical test treats all 64 saved measurement vectors as a
fixed discrete population and resamples with replacement. This establishes
conditional algebra tests, not additional independent N-body skies. It uses
known discrete-population covariance (denominator 64) and ordinary sample
covariances (denominator N-1). Keep both polynomial-target and exact-inverse
errors, and report when higher order worsens RMS. More synthetic maps reduce
sampling error for the fitted model, not its calibration uncertainty or
missing physical correlations. Do not present raw array file counts as a
known count of independent cosmological realizations.

### Generated-map storage precision

**User directive (21 November 2026): save generated maps as float32.**
Keep harmonic generation, normalization, filter weights, aperture accumulation,
measured vectors, covariance and precision arithmetic in float64. Decode saved
float32 pixels to float64 before measurement, so saved measurements reproduce
the files users will later read. Do not change FFT or covariance arithmetic
to float32 merely because map storage is float32.

The 64-map storage check at NSIDE512 and 30/60-arcminute apertures found a
maximum covariance-weighted vector change of 2.564e-7, fractional variance
change of 1.499e-8, and relative precision-eigendirection change of 5.252e-8.
These errors are negligible for this tested use. Keep the original float64
pilot only as a small precision-test reference; new ensembles use float32.
Repeat the storage comparison when resolution, dynamic range or observables
change substantially. Compare differences in covariance-weighted units and
matrix directions; relative errors of near-zero cubic moments are misleading.

The 10,000 comparison + 5,000 independent validation design freezes the fitted
model, uses disjoint seed ranges, and keeps validation out of reference/order
tuning. Float32 payload is about 177 GB for all 15000 four-bin footprint maps.
`generate_ensemble.py` journals each completed map and preserves a 50 GB reserve.
The validation quadratic/log-determinant score does not identify the direct
sample inverse as truth. Model validation against N-body simulations remains
separate from estimator validation conditional on a fitted generator.

The current laptop priority is map generation: three CPU workers and one
Metal scalar-transform worker, each with three native CPU threads, plus a
coordinator. BLAS stays one. See the optional arithmetic section below.
Full TreeCorr measurement is paused. HPC launchers in scripts/ forward Cocoa's
MPI keys and allocate one CPU per measurement rank. Global measurement IDs
10000..14999 select validation maps 0..4999. Do not run two coordinators
against the same output directory. Preserve older timing records as history.

## Exact analytical two-point block

For the fixed-block covariance experiment, the user assumes the production
G+SSC+cNG two-point block is exact. Do not fit, perturb or run robustness
scans of that block. Correct only cross and three-point information.
Match observables: the 16-row map journal contains aperture moments, not
xi-plus/minus. `measure_joint_xip.py` measures four source-bin-four xi-plus
bins on saved GPTG maps; `analyze_fixed_block.py` joins the cubic moments.
The production DES Y3 example supplies the analytical submatrix. Record its
survey inputs rather than claiming bin matching alone matches all physics.
`fixed_block_precision.py` uses distinct independent pair factors. The raw
finite polynomial is unbiased for its finite target. Its conditional
reparameterization keeps A exactly but is nonlinear and has no corresponding
unbiasedness guarantee; evaluate it independently. Never invert a non-positive
residual precision or conceal a failed expansion by clipping eigenvalues.

The main fixed-block comparison must use all 480 entries, not the former
16-entry or four-xi-bin exercises. `measure_full_joint_mpi.py` uses TreeCorr
5.1.2 GGCorrelation, GGGCorrelation, toSAS and calculateMap3 for all pairs
and triples. The updated TreeCorr build supports OpenMP; mpi4py/Open MPI supplies
process parallelism. Count native threads in the MPI CPU allocation. `run_full_joint_mpi.py` runs first-1000 measurement,
analysis, next-500 measurement, then independent validation without running
matrix analysis alongside the eight active MPI workers. The fixed 80x80
starting reference is the archived simulation covariance, not an analytical
three-point calculation. Count that prior input in any simulation-saving
claim. State NSIDE512, equal pixel weights, original-bank lower-edge >=8
cut, max_n=100, 63 SAS bins and the unresolved exact publication match.
Do not interpret earlier small-vector performance as a result for this matrix.

## Optional arithmetic and Metal map generation

Float32 file storage does not by itself validate float32 computation.
The computation tests and their records are under notes/gptg_simulations/
and tests/gptg_pilot/compute_precision/. The optional --precision float32
uses single-precision map/harmonic arrays with double-precision calibration
and the original random stream. The --point-device metal flag uses MLX only
for the nonlinear point transformation; ducc0 spherical transforms and
iterative inversion remain on CPU. Do not describe this as full GPU map
synthesis. The default arithmetic remains float64 unless explicitly selected.
MPI records numerical-mode changes and preserves already saved maps.
--point-batch 4 batches the four source-bin fields within one realization;
it is not independent-realization batching. It passed the agreement checks
but did not establish better throughput than the single-field GPU path.
The point-only route is retained for comparison; the later scalar-transform
GPU route below is the current mixed production allocation. Keep notebook,
refreeze and full TreeCorr runs paused.
The Apple GPU SHT probe must distinguish actual device execution from a
CPU fallback. Torch's macOS limitation and S2FFT's MPS/DLPack failure do not
prove that a custom Metal spherical transform is impossible. MLX FFT works.
On completion verify_ensemble.py checks all 15000 IDs, seeds, headers and
sizes; it does not perform a full pixel scan or launch TreeCorr measurements.


### Validated optional Metal scalar spherical transforms

`gpu_scalar_sht.py` implements scalar latitude recurrences, exact ring FFTs,
the pixel adjoint and matrix-free CG normal equations on MLX/Metal.
`gpu_map_generation.py` keeps the nonlinear scalar mapping on device, then
returns coefficients to CPU ducc0 for spin-two shear. Do not describe this as
a GPU spin-two transform. Float32 arithmetic is explicitly selected; frozen
calibration and random draws remain double precision, measurements float64.
The default CPU route remains float64 unless a flag changes it.

The user authorized mixed production on 21 November: `mpiexec -n 5`,
`--threads 3 --precision float32 --gpu-workers 1`. This means one coordinator,
three CPU workers and one GPU worker, at most twelve native transform CPU
threads and BLAS=1. Preserve saved maps, seeds, numerical history and per-map
`arithmetic_workers.txt`. GPU dependencies tested: MLX/Metal 0.29.3.

Evidence under `tests/gptg_pilot/gpu_sht/` includes scalar NSIDE512/lmax1024,
adjoint identity, full noiseless maps and forward/reversed 32-map MPI tests.
Elapsed time improved about 20% against CPU float64. Numerical agreement
is not a physical validation or a full TreeCorr covariance comparison.
Repeat gates for changed geometry or algorithm. The companion teaching
section in `notes/gptg_simulations/gptg_simulations.tex` derives every
stage with small examples. Keep it consistent with the actual backend;
the normal residual is not the map residual, and pixel-area scaling alone
does not improve the normal matrix's condition number.


The follow-up scheduling test selects `--gpu-pipeline` with one GPU worker.
A single CPU consumer synthesizes shear/noise for bin i while the main
thread executes Metal scalar work for bin i+1. Coefficients cross into the
consumer as owned NumPy arrays; GPU context ownership stays on the main
thread. Noise draws remain serial in source order. All tasks finish before
a map returns, including on exceptions. Do not equate pipeline stage-time
sums with elapsed wall time: CPU and GPU stages overlap.

Two GPU workers replaced one CPU worker at the same twelve-native-thread
allocation and were slower in both 48-map trials. One GPU worker with overlap
reduced elapsed time by about 6.7% in both orders. Record numerical identity
for common GPU seeds, noiseless agreement, and timing in
`tests/gptg_pilot/gpu_sht/scheduling/`. Defaults stay sequential unless the
flag is selected; preserve the production arithmetic-history entry.


### Forty-thousand-map extension

User authorized 30000 comparison + 10000 validation maps on 21 November.
Retain chains/gptg_ensemble_15000/ and both original seed starts. MPI flags
--comparison-count 30000 --validation-count 10000 set the enlarged target;
keep --gpu-workers 1 --gpu-pipeline --threads 3, five ranks, BLAS=1.
The verifier reads counts/seeds from ensemble.yaml. Earlier completion
records are archived before extension; do not report them as current success.
The existing TreeCorr/HPC selection still has the original 10000 split and
15000 target. Update that selection explicitly before measuring all 40000;
never assume its global IDs changed when generator counts changed.


HPC measurement extension: --count 40000 retains original measurement IDs
0..9999=comparison0..9999, 10000..14999=validation0..4999; appended IDs
15000..34999=comparison10000..29999, 35000..39999=validation5000..9999.
Do not reinterpret existing saved vectors using a new 30000 split.
GPTG_MAP_DIR selects an absolute HPC folder (e.g. chains/gptg); the launcher
passes --maps-dir and outputs remain in its full_joint_480 subfolder.
The earlier paragraph describing the 15000-only HPC limit is superseded.
Keep the selected folder identical on resubmission, with only one writer.


TreeCorr scheduling uses a shared queue of whole ready maps. Each MPI worker
owns its map through all unfinished tomographic jobs; intermediate results
are checkpointed by rank zero without releasing the worker. Existing finite
rows and map IDs are retained. Arrival-aware mode requires successful-rsync
.npy.ready markers. tests/gptg_pilot/test_map_dispatch.py executes the actual
coordinator AST with fake MPI messages, never TreeCorr, covering late arrivals
and partial resume. Do not revert to static strides or task-level dispatch.


HPC measurement submission follows LSST Y1: use one standalone
`scripts/EXAMPLE_GPTG_TREECORR_UOFA.sbatch` containing the Slurm directives,
Cocoa activation, exported MPI environment and direct mpirun command.
Do not split it into a Slurm file plus a shell launcher. Keep the 40000-map
selection, existing journal IDs and arrival-aware resume behavior intact.


GPTG note layout follows LSST Y1 x DESI's Fisher machinery note: one
companion notebook beside TeX/PDF, with commented scientific analysis and
figure cells. MPI programs, TreeCorr measurements and production dependencies
belong in scripts/. Do not accumulate small analysis .py files under notes/.
Retire one-off hardware probes instead of exposing internal exploration as
a public workflow. Notebook code needs explanations of inputs, array shapes,
normalization, each mathematical step and what each result establishes.

Keep this note in one TeX source, `gptg_simulations.tex`; do not split
its sections into additional TeX files. It has one PDF and one companion notebook.

Use one ensemble entry point, scripts/generate_ensemble_mpi.py. With one MPI
process rank zero computes and journals directly; with multiple processes
rank zero coordinates. Do not recreate a separate serial driver. --threads
controls native transform threads per computing process; BLAS stays one.

Script-count rule: avoid an explosion of Python files. scripts/ is for
expensive production jobs such as map generation and TreeCorr, plus the
necessary support code they actually call. Put lightweight analysis and
plotting in the associated example or note notebook, with explanatory
comments. Keep exploratory development diaries in an ignored folder.
Review existing scripts against this rule; do not merely rename or move
an unwanted framework into another package.
