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

## Consistency comes first

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
- Reusable numerical helpers belong in core's `cosmolike_3pcf/` package;
  reusable plotting functions belong in `cosmolike_notebook_utils/`.
  Neither may silently select a particular survey's compiled interface.
- General Fisher algebra belongs in the existing
  `cosmolike_notebook_utils/fisher.py`, not in a 3PCF-specific duplicate.
  DES saved-file handling stays in project scripts.
- Matter-bispectrum prescriptions belong in `cosmolike_core/theory/`.
  BiHalofit and BACCO are separate theory blocks that can be connected through
  Cobaya's provider dependencies. Do not fold BACCO into the existing `bfmt`
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
one theory producer, not a definition of the project. The current physical
adapter calls CAMB and that producer directly; do not describe it as a fully
interchangeable Cobaya theory dependency graph until that graph is wired
and tested. A matching file layout does not establish scientific agreement
with a paper's estimator or likelihood.

## Writing and validation

Keep one top-level README with a table of contents and appendices. Explain
how to run an example and what its result establishes. General numerical
controls and Fisher derivations belong in core documentation; model-specific
equations belong beside the theory code. Give prominent bold thanks to Sunao
Sugiyama and Rafael Gomes, with their original repositories and papers.

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
