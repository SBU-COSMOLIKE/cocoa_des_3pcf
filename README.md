# DES two-point and three-point calculations

1. [Purpose](#scope)
2. [Build](#build)
3. [Prepare and evaluate](#evaluate)
4. [Run Nautilus](#sampling)
5. [Notebooks](#notebooks)
6. [Tests](#tests)
7. [Acknowledgements](#acknowledgements)
8. [Appendix: input and output files](#files)
9. [Appendix: the theory interface](#theory)
10. [Appendix: accuracy controls](#accuracy)
11. [Appendix: Fisher forecasts and response plots](#fisher)
12. [Appendix: covariance and simulations](#covariance)
13. [Appendix: feedback and BACCO](#feedback)
14. [Appendix: optional compression](#compression)
15. [Appendix: estimator and publication comparisons](#estimator)
16. [Appendix: original-code comparisons and timing](#comparison)

## Purpose <a name="scope"></a>

This project calculates a joint cosmic-shear vector: two-point correlations
$`\xi_\pm`$ and third-order aperture moments $`\langle M_{\rm ap}^3\rangle`$.
The latter measure the skewness of shear through a compensated angular filter.
Combining the observables can distinguish parameter responses that look
similar in two-point data alone. The joint covariance determines how much
independent information is added.

The examples use DES Y3 inputs and include a Y6 forecast template with
missing survey inputs explicitly unset. They demonstrate calculations;
matching a published posterior also requires matching its estimator,
selection, priors and covariance treatment. The full two-point vector is
kept unless the user selects compression.

## Build <a name="build"></a>

We assume an installed Cocoa environment, a Bash shell in `Cocoa/`,
and the project and core checkouts. Each command below runs from there.

**Step :one:**: activate Cocoa.

```bash
source start_cocoa.sh
```

**Step :two:**: compile the project extension.

```bash
make -C projects/des_3pcf/interface -f MakefileCosmolike -j1
```

The extension is written to `interface/`. The production and notebook
adapters call the same numerical library; their array ownership differs.
See the [core guide](../../external_modules/code/cosmolike_core/cosmolike/3pcf/README.md)
for implementation details.

## Prepare and evaluate <a name="evaluate"></a>

The examples read included survey inputs from `data/`. Fiducial theory
tables and predictions are calculated locally in `chains/`.

**Step :one:**: select up to six OpenMP workers and keep BLAS serial.

```bash
export OMP_NUM_THREADS=6
export OMP_THREAD_LIMIT=6
export OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
```

Set the configuration's `threads` field to the intended worker count too.
For MPI sampling, choose ranks and OpenMP workers within the available cores.

**Step :two:**: inspect the input paths without running a model.

```bash
python projects/des_3pcf/scripts/run_3pcf.py prepare \
  projects/des_3pcf/configs/des_y3_nla_demo_fisher.yaml --validate-config
```

**Step :three:**: prepare the fiducial tables shared by the NLA and TATT examples.

```bash
python projects/des_3pcf/scripts/run_3pcf.py prepare \
  projects/des_3pcf/configs/des_y3_nla_demo_fisher.yaml
```

This writes `chains/y3_demo_pchip9601_physics.npz`. NLA is nonlinear tidal
alignment; TATT adds tidal-torquing and density-weighting contributions.

**Step :four:**: calculate the joint mean.

```bash
python projects/des_3pcf/scripts/run_3pcf.py evaluate \
  projects/des_3pcf/configs/des_y3_nla_demo_fisher.yaml
```

`--overwrite`: explicitly replace an existing generated output. Recompute
dependent results after changing inputs or numerical settings; do not
relabel an old forecast as a result of the changed calculation.

Paths inside each YAML are relative to that file. See the
[configuration reference](configs/README.md) for its fields.

For an ordinary Cobaya evaluation, use `EXAMPLE_EVALUATE1.yaml`. Its
theory components calculate the tables; no preparation command is needed.
Cocoa's project-link setup exposes
`likelihood/joint_shear.py` as `des_3pcf.joint_shear`, just as it exposes
the LSST Y1 and DES Y3 likelihoods. The YAML selects a local dataset and
specifies the parameter point; no custom sampler launcher is needed.

```bash
cobaya-run projects/des_3pcf/EXAMPLE_EVALUATE1.yaml
```

The example uses synthetic data and writes its evaluation to `chains/`.
`EXAMPLE_EVALUATE1–4.yaml`, `EXAMPLE_MCMC1–4.yaml` and the four Nautilus
examples use the same numbered cases in the table below. Changing sampler
does not change the data, covariance, theory or priors.

For ordinary Cobaya MCMC, as in LSST Y1:

```bash
cobaya-run projects/des_3pcf/EXAMPLE_MCMC1.yaml
```

Replace 1 with 2, 3 or 4 to select another case. Each YAML writes to its
own `chains/EXAMPLE_MCMC<number>` prefix. Proposal learning and convergence
settings follow LSST Y1; Cobaya measures the joint model's speeds instead
of using LSST's parameter blocking. A single evaluation does not establish
chain convergence.

## Run Nautilus <a name="sampling"></a>

The four launchers follow `lsst_y1/EXAMPLE_EMUL2_NAUTILUS1.py`: import
`run` from the core's `cocoa_hybrid_sampling` and pass the project and
example number. DES sets `hybrid=False` to use physical theory. The
sampler, priors, command-line options and MPI pool are shared with LSST.

| Case number | IA model | Fitted data | Likelihood |
|---|---|---|---|
| 1 | NLA | Synthetic fiducial mean | Fixed Gaussian |
| 2 | TATT | Synthetic fiducial mean | Fixed Gaussian |
| 3 | NLA | Released observed Y3 measurements | Fixed Gaussian |
| 4 | TATT | Synthetic mean with independently trained compression | Percival |

All four use the supplied **simulation-estimated covariance**, including
its two-/three-point cross terms. Cases 1–3 treat that matrix as fixed;
case 4 uses the finite-simulation likelihood and requires its independent
pilot compression file. These examples exercise the methods associated
with Rafael's analyses; matching their published posterior is still subject
to the estimator, scale-cut and prior differences described below.

Each matching YAML defines Cobaya `params` and
`sampler.evaluate.override`. Its likelihood selects a `.dataset` in
`data/`, which names the source distributions, measurements and covariance.
The YAML `theory:` section selects CAMB and the bispectrum provider.
Example 4 first needs the matrix generated by the compression notebook.

**Step :one:**: check the fiducial without sampling.

```bash
python projects/des_3pcf/EXAMPLE_NAUTILUS1.py --check
```

**Step :two:**: sample with an explicit budget.

```bash
python projects/des_3pcf/EXAMPLE_NAUTILUS1.py \
  --nlive 1000 --neff 10000 --maxfeval 100000
```

Replace the example number for the other models. `--input` selects a
Cobaya YAML; `--outroot` changes the chain basename; `--root` selects the
directory containing `chains/`. Existing records and chains are refused,
as in the LSST driver; there is no project-specific resume command.
`--check` evaluates one fiducial, not derivative accuracy or posterior convergence.

For four MPI ranks, keep `omp_threads: 1` in the likelihood settings. One rank coordinates
Nautilus and three evaluate separate models.

Linux:

```bash
mpirun -n 4 --oversubscribe \
  --mca pml ob1 --mca btl vader,tcp,self \
  --bind-to core:overload-allowed --report-bindings \
  --rank-by slot --map-by slot \
  python projects/des_3pcf/EXAMPLE_NAUTILUS1.py
```

macOS:

```bash
mpirun -n 4 --oversubscribe \
  python projects/des_3pcf/EXAMPLE_NAUTILUS1.py
```

Outputs are GetDist `.1.txt` rows and `.paramnames`, an HDF5 checkpoint,
and a YAML record with the evidence and convergence status. A run that
reaches its budget can have `converged: false`; no retained posterior
samples means no chain is written.

## Notebooks <a name="notebooks"></a>

Open these with the Cocoa kernel and `projects/des_3pcf` as the notebook
directory. Generated results and figures use `chains/`.

```bash
jupyter lab projects/des_3pcf --no-browser
```

| Notebook | Question addressed |
|---|---|
| [EXAMPLE_FISHER_IA.ipynb](EXAMPLE_FISHER_IA.ipynb) | How do NLA/TATT responses and joint Fisher constraints compare? |
| [EXAMPLE_FISHER_ACCURACY.ipynb](EXAMPLE_FISHER_ACCURACY.ipynb) | Which sampling controls and derivative steps stabilize the forecast? |
| [EXAMPLE_IA_COMPONENTS.ipynb](EXAMPLE_IA_COMPONENTS.ipynb) | How do ordered intrinsic-alignment terms enter the aperture signal? |
| [EXAMPLE_JOINT_FEEDBACK.ipynb](EXAMPLE_JOINT_FEEDBACK.ipynb) | How do separate power and bispectrum responses change the joint mean? |
| [EXAMPLE_FEEDBACK_SCENARIOS.ipynb](EXAMPLE_FEEDBACK_SCENARIOS.ipynb) | What changes among the installed Cocoa feedback scenarios? |
| [EXAMPLE_FISHER_BACCO.ipynb](EXAMPLE_FISHER_BACCO.ipynb) | How do neutrino conversion and extrapolation affect BACCO forecasts? |
| [EXAMPLE_COMPRESSION.ipynb](EXAMPLE_COMPRESSION.ipynb) | What local information does each optional compression preserve? |
| [EXAMPLE_COVARIANCE_PME.ipynb](EXAMPLE_COVARIANCE_PME.ipynb) | How can simulations correct an analytic precision matrix? |
| [EXAMPLE_COMPARE_CODES.ipynb](EXAMPLE_COMPARE_CODES.ipynb) | Which calculations agree with the original libraries? |

A saved notebook output describes its recorded inputs and implementation.
Editing a path does not recompute that result. The [study index](tests/studies/README.md)
identifies executed comparisons and unresolved accuracy questions.

## Tests <a name="tests"></a>

Use the [test guide](tests/README.md) for routine checks and compilation
variants, and [optional studies](tests/studies/README.md) for expensive
refinement calculations. Frozen vectors, their settings and timing records
live under `tests/`. They check particular numerical comparisons; they do
not establish a survey posterior or extend a theory model's calibration.

## Acknowledgements <a name="acknowledgements"></a>

**We thank Sunao Sugiyama and Rafael C. H. Gomes for their scientific work
and for making their methods and code available. Their work is the
foundation of this three-point port.**

- **Sunao Sugiyama:** [fastnc](https://github.com/git-sunao/fastnc), including
  its [TATT branch](https://github.com/git-sunao/fastnc/tree/tatt), and the
  shear three-point multipole method.
- **Rafael C. H. Gomes:** [3pcf_integrator](https://github.com/rchgomes/3pcf_integrator),
  [desy6-3pcf](https://github.com/rchgomes/desy6-3pcf), and the joint
  two-/three-point and intrinsic-alignment analyses.

Please cite their [fast-modeling paper](https://arxiv.org/abs/2407.01798),
[Y3 methods paper](https://arxiv.org/abs/2503.03964), and
[TATT/velocity-shear paper](https://arxiv.org/abs/2601.09133) when using
these methods, alongside Cocoa/CosmoLike and the selected theory model.

We also thank **Mike Jarvis** for TreeCorr, **Xiao Fang** for the FFTLog
method, and **Tim Eifler, Elisabeth Krause and the CosmoLike contributors**
for the numerical framework. Aperture formulas follow
[Jarvis, Bernstein and Jain](https://arxiv.org/abs/astro-ph/0307393) and
[Schneider, Kilbinger and Lombardi](https://arxiv.org/abs/astro-ph/0308328).
Covariance methods include [Linke et al.](https://arxiv.org/abs/2212.04485)
and [Wielders et al.](https://arxiv.org/abs/2509.20443); the precision
expansion follows [Friedrich and Eifler](https://arxiv.org/abs/1703.07786).
The [core credit record](../../external_modules/code/cosmolike_core/cosmolike/3pcf/CREDITS.md)
preserves source revisions and license notices.

## Appendix: input and output files <a name="files"></a>

| Location | Contents |
|---|---|
| `data/` | Released measurements, intentional mock inputs, sources, covariance matrices, pixel window and their provenance |
| `chains/` | Generated theory tables, predictions, derivatives, compression matrices, figures and sampler results; ignored by Git |
| `tests/` | Frozen predictions and recorded numerical comparisons |
| `likelihood/` | Cobaya likelihood and YAML defaults |
| `scripts/` | DES-specific preparation and run scripts |

No fetch script is needed. The [input guide](data/README.md) identifies
included files and regeneration commands. The processed demonstration
source clips negative density samples and renormalizes each bin, with a
YAML record. That declared input is a teaching assumption, not a survey
recalibration. Original released files remain unchanged.

## Appendix: the theory interface <a name="theory"></a>

The observable calculation consumes background quantities, power spectra
and a supplied matter bispectrum. BiHalofit is a choice of theory, not the
definition of a shear three-point observable. Another producer must supply
the same units, table coverage and background conventions; no internal C
BiHalofit fallback is used.

The evaluation and Nautilus YAMLs select `camb` and `bihalofit` as separate
Cobaya theory blocks. The likelihood reads their products through
`self.provider`; replacing the bispectrum block requires the same supplied
array contract. The older explicit-table Fisher studies still prepare their
inputs directly and do not exercise this Cobaya dependency graph.
The [theory-component documentation](../../external_modules/code/cosmolike_core/theory/bihalofit/README.md) owns its
fit equations, sampling and calibration limits. Survey assumptions belong
in the project configuration independently of that theory choice.

## Appendix: accuracy controls <a name="accuracy"></a>

Use `EXAMPLE_FISHER_ACCURACY.ipynb` with the settings in
`configs/des_y3_fisher_accuracy.yaml`. It compares individually refined
derivative columns with a default-accuracy base; selected candidates
still need full paired forecasts.

The [core accuracy reference](../../external_modules/code/cosmolike_core/cosmolike/3pcf/README.md#accuracy-controls)
defines global and individual boosts. The
[shared Fisher guide](../../external_modules/code/cosmolike_core/cosmolike_notebook_utils/README.md#fisher-noise)
explains the noise estimate, step conventions and limitations. DES-specific
measurements are in the [study record](tests/studies/implementation_record.md#accuracy).

## Appendix: Fisher forecasts and response plots <a name="fisher"></a>

After preparing the shared fiducial tables, run the TATT demonstration
forecast with the joint covariance:

```bash
python projects/des_3pcf/scripts/run_3pcf.py fisher \
  projects/des_3pcf/configs/des_y3_tatt_demo_fisher.yaml
```

`EXAMPLE_FISHER_IA.ipynb` plots the saved parameter responses in units of
each measurement's error and compares their separation with the full
cross-covariance. The [shared Fisher guide](../../external_modules/code/cosmolike_core/cosmolike_notebook_utils/README.md#fisher)
defines derivatives, priors, marginalized errors and figures of merit.

## Appendix: covariance and simulations <a name="covariance"></a>

The [covariance guide](covariance/README.md) gives the analytic terms,
noise, survey-window inputs and commands. The examples can instead read a
fixed released joint covariance. A fixed Gaussian likelihood retains its
normalization and uses a Cholesky solve; a positive log density is possible
because these observables have small numerical units.

The PME notebook teaches the [Friedrich–Eifler precision expansion](https://arxiv.org/abs/1703.07786)
with controlled Gaussian simulations. It is not a DES covariance
calibration. Survey simulations must reproduce the selection and estimator;
independent validation must check the corrected precision. The released
covariance does not supply those individual realizations. Four footprints
cut from one full sky also require an independence assessment.

## Appendix: feedback and BACCO <a name="feedback"></a>

A joint feedback model needs separate responses $`R_P=P_H/P_D`$ and
$`R_B=B_H/B_D`$ from compatible hydro and gravity-only models. A power
response alone does not determine the bispectrum response. The artificial
feedback notebook writes `chains/paired_feedback_demo.npz`; it is not a
calibrated constraint on feedback.

The BACCO example uses paired cold-matter responses. With massive neutrinos,
converting them to total matter requires the matching component spectra
and an explicit approximation for mixed bispectra. Treating a cold response
as a total-matter response is a separate model choice. The notebook compares
those choices and the continuation beyond the emulator's trained domain.
Quadratic IA and the fiducial covariance retain their declared DMO inputs.

```bash
python projects/des_3pcf/scripts/run_3pcf.py prepare \
  projects/des_3pcf/configs/des_y3_bacco_fisher.yaml
```

This creates `chains/y3_bacco_physics.npz`; its neutrino fields differ from
the ordinary demonstration table. Parameter meanings and continuation
settings are explicit in the YAML and [BACCO study](tests/studies/README.md#bacco).

## Appendix: optional compression <a name="compression"></a>

Compression replaces the data and every predicted mean by $`Td`$ and
$`T\mu`$, and the covariance by $`TCT^T`$. Train and then hold $`T`$ fixed;
changing it during inference changes the statistic. The notebook compares
xi-only, conditional-xi and full-joint choices. Preserving a local Fisher
matrix does not guarantee preservation of a nonlinear posterior.

Example 4 uses a matrix generated in `chains/` by the compression notebook.
Its Percival likelihood assumes a sample covariance from independent
Gaussian mocks, independent data and an independently trained compression.
It is not a Hartlap-rescaled Gaussian and must not be combined with that
correction. The stated 796 footprints are an analysis assumption, not
proof of 796 independent skies or a reproduction of the published matrix.

## Appendix: estimator and publication comparisons <a name="estimator"></a>

The included Y3 products have 200 $`\xi_+`$, 200 $`\xi_-`$ and 80 Map3 rows.
Matching those counts does not match their estimator. Finite triangle
bounds, side-bin weights, representative lengths, pixel smoothing and
angular cuts all affect a prediction. The [estimator study](tests/studies/estimator_conventions.md)
records the remaining measurement and Map3 cross-covariance sign questions.
Finite-cut E/B tests are separate from the unrestricted E-only mean.

Exact published inference also needs the original compression, trained
emulator, scale cuts, priors and finite-simulation likelihood. A Gaussian
Fisher improvement is not the published posterior FoM improvement.
The [comparison record](tests/studies/implementation_record.md#paper_targets)
separates figure-level tests from inference claims. Y6 source curves,
noise, area and calibration priors remain required inputs; the Y3 files
must not silently stand in for them.

## Appendix: original-code comparisons and timing <a name="comparison"></a>

`EXAMPLE_COMPARE_CODES.ipynb` declares its source paths and numerical settings
in one Python cell. It reads independent original-code copies from Cocoa's
`external_modules/code/fastnc_tatt`, `fastnc_main` and `3pcf_integrator`.
Use the revisions in `data/code_comparison_manifest.yaml`; missing or modified
files are reported before execution. No other survey project is a dependency.
Results go to `chains/code_comparison/`. Frozen regression inputs stay in `tests/`.

For timing, use the CLI's ten changed parameter points. CAMB, file reads,
imports and initialization are excluded from the data-vector interval;
a separate warmup precedes the measured points. Validate changed outputs
and thread agreement before interpreting scaling.

```bash
python projects/des_3pcf/scripts/benchmark_scaling.py preflight
```

A timing run requires a quiet machine and the explicit allocation in
`configs/openmp_scaling.yaml`. It is not part of routine tests. Historical
figures and their vectors are in [tests/timing/](tests/timing); the
[timing guide](tests/timing/README.md) gives run and plot commands.

The joint likelihood uses the same extension for ordinary two-point shear
and three-point apertures. Numerical preparation and IA decomposition are
in its C++ interfaces. Survey readers stay in `scripts/des3pcf/`; there is no
separate shared Python `cosmolike_3pcf` package. The optional component
notebook calls `TransformPlan.aperture_components` and plots its signed
operator/placement contributions and closure.
