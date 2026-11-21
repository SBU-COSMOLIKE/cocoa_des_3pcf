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
17. [Fixed analytical two-point block](#fixed-analytical-two-point-block-in-the-map-experiment)

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
| [EXAMPLE_COVARIANCE_PME.ipynb](EXAMPLE_COVARIANCE_PME.ipynb) | How do we compute the noisy-square correction, one matrix operation at a time? |
| [EXAMPLE_COVARIANCE_MAPS.ipynb](EXAMPLE_COVARIANCE_MAPS.ipynb) | What can a bounded aperture-moment experiment establish with Rafael’s maps? |
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

The [PME notebook](EXAMPLE_COVARIANCE_PME.ipynb) teaches the
[Friedrich–Eifler precision expansion](https://arxiv.org/abs/1703.07786)
with scalar and two-by-two worked calculations, then controlled Gaussian
and lognormal experiments. It derives unbiased matrix powers through sixth order
under Gaussian sampling, tests a distinct-pair alternative using
one complete ensemble, and explains jackknife error bars step by step. The [map study](EXAMPLE_COVARIANCE_MAPS.ipynb) measures coarse
second- and third-order aperture moments and tests a separately calibrated
Gaussian reference. It also compares four measured xi-plus bins with the
released joint covariance and tests shared shear between realization labels.
Its initial 48-run correction was not positive definite; it does not establish
a usable DES precision. The [undergraduate notes](notes/covariance_from_simulations/covariance_from_simulations.pdf)
derive the method and explain repeated noise realizations, pixel resolution
and independent validation; their [LaTeX source](notes/covariance_from_simulations/covariance_from_simulations.tex)
follows the Fisher notes' style. The [simulation guide](maps_cov_512_v2/README.md)
and [viewing notebook](maps_cov_512_v2/EXAMPLE_VIEW_MAPS.ipynb) inspect two maps
and plot the DES footprint using `indices_512.npy`. A comparison of both
HEALPix orderings identifies RING geometrically. The maps are ignored by Git; only their guide and notebook
are tracked. No DES covariance has yet been calibrated from this download.

The [comparison note for Rafael](notes/covariance_comparison/covariance_comparison.pdf)
uses all 15,997 measured aperture vectors for both the direct covariance
and the second-, fourth- and sixth-order Eifler calculations, with one fixed starting matrix and
no sample partition. Difference maps, histograms and split-triangle
correlations show how the methods differ. It also retains the separate
four-bin comparison with the released DES covariance, using all 32 runs
measured with the matching pair estimator. The aperture moments and those
DES rows are different observables.
Its [companion notebook](notes/covariance_comparison/EXAMPLE_COVARIANCE_COMPARISON.ipynb)
reproduces these two comparisons from compact records in `tests/`;
optional switches reread the complete moment table or repeat the bounded
pair measurements. The broader map notebook retains the earlier pilot
experiments as history. The current note does not divide the sample into
calibration, correction and validation groups.
Survey simulations must reproduce the selection and estimator; four cuts
from one sky and repeated noise draws are not automatically independent skies.

The [map-generation note](notes/gptg_simulations/gptg_simulations.pdf) and
[notebook](notes/gptg_simulations/EXAMPLE_GPTG_SIMULATIONS.ipynb) teach
[Zhong et al.'s point-transformed Gaussian method](https://arxiv.org/abs/2411.04759),
our initial 32-map fit, and 64 saved DES-footprint maps with added Gaussian
shape noise. They derive the noise subtraction for squared, cubed and mixed-bin
apertures, then compare spectra and covariance. The pilot is not yet a validated
DES covariance model. A conditional resampling experiment tests second-,
fourth- and sixth-order precision estimates without claiming new physical skies.
Scripts and compact records are committed; generated maps remain in `chains/`.
New maps use float32 storage after a measured rounding check; computations
remain float64. A separate run is generating 10,000 comparison and 5,000
independent validation maps. Its fixed validation score does not assume the
direct-sample inverse is true; results are pending completion.

### Generate and analyze the fitted maps

Run these commands from `projects/des_3pcf/` with Cocoa's Python activated.
The generator requires NumPy, SciPy, PyYAML, ducc0 (tested with 0.42.0), and
astropy-healpix (tested with 1.0.3); precision analysis also uses SymPy and
plotting uses Matplotlib. Install missing dependencies into that environment.
Supply Rafael's maps and `indices_512.npy` in `maps_cov_512_v2/` first.
These simulation inputs are not distributed through Git.

Fit the transformation and correlated spectra using 32 primary runs:

```bash
python notes/gptg_simulations/generate_maps.py fit --runs 32 --lmax 1024 --workers 4
```

This writes `calibration.yaml` and `calibration.npz` under
`chains/gptg_method/`. To inspect a small generated sample first:

```bash
python notes/gptg_simulations/generate_maps.py generate --count 64 --workers 4
```

Generate the full 10,000 comparison plus 5,000 validation ensemble:

```bash
python notes/gptg_simulations/generate_ensemble.py --workers 4
```

The ensemble copies and freezes the calibration on its first launch. Repeating
this command resumes from its saved-map journal. Use `--limit 1000` for a
bounded first run, then omit that option to continue. Do not launch two
writers to the same ensemble directory. BLAS stays at one worker.
The 15,000 float32 maps need about 177 GB, plus a 50 GB free-space reserve.
Their shape is `(4 source bins, 2 shear components, 369190 footprint pixels)`;
`indices_512.npy` identifies the RING pixels. Computation remains float64.
Maps, measured aperture moments and progress records stay in the ignored
`chains/gptg_ensemble_15000/` directory.

After all maps finish, compute the fixed-reference precision comparisons and
independent validation scores:

```bash
python notes/gptg_simulations/analyze_ensemble.py
```

For an early look, the following process can run alongside generation. It
freezes estimates from the first 1,000 maps and waits for the next 500 to
validate them. Those 500 are independent of the preliminary estimates; they
later belong to the full 10,000 comparison sample, not its final holdout.

```bash
python notes/gptg_simulations/analyze_first_1500.py
```

Once the first 1,000 estimates are written, plot covariance differences and
split-triangle correlations:

```bash
python notes/gptg_simulations/plot_first_1000.py
```

The preliminary tables and PNG/PDF figures go to
`chains/gptg_ensemble_15000/preliminary_1000_500/`. These tests concern 16 raw
aperture moments within the initial fitted model, not the published DES
measurement vector. The direct sample covariance is an estimated comparator.

## Appendix: feedback and BACCO <a name="feedback"></a>

The numbered evaluate, MCMC and Nautilus YAMLs contain an optional,
commented BACCO setup, like LSST Y1's optional theory blocks. In
[EXAMPLE_EVALUATE1.yaml](EXAMPLE_EVALUATE1.yaml), uncomment
`external_baryon_response: true`, the five BACCO parameters, and the
`bacco_bispectrum` theory block together. Keep `bihalofit` enabled: it
supplies the absolute gravity-only bispectrum, while BACCO supplies the
paired power/bispectrum ratios. CAMB supplies their common cosmology.

The existing simulation covariance is retained. The supplied fiducial point
is inside the emulator domain; before sampling, restrict the cosmological
and feedback priors to that domain. Do not also apply the power-only `bfmt`
response. The [BACCO theory guide](../../external_modules/code/cosmolike_core/theory/bacco_bispectrum/README.md#cobaya)
explains registration, parameter bounds and refinement controls.

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

### Fixed analytical two-point block in the map experiment

Section 13 of the [covariance teaching note](notes/covariance_from_simulations/covariance_from_simulations.pdf)
derives the fixed-block correction step by step. The main test uses **all
480 entries**: 400 xi-plus/minus entries and 80 third-order aperture entries.
The full analytical G+SSC+cNG 400-by-400 block stays fixed. Only cross and
three-point information is corrected. The earlier 16-entry and four-xi-bin
exercises remain historical algebra diagnostics, not this science comparison.

**TreeCorr 5.1.2**, by Mike Jarvis and contributors, measures the vector:
`GGCorrelation` for ten source pairs and twenty angular bins, then
`GGGCorrelation`, `toSAS()` and `calculateMap3()` for twenty source triples
and four equal aperture radii. **mpi4py and Open MPI** distribute independent
measurements across ranks. Each measurement rank uses one CPU, even when
TreeCorr was built with OpenMP. A separate coordinator writes the results.
The laptop currently prioritizes map generation: three MPI workers each use
four transform threads; the full measurement is paused. BLAS stays one.

From activated **Cocoa/**, with DES Y3 covariance support compiled, calculate
and save all analytical components:

```bash
python projects/des_y3/covariance/compute_covariance.py projects/des_y3/EXAMPLE_EVALUATE_COVARIANCE.yaml --output projects/des_3pcf/chains/gptg_ensemble_15000/des_y3_full_covariance.npz
```

The small staged test remains available from **projects/des_3pcf/**:

```bash
python notes/gptg_simulations/run_full_joint_mpi.py
```

It allocates nine MPI ranks: eight single-CPU measurement workers and a
coordinator. It measures 1,000 maps, analyzes them, then measures 500 further
maps for independent validation. Both stages resume saved partial vectors.
Do not run it alongside the current twelve-core generation job.

### MPI generation and TreeCorr measurements on HPC

TreeCorr now follows Cocoa's package installation. From **Cocoa/**, leave
`IGNORE_TREECORR_CODE` unset. Setup retrieves pinned TreeCorr 5.1.2 and its
Coord/pybind11 dependencies; compilation uses no network. It uses Cocoa's
selected C/C++ compilers and removes upstream's fast-math option.

```bash
source installation_scripts/setup_treecorr.sh
```

```bash
source installation_scripts/compile_treecorr.sh
```

These sourced installation steps return 55 on success, following Cocoa's
usual convention. Full `setup_cocoa.sh` and `compile_cocoa.sh` also include
them. TreeCorr's keys are removed by `stop_cocoa.sh`. Install ducc0,
astropy-healpix and mpi4py in the same environment before map generation;
mpi4py must use the cluster's MPI installation. Build dependencies on the
cluster rather than copying the laptop's compiled Python extensions.

Use shared storage visible at the same path on every node. Transfer the
project and its actual `data/` inputs, `maps_cov_512_v2/indices_512.npy`, and
`chains/gptg_ensemble_15000/`. That last directory must contain the frozen
`calibration.yaml`, `calibration.npz`, `ensemble.yaml`, footprint indices,
existing comparison/validation maps and their measurement journals. Preserve
these together when resuming. Recalibration is unnecessary, so Rafael's
hundreds of GB of original maps need not be transferred for this step.
Retain `des_y3_full_covariance.npz` for the later fixed-A analysis.

The launch scripts below follow Cocoa's MPI environment forwarding. Source
`start_cocoa.sh` on the allocated node first. Under Slurm request
`--export=ALL`; request four CPUs per task for generation, one for measurement.
The rank count includes one coordinator. For example, 16 generation ranks
reserve 64 CPUs and run 15 workers with four transform threads each. Ensure
that tasks per node times CPUs per task fits the allocated node. Do not
launch these computations on a login node.

```bash
bash projects/des_3pcf/scripts/generate_gptg_hpc.sh 16 15000
```

This resumes the existing seeds and journals, saves float32 maps, and keeps
at least 50 GB free. Computation remains float64. A three-map test agreed
bitwise with the serial generator in both saved pixels and aperture moments.
The laptop uses three generation workers with four transform threads each;
the Linux HPC launcher additionally binds ranks to their allocated cores.

After generation, measure the complete 15,000-map ensemble, for example with
64 allocated single-CPU tasks:

```bash
bash projects/des_3pcf/scripts/measure_gptg_hpc.sh 64 15000
```

There are 63 measurement workers and one coordinator in this example.
Global IDs 0--9999 are the comparison maps; IDs 10000--14999 refer to
validation maps 0--4999. Workers read saved maps; only the coordinator writes
small 480-entry vectors. No per-rank map copies or large triangle outputs
are saved. Repeat the command to resume. Never run two coordinators against
the same journals or measurement directory. The existing analysis driver
still analyzes its early 1,000+500 experiment; measuring all maps does not
silently change that statistical selection.

The scripts leave network transport to the site's MPI configuration and
forward Cocoa's paths plus OpenMP/BLAS keys explicitly. Consult Cocoa's
main MPI README for site-specific network choices. The launcher must be
compatible with the MPI library used to build mpi4py. These are Open MPI
launchers; a site requiring `srun` must use its supported PMI/PMIx settings.
Actual multi-node execution still requires testing on that cluster.

### Optional float32 computation and Apple GPU point transforms

Float32 **storage** and float32 **computation** have separate tests. The
candidate keeps the fitted calibration and random draws in float64, but uses
float32 map arrays and complex64 harmonic arrays. Four noiseless test skies
show a maximum signal RMS difference of 2.3 parts per million; aperture
moments through sixth order differ by at most 3.3 parts per million when
normalized by their absolute moments. Masked E/B cross-spectrum checks also
pass. The [test record](tests/gptg_pilot/compute_precision/summary.txt) states
what was measured; this is numerical agreement, not a physical validation of
the fitted map model or convergence of its covariance.

The default remains float64 CPU computation. Optional `--precision float32`
selects the tested smaller arrays. On Apple silicon, `--point-device metal`
uses MLX for the nonlinear pixel transformation. **Spherical transforms and
the iterative harmonic solve remain on CPU.** MPI workers can overlap their
CPU work with another worker's GPU submission. Calibration, random-number
draws and file handling also remain on CPU; moving these small operations
would change reproducibility or add overhead.

Install the optional tested MLX version in the activated Cocoa environment:

```bash
python -m pip install mlx==0.29.3
```

From `projects/des_3pcf/`, a four-rank hybrid launch is:

```bash
mpiexec -n 4 --mca mpi_yield_when_idle 1 python -u notes/gptg_simulations/generate_ensemble_mpi.py --threads 4 --precision float32 --point-device metal
```

This uses three CPU workers with four transform threads each and one
coordinator. BLAS remains one. Stop the previous generator before launching
a replacement. Existing maps are preserved; `ensemble.yaml` records an
arithmetic change, its completed-map boundary and any saved maps awaiting
journal recovery. Thus switching numerical settings does not conceal which
part of the ensemble used the earlier settings.

Add `--point-batch 4` to submit all four source-bin fields together. That
batch produced bitwise-identical maps to single-field GPU submissions in
four noiseless and 24 noisy examples. It did not demonstrate a speed advantage,
so single-field submissions remain the default. This batches four source
fields from one realization, not independent whole survey realizations.
Batching across workers would also require transferring their field arrays;
the present implementation avoids that communication.

Measured MPI wall times, including process initialization:

| Test | Float64 CPU | Float32 CPU | Float32 CPU + Metal |
|---|---:|---:|---:|
| 12 maps, single-field GPU | 18.79 s | 17.53 s | 17.38 s |
| 24 maps, reverse order, single-field GPU | 36.42 s | 33.03 s | 31.94 s |
| 24 maps, four-field GPU batches | 35.09 s | 32.84 s | 32.24 s |

The hybrid reduces elapsed time by about 8--12% against float64 in the
24-map tests. The GPU's extra gain beyond float32 CPU is only about 2--3%.
The large remaining cost is the harmonic solve. These short measurements
are not a guarantee under a different concurrent laptop workload.

Reproduce the noiseless arithmetic test:

```bash
python notes/gptg_simulations/compute_precision_tests.py --count 4 --threads 1 --metal
```

Reproduce MPI timings when those CPUs are free; use a fresh output directory
so saved maps cannot turn a timing into a cache-hit measurement:

```bash
python notes/gptg_simulations/benchmark_compute_precision.py --count 24 --reverse --output chains/compute_precision_mpi_new
```

Check the masked spectra, common seeds and the unchanged float64 reference:

```bash
python notes/gptg_simulations/check_compute_precision_outputs.py --mpi-output chains/compute_precision_mpi_new
```

The GPU spherical-transform probe is also committed as
`notes/gptg_simulations/probe_gpu_sht.py`. With Torch 2.6.0 on this machine's
macOS 13.7.5, GPU FFT requires macOS 14 or later. This is Cocoa's installed
Torch version. [Torch 2.2](https://github.com/pytorch/pytorch/blob/v2.2.0/aten/src/ATen/native/native_functions.yaml#L2952)
has no MPS FFT dispatch; [Torch 2.3](https://github.com/pytorch/pytorch/blob/v2.3.0/aten/src/ATen/native/mps/operations/FastFourierTransform.mm#L90)
already has the macOS14 requirement. S2FFT 1.5.0, tested with
compatible JAX/JAXLIB 0.5.3 in isolation, rejects MPS at its DLPack transfer.
MLX's Metal FFT itself passes a NumPy comparison. Consequently these tests
rule out the tested ready-made GPU route on this installation, not a future
custom Metal spherical transform. Do not upgrade Cocoa's JAX for this probe;
use an isolated environment for its optional dependencies.

### Optional Metal scalar transforms and mixed CPU/GPU generation

The later implementation in `notes/gptg_simulations/gpu_scalar_sht.py`
moves scalar spherical synthesis, its adjoint and the iterative harmonic
solve to Metal through MLX. The final spin-two shear transform still uses
ducc0 on the CPU. This route does not use Torch or JAX. The earlier
point-only timings above remain a record of that different implementation.

The [map teaching notes](notes/gptg_simulations/gptg_simulations.pdf),
Section 15, explain longitude FFTs, latitude Legendre recurrences, the
adjoint and the least-squares solve with worked examples. The implementation
uses MLX and mlx-metal 0.29.3 in the tested environment, alongside the
existing ducc0 and map-generation dependencies. Make these dependencies
available to the Python interpreter used by every MPI rank.

From this project directory, the tested twelve-CPU-thread allocation is:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=3 mpiexec -n 5 --mca mpi_yield_when_idle 1 python -u notes/gptg_simulations/generate_ensemble_mpi.py --threads 3 --precision float32 --gpu-workers 1 --gpu-pipeline
```

This launches a coordinator, three CPU workers and one GPU worker. The GPU
worker also uses three CPU threads for shear synthesis. Completed maps are
preserved on resume; the numerical-mode history and `arithmetic_workers.txt`
record the transition and each map's backend. Do not start a second
coordinator on the same output directory. Metal schedules the GPU internally.

Two controlled 32-map tests reduced elapsed time from 47.15 to 38.18 seconds
and, reversing run order, from 48.12 to 38.56 seconds. That is approximately
20% less elapsed time, not a guarantee for a busy machine. Scalar-transform
RMS differences against ducc0 were about one part per million; tested
noiseless map and aperture-moment differences stayed below three parts per
million. The records are in `tests/gptg_pilot/gpu_sht/`. These are numerical
checks, not validation of the fitted sky model or full TreeCorr covariance.

The later overlap test keeps one GPU worker and adds `--gpu-pipeline`.
While the CPU synthesizes shear for one source bin, the GPU calculates
scalar harmonics for the next bin. A single CPU consumer preserves noise
ordering and the native-thread allocation. This reduced two 48-map runs
from 56.17/56.12 seconds to 52.42/52.33 seconds, about 7% higher throughput.
Two GPU workers were slower than one in both worker-count trials; the
production command above retains one GPU worker. The scheduling comparison
requires fresh output directories:

```bash
python notes/gptg_simulations/benchmark_gpu_sht_mpi.py --compare-pipeline --count 48 --output chains/pipeline_timing_new
```

Compare the generated maps with the saved original CPU ensemble:

```bash
python notes/gptg_simulations/check_gpu_worker_comparison.py chains/pipeline_timing_new --pipeline
```

For the alternative two-GPU experiment use `--compare-gpu-workers` in the
benchmark and omit `--pipeline` in the checker. Add `--reverse` to the
benchmark with another fresh directory to reverse the execution order.
Neither benchmark stops an existing production process; free its CPU/GPU
allocation before timing. Compact records live under
`tests/gptg_pilot/gpu_sht/scheduling/`.

Recheck the scalar transform at production resolution:

```bash
python notes/gptg_simulations/test_gpu_scalar_sht.py --nside 512 --threads 3
```

Check two paired noiseless four-bin maps:

```bash
python notes/gptg_simulations/test_gpu_map_generation.py --count 2 --threads 3
```

The map test writes under `chains/gpu_sht_maps/`. For timing, use a fresh
output directory and free the required CPUs first:

```bash
python notes/gptg_simulations/benchmark_gpu_sht_mpi.py --count 32 --output chains/gpu_sht_timing_new
```

When generation reaches 15,000 maps, its coordinator checks exact file IDs,
journal seeds, finite journal entries, array headers, dtype and file lengths,
then writes `COMPLETENESS.txt`. This checks missing/truncated maps without
reading all 177 GB of pixels again. It does not start TreeCorr. The same audit
can be run separately:

```bash
python notes/gptg_simulations/verify_ensemble.py chains/gptg_ensemble_15000
```

The declared side cut keeps lower edges at least 8 arcmin from the original
20-bin 0.5–80 arcmin bank (first retained edge 8.151471). It uses multipoles
through 100, 63 SAS angle bins with edges 0.016–3.126 radians, and zero
TreeCorr bin/angle slop. NSIDE512 equal-pixel measurements are not claimed
to reproduce exactly the published NSIDE1024 catalogue/noise procedure.

The initial cross block is zero. The initial 80-by-80 block is the archived
**simulation-estimated** covariance supplied with Rafael's public data,
fixed independently of these new maps. It is not an analytical three-point
covariance; any simulation-saving claim must count that prior input. The
analytical A is never fitted or perturbed. The precision polynomial has a
finite-target unbiasedness proof; imposing A on the final covariance is
nonlinear and must be assessed by independent validation.

Results go under `chains/gptg_ensemble_15000/full_joint_480/` and
`full_joint_comparison/`. Missing entries in partial text vectors are NaNs;
only complete vectors enter the covariance. No extra map copies or large
triangle grids are saved. At this revision, full-map results are pending.

The reproducible algebra checks are:

```bash
python notes/gptg_simulations/check_fixed_block.py
```

```bash
python notes/gptg_simulations/check_full_joint.py
```

**Later TODO — correct only the cross block.** Keep analytical A fixed,
use the ordinary sample covariance for the full 80-by-80 D block, and
construct an Eifler-based estimator for the full 400-by-80 B block.
Derive that estimator explicitly: PME estimates precision, so extracting
an off-diagonal block from a corrected inverse is not automatically an
unbiased estimate of B. Keep the same simulation ensemble and 480 rows,
account for the shared sampling fluctuations of B and D, and compare on
independent validation maps. Check joint positivity through
`D_sample - B_corrected.T @ solve(A, B_corrected)` without clipping failed
matrices. This is a deferred alternative, not a change to the running test.
