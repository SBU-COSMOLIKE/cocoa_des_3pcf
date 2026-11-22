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
17. [CPU map generation on the HPC](#cpu-map-generation)
18. [Fixed analytical two-point block](#fixed-analytical-two-point-block-in-the-map-experiment)

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

## Evaluate with Cobaya <a name="evaluate"></a>

Follow the numbered YAML examples, as in LSST Y1. Their `theory` blocks
calculate CAMB power spectra and the supplied matter bispectrum; the
`likelihood` block selects the joint shear dataset. Survey inputs are
included in `data/`, and results go to `chains/`. No separate theory-table
preparation command is needed.

Activate Cocoa, then evaluate the declared fiducial point:

```bash
cobaya-run projects/des_3pcf/EXAMPLE_EVALUATE1.yaml
```

Cocoa exposes `likelihood/joint_shear.py` as `des_3pcf.joint_shear` through
its normal project links. Set `omp_threads` in that likelihood's YAML
settings to choose the OpenMP worker count; the examples default to one.
Keep BLAS at one worker and account for all ranks when running MPI.
[EXAMPLE_EVALUATE1.ipynb](EXAMPLE_EVALUATE1.ipynb) shows the same model call,
its residuals and a small Fisher calculation using the shared core routines.

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

Open these with the Cocoa kernel. The IA and baryonic-feedback studies live
under `notes/`, each with its TeX source, PDF and companion notebook, as in
LSST Y1 × DESI. Their visible cells reproduce the calculations explained in
the notes; they use the numbered Cobaya likelihoods and shared Fisher
functions. Physical reruns are deferred, so these refactored notebooks have
empty outputs and the notes report no new forecast values. Generated results
and figures use the project’s `chains/` directory.

```bash
jupyter lab projects/des_3pcf --no-browser
```

| Notebook | Question addressed |
|---|---|
| [IA notebook](notes/intrinsic_alignments/EXAMPLE_FISHER_IA.ipynb) and [teaching note](notes/intrinsic_alignments/intrinsic_alignments.pdf) | How do NLA/TATT responses and joint Fisher constraints compare? |
| [EXAMPLE_FISHER_ACCURACY.ipynb](EXAMPLE_FISHER_ACCURACY.ipynb) | Which sampling controls and derivative steps stabilize the forecast? |
| [EXAMPLE_IA_COMPONENTS.ipynb](EXAMPLE_IA_COMPONENTS.ipynb) | How do ordered intrinsic-alignment terms enter the aperture signal? |
| [EXAMPLE_JOINT_FEEDBACK.ipynb](EXAMPLE_JOINT_FEEDBACK.ipynb) | How do separate power and bispectrum responses change the joint mean? |
| [EXAMPLE_FEEDBACK_SCENARIOS.ipynb](EXAMPLE_FEEDBACK_SCENARIOS.ipynb) | What changes among the installed Cocoa feedback scenarios? |
| [BACCO notebook](notes/baryonic_feedback/EXAMPLE_FISHER_BACCO.ipynb) and [teaching note](notes/baryonic_feedback/baryonic_feedback.pdf) | How do neutrino conversion and extrapolation affect BACCO forecasts? |
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
array contract. The IA, accuracy and BACCO Fisher notebooks use this same
Cobaya dependency graph; they do not prepare a separate CAMB calculation.
The [theory-component documentation](../../external_modules/code/cosmolike_core/theory/bihalofit/README.md) owns its
fit equations, sampling and calibration limits. Survey assumptions belong
in the project configuration independently of that theory choice.

## Appendix: accuracy controls <a name="accuracy"></a>

Use `EXAMPLE_FISHER_ACCURACY.ipynb`. Its cells load a numbered evaluate
YAML, state physical parameter steps and priors, and change ordinary
likelihood settings for each refinement. The full default-accuracy base
is computed once; candidate screening varies selected columns. A final
full derivative comparison at both step sizes is still required.

The [core accuracy reference](../../external_modules/code/cosmolike_core/cosmolike/3pcf/README.md#accuracy-controls)
defines global and individual boosts. The
[shared Fisher guide](../../external_modules/code/cosmolike_core/cosmolike_notebook_utils/README.md#fisher-noise)
explains the noise estimate, step conventions and limitations. DES-specific
measurements are in the [study record](tests/studies/implementation_record.md#accuracy).

## Appendix: Fisher forecasts and response plots <a name="fisher"></a>

Use [EXAMPLE_EVALUATE1.ipynb](EXAMPLE_EVALUATE1.ipynb) for the ordinary
workflow: evaluate the numbered Cobaya model, differentiate its retained
data vector, then call the shared Fisher functions. Its three-parameter
example holds the other parameters fixed and is a conditional demonstration.
A full survey forecast must vary the intended nuisance parameters and state
its priors. Derivatives and matrices are saved under `chains/`.

The IA, accuracy and BACCO Fisher notebooks now follow that calling pattern:
load the numbered YAML, evaluate through Cobaya, and call the shared core
Fisher routines in visible cells. The IA notebook retains finite sign
comparisons and normalized derivative plots. The accuracy notebook separates
cheap noise screening from full convergence checks. The BACCO notebook
separates physical extrapolation choices from numerical sampling and labels
its restricted parameter forecast as conditional. These examples use the
unmasked, uncompressed rows of numbered examples 1 and 2.

The migrated notebook cells passed small API and algebra tests; their full
physical forecasts have not been rerun. Old numerical outputs were cleared
to avoid presenting results from the retired route as new calculations.
Historical study scripts still use the older Fisher/accuracy wrappers;
those dependencies remain migration work, not the recommended workflow. The
[shared Fisher guide](../../external_modules/code/cosmolike_core/cosmolike_notebook_utils/README.md#fisher)
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
remain float64. The ensemble generator produces separate comparison and
independent validation groups. Its fixed validation score does not assume the
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
python scripts/generate_maps.py fit --runs 32 --lmax 1024 --workers 4
```

This writes `calibration.yaml` and `calibration.npz` under
`chains/gptg_method/`. To inspect a small generated sample first:

```bash
python scripts/generate_maps.py generate --count 64 --workers 4
```

Generate an ensemble using the generator's default group counts:

```bash
mpirun -n 1 python scripts/generate_ensemble_mpi.py --threads 4
```

This is the same generator used for larger MPI runs. With `-n 1`, the sole
process generates maps and writes the journals itself; it does not reserve
a coordinator-only process. `--threads 4` gives that process four native
transform threads. Use `--threads 1` for one computing thread. With `-n N`
and N greater than one, rank zero coordinates and the other N-1 processes
generate maps. BLAS stays at one in both cases.


The ensemble copies and freezes the calibration on its first launch. Repeating
this command resumes from its saved-map journal. Use `--limit 1000` for a
bounded first run, then omit that option to continue. Do not launch two
writers to the same ensemble directory. BLAS stays at one worker.
Budget storage from the requested count and saved map size, plus the
generator's configured free-space reserve.
Their shape is `(4 source bins, 2 shear components, 369190 footprint pixels)`;
`indices_512.npy` identifies the RING pixels. Computation remains float64.
Maps, measured aperture moments and progress records stay in the ignored
`chains/gptg_ensemble_15000/` directory.

Use the [companion notebook](notes/gptg_simulations/EXAMPLE_GPTG_SIMULATIONS.ipynb)
for analysis and plots. Its small aperture examples explain the algebra;
the full scientific comparison uses the 480-entry TreeCorr vector. Earlier
16-entry timing and preliminary-validation runners are kept only in the
local, ignored development archive. They are not steps in this workflow.

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

### Fixed analytical two-point block in the map experiment <a name="fixed-analytical-two-point-block-in-the-map-experiment"></a>

Section 13 of the [covariance teaching note](notes/covariance_from_simulations/covariance_from_simulations.pdf)
derives the fixed-block correction step by step. The checkpoint layout has
400 xi-plus/minus entries and 80 third-order aperture entries. **At NSIDE512,
the first three angular bins of every xi pair contain no distinct pixel
pairs.** Their zero entries are missing measurements, so the full sample
covariance cannot be inverted. The [map note](notes/gptg_simulations/gptg_simulations.pdf)
and its companion notebook explain the geometry and explicitly select the
varying rows for diagnosis. Both fixed-block methods keep the same analytical
G+SSC+cNG two-point block on those rows; only cross and three-point information
is estimated. The immediate target is the covariance of the generated ensemble,
not reproduction of Rafael's published covariance.

**TreeCorr 5.1.2**, by Mike Jarvis and contributors, measures the vector:
`GGCorrelation` for ten source pairs and twenty angular bins, then
`GGGCorrelation`, `toSAS()` and `calculateMap3()` for twenty source triples
and four equal aperture radii. **mpi4py and Open MPI** distribute independent
measurements across ranks. Each measurement rank uses one CPU, even when
TreeCorr was built with OpenMP. A separate coordinator writes the results.
BLAS stays one. Before a large measurement run, check angular-bin support
and match the starting covariance to the generated-map experiment. A finite
output value alone does not establish that a bin contained pairs.

From activated **Cocoa/**, with DES Y3 covariance support compiled, calculate
and save all analytical components:

```bash
python projects/des_y3/covariance/compute_covariance.py projects/des_y3/EXAMPLE_EVALUATE_COVARIANCE.yaml --output projects/des_3pcf/chains/gptg_ensemble_15000/des_y3_full_covariance.npz
```

The single Slurm file below describes submission and resume. For existing
measurements, use the **Audit the downloaded full-vector measurements** section
of the companion notebook. It reads all complete checkpoints, excludes partial
vectors, checks duplicate IDs, and saves the selected IDs. It runs no TreeCorr
measurement. Additional downloaded checkpoints change this explicitly reported
sample on rerun. The starting Map3 block currently comes from a released
simulation covariance, not an analytical calculation. The audit tests whether
that reference is close enough for precision expansion and refuses to present
an indefinite precision as a valid covariance.

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

Measure the requested ensemble with the single submission file
below. It contains both the Slurm allocation and the MPI command, following
the LSST Y1 examples; there is no separate shell launcher. Workers read saved
maps, and only the coordinator writes the small 480-entry vectors. No
per-rank map copies or large triangle outputs are saved. The existing
analysis driver still selects its explicitly labeled pilot experiment; measuring all
maps does not silently change that statistical selection.

### UofA Slurm submission

[EXAMPLE_GPTG_TREECORR_UOFA.sbatch](scripts/EXAMPLE_GPTG_TREECORR_UOFA.sbatch)
contains the allocation and requested measurement count. Adjust its Slurm
node, task, account, queue and time directives for your allocation before
submission. One MPI rank coordinates; the remaining ranks measure whole
maps. Each worker uses one CPU, with OpenMP and BLAS set to one. Acceptance
depends on the site's allocation limits; no socket count is assumed.

The same script also accepts Slurm array indices. With its default range
size, array task `0` handles map IDs `[0, 5000)`, task `1` handles
`[5000, 10000)`, and task `2` handles `[10000, 15000)`. Each range writes
under `full_joint_480_shards/<start>_<stop>/`, separate from the original
job's `full_joint_480/`. The range size is a setting in the script; keep
the same value when resubmitting an array task so it finds its checkpoints.
Do not run two copies of the **same** array index at once.

The original, already running job has no upper map limit. Start another
allocation with array task `1` to begin at map 5000:

```bash
sbatch --array=1 projects/des_3pcf/scripts/EXAMPLE_GPTG_TREECORR_UOFA.sbatch
```

To queue the next range for a later start, submit a separate array index:

```bash
sbatch --begin=now+1hour --array=2 projects/des_3pcf/scripts/EXAMPLE_GPTG_TREECORR_UOFA.sbatch
```

These jobs may overlap in time because their checkpoint directories differ.
The older job may eventually compute a map from one of those ranges as well;
that duplicates work but cannot overwrite a shard checkpoint. The example
output below stays in the selected `$GPTG_MAP_DIR`. Do not submit array task
`0` alongside the original job unless that duplicate work is intentional.

Set `GPTG_MAP_DIR` to the ensemble's absolute parent path on the HPC:

```bash
export GPTG_MAP_DIR="/absolute/path/to/gptg"
```

Replace this example path with your shared storage directory or a project
`chains/gptg/` symlink. Retain `comparison/`, `validation/`,
`indices_512.npy`, calibration files and journals. The project also needs its
bundled `data/upstream_y3/` inputs. These paths must be visible at the same
location on every node. Build/install Cocoa, TreeCorr and mpi4py for the
cluster; do not copy the Mac's Python binaries. The MPI installation used by
mpi4py must match the launcher's Open MPI installation.

From the HPC Cocoa directory containing `start_cocoa.sh`, activate the
installed environment (replace `cocoa` if your HPC environment has another name):

```bash
conda activate cocoa
```

Initialize Cocoa once before submission, so the batch job inherits its paths
and does not rebuild shared links during another job:

```bash
source start_cocoa.sh
```

Submit the allocation:

```bash
sbatch projects/des_3pcf/scripts/EXAMPLE_GPTG_TREECORR_UOFA.sbatch
```

Slurm writes `D3PCF_TC-<jobid>.out` and `.err` in the submission
directory. Inspect your jobs with:

```bash
squeue -u "$USER"
```

#### Reading measurement progress

Progress is printed when a map finishes a measurement block:

```text
map 60 2pt done compute seconds this run 1040.5 completed entries 400/480
map 60 3pt done compute seconds this run 24000.8 completed entries 480/480
```

The times above illustrate the format; they are not performance estimates.
`compute seconds this run` sums the worker's task times for that block only,
separately for each map. It excludes queue waits, other maps, MPI sends and
coordinator checkpoint writes. The first task on a map includes loading the
map and constructing catalogs. On resume, only tasks computed in the current
allocation contribute; earlier saved work has no retained timing.

`map` is the simulation ID. `2pt done` means all xi-plus and xi-minus
entries are saved; `3pt done` means all Map3 entries are saved.
`completed entries` counts finite entries for that map, not completed maps.
The current vector has 400 two-point and 80 three-point entries; the map
is complete when numerator and denominator agree.

Every task still writes its checkpoint, even though individual tasks no
longer print progress. A quiet log can therefore mean workers are computing
within a block. Resume retains saved entries and announces only blocks that
become complete during this run. Blocks already complete are not announced
again. Older jobs retain the earlier per-task log format. The number of
saved entries is not a fraction of elapsed work: three-point entries cost
much more to measure than two-point entries.

#### Resuming after the wall-time limit

If the configured Slurm wall-time limit is reached,
Slurm stops the job; the script does **not** automatically requeue itself.
Keep the existing output folder and its partial vectors. From the same
activated HPC Cocoa checkout, first check that the old job has ended:

```bash
squeue -u "$USER"
```

Wait until the old job has left the queue, including its `COMPLETING` state.
Then submit the same script again for another allocation:

```bash
sbatch projects/des_3pcf/scripts/EXAMPLE_GPTG_TREECORR_UOFA.sbatch
```

No additional resume flag is needed. Use the same input maps, measurement
settings and output folder. Do not delete or rename the saved vectors
between submissions. Resume detects saved entries; it does not verify that
you have kept the input maps and estimator settings unchanged.

For each map, the script reads its saved 480-entry vector:

- **All entries finite:** skip the entire map; TreeCorr does not measure it again.
- **Some tasks saved:** skip tasks whose output entries are all finite and
  compute only tasks with missing or `NaN` entries.
- **No saved vector:** compute all tasks for that map.

Each completed tomographic task is saved by writing a temporary vector and
atomically replacing the map's output file. A task interrupted before its
result was saved must run again. Results already saved remain available to
the next allocation. This is task-level resumption, not resumption inside
an unfinished TreeCorr calculation.

**Never run two measurement jobs against the same checkpoint directory at
the same time.** There is no inter-job lock. Different array indices have
separate directories, but repeating one index concurrently is unsafe. The
original job's directory is also exclusive to that job while it runs.

After all measurement jobs have ended, combine the shard checkpoints into
`full_joint_480/` from the activated Cocoa directory:

```bash
python projects/des_3pcf/scripts/measure_full_joint_mpi.py --merge-shards --maps-dir "$GPTG_MAP_DIR"
```

The merge fills missing entries in the original vectors and refuses
conflicting finite values. It saves each updated vector atomically and can
be repeated after later allocations. Run it only when **all** writers have
stopped; a running job could otherwise overwrite a merged vector. A shard
completion marker covers only its own range. The main `COMPLETE.txt` is
written by a full main-directory measurement job, not by this merge.

Measurements live under `$GPTG_MAP_DIR/full_joint_480/`.
Successful completion writes `COMPLETE.txt`; its vector count must match the
requested `--count` in the submission script, since an older smaller run
may have its own completion marker. Generation journals and float32 maps are not modified by measurement.

Estimate wall time as the remaining map count times the measured time per
map, divided by the number of measurement workers and parallel efficiency.
Use timings from your first HPC allocation: laptop timings do not establish
cluster throughput. Check peak memory before filling every core on a node.
The script launches measurements only; it does not generate maps or run the
covariance analysis automatically.

The Slurm command uses Cocoa's single-thread Linux MPI pattern:
`--oversubscribe`, `ob1`, `vader,tcp,self`, explicit environment forwarding,
`mpi_yield_when_idle`, the TCP interface exclusion list,
`--bind-to core:overload-allowed --report-bindings`, and
`--rank-by slot --map-by slot`. TreeCorr uses one thread per worker here;
the general hybrid OpenMP guidance below applies when allocating multiple
threads per rank. The MPI library used by mpi4py must match the launcher.
Actual multi-node execution still requires testing on the cluster.

TCP interface inclusion and exclusion lists are mutually exclusive. The
batch script unsets inherited `OMPI_MCA_btl_tcp_if_include` before launching
MPI and retains the explicit `--mca btl_tcp_if_exclude lo,docker0,virbr0,ib0,eno1,enp1s0f0np0`
line. This keeps Cocoa's exclusion policy even when shell startup or modules
supply an inclusion setting.
A successful single-node run does not establish that the selected interface
is routable between nodes.

> [!NOTE]
> **Running on more than one node.**
>
> 1. **Network interface.** TCP must use an interface routable between compute
>    nodes. The tested UofA exclusion list is
>    `--mca btl_tcp_if_exclude lo,docker0,virbr0,ib0,eno1,enp1s0f0np0`; adapt it to the cluster.
>    On the tested UofA nodes this leaves `enp1s0f1np1` available.
>    Before launching, run `unset OMPI_MCA_btl_tcp_if_include` so an
>    inherited inclusion list cannot conflict with the explicit exclusion.
>    Keep `ib0` if routable IP-over-InfiniBand is the intended network. These
>    examples exchange parameter vectors and scalar scores, so communication
>    volume is small; actual scaling still depends on the machine.
> 2. **Environment.** Remote ranks need the same Cocoa paths and libraries:
>    `ROOTDIR`, `PATH`, `LD_LIBRARY_PATH`, `PYTHONPATH`, `CONDA_PREFIX`, OpenMP
>    and BLAS settings, and `CLIK_PATH`/`CLIK_DATA`/`CLIK_PLUGIN` when used.
>    Slurm normally exports the submitting environment (`--export=ALL`).
>    Explicit `-x` options also forward these variables with SSH launchers.
>    Activate Cocoa before launching; build/download flags do not replace
>    runtime paths. All nodes must see the same files at the same paths.
> 3. **Slurm geometry.** Keep `ntasks-per-node × cpus-per-task` within the
>    allocated physical cores per node. Set `OMP_NUM_THREADS` to
>    `SLURM_CPUS_PER_TASK` and use `--map-by numa:pe=${OMP_NUM_THREADS}`.
>    The minimization, profile and Nautilus pool reserves one MPI rank as
>    coordinator; the remaining ranks evaluate the model.

### Extending the ensemble

Set comparison and validation counts in the generation command below.
The existing `chains/gptg_ensemble_15000/` directory keeps its historical
name; extending an ensemble does not require duplicating or renaming it.
Existing maps retain their IDs and seeds. Keep the recorded seed ranges
disjoint when extending either group.

From the project directory, with the tested MLX and map dependencies available:

```bash
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=3 mpiexec -n 5 --mca mpi_yield_when_idle 1 python -u scripts/generate_ensemble_mpi.py --threads 3 --precision float32 --gpu-workers 1 --gpu-pipeline --comparison-count 30000 --validation-count 10000
```

Repeat this exact command to resume the extended target. The final audit
reads group counts from `ensemble.yaml`; old completion markers are archived
with `.before_40000` when extension starts. Estimate storage from the
requested map count times the saved map size,
and retain the generator's configured free-space reserve.

Set the measurement script's `--count` to the intended supported target.
Its current ID mapping below is an implementation constraint, not a suggested
ensemble size. Extending beyond this mapping requires updating the reader
while preserving existing IDs so partial and complete measurements remain
valid. Measurement IDs are not a contiguous comparison/validation split:

| Measurement ID | Input group | Local map ID |
|---|---|---|
| 0–9999 | comparison | 0–9999 |
| 10000–14999 | validation | 0–4999 |
| 15000–34999 | comparison | 10000–29999 |
| 35000–39999 | validation | 5000–9999 |

Use this mapping when selecting groups for later analysis. The earlier
pilot analysis is unchanged; it does not automatically analyze the larger
ensemble. Measurement resumption retains finite entries under the original
IDs; old smaller-target completion markers are archived when tasks remain.

#### Copying maps and choosing the HPC folder name

You may name the HPC directory `chains/gptg/`, including a symlink to
storage outside the checkout. Set `GPTG_MAP_DIR` to that parent directory,
not to its `comparison/` child. The parent must exist and be writable.

**Before submission, check this layout under `$GPTG_MAP_DIR`:**

| File or folder | Requirement |
|---|---|
| `indices_512.npy` | Required at startup; the footprint pixel indices matching these maps |
| `comparison/` | Create before uploading comparison maps; may initially be empty with `GPTG_WAIT_FOR_READY=1` |
| `validation/` | Create before uploading validation maps, even if its upload has not started; may initially be empty with `GPTG_WAIT_FOR_READY=1` |
| `comparison/GPTG_DES_*.npy` and `validation/GPTG_DES_*.npy` | Completed map files; they may arrive during the job in waiting mode |
| Matching `GPTG_DES_*.npy.ready` files | Required for dispatch with `GPTG_WAIT_FOR_READY=1`; create only after successful transfer |
| `full_joint_480/` | Output, created automatically; preserve it when resuming, including its partial measurement files |

Create both input folders on the HPC. These commands also work when the
parent path is a symlink and leave existing files intact:

```bash
mkdir -p "$GPTG_MAP_DIR/comparison"
```

```bash
mkdir -p "$GPTG_MAP_DIR/validation"
```

The reader tolerates a missing group folder in waiting mode, so a missing
`validation/` does not itself prevent comparison maps from being measured.
Creating both folders in advance is the transfer setup convention, not an
extra startup check in the Python program. Empty folders are allowed; if
neither group has ready maps, workers wait and still consume allocated wall
time. The job cannot finish until all requested maps are supplied.

**Also required in the project checkout:** the installed TreeCorr, mpi4py,
NumPy, Astropy and astropy-healpix dependencies, plus these two bundled FITS
files under `data/upstream_y3/data/dv/`. They define the measurement row order:

- `sim_2pt-NLA-cosmoCosmogrid-04Nov24.fits`
- `sim_map3_emu-NLA-cosmoCosmogrid-24FEBRUARY2025_WITH_COV.fits`

**Keep these provenance files in `$GPTG_MAP_DIR` as well:**

- `ensemble.yaml`
- `calibration.yaml`
- `calibration.npz`
- `comparison_measurements.txt`
- `validation_measurements.txt`

The TreeCorr measurement script does not read those five provenance files,
but they retain the generation settings and earlier measured diagnostics
needed to interpret the ensemble. `joint_xip_measurements.txt` and other
diagnostic folders are optional for the full 480-row measurement.

While transfers continue, use the readiness mode below. Without that mode,
complete the requested transfers before submission. Never mark a partially
uploaded map as ready, and never run two measurement jobs writing to the
same checkpoint directory.

#### Starting measurement while uploads continue

Set `GPTG_WAIT_FOR_READY=1` before `sbatch` to enable arrival-aware dispatch:

```bash
export GPTG_WAIT_FOR_READY=1
```

The coordinator maintains one shared queue of **whole maps**. Each free
worker takes the next uploaded, unfinished map and computes its remaining
tomographic tasks. There are no fixed strides or predetermined map ranges
per rank. The worker retains that map's catalogs and sends each task result
to the coordinator for an immediate checkpoint; it receives another map only
after finishing the current one. Saved tasks are skipped on resume.
Unavailable maps are skipped until they arrive. Supply at least as many
ready maps as measurement workers to keep the initial allocation occupied.
Fewer ready maps will leave some workers waiting; the coordinator polls
for later arrivals.
Idle ranks wait when no ready work remains. Waiting consumes allocated wall
time, so upload a useful initial batch before submission.

In this mode a map needs a companion `.npy.ready` file created **only after
a successful transfer**. File existence alone cannot distinguish a complete
map from an interrupted `rsync -P` upload. Use the shared-SSH transfer
commands below: they upload each map and then create its marker. Already
uploaded matching maps also receive missing markers. Do not create markers
merely from a remote file listing, and never overwrite a marked map while
measurements run.
Transfer `indices_512.npy` and the required project inputs before submitting.
Keep `GPTG_MAP_DIR` set to the remote parent directory as described below.
The default mode still assumes transfers are complete. The coordinator was tested with fake MPI messages for whole-map ownership,
late arrivals, partial resume and preservation of existing entries. Run that
non-numerical test with:

```bash
python projects/des_3pcf/tests/gptg_pilot/test_map_dispatch.py
```

Actual multi-node execution still needs HPC validation; no local TreeCorr
run was made.

#### Transferring the maps to the HPC

Run these commands on the laptop. The remote `comparison/` and
`validation/` directories must already exist. Stop an earlier transfer loop
before starting its replacement; completed files are retained.

Set these variables on the laptop, replacing the host and path with your
HPC transfer login and ensemble parent directory. Use ordinary paths without
spaces or shell metacharacters in these transfer examples. `export` makes
the values available to the shells launched by `xargs`.

```bash
export GPTG_TRANSFER_HOST="your_username@your_transfer_host"
```

```bash
export GPTG_REMOTE_DIR="/absolute/path/to/gptg"
```

```bash
export GPTG_TRANSFER_JOBS=20
```

`GPTG_TRANSFER_JOBS` controls concurrent uploads, independently of MPI.
Reduce it if the transfer server closes connections or limits sessions.

**First terminal: upload using independent SSH connections.** From the local
`comparison/` folder, run:

```bash
ls *.npy | xargs -P "$GPTG_TRANSFER_JOBS" -I {} sh -c 'rsync -avP "$1" "${GPTG_TRANSFER_HOST}:${GPTG_REMOTE_DIR}/comparison/" && ssh "$GPTG_TRANSFER_HOST" "touch ${GPTG_REMOTE_DIR}/comparison/$1.ready"; result=$?; if [ "$result" -ne 0 ]; then echo "Retry needed: $1" >&2; exit 1; fi' sh {}
```

`ls` supplies the filenames; `xargs -P` limits parallel jobs, and `-I {}`
passes each filename as `$1` to its shell. Rsync's `-a` preserves timestamps,
`-v` prints progress details, and `-P` retains partial uploads and displays
transfer progress. The `&&` creates a ready marker only after rsync succeeds.
The failure handler reports the filename and converts SSH's exit status so
`xargs` can continue processing other files. Rerun to retry failures.

This uses independent connections rather than forcing all uploads through
one shared SSH connection. For `validation/`, run from that local folder
and replace both `/comparison/` occurrences with `/validation/`.

**Second terminal: recover missing ready markers.** Set the same host and
remote-directory variables in this terminal; exports in the first terminal
are not automatically shared. Open a reusable connection:

```bash
ssh -MNf -o ControlPersist=600 -S "$HOME/.ssh/d3pcf-%C" "$GPTG_TRANSFER_HOST"
```

`-M` creates a shared connection, `-N` runs no remote command, `-f` puts it
in the background, and `-S` names the control socket. `%C` identifies the
connection. Repeat this command if the connection expires or disconnects.

From the local `comparison/` folder, check unmarked files:

```bash
for file in *.npy; do
  ssh -S "$HOME/.ssh/d3pcf-%C" "$GPTG_TRANSFER_HOST" "test -f ${GPTG_REMOTE_DIR}/comparison/$file.ready" && continue
  changes=$(rsync -acni -e "ssh -S $HOME/.ssh/d3pcf-%C" "$file" "${GPTG_TRANSFER_HOST}:${GPTG_REMOTE_DIR}/comparison/") || continue
  if [ -z "$changes" ]; then
    ssh -S "$HOME/.ssh/d3pcf-%C" "$GPTG_TRANSFER_HOST" "touch ${GPTG_REMOTE_DIR}/comparison/$file.ready"
  fi
done
```

This loop never uploads or modifies a map. Rsync's `-n` is a dry run,
`-c` compares file checksums, and `-i` reports differences. Only a successful
check with no reported changes creates a marker. Missing, incomplete or
different files remain unmarked; a later pass can check them again. Checksums
read both copies, so this recovery pass adds disk and network work and may
be slower than a size check. A remote listing or equal byte count alone is
not enough to mark a file safely while uploads continue.

Use the corresponding local folder and replace `/comparison/` with
`/validation/` to recover that group's markers. Both workflows assume the
generated filenames contain no spaces or newlines and completed local maps
are immutable. Never overwrite marked maps while TreeCorr reads them.

The upload command already creates markers; the second terminal is for
uploads whose marker command failed or which were transferred previously.
Each pass lists only maps present when it starts; repeat for later arrivals.
With `GPTG_WAIT_FOR_READY=1`, TreeCorr can consume marked maps while uploads
continue. These transfer commands do not launch TreeCorr or delete maps.

From the local parent `chains/gptg_ensemble_15000/` directory, copy the pixel
indices and provenance files to the remote parent directory:

```bash
rsync -avP indices_512.npy ensemble.yaml calibration.yaml calibration.npz comparison_measurements.txt validation_measurements.txt "${GPTG_TRANSFER_HOST}:${GPTG_REMOTE_DIR}/"
```

To retain existing TreeCorr measurements, also copy `full_joint_480/` from
that local parent directory, before any HPC measurement job starts:

```bash
rsync -avP full_joint_480/ "${GPTG_TRANSFER_HOST}:${GPTG_REMOTE_DIR}/full_joint_480/"
```

Do not repeat this last copy over newer HPC measurements: choose one
authoritative output folder. Maps already fully saved locally can be copied
while generation continues, but each invocation lists only the files present
at that time. Repeat the map transfers and refresh the parent journals after
generation finishes. Never start TreeCorr on files still being transferred;
partial remote `.npy` files are not valid completed maps.

On the HPC, set `GPTG_REMOTE_DIR` to the same absolute destination chosen
on the laptop; shell variables are not copied by SSH. Then select it for
measurement:

```bash
export GPTG_MAP_DIR="${GPTG_REMOTE_DIR:?Set GPTG_REMOTE_DIR on the HPC first}"
```

Measurements are written to `$GPTG_MAP_DIR/full_joint_480/`. This explicitly
selected storage replaces the default project location; a second copy of
the maps inside the project is unnecessary. Keep the same setting on resume.

Alternatively, if you copied into the project's `chains/gptg/` directory,
select that location from activated Cocoa on the HPC:

```bash
export GPTG_MAP_DIR="${ROOTDIR}/projects/des_3pcf/chains/gptg"
```

Submit using the same Slurm script:

```bash
sbatch projects/des_3pcf/scripts/EXAMPLE_GPTG_TREECORR_UOFA.sbatch
```

Slurm exports this variable to the batch job, which passes the absolute path
to every measurement rank. Results are written to
`$GPTG_MAP_DIR/full_joint_480/`. Set the same variable before each resubmission.
Without it, the historical `chains/gptg_ensemble_15000/` path is used.
Never run two jobs against the same checkpoint directory, even if submitted
with different job names. Array indices write to separate shard directories;
merge their results after the jobs stop as described above.

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
mpiexec -n 4 --mca mpi_yield_when_idle 1 python -u scripts/generate_ensemble_mpi.py --threads 4 --precision float32 --point-device metal
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
# Companion notebook: compute_precision_tests()
```

Reproduce MPI timings when those CPUs are free; use a fresh output directory
so saved maps cannot turn a timing into a cache-hit measurement:

```bash
# Companion notebook: benchmark_compute_precision()
```

Check the masked spectra, common seeds and the unchanged float64 reference:

```bash
# Companion notebook: check_compute_precision_outputs()
```

The GPU spherical-transform probe is also committed as
the archived exploratory GPU probe. With Torch 2.6.0 on this machine's
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

The later implementation in `scripts/gpu_scalar_sht.py`
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
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=3 mpiexec -n 5 --mca mpi_yield_when_idle 1 python -u scripts/generate_ensemble_mpi.py --threads 3 --precision float32 --gpu-workers 1 --gpu-pipeline
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
# Companion notebook: benchmark_gpu_sht_mpi()
```

Compare the generated maps with the saved original CPU ensemble:

```bash
# Companion notebook: check_gpu_worker_comparison()
```

For the alternative two-GPU experiment use `--compare-gpu-workers` in the
benchmark and omit `--pipeline` in the checker. Add `--reverse` to the
benchmark with another fresh directory to reverse the execution order.
Neither benchmark stops an existing production process; free its CPU/GPU
allocation before timing. Compact records live under
`tests/gptg_pilot/gpu_sht/scheduling/`.

Recheck the scalar transform at production resolution:

```bash
# Companion notebook: test_gpu_scalar_sht()
```

Check two paired noiseless four-bin maps:

```bash
# Companion notebook: test_gpu_map_generation()
```

The map test writes under `chains/gpu_sht_maps/`. For timing, use a fresh
output directory and free the required CPUs first:

```bash
# Companion notebook: benchmark_gpu_sht_mpi()
```

When generation reaches its configured target, the coordinator checks exact file IDs,
journal seeds, finite journal entries, array headers, dtype and file lengths,
then writes `COMPLETENESS.txt`. This checks missing/truncated maps without
reading every pixel again. It does not start TreeCorr. The same audit
can be run separately:

```bash
python scripts/verify_ensemble.py chains/gptg_ensemble_15000
```

**Measurement review (22 November 2026).** Keep production measurements
paused while the revised estimator and replacement resolution are validated.
The old mixed-catalogue multipole calls retained coincident-pixel products.
The repaired runner recovers mixed source moments from auto correlations of
field sums and differences. It assumes the same positions, mask and weights
in all source bins, and equal aperture radii. Small direct-triangle tests
validate the algebra; production angular-cutoff convergence remains open.

The runner records `shared-pixel-auto-polarization-v1` in metadata and vector
headers. It refuses legacy results and smoke/full mismatches before resuming
or merging; use a fresh measurement directory for the repaired definition.
Keep old maps and checkpoints for the documented comparisons. Map IDs must
refer to unchanged input maps; a replacement ensemble also needs a fresh
measurement directory. Changing rank counts or the requested ID range does
not change the observable. The runner still uses NSIDE512 geometry; do not
feed it new-resolution maps until that geometry support has been updated.

The [map note and companion notebook](notes/gptg_simulations/) explain the
repair, the covariance controls and the proposed resolution tests. The
356-entry auto-only resampling control uses a known finite population;
its larger resampled counts are not new physical skies or a precision
forecast for the future ensemble. Both direct fixed-block constructions
and orders two, four and six are compared. Two-stage correction with
Rafael's maps remains a proposal requiring matched observables and an
independent first-stage reference.

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
triangle grids are saved. Downloaded legacy results are analyzed in the
companion notebook, with the measurement qualification above.

The reproducible algebra checks are:

```bash
# Companion notebook: check_fixed_block()
```

```bash
# Companion notebook: check_full_joint()
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


### Optional analytical covariance build

DES 3PCF follows LSST Y1's build switch: the same native extension is built
with or without covariance generation. `set_installation_options.sh` sets
`IGNORE_COSMOLIKE_DES_3PCF_COVARIANCE=1` by default. This omits covariance
kernels and bindings and exposes `has_covariance=False`. Supplied covariance
matrices remain usable by the joint likelihood. Fixed quadrature, xi-bin
weights, shear spectra and xi means are data-vector operations and remain
available in either build. This key does not control TreeCorr or map generation.

From activated Cocoa, enable analytical covariance generation:

```bash
unset IGNORE_COSMOLIKE_DES_3PCF_COVARIANCE
```

Rebuild through the normal project compilation script:

```bash
source projects/des_3pcf/scripts/compile_des_3pcf.sh
```

To keep covariance enabled across Cocoa sessions, comment out its export in
`set_installation_options.sh`. To disable it again, set the key and recompile.
`stop_cocoa` unsets the key through the shared cleanup file. Covariance and
non-covariance objects use separate directories to avoid stale compilation
flags when changing modes; restart Python after rebuilding its extension.

### Map programs and the companion notebook

The map note has one companion notebook:
[EXAMPLE_GPTG_SIMULATIONS.ipynb](notes/gptg_simulations/EXAMPLE_GPTG_SIMULATIONS.ipynb).
It contains the small worked examples, figure generation from frozen text
records, covariance calculations and analyses of completed measurements.
Read each section's assumptions before calling its analysis function. The
480-row analysis requires completed TreeCorr measurements from the HPC;
opening or running the teaching cells does not start those measurements.

Production programs live in `scripts/`: `generate_maps.py` prepares the
calibration, `generate_ensemble_mpi.py` generates maps, and
`measure_full_joint_mpi.py` measures the full vector with TreeCorr.
The GPU transform implementation and completion verifier
are support code for those programs. `measure_joint_xip.py` retains the
bounded TreeCorr measurement used in the early comparison. Submit the full
measurement with `EXAMPLE_GPTG_TREECORR_UOFA.sbatch` as explained above.
All simulation data and measurement journals keep their existing paths.

One-off hardware probes and thread/scheduling benchmarks are retired;
their local source is kept in the git-ignored
`notes/gptg_simulations/development_archive/` and remains in Git history and their recorded numerical evidence
remains in `tests/gptg_pilot/`. They are not prerequisites for using the maps
or reproducing the note's scientific figures.

### Which calculations belong in scripts?

Keep expensive production calculations in `scripts/`; put short analysis
and plotting steps in the notebook that explains their science. Support
modules stay only when those production programs actually need them.

| Calculation | Location |
|---|---|
| Calibrate and generate maps; CPU/GPU spherical transforms; completion checks | Map programs and their support in `scripts/` |
| Measure survey correlations with TreeCorr across MPI ranks | `scripts/measure_full_joint_mpi.py` and the single Slurm example |
| Bounded map measurements needed by the covariance comparison | Existing measurement programs in `scripts/`; they still read many maps or run TreeCorr |
| Source clipping, response plots, covariance algebra and small examples | Visible cells in the associated notebook |
| Ordinary evaluation or inference | Numbered Cobaya examples and `likelihood/` |
| Ordinary Fisher calculation | Evaluate notebook using shared core Fisher functions |
| Explicitly expensive Fisher study | Only when needed: an `interface/cosmolike_des_3pcf_fisher.py` following LSST Y1 × DESI |
| Hardware probes and superseded exploration | Ignored local diary; retained results in `tests/` |

The old Fisher CLI and Fisher/accuracy/template wrappers have been removed.
The accuracy notebook contains the step, angular-order and conditioning
studies directly. Remaining non-Fisher `run_3pcf.py` preparation and native
mean/covariance diagnostics still have test consumers; they are not the
public forecast path. Timing runs may use a script, but plotting completed
timing records belongs in a notebook.


## CPU map generation on the HPC <a name="cpu-map-generation"></a>

The standalone [Slurm example](scripts/EXAMPLE_GPTG_GENERATE_UOFA.sbatch)
distributes independent maps with MPI, one CPU thread per rank and BLAS=1.
One rank coordinates; the others generate maps. The allocation and default
ensemble counts are declared in the script. TreeCorr runs separately.

The [map notebook](notes/gptg_simulations/EXAMPLE_GPTG_SIMULATIONS.ipynb),
section “Prepare the physical-power model through multipole 4096”, prepares
`chains/gptg_physical_l4096/`. Its numerical checks are recorded in the
[calibration test](tests/gptg_pilot/physical_calibration_validation.txt).
The model uses physical lensing power and prescribed GPTG point functions.
It is ready for a **controlled band-limited covariance experiment**; it is
not a validated reproduction of DES small-scale covariance. The
[map note](notes/gptg_simulations/gptg_simulations.pdf) explains the cutoff
test and the existing Eifler results.

Maps use total Omega_m=0.26 and mnu=0.06 eV. The fixed analytical two-point
block uses that same total Omega_m with mnu=0; do not subtract the neutrino
matter fraction. Only cross and Map3 blocks receive simulation corrections.
The notebook computes the full Gaussian+SSC+cNG two-point block separately.

**Step :one:**: Prepare or use the tested calibration in an activated local
Cocoa directory. These are the only generation inputs to transfer:

| File | Meaning |
|---|---|
| `calibration.yaml` | Cosmology, resolution, noise and model assumptions |
| `calibration.npz` | Target convergence spectra, point parameters and pixel noise |
| `indices_2048.npy` | RING pixel addresses of the refined footprint |

The package already contains noise at its stated resolution. The generator
retains its harmonic cutoff; `GPTG_LMAX` checks the input and any frozen
output copy before MPI starts. It does not extend the old calibration.
Keep the package files together and use a fresh output directory.

**Step :two:**: Copy the package to the HPC. Set `GPTG_TRANSFER_HOST` to
your transfer login and `GPTG_REMOTE_CALIBRATION_DIR` to its absolute
calibration directory. The directory must exist before rsync:

```bash
export GPTG_CALIBRATION_DIR="${ROOTDIR}/projects/des_3pcf/chains/gptg_physical_l4096"
export GPTG_TRANSFER_HOST="your_login@your_transfer_host"
export GPTG_REMOTE_CALIBRATION_DIR="/absolute/path/to/gptg_calibration"
```

```bash
ssh "$GPTG_TRANSFER_HOST" "mkdir -p '$GPTG_REMOTE_CALIBRATION_DIR'"
```

```bash
rsync -avP "$GPTG_CALIBRATION_DIR/calibration.yaml" "$GPTG_CALIBRATION_DIR/calibration.npz" "$GPTG_CALIBRATION_DIR/indices_2048.npy" "$GPTG_TRANSFER_HOST:$GPTG_REMOTE_CALIBRATION_DIR/"
```

The output maps, old ensemble and analytical covariance are not needed to
generate this new ensemble. Calibration arrays and map files are binary
NumPy files because text would increase their size and parsing cost.

**Step :three:**: On the HPC, activate Cocoa from its top directory:

```bash
source start_cocoa.sh
```

Set the calibration path to the directory just transferred. Choose the
output on a volume with space for the requested maps and the generator's
free-space reserve. Keep this path unchanged on resubmission.

```bash
export GPTG_CALIBRATION_DIR="/absolute/path/to/gptg_calibration"
export GPTG_OUTPUT_DIR="${ROOTDIR}/projects/des_3pcf/chains/gptg_refined"
export GPTG_NSIDE=2048
export GPTG_LMAX=4096
```

To use the script's count defaults rather than values left in the shell:

```bash
unset GPTG_COMPARISON_COUNT GPTG_VALIDATION_COUNT
```

Alternatively, export `GPTG_COMPARISON_COUNT` and `GPTG_VALIDATION_COUNT`
with your chosen counts. Their seed ranges are separate. Validation may be zero
for a pilot. Do not point these commands at the old ensemble.

**Step :four:**: Check one worker on the HPC before filling the allocation.
Use a separate pilot directory; command-line Slurm options override the
header for this submission:

```bash
sbatch --nodes=1 --ntasks=2 --ntasks-per-node=2 --ntasks-per-socket=1 --export="ALL,GPTG_OUTPUT_DIR=${GPTG_OUTPUT_DIR}_pilot,GPTG_COMPARISON_COUNT=1,GPTG_VALIDATION_COUNT=0" projects/des_3pcf/scripts/EXAMPLE_GPTG_GENERATE_UOFA.sbatch
```

Check the completed job's `MaxRSS` and map-generation time with `sacct`.
The local measured peak is recorded in the calibration test; Linux
allocator behavior and cluster timing must be measured on the cluster.
Fresh Cocoa setup installs ducc0. Do not install packages inside this job.

**Step :five:**: Submit the production allocation after the pilot succeeds:

```bash
sbatch projects/des_3pcf/scripts/EXAMPLE_GPTG_GENERATE_UOFA.sbatch
```

Resubmit the same command with the same output and calibration after the
time limit. Completed IDs are skipped. A complete atomic map saved before
its journal entry is remeasured for the inexpensive aperture diagnostics;
it is not regenerated. Never run two jobs writing to the same ensemble.
The output includes maps, journals, frozen calibration, ensemble metadata
and footprint indices. Keep all of them for resume and later measurement.

**Step :six:**: After completion, audit expected IDs, seeds, map headers
and lengths. This does not run TreeCorr or scan every pixel:

```bash
python projects/des_3pcf/scripts/verify_ensemble.py "$GPTG_OUTPUT_DIR"
```

The current full-vector TreeCorr driver still selects NSIDE512 geometry.
Its geometry and angular settings need separate validation before measuring
this new ensemble; the generation job does not invoke that driver.

The [earlier resolution pilot](tests/gptg_pilot/nside2048_generation.txt)
kept the old harmonic cutoff. It is separate from the new physical-power
calibration. Numerical projection accuracy, sky cutoff and agreement with
DES higher-order statistics are separate questions; finer pixels alone
answer none of the latter two.
