# Paired baryonic responses

1. [What changes in the joint mean](#model)
2. [Response-file format](#input)
3. [Demonstration notebook](#demonstration)
4. [Accuracy and state changes](#accuracy)
5. [Physical calibration](#calibration)
6. [Existing Cocoa feedback scenarios](#scenarios)

## What changes in the joint mean <a name="model"></a>

Gas cooling and feedback redistribute matter. Their effects on the power
spectrum and bispectrum need separate inputs: knowing the change in a
two-field correlation does not determine the change in a three-field
correlation. A paired input supplies positive ratios
$`R_P=P_{\rm hydro}/P_{\rm DMO}`$ and
$`R_B=B_{\rm hydro}/B_{\rm DMO}`$ for the same total-matter definition.
DMO denotes the gravity-only reference that defines those ratios.

The parameter `feedback_amplitude` applies the response family

```math
P(A)=P_{\rm DMO}\exp(A\ln R_P),\qquad
B(A)=B_{\rm DMO}\exp(A\ln R_B).
```

At $`A=0`$ the prediction returns to DMO; at $`A=1`$ it uses the input
pair. The amplitude interpolates logarithmic responses. It has no calibrated
interpretation as gas temperature or another feedback parameter. The ratios
stay fixed when one loaded response file serves several cosmologies, while
the underlying DMO spectra change. The separate [BACCO producer](BACCO.md)
regenerates both responses at each physical parameter point and fixes this
extra amplitude to one.

The two-point calculation modifies its nonlinear matter power, and the
three-point calculation modifies its matter-bispectrum baseline. NLA
intrinsic alignments share those baselines, while the quadratic IA terms
keep DMO inputs: matter-only ratios do not predict how feedback changes
galaxy shapes. The covariance stays at the fiducial reference model.
The older `fb` parameter is a bispectrum-only response; the code refuses
nonzero `fb` together with `feedback_amplitude`.

## Response-file format <a name="input"></a>

The NPZ file contains numerical arrays and a scalar JSON `metadata` string.
The reader loads no pickled objects. Each axis is uniform, increasing and
contains at least four points. For a matter triangle, sort its sides as
$`a\leq b\leq c`$, then set $`k=b`$, $`u=a/b`$ and $`v=(c-b)/a`$.
Sorting is appropriate for matter; ordered intrinsic-shape legs keep their
original identities elsewhere in the calculation.

| Array | Definition and shape |
|---|---|
| `z` | Redshift, nonnegative, `[nz]` |
| `logk` | Natural log of wavenumber in h/Mpc, `[nk]` |
| `logu` | Natural log of the shortest/middle side ratio, `[nu]` |
| `v` | Longest-minus-middle side divided by shortest side, `[nv]`, within 0 to 1 |
| `log_power` | Natural log of the positive power response, `[nz,nk]` |
| `log_bispectrum` | Natural log of the positive bispectrum response, `[nz,nk,nu,nv]` |

The metadata specifies these fields without inferred defaults:

    name: descriptive model name
    source_url: source of the paired responses
    matter_field: total
    cosmology_dependence: fixed_ratio
    ia_response: nla_only
    amplitude_model: log_response
    domain: error

`domain: error` refuses queries outside the file's axes. Choosing
`domain: clamp` instead holds each coordinate at its nearest boundary.
That extension is a physical extrapolation assumption and needs a test
over the Fourier and redshift ranges that contribute to the observables.

The prediction YAML selects the file and dense interpolation multiplier:

```yaml
feedback:
  path: ../runs/paired_feedback_demo.npz
  boost: 2
```

## Demonstration notebook <a name="demonstration"></a>

[EXAMPLE_JOINT_FEEDBACK.ipynb](EXAMPLE_JOINT_FEEDBACK.ipynb) creates explicitly
artificial power and bispectrum suppressions, evaluates all 480 joint rows
at three amplitudes, and plots their effects on xi+, xi− and Map3. It also
returns to zero amplitude and requires exactly the original prediction.
The associated [YAML](configs/des_y3_joint_feedback_demo.yaml) supplies a
Fisher step for this parameter. These examples teach the paired response;
they are not constraints on a hydrodynamically calibrated model.

## Accuracy and state changes <a name="accuracy"></a>

Native construction copies the inputs and prepares natural cubic splines
one axis at a time. The resulting uniform dense arrays use linear reads
with arithmetic indices during projection. Preparation is serial; parallel
workers only read the completed arrays. `feedback_table_boost` multiplies
the file's dense-grid boost together with `accuracy_boost`.

Changing amplitude reuses the DMO physical model but invalidates the mean.
Changing a response requires a new immutable input owner; editing a source
array after construction cannot change its native copy. Saved predictions
record the response-file hash. Raw matter/IA building-block queries remain
DMO; the projected baseline applies the paired response and exposes it
through separate response queries.

Tests use independent constant responses, unequal axis lengths and
multilinear log responses, including squeezed triangles. They also check
both array adapters, fixed quadratic IA terms, the ordinary two-point
CFASTPT bridge and amplitude restoration within one process.

## Physical calibration <a name="calibration"></a>

Physical power-response inputs already exist in Cocoa's `bfmt` theory
block. Its BACCO dependency also includes the paired Burger2025
power/bispectrum emulator and both weight files. The artificial notebook
above tests the native interface; it is not evidence that physical
feedback models are unavailable. The scenario section below uses those
installed models directly.

Suitable inputs need a common matter definition, hydro/DMO normalization,
redshifts and a supported triangle domain. The
[public hydro power/bispectrum measurements](https://github.com/sjforeman/hydro_bispectrum)
are a possible preparation source. The methods paper's OWLS power response
and rescaled TNG bispectrum response define an approximate contamination
test, not a jointly calibrated feedback parameter.
[Gomes et al., Section IV.3](https://arxiv.org/html/2503.03964v2#S4.SS3)

A cold-matter emulator cannot be labeled total matter without an explicit
conversion, and a ratio prepared at one cosmology does not establish its
accuracy elsewhere. The [BACCO guide](BACCO.md) defines the conversion,
continuation and parameter handling this project uses for those installed
emulators.

## Existing Cocoa feedback scenarios <a name="scenarios"></a>

[EXAMPLE_FEEDBACK_SCENARIOS.ipynb](EXAMPLE_FEEDBACK_SCENARIOS.ipynb)
follows the LSST Y1 feedback notebook
(`projects/lsst_y1/EXAMPLE_EVALUATE3.ipynb`) and its Roman counterpart
(`projects/roman_real/EXAMPLE_EVALUATE3.ipynb`). The
[scenario YAML](configs/feedback_scenarios.yaml) selects three SP(k)
relations, BCEmu, FLAMINGO, BACCO and BCemu2025. The notebook evaluates
each at the DES demonstration cosmology through the existing `bfmt` Cobaya
theory block. Its power response changes all 400 two-point rows; the
notebook plots source-bin pair (1,1) and checks exact restoration to DMO.
The quadratic IA loops keep their linear DMO input. These are fixed-point
model comparisons, not fitted feedback parameters or joint constraints.

The adapter converts wavenumbers from h/Mpc to bfmt's 1/Mpc request and
keeps the saved CAMB amplitude and neutrino density. It refuses a changed
cosmology whose preparation metadata no longer matches, because the
preparation must then be redone. It catches warnings from the actual
theory logger, including error paths that otherwise return a plausible
unity response, and it checks the installed weights before the pinned
dependencies can trigger downloads. BLAS and OpenMP each use one worker;
this example needs no covariance simulations.

The separate BACCO figure evaluates both installed Burger2025 models at
identical cosmology and feedback parameters. The bispectrum curves use
equilateral and scalene triangles within the installed side bounds. The
five feedback coordinates are logarithmic: gas-retention mass, its mass
dependence, gas-ejection scale, central stellar-mass relation and inner
gas-profile scale. Both responses describe CDM plus baryons. With massive
neutrinos, total matter also contains neutrino correlations, including
mixed three-field correlations that these ratios do not specify.
[BACCO documentation](https://baccoemu.readthedocs.io/en/latest/),
[Burger et al.](https://arxiv.org/abs/2506.18974)

The ordinary Cocoa power prescription applies the BACCO cold response
to total nonlinear power. The notebook follows that prescription for
the two-point comparison and labels the approximation; it does not pass
the cold bispectrum response into a total-matter `FeedbackPair` or invent
a bispectrum for the other six scenarios. The [BACCO guide](BACCO.md)
converts cold responses to total matter and regenerates them when a
physical parameter changes.

The scenario Fourier tables extend far beyond the emulator training
ranges. The notebook therefore compares FLAMINGO's existing continuation
with a separate choice that holds its response constant above 10 h/Mpc.
Both are modeling choices outside the calibration; their effect belongs
in the scale-cut and accuracy study.

[EXAMPLE_FISHER_BACCO.ipynb](EXAMPLE_FISHER_BACCO.ipynb) connects the paired
Burger2025 responses to the full joint mean. It makes the neutrino
conversion and triangle-domain continuation explicit, varies physical
feedback coordinates, and compares model choices separately from numerical
refinement. The [BACCO guide](BACCO.md) derives the assumed mixed-neutrino
response and explains why it still needs physical calibration.

When run, the notebook writes `runs/feedback_scenarios.npz` with every
plotted array, model point, theory and weight hash, and restoration
result; `numpy.load` reads it with `allow_pickle=False`. The
artificial-pair notebook and this physical scenario notebook serve
different checks, and the README links both.
