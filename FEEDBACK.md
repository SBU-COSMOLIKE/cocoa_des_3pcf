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
DMO denotes the gravity-only reference used to define those ratios.

The parameter `feedback_amplitude` applies the following declared family:

```math
P(A)=P_{\rm DMO}\exp(A\ln R_P),\qquad
B(A)=B_{\rm DMO}\exp(A\ln R_B).
```

At $`A=0`$ the prediction returns to DMO; at $`A=1`$ it uses the supplied
pair. The amplitude interpolates logarithmic responses. It has no calibrated
interpretation as gas temperature or another feedback parameter. The ratios
stay fixed when cosmology changes, while the underlying DMO spectra change.

The two-point calculation modifies its nonlinear matter power. The
three-point calculation modifies its matter-bispectrum baseline. NLA
intrinsic alignments share those baselines. The additional quadratic IA
terms retain DMO inputs: matter-only ratios do not predict how feedback
changes galaxy shapes. The covariance remains the separately selected
fiducial covariance. The older `fb` parameter is a bispectrum-only response;
combining nonzero `fb` and `feedback_amplitude` is refused.

## Response-file format <a name="input"></a>

The NPZ file contains numerical arrays and a scalar JSON `metadata` string.
Pickled objects are not read. Each axis is uniform, increasing and contains
at least four points. For a matter triangle, sort its sides as
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

`domain: error` refuses queries outside the supplied axes. Choosing
`domain: clamp` instead holds each coordinate at its nearest boundary.
That extension is a physical extrapolation assumption and must be tested
over the Fourier and redshift ranges contributing to the observables.

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
DMO; the paired response is applied in the projected baseline and exposed
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
feedback models are unavailable. The next section uses those installed
models directly.

Suitable inputs need a common matter definition, hydro/DMO normalization,
redshifts and supported triangle domain. The
[public hydro power/bispectrum measurements](https://github.com/sjforeman/hydro_bispectrum)
are a possible preparation source. The methods paper's OWLS power response
and rescaled TNG bispectrum response define an approximate contamination
test, not a jointly calibrated feedback parameter.
[Gomes et al., Section IV.3](https://arxiv.org/html/2503.03964v2#S4.SS3)

A cold-matter emulator cannot be labeled total matter without an explicit
conversion and validation. Likewise, supplying a ratio at one cosmology
does not establish its accuracy elsewhere. These calibration questions
remain separate from the tested numerical response reader.

## Existing Cocoa feedback scenarios <a name="scenarios"></a>

[EXAMPLE_FEEDBACK_SCENARIOS.ipynb](EXAMPLE_FEEDBACK_SCENARIOS.ipynb)
follows the [LSST Y1 feedback notebook](../lsst_y1/EXAMPLE_EVALUATE3.ipynb)
and [Roman example](../roman_real/EXAMPLE_EVALUATE3.ipynb). The
[scenario YAML](configs/feedback_scenarios.yaml) selects three SP(k)
relations, BCEmu, FLAMINGO, BACCO and BCemu2025. Each is evaluated at the
DES demonstration cosmology through the existing `bfmt` Cobaya theory
block. Its power response changes all 400 two-point rows; the notebook
plots source-bin pair (1,1) and checks exact restoration to DMO. The
quadratic IA loops retain their linear DMO input. These are fixed-point
model comparisons, not fitted feedback parameters or joint constraints.

The adapter converts wavenumbers from h/Mpc to bfmt's 1/Mpc request,
retains the saved CAMB amplitude and neutrino density, and refuses a changed
cosmology with stale preparation metadata. It catches warnings from the
actual theory logger, including error paths that otherwise return a
plausible unity response. Installed weights are checked before the pinned
dependencies can trigger their downloads. BLAS and OpenMP each use one
worker in this example; no covariance simulations are required.

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
the two-point comparison and labels the approximation. It does not pass
the cold bispectrum response into a total-matter `FeedbackPair` or invent
a bispectrum for the other six scenarios. A massless-neutrino example
would remove this field distinction but would still need validated
triangle-domain extensions. A cosmology-dependent forecast must also
regenerate both responses when its physical parameters change.

The supplied Fourier tables extend much farther than emulator training
ranges. The notebook therefore compares FLAMINGO's existing continuation
with a separate choice that holds its response constant above
10 h/Mpc. Both are modeling choices outside calibration. Sensitivity
to them belongs in the eventual scale-cut and accuracy study; the
availability of physical response models does not settle those tests.

The saved `runs/feedback_scenarios.npz` contains every plotted array,
model point, theory/weight hash and restoration result. It can be read
with `allow_pickle=False`. The artificial-pair notebook and this physical
scenario notebook serve different checks and are linked from the README.
