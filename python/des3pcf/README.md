# BiHalofit theory component

BiHalofit is one producer for the supplied-bispectrum interface. Its fit
and calibration belong to this theory component; they are not requirements
on an alternative matter-bispectrum model.

The Python theory producer evaluates the dark-matter-only BiHalofit fit and
always supplies its positive, full matter bispectrum as a table. C performs
the line-of-sight projection, intrinsic alignment, aperture integration and
covariance calculation. The internal C matter fit has been removed.
`physics.bispectrum_source` therefore accepts only `supplied`. Existing
configurations without a selector use that value; those without explicit
axes use the supplied example's uniform sampling, with the background z
endpoint belonging to their own project.

The provider owns the background and matter prescription. It supplies the
matter-density fraction `omega_m_z` from the same cosmology as P, B, growth
and distances. The C model requires B and this fraction; missing inputs are
errors. Tree-level F2 and power calculations remain in C because TATT needs
them. One-/three-halo fit diagnostics are available from Python BiHalofit,
not from a full-B native projection model.

The required table introduces a separate interpolation error and preparation
cost. Earlier comparisons below record the retired C fit as a historical
reference; their dates and numerical results are retained. They do not
certify current full-forecast accuracy or timing. The regression tests now
compare the Python equation with frozen pre-removal C outputs, and test C
projection/state behavior with explicitly supplied tables.

## Acknowledgements and origin

**We thank Sunao Sugiyama for fastnc, including its TATT branch, and Rafael
Gomes for 3pcf_integrator, the DES Y6 pipeline, and the analyses that motivate
this joint two/three-point implementation. Their original work and code are
the basis for this port.**

- [Sunao Sugiyama's fastnc TATT branch](https://github.com/git-sunao/fastnc/tree/tatt),
  with comparison revision `7b2336ce37a909bed51f07f32a2a4f096e71b6ce`.
- [Rafael Gomes's 3pcf_integrator](https://github.com/rchgomes/3pcf_integrator)
  and [DES Y6 pipeline](https://github.com/rchgomes/desy6-3pcf).
- The [methods analysis](https://arxiv.org/abs/2503.03964) and
  [subsequent analysis](https://arxiv.org/abs/2601.09133) motivate the
  observable and inference work. The BiHalofit fitting formula itself is
  from [Takahashi et al., *Fitting the nonlinear matter bispectrum by the
  Halofit approach*](https://arxiv.org/abs/1911.07886).

The Python module includes Sunao's MIT notice. The equation comparison pins
the pre-change `physics_3pcf.c` SHA-256 to
`af3c533d7a85ddf493ab5840823134206fcd154ac2d5cd648875185de41ac351`.
That identifies the implementation whose numerical conventions were
ported; it is not a replacement for citing the fit or the original codes.

## What the fit receives and computes

The inputs are the linear power spectrum at redshift zero, a growth factor,
the primordial spectral index, the present-day fluctuation amplitude and
the matter-density fraction as a function of redshift. All are supplied
explicitly. The units are

| Quantity | Convention |
| --- | --- |
| `k` | $`h/\mathrm{Mpc}`$ |
| `p_linear0` | $`(\mathrm{Mpc}/h)^3`$ |
| `growth` | dimensionless $`D(z)`$, with $`D(0)=1`$ |
| `omega_m_z` | dimensionless $`\rho_m(z)/\rho_{\mathrm{crit}}(z)`$ |
| returned $`B`$ | $`(\mathrm{Mpc}/h)^6`$ |

The fit begins with the dimensionless spectrum

```math
\Delta^2(k,z)=\frac{k^3 P_{\mathrm{L}}(k,0)D(z)^2}{2\pi^2}.
```

It finds a nonlinear radius $`R_\sigma(z)`$ from

```math
\sigma_G^2(R,z)=\int d\ln k\,\Delta^2(k,z)e^{-k^2R^2}=1.
```

At this radius, the logarithmic slope of the variance gives an effective
spectral slope. The fitted coefficients depend on that slope, the
redshift-dependent top-hat fluctuation amplitude and, for one coefficient,
$`\Omega_m(z)`$. The final result is the sum of the fit's one-halo and
three-halo pieces. These names describe the terms in the fitting formula;
they do not imply that the implementation performs a separate halo-mass
integral for each queried triangle.

The producer uses the same input-grid convention as the Cocoa likelihoods:
`log10k = np.linspace(...)`, followed by `k = 10.0**log10k`, and redshift
segments constructed with `np.linspace`. Production power and growth
preparation uses a natural cubic within each uniform segment, followed by
dense linear interpolation with arithmetic cell indices. Refinement keeps
every segment endpoint, including a join between fine low-redshift sampling
and coarser high-redshift sampling. Each join needs a coarse-grid refinement
check because natural end conditions need not match the physical curvature.
The explicit `source_readers=True` comparison retains fastnc's original
not-a-knot preparation. Python and C must select the same convention when
comparing their fitted equations; changing a dense sampling factor cannot
remove a discrepancy between different cubic boundary conditions.

The old C normalization is deliberately retained. First, a top-hat integral
over $`10^{-3}\leq k\leq100\ h/\mathrm{Mpc}`$ rescales the input power to the
supplied `sigma8`. The later fit variances use a lower limit of
$`10^{-4}\ h/\mathrm{Mpc}`$. The Gaussian variance includes the same analytic
power-law tail beyond $`100\ h/\mathrm{Mpc}`$ as C. Its incomplete-gamma
integral is evaluated for positive, zero and negative shape parameters.
There is no adaptive integration in the production producer: the numerical
rules use explicit uniform logarithmic nodes and Simpson integration, with
the same even-node closing correction as C.

A special-function check found that naive downward recurrence loses
precision when its gamma shape approaches zero: a subtraction of nearly
equal numbers is divided by that small shape. At a shape of minus one part
in a trillion, its relative error reached 0.00137 in the small probe.
The corrected evaluation removes that cancellation analytically for
absolute shape below 0.001. It uses a convergent series when the
dimensionless lower bound is at most one, and a continued fraction above
one. Negative shapes with larger lower bounds also use that fraction to
avoid a second recurrence cancellation. These are evaluations of the same
integral, with explicit iteration limits; the physical Simpson grids and
root tolerance are unchanged.

The [special-function tests](../../tests/data_vector/test_bihalofit_python.py)
compare with 80-digit mpmath values. The saved validation covers 24 shapes
on both sides of zero and the algorithm boundaries, a negative-integer
recurrence crossing, and 12 lower bounds from $`10^{-300}`$ to 1000.
The largest relative error was $`1.03\times10^{-12}`$, at the shape-branch
boundary. The values near zero agreed within $`4.93\times10^{-14}`$; underflow
to zero was checked separately. mpmath is a validation dependency already
present in Cocoa, and does not enter the production calculation.

These conventions matter for an implementation comparison. Removing the
power renormalization, changing the integration bounds or supplying a
different density history can change a prediction even if the equation
port is correct. The producer records its normalization factor so a user
can inspect it. A fit that trusts the raw power normalization would be a
separate, tested change.

## Uniform triangle coordinates and the C reader

Sort the three wavenumber magnitudes as $`a\leq b\leq c`$. The stored
coordinates are

```math
z,\qquad \ln r=\ln b,\qquad \ln u=\ln(a/b),\qquad v=(c-b)/a.
```

The triangle inequality makes $`0\leq v\leq1`$, and $`u\leq1`$. At $`v=0`$
the two longer sides coincide; at $`v=1`$ the triangle is flattened. The
native key `logk` denotes $`\ln r`$, the middle side. This spelling matches
the existing feedback-table interface. The shortest side can be much
smaller than the middle side, so `logu` is logarithmic. Every axis must have
at least four finite, increasing, uniformly spaced nodes in its stored
coordinate.

The array is `log_bispectrum[nz, nk, nu, nv]` in C order, with `v` varying
fastest. A positive full $`B`$ is required to take its logarithm. A signed
diagnostic piece cannot be substituted for the full table or clipped to
make the logarithm possible. The supplied C route retains the power-derived
tree diagnostic and explicitly refuses the unavailable one-halo/three-halo
diagnostics. The Python equation object can still evaluate all four pieces
for reference checks.

Preparation uses cubic interpolation to construct the dense uniform C
table. Repeated C reads use an arithmetic cell index and linear weights;
they need no binary search or mutable interpolation accelerator. Python's
power/background readers use the same cubic-preparation/uniform-linear-read
strategy in production mode. `source_readers=True` is an explicit comparison
mode that retains the upstream cubic conventions. It is not the default.

The default table policy refuses a query outside its domain. The standalone
producer also accepts `domain="clamp"`, which holds boundary values. Clamping
changes the prediction and requires its own accuracy study; it must not be
used to conceal missing table coverage. The project configuration exposes
the refusing policy. The chosen domain must cover all triangles reached by
the mean and covariance, including squeezed triangles and the lowest
line-of-sight distance used in a projection.

## Producer and ownership

The reusable implementation is
[bihalofit.py](../../python/des3pcf/bihalofit.py). `BiHalofit` evaluates the
equation, `BiHalofitProducer` prepares a uniform table, and
`BaselineBispectrum` owns the result passed to C. A minimal preparation is:

```python
from des3pcf.bihalofit import BiHalofitProducer

producer = BiHalofitProducer(
    z=baseline_z,
    logk=baseline_logk,
    logu=baseline_logu,
    v=baseline_v,
    boost=2,
    domain="error",
)
baseline = producer.prepare(
    tables=physical_tables,
    fit_nz=33,
    table_boost=8,
)
```

`physical_tables` contains `k`, `p_linear0`, `z`, `growth`, `omega_m_z`,
`ns` and `sigma8`. An optional `omega_m_redshift` gives the coordinate axis
of the density-fraction samples when it differs from `z`. For equal axes,
the returned background preserves the supplied `omega_m_z` bytes. For
different axes, `baseline.background_omega_m_z` is the explicitly resampled
array on `baseline.background_z`. A direct native caller must use those
returned background arrays, so the baseline and its C model share one
background. The project helper checks this agreement.

At the native boundary use `dict(baseline.native)`: the Python owner exposes
an immutable mapping, and the binding accepts a dictionary. The dictionary
has exactly `z`, `logk`, `logu`, `v`, `log_bispectrum`, `boost` and `clamp`.
The C constructor copies and prepares its inputs. Reusing the Python owner
avoids another Python BiHalofit evaluation; independent C models still own
and prepare separate native tables. Sharing those prepared C tables between
models is a separate follow-up. The producer's arrays are also backed
by immutable bytes, so resetting a NumPy write flag cannot make an old
result writable. A retained baseline remains valid after another cosmology
is prepared or the producer is deleted.

The producer keeps its latest result. It hashes every consumed physical
array, its dtype and shape, the output axes, and every numerical control.
An unchanged call returns that owner. A changed call constructs a new owner
before replacing the cache. The input key and its owner are published
together, so independent Python preparations cannot pair one cosmology
with another cosmology's table. An invalid replacement leaves the last valid
owner usable. The A-to-B-to-A sequence therefore recovers the same A values
without requiring a notebook restart. Source distributions and IA
amplitudes do not enter the dark-matter-only fit; they remain dependencies
of the projected C model.

`baseline.record()` returns an independent JSON-compatible dictionary.
It includes source credits, the pinned C source hash, physical-input
hashes, numerical controls, the normalization factor, and hashes of every
actual native table/background array. Data-vector, covariance and mask
files are outside this producer; a likelihood comparison must hash them
separately.

## Refinement controls

There are several different approximations. Increasing only a dense-table
boost does not repair an inadequately sampled coarse function.

| Control | Approximation it changes |
| --- | --- |
| output `z`, `logk`, `logu`, `v` axes | Sampling of the full bispectrum before four-dimensional interpolation |
| producer `boost` | C uniform upsampling of that full table |
| `fit_nz` | Redshift anchors of the fitted radius and coefficients |
| `table_boost` | Python dense readers for input power, growth and fit coefficients |
| `variance_nodes` | Normalization and Gaussian/top-hat integration; default 1000 |
| `slope_nodes` | Effective-slope integration; default 2000 |
| `radius_rtol` | Nonlinear-radius bisection; default $`10^{-5}`$ |
| `block_size` | Number of vectorized triangle queries at once; it controls temporary memory, not numerical resolution |

The project `Accuracy` integration multiplies the full-table interval counts
by `accuracy_boost * bispectrum_boost`. The individual `bispectrum_boost`
changes this table alone. The global `accuracy_boost` also refines other
prepared sampling grids, including the fit readers. Use the individual
controls to isolate an error source before applying a global refinement.
The direct producer exposes quadrature/root controls for additional studies.
The final four-dimensional table is always stored explicitly; block
evaluation bounds temporary arrays, not the size of that output or its
dense C copy. Count the dense nodes before choosing a large boost on four
axes.

## Cobaya Theory adapter

[bihalofit_theory.py](../../python/des3pcf/bihalofit_theory.py) is an actual
`cobaya.theory.Theory` subclass. It declares requirements for linear
total-matter `Pk_interpolator`, `sigma8_z`, `Hubble`, `Omega_b`, `Omega_cdm`,
`Omega_nu_massive` and `ns`. It provides `bihalofit_baseline` and
`bihalofit_background`. A consuming component requests these products
through the normal Cobaya provider. Merely importing the module enables
nothing.

The required configuration values are `baseline_z`, `baseline_logk`,
`baseline_logu` and `baseline_v`. `background_z` can specify the C model's
different background grid. The provider is asked for matter fractions on
both that grid and the exact fit anchors. Their sum is used directly,
including the provider's massive-neutrino and expansion treatment. The
adapter uses $`D(z)=\sigma_8(z)/\sigma_8(0)`$.

The adapter converts physical units exactly once:

```math
k_{\mathrm{physical}}=h\,k_{h},\qquad
P_{h}=h^3 P_{\mathrm{physical}}.
```

Its default Boltzmann computation limit is `kmax_boltzmann=5` in
$`\mathrm{Mpc}^{-1}`$. The wider output `power_k` grid is sampled through the
provider interpolator with explicit power-law extrapolation. Asking for
large output wavenumbers therefore does not silently request a Boltzmann
calculation up to $`1000\ h/\mathrm{Mpc}`$. The extrapolation remains a physical
assumption whose effect requires testing; a fast interpolation is not a
calibration of high-$`k`$ power.

The direct project pipeline uses the reusable producer without Cobaya's
component manager. The focused unit test instantiates Cobaya's real base
class with an explicit array provider. The separate
[actual provider-graph check](../../tests/studies/bihalofit_cobaya_check.py) creates
a real Cobaya Model, loads the installed CAMB 1.6.7, and uses a validating
likelihood consumer to request both baseline and background products.
Its A, unchanged A, B, restored A sequence passed: unchanged inputs retain
the same owner, a changed primordial amplitude replaces it, and restoring
the amplitude restores identical table values.

The [saved provider result](../../tests/studies/results/bihalofit_cobaya_provider.json)
records all component settings, requirements and input hashes. CAMB was
asked for a physical wavenumber limit of 5 per Mpc; its internally extended
raw grid ended at 5.69845 per Mpc. The much wider output grid was obtained
through the explicitly requested interpolator extrapolation. This is a
small component and cache check with a zero-valued validating likelihood.
It does not test a survey likelihood or replace a full inference run.

## Validation and physical limits

Component regression, supplied-table projection and refinement are
different checks. The [recorded comparisons](../../tests/studies/implementation_record.md#bihalofit_python)
retain their input choices and numerical results. They do not establish
convergence of every survey Fisher derivative. The fit's use of
P(k,0)D(z)^2 assumes scale-independent growth; agreement with its
implementation does not extend its dark-matter-only calibration to
arbitrary neutrino, dark-energy or feedback models.

## Shared variance inputs

Both the direct project producer and the optional Cobaya theory block can
consume Gaussian and top-hat variances from the ordinary CosmoLike FFTLog
engine. Select `variance_source: shared` in the theory block, or
`physics.variance_source: shared` in the CLI configuration. The default
remains `local` until full joint mean and Fisher acceptance is established.
The shared route supplies owned z/rnl/neff/sigma8 arrays to the fit and
retains the amplitude of the supplied power spectrum. A separate sigma8
scalar no longer renormalizes that power.

The default `variance_growth: factorized` uses P(k,0)D(z)^2. The optional
`evolving` choice uses the supplied full P(k,z). Those prescriptions can
differ with massive neutrinos; neither changes the existing growth choice
on the perturbative tree and IA power legs. For the Cobaya adapter, D is
still the provider's sigma8(z)/sigma8(0), and Omega_m(z) remains the sum of
its baryon, cold-dark-matter and massive-neutrino fractions. All h-unit
conversions occur at the Python boundary.

The theory's `variance_accuracy` mapping accepts the shared controls
described in [Accuracy](../../../../external_modules/code/cosmolike_core/cosmolike/3pcf/README.md#shared-smoothing). Table counts derive
from an unchanged baseline. The cached bispectrum includes every supplied
variance array in its input identity, and old outputs retain their own
copies after replacement. The physical-model and CLI tests compare C and
Python equations with identical anchors, exercise both covariance adapters,
and check unchanged power when only the redundant sigma8 scalar changes.
These tests establish the interface contract; they do not establish survey
forecast convergence or a new neutrino halo prescription.

## BACCO response producer <a name="bacco"></a>

# BACCO feedback in a joint shear forecast

1. [Physical inputs and parameter meanings](#bacco-inputs)
2. [Converting the power response](#bacco-power)
3. [Converting the bispectrum response](#bacco-bispectrum)
4. [Continuing beyond the emulator domain](#bacco-continuation)
5. [Changing parameters in a Fisher calculation](#bacco-fisher)

Gas ejection and cooling change the distribution of matter inside and
around halos. BACCO's Burger2025 emulators predict the corresponding
changes in the power spectrum and bispectrum using the same baryonic
correction model. The joint calculation needs both predictions: the power
spectrum determines the shear correlations $`\xi_+`$ and $`\xi_-`$, while
the bispectrum determines the third-order aperture statistic Map3.

The emulators describe cold dark matter and baryons together. Lensing
also responds to massive neutrinos. This note defines the approximation
used to connect those matter fields, the continuation outside emulator
training ranges, and the checks needed before interpreting a forecast.
The separate [executed validation record](../../tests/studies/bacco_validation.md)
reports interpolation checks and a five-parameter Fisher comparison.
Numerical agreement between implementations cannot establish the missing
physical calibration described below.

## Physical inputs and parameter meanings <a name="bacco-inputs"></a>

Write $`c`$ for cold dark matter plus baryons, $`\nu`$ for massive neutrinos,
and $`m`$ for their total. A density contrast is
$`\delta_x=(\rho_x-\bar\rho_x)/\bar\rho_x`$ for component $`x`$.
The denominator uses that component's own mean density. This distinction
explains why summing density contrasts requires mass fractions:

```math
\delta_m=f_c\delta_c+f_\nu\delta_\nu,\qquad
f_\nu=\frac{\Omega_\nu}{\Omega_m},\qquad f_c=1-f_\nu.
```

These fractions use the nonrelativistic matter description over the
redshifts used for the feedback calculation. CAMB supplies the recorded
neutrino density $`\Omega_\nu h^2`$ and
$`h=H_0/(100\,{\rm km\,s^{-1}\,Mpc^{-1}})`$. The calculation must use
that density, the neutrino species settings, and the primordial amplitude
from the same cosmology as its matter tables.

BACCO returns dimensionless ratios

```math
R_{P,c}(k,z)=\frac{P_c^{H}(k,z)}{P_c^{D}(k,z)},\qquad
R_{B,c}(k_1,k_2,k_3,z)=
\frac{B_{ccc}^{H}(k_1,k_2,k_3,z)}{B_{ccc}^{D}(k_1,k_2,k_3,z)}.
```

The superscript $`H`$ denotes the matter distribution after the baryonic
correction; $`D`$ denotes its gravity-only reference, also called DMO.
The two emulators have separate trained weights. No power of $`R_{P,c}`$
determines $`R_{B,c}`$. Their common cosmology and feedback parameters make
them a paired physical input. The [BACCO documentation](https://baccoemu.readthedocs.io/en/latest/)
specifies the cold-matter convention.

Power measures the average product of two Fourier density fluctuations.
The bispectrum measures the average product of three. A Gaussian density
field has no connected three-point correlation, whereas gravitational
evolution produces one by coupling Fourier modes. Baryonic rearrangement
can change those couplings as well as each mode's power. That is why the
joint forecast needs a separately predicted bispectrum response.

The five sampled feedback coordinates retain the names used by Cocoa's
`bfmt` theory block. Each is a base-ten logarithm. Their physical meaning
comes from the model that redistributes halo gas and stellar mass.

| Sampled parameter | Quantity sent to BACCO | Physical role |
|---|---|---|
| `M_c_baccoemu` | $`\log_{10}[M_c/(h^{-1}M_\odot)]`$ | Halo mass scale governing gas retention |
| `eta_baccoemu` | $`\log_{10}\eta`$ | Extent of ejected gas |
| `beta_baccoemu` | $`\log_{10}\beta`$ | Mass dependence of the retained gas fraction |
| `M1_z0_cen_baccoemu` | $`\log_{10}[M_1/(h^{-1}M_\odot)]`$ | Mass scale in the central-galaxy stellar-mass relation |
| `theta_inn_baccoemu` | $`\log_{10}\theta_{\rm inn}`$ | Inner gas-profile scale |

For example, `eta_baccoemu: -0.3` means $`\eta=10^{-0.3}`$.
A negative logarithm describes a positive parameter smaller than one.
It does not describe negative gas mass. Burger2025 fixes the outer gas
profile scale and a second gas-profile mass parameter; their physical
values are $`\theta_{\rm out}=1`$ and
$`M_{\rm inn}=2.3\times10^{13}\,h^{-1}M_\odot`$.
[Burger et al., Section 2.1.2](https://arxiv.org/html/2506.18974v2#S2.SS1.SSS2)

The paired models also require $`\Omega_c=\Omega_m-\Omega_\nu`$,
$`\Omega_b`$, $`\sigma_{8,c}`$, and the expansion factor
$`a_{\rm exp}=1/(1+z)`$. Here $`\sigma_{8,c}`$ is the root mean square
linear cold-matter contrast in spheres of radius $`8\,h^{-1}{\rm Mpc}`$.
CAMB computes it from the cold transfer functions at the saved primordial
amplitude. The project's total-matter $`\sigma_8`$ cannot replace it when
neutrinos are massive.

Burger2025 has no independent input for $`h`$, $`n_s`$, $`w_0`$, $`w_a`$,
or neutrino mass. Recomputing CAMB at a new cosmology changes the DMO
spectra fully and moves the response through the supported
$`\Omega_c`$, $`\Omega_b`$ and $`\sigma_{8,c}`$ coordinates. A dark-energy
forecast therefore assumes that the remaining response directions are
negligible; keep neutrino mass fixed in the example until that assumption
has been tested.
[Burger et al., Appendix C](https://arxiv.org/html/2506.18974v2#A3)

The installed weight files determine the accepted parameter bounds.
Rounded ranges in a paper or example are insufficient for a derivative
point near a boundary. The local Burger2025 weights have these limits:

| Coordinate | Lower bound | Upper bound |
|---|---:|---:|
| $`\Omega_c`$ | 0.23009 | 0.39969 |
| $`\Omega_b`$ | 0.04001 | 0.05999 |
| $`\sigma_{8,c}`$ | 0.73007 | 0.89998 |
| $`a_{\rm exp}`$ | 0.25 | 1.0 |
| `M_c_baccoemu` | 10.00919 | 15.99716 |
| `eta_baccoemu` | -0.69778 | 0.19919 |
| `beta_baccoemu` | -0.99677 | 0.69923 |
| `M1_z0_cen_baccoemu` | 9.0005 | 12.99789 |
| `theta_inn_baccoemu` | -1.9998 | -0.00021 |
| Individual Fourier side, in h/Mpc | 0.01785654 | 17.655489 |

Preparation records the installed weight hashes and reads their bounds.
Changing the weights requires checking the bounds again. All Fourier
wavenumbers in this note use h/Mpc; power spectra use $`({\rm Mpc}/h)^3`$.
The ratio is dimensionless, but its Fourier argument still needs the
correct units. Cocoa's separate `bfmt` request uses 1/Mpc and receives
$`h k`$ when the input number $`k`$ uses h/Mpc.

## Converting the power response <a name="bacco-power"></a>

The default prescription is `coherent_linear_neutrinos`.
Here **linear refers to the neutrino power spectrum that CAMB computes**.
The calculation uses a linear neutrino auto-spectrum and approximates
the cold-neutrino cross-spectrum by

```math
P_{c\nu}^{D}(k,z)\simeq
\sqrt{P_c^{D}(k,z)\,P_{\nu\nu,L}(k,z)}.
```

The subscript $`L`$ means linear theory. This equation assigns a
cross-correlation coefficient of one: cold and neutrino fluctuations
have aligned Fourier phases in this approximation. Cataneo et al. use
this cross-spectrum approximation in their matter-power calculation.
That precedent does not calibrate its baryonic or bispectrum extension.
[Cataneo et al., arXiv:1909.02561, equations 1 and 2](https://arxiv.org/abs/1909.02561)

Expanding the total-density relation gives

```math
P_m^D=f_c^2P_c^D+2f_cf_\nu P_{c\nu}^D+f_\nu^2P_{\nu\nu,L}.
```

The factor of two counts the two cold-neutrino orderings in a pair.
Under the cross-spectrum approximation, the right side is a square.
Define three quantities with units of the square root of a power spectrum:

```math
A=\sqrt{P_m^D},\qquad
N=f_\nu\sqrt{P_{\nu\nu,L}},\qquad C=A-N.
```

Here $`P_m^D`$ is the project's existing nonlinear total-matter spectrum.
$`N`$ is the mass-weighted neutrino fluctuation amplitude, and $`C`$ is
the cold amplitude needed to recover that total spectrum. The construction
preserves the DMO baseline exactly. It infers a cold nonlinear
spectrum through $`f_c\sqrt{P_c^D}=C`$; this inferred spectrum is part
of the approximation.

Use the explicit CAMB massive-neutrino spectrum for $`P_{\nu\nu,L}`$.
Reconstructing it by subtracting nearly equal total and cold transfer
amplitudes would discard precision when the neutrino contribution is
small. CAMB distinguishes its massive-neutrino density variable `delta_nu`
from its massless-neutrino variable `delta_neutrino`.

The cold power response changes its amplitude by
$`q=\sqrt{R_{P,c}}`$. Holding the neutrino auto-spectrum fixed and applying
the same cross-spectrum approximation after feedback gives

```math
P_m^H=(qC+N)^2,\qquad
R_{P,m}=\frac{P_m^H}{P_m^D}.
```

For tabulation, define $`a=C/A`$ and $`n=N/A=1-a`$. These are fractional
fluctuation amplitudes at a particular $`k,z`$; $`a`$ is not the cosmic
expansion factor $`a_{\rm exp}`$. The response becomes

```math
R_{P,m}=(a q+n)^2,
\qquad a=1-f_\nu\sqrt{\frac{P_{\nu\nu,L}}{P_m^D}}.
```

The calculation requires $`0\leq N\leq A`$. A violation rejects the
inputs: replacing a negative
inferred cold amplitude by zero would change the model. For a massless
neutrino calculation, set $`N=0`$ directly; no ratio of zero neutrino
spectra is needed.

A worked arithmetic example illustrates the dilution. Choose $`a=0.99`$
and $`R_{P,c}=0.81`$, so $`q=0.9`$. The total response is
$`(0.99\times0.9+0.01)^2=0.811801`$. The neutrino contribution reduces
the suppression slightly. These chosen numbers explain the formula;
they are not an emulator measurement.

The alternate `cold_on_total` prescription sets $`R_{P,m}=R_{P,c}`$.
It matches the assumption in Cocoa's existing BACCO power-response
application. Comparing it with the default isolates the consequence
of the chosen neutrino treatment while retaining the same DMO spectrum.

## Converting the bispectrum response <a name="bacco-bispectrum"></a>

Three Fourier wavevectors contribute to a bispectrum when they sum to
zero. Their positive lengths $`k_1,k_2,k_3`$ form a triangle. Each leg
can carry cold matter or neutrinos, giving eight ordered contributions:

```math
\begin{aligned}
B_m={}&f_c^3B_{ccc}
 +f_c^2f_\nu(B_{cc\nu}+B_{c\nu c}+B_{\nu cc})\\
 &+f_cf_\nu^2(B_{c\nu\nu}+B_{\nu c\nu}+B_{\nu\nu c})
 +f_\nu^3B_{\nu\nu\nu}.
\end{aligned}
```

The position of a subscript identifies its Fourier leg. For example,
$`B_{c\nu c}`$ places neutrinos on $`k_2`$. Unequal sides generally
make the three placements unequal. The density-field decomposition
is exact; BACCO supplies the cold contribution's response but does not
supply the seven other responses.
[Ruggeri et al., Equations 2.8--2.10](https://arxiv.org/pdf/1712.02334)

The default prescription assigns those mixed responses using the
amplitude fractions defined for power. For leg $`i=1,2,3`$, evaluate
$`a_i,n_i`$ and $`q_i=\sqrt{R_{P,c}(k_i,z)}`$ at that leg's own
wavenumber. Give a cold leg the factor $`a_iq_i`$ and a neutrino leg
the factor $`n_i`$. Replace the pure-cold three-leg factor by the
independently emulated bispectrum response:

```math
\begin{aligned}
R_{B,m}={}&a_1a_2a_3R_{B,c}\\
 &+a_1q_1a_2q_2n_3
  +a_1q_1n_2a_3q_3
  +n_1a_2q_2a_3q_3\\
 &+a_1q_1n_2n_3
  +n_1a_2q_2n_3
  +n_1n_2a_3q_3\\
 &+n_1n_2n_3.
\end{aligned}
```

For example, $`a_1q_1n_2a_3q_3`$ is the assigned contribution for
cold matter on legs 1 and 3 and neutrinos on leg 2. The three terms
on that line visit each placement of one neutrino. The next line
visits each placement of one cold leg. The last line assigns an
all-neutrino contribution. Every term multiplies the existing total
DMO bispectrum to construct the corresponding contribution to the
predicted hydro bispectrum.

This is a **phenomenological model of mixed bispectrum responses**.
It uses linear neutrino power to choose amplitudes; it does not calculate
neutrino perturbations to second order. In particular, the term
$`n_1n_2n_3B_m^D`$ is an assigned all-neutrino bispectrum. It would be
zero for a strictly Gaussian linear neutrino field, which is not the
field assumption made by this bispectrum prescription.

Linear power and a nonzero bispectrum are compatible at different
orders of perturbation theory: a second-order density contribution can
generate a bispectrum without changing the leading-order power.
Ruggeri et al. show why dropping nonlinear neutrino contributions can
misestimate the large-scale matter bispectrum. Their perturbative
calculation is not a derivation of the assignment above.
[Ruggeri et al., Section 2.2](https://arxiv.org/pdf/1712.02334)

For checking the eight-term expression, the same response can be written

```math
R_{B,m}=\prod_{i=1}^{3}(a_iq_i+n_i)
 +\left(\prod_{i=1}^{3}a_i\right)
   \left(R_{B,c}-\prod_{i=1}^{3}q_i\right).
```

The correction removes the pure-cold product inferred from power and
inserts BACCO's independent cold bispectrum response. The expanded
expression shows that positive emulator ratios and nonnegative
amplitude fractions give a positive result without clipping a negative
answer. The alternate `cold_on_total` prescription uses
$`R_{B,m}=R_{B,c}`$ and assigns no separate mixed contributions.

Both prescriptions return the cold responses when neutrinos have zero
mass fraction or zero clustering power. Unity cold responses give unity
total responses. These limits check the algebra; agreement in the
limits does not measure the error at a finite neutrino mass.

## Continuing beyond the emulator domain <a name="bacco-continuation"></a>

Lensing projections integrate over redshift and Fourier scale. Their
requested wavenumbers can exceed BACCO's training interval even when
the displayed angular separations seem large. A short comoving lens
distance converts a given angular multipole to a large spatial
wavenumber. The continuation belongs in the saved model definition,
separately from the interpolation resolution.

The diagnostic emulator calls refuse out-of-domain requests. Projection
preparation uses the continuation defined below. The high
anchor $`K`$ is the largest wavenumber sent to the emulator, and the low
floor $`L`$ is its smallest. Both lie inside the installed bounds. Choosing
$`K=10\,h/{\rm Mpc}`$ or $`K=17.6\,h/{\rm Mpc}`$ defines two distinct
assumptions to compare; it does not change the emulator's calibration.
The settings use $`L=0.02\,h/{\rm Mpc}`$ and
$`K=10\,h/{\rm Mpc}`$ by default.

**High wavenumbers.** For the default mapping, let
$`k_{\max}=\max(k_1,k_2,k_3)`$ and set

```math
s=\min\left(1,\frac{K}{k_{\max}}\right),\qquad
k_i'=\max(L,s k_i).
```

The common factor $`s`$ preserves triangle shape while moving its
largest side to the high anchor. Raising any remaining short side
to $`L`$ changes the shape; the saved model definition records this
choice. The alternative `side_clamp`
maps each original side independently into $`[L,K]`$. It generally
changes the ratios of high sides. Both choices require checking the
triangle after finite-precision conversion.

**A single long-wavelength leg.** A triangle with one small $`k_i`$
and two large sides still probes short-scale matter. Its bispectrum
response therefore retains the value at the mapped triangle. Setting
the whole response to unity whenever one side is small would discard
that effect. Burger et al. discuss this issue and the extrapolation
of squeezed configurations absent from the measured training set.
[Burger et al., Section 4](https://arxiv.org/html/2506.18974v2#S4)

**All three wavelengths large.** When every side is below $`L`$, taper
the logarithmic response to zero. Define

```math
t=\frac{k_{\max}}{L},\qquad
g(t)=t^2(3-2t)\quad(0\leq t\leq1),\qquad
\ln R_{B,c}^{\rm cont}=g(t)\ln R_{B,c}(k_1',k_2',k_3',z).
```

The function rises from zero to one and has zero slope at both endpoints.
It gives $`R_{B,c}^{\rm cont}\to1`$ as the whole triangle shrinks,
with a leading change proportional to $`k_{\max}^2`$. The calculation
applies this taper only when all sides are small. For power, use the corresponding
one-wavenumber taper below $`L`$ and a constant response above $`K`$.
The joins are continuous; boundary clamping still introduces changes
of slope that require numerical refinement checks.

**High redshift.** The installed expansion-factor range ends at $`z=3`$.
The boundary-hold choice uses the cold response at that redshift for
higher redshifts. The alternative multiplies its logarithm by
$`1-g(t_z)`$, where $`t_z=(z-3)/1`$ between $`z=3`$ and $`z=4`$,
and uses unity beyond $`z=4`$. The width of one in redshift is a chosen
continuation setting. Conversion to total matter uses the CAMB spectra
and mass fractions at the actual requested redshift.

**Collapsed triangles.** Response grids can include the endpoint
$`k_3=k_1+k_2`$. A one-sided limit from nondegenerate triangles supplies
its value. The chosen interior point must remain nondegenerate after
BACCO's float32 conversion. Silent removal of a triangle would shift
every subsequent table entry. Preparation approaches an exact zero side
through a limit instead of sending it into the emulator.

Preparation continues the cold responses first and then converts them
to total matter at the physical query. In the mixed-bispectrum formula,
each $`q_i`$ uses the same continued power response that the two-point
prediction uses at $`k_i`$. A triangle's mapped power coordinates must
not silently replace those physical leg coordinates.

The CAMB spectra have their own saved wavenumber interval, which differs
from the emulator interval. Inside it, the neutrino fraction uses the
actual physical $`k_i,z`$. Outside it, preparation holds
$`P_{\nu\nu,L}/P_m^D`$ at that CAMB interval's nearest wavenumber
boundary. This is an additional continuation assumption. Its influence
requires checking the saved CAMB interval separately from the BACCO
anchors; the emulator mapping must not select the neutrino weight.

## Changing parameters in a Fisher calculation <a name="bacco-fisher"></a>

A Fisher forecast differentiates the observable mean around a stated
parameter point. Let $`\mu_i`$ be observable row $`i`$, such as one
bin-averaged $`\xi_+`$ measurement, and let $`p_\alpha`$ be parameter
$`\alpha`$. With a fixed covariance $`C_{ij}`$, the local information is

```math
J_{i\alpha}=\frac{\partial\mu_i}{\partial p_\alpha},\qquad
F_{\alpha\beta}=\sum_{i,j}J_{i\alpha}(C^{-1})_{ij}J_{j\beta}.
```

The covariance couples measurement rows, including two-point/Map3
cross-correlations. Keeping all other parameters fixed gives a different
question from marginalizing their uncertainty. The inverse Fisher
matrix gives marginalized parameter covariances only when the chosen
parameters and stated priors make it invertible. The
[Fisher guide](../../../../external_modules/code/cosmolike_core/cosmolike_notebook_utils/README.md#fisher) describes the project's derivative and
covariance conventions.

For the five physical BACCO parameters, the calculation regenerates
both emulator responses at every displaced point. A cosmological
displacement also changes the CAMB spectra, $`\sigma_{8,c}`$, the
mass fractions, and the response conversion. Reusing ratios prepared
at another cosmology would omit part of the derivative.

The generic native response table uses a separate logarithmic amplitude,
defined in [the paired-response guide](../../../../external_modules/code/cosmolike_core/cosmolike/3pcf/README.md#paired-feedback). The physical
BACCO calculation requires

```yaml
feedback_amplitude: 1
```

Changing that amplitude would vary a phenomenological logarithmic
response after conversion. It would not vary a BACCO gas parameter.
Exponentiating a cold response before the neutrino conversion also
differs from exponentiating the total response afterward. The physical
parameter calculation fixes the amplitude and regenerates the ratios.

The C integration uses the existing paired-response tables. Preparation
samples uniform redshift, logarithmic scale, and triangle-shape axes,
then the existing interpolation prepares dense arrays for linear reads.
A single prepared table is fixed during its native calculation. The
project replaces that table when the cosmology or a feedback parameter
changes; a fixed native object does not imply cosmology-independent
BACCO physics.

| Input that changes | Required preparation |
|---|---|
| Cosmological parameter | CAMB spectra, BACCO coordinates, and both converted responses |
| One physical BACCO parameter | Both cold responses and their total-matter conversion |
| Neutrino prescription | Both total-matter response tables |
| Fourier continuation setting | Both response tables |
| Redshift continuation setting | Both response tables |
| Response-grid resolution | The prepared response arrays |
| Emulator weights | Bounds, recorded hashes, and both responses |

The saved identity includes these choices and the exact CAMB neutrino
density and source spectra. Returning to a parameter point in a live
notebook must reproduce the corresponding newly constructed model.
The five feedback parameters remain shared between the power and
bispectrum responses throughout this sequence.

The NLA intrinsic-alignment baseline follows the converted matter
responses. NLA means nonlinear alignment, in which the shape response
uses a nonlinear matter spectrum. The quadratic terms of the tidal
alignment and tidal torquing model, TATT, and the covariance keep their
DMO reference; the [paired-response guide](../../../../external_modules/code/cosmolike_core/cosmolike/3pcf/README.md#paired-feedback) explains
that convention.
