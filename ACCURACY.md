# Sampling, integration and notebook changes

Increasing a numerical resolution answers a specific question: does the
calculated observable change when that approximation is made more accurate?
It does not test a missing physical contribution or an insufficient integration
range. Keep the range fixed for a resolution comparison, then vary the range
in a separate comparison. This separation is especially useful for a signed
IA contribution that crosses zero: compare its error with a declared component
amplitude, rather than dividing by each nearly zero entry.

## Contents <a name="contents"></a>

- [Two global controls](#global)
- [Individual stages](#stages)
- [CAMB solve and extrapolation limits](#camb-limits)
- [Changing settings in a notebook](#notebook)
- [What remains fixed](#fixed)

## Two global controls <a name="global"></a>

`des3pcf.Accuracy` has two global multiplicative controls. Their defaults are
both one. `accuracy_boost` increases prepared sampling grids, while
`integration_accuracy_boost` increases the node counts of fixed quadratures.
An integration boost does not select an adaptive routine or change a tolerance
inside an integrand. For a quadrature of baseline size $`N`$, the new size is
$`\lceil N b\rceil`$. For an interpolation grid with endpoints, it is
$`\lceil(N-1)b\rceil+1`$. Each individual stage multiplier multiplies its
applicable global factor.

The line-of-sight calculation uses a fixed trapezoidal integral in comoving
distance. Its supplied redshift nodes are subdivided interval by interval,
and the physical model evaluates the distance at every new node. The
subdivision is the next power of two above
`accuracy_boost * integration_accuracy_boost * chi_boost`. Thus all old nodes
and both limits remain present, including for a nonuniform baseline grid.
A requested subdivision above 1024 is refused. A boost of three requests four
subintervals; the resolved node count records that distinction.

The counts passed into the constructor remain the unboosted baseline. Applying
the same Accuracy object again to that baseline gives identical counts. It
never multiplies previously boosted values. Returning the controls to one
restores the baseline without restarting Python. As in the ordinary core's
accuracy controls, this rule prevents a notebook cell run twice from silently
doubling its resolution twice.

## Individual stages <a name="stages"></a>

| Control | Quantity refined | Global factor |
|---|---|---|
| `chi_boost` | LOS intervals, with nested old nodes | Both |
| `kernel_boost` | Source-window redshift grid | Sampling |
| `fit_z_boost` | Matter-fit redshift anchors | Sampling |
| `power_table_boost` | Physical model's dense interpolation ratio | Sampling |
| `triangle_boost` | Coarse projected triangle coordinates | Sampling |
| `multipole_boost` | Coarse projected multipole coordinates | Sampling |
| `fft_boost` | Minimum FFT length before exact angular alignment | Sampling |
| `angular_boost` | Bispectrum angular quadrature nodes | Integration |
| `aperture_ell_boost`, `aperture_phi_boost` | Full-filter mean quadratures | Integration |
| `xi_ell_boost`, `xi_theta_boost` | Xi Fourier and within-bin quadratures | Integration |
| `covariance_ell_boost`, `covariance_window_boost` | Short Fourier modes and separate survey-window modes | Integration |
| `covariance_phi_boost`, `covariance_radius_boost` | Relative angle and center/halo offset quadratures | Integration |
| `covariance_k_boost`, `covariance_mass_boost` | Halo variance and mass quadratures | Integration |
| `profile_boost` | Halo-profile input radius intervals | Sampling |
| `profile_upsample_boost` | Halo-profile dense interpolation ratio | Sampling |

The physical C model shares its dense-reader ratio between power and the
prepared matter-fit interpolation. Increasing `power_table_boost` refines both
dense readers. Increasing `fit_z_boost` instead adds expensive matter-fit
anchors. A global sampling boost increases both the anchor count and the
dense-reader ratio, so the final dense fit table can grow faster than either
one individually. The analogous distinction holds between covariance profile
samples and their dense interpolation ratio. Use the individual controls to
determine which approximation limits a calculation before increasing all of
them together.

Projection settings still expose the individual axis counts (`nr`, `nu`,
`nv`, `np`, `nshape`, `nell`, `npsi`, and the reference-coordinate counts).
These allow an even narrower test than a whole triangle-grid boost. The
direct reference route and the prepared production route remain explicit
choices; a boost does not switch between them.

## CAMB solve and extrapolation limits <a name="camb-limits"></a>

The physical input producer separates two wavenumber limits, following the
LSST Y1 likelihood. `physics.kmax_boltzmann` controls the transfer functions
CAMB actually evolves; its default is 5 in $`\mathrm{Mpc}^{-1}`$.
`physics.extrap_kmax` controls how far its power interpolator extends the
high-wavenumber tail, also in $`\mathrm{Mpc}^{-1}`$. If omitted, this second
limit covers at least 250 and 1.01 times the physical endpoint of the saved
table. CAMB supplies its log-power extrapolation; no additional transfer
evolution is requested merely to sample the tail.

The saved `physics.k` axis instead uses $`h\,\mathrm{Mpc}^{-1}`$. At $`h=0.7`$,
a saved node at 1000 therefore queries the interpolator at
$`700\,\mathrm{Mpc}^{-1}`$, while the Boltzmann solve can still end at
$`5\,\mathrm{Mpc}^{-1}`$. Increasing the saved table's upper endpoint must
not silently increase the transfer calculation. An explicit extrapolation
limit that does not cover the requested nodes is refused.

The producer fixes the requested $`\sigma_8`$ by adjusting $`A_s`$ and
recalculating matter powers from retained transfer functions. CAMB recomputes
the nonlinear correction at that amplitude; nonlinear power is not rescaled
by a constant. Metadata records both wavenumber limits, their units and this
amplitude procedure. Changing either limit invalidates the provider and
outer mean cache. Reference tables without this limit provenance must be
regenerated before varying cosmology in CAMB mode.

Vary the solved cutoff separately when testing sensitivity to the extrapolated
tail. A denser output grid only samples that same tail more closely; it does
not establish its physical accuracy. Neither limit is changed by the C
sampling and quadrature boosts above.

## Changing settings in a notebook <a name="notebook"></a>

The project Python package is under `python/des3pcf`. After the notebook has
added that directory to its import path, an existing physical provider can
change its numerical controls directly:

```python
from des3pcf import Accuracy

provider.accuracy = Accuracy(chi_boost=2.0)
refined = prediction(parameters)

provider.accuracy = Accuracy()
baseline_again = prediction(parameters)
```

The provider includes every numerical control in its exact model key. A
changed control constructs a new owned C model. A retained old model remains
valid and unchanged. `ExactMeanCache` also checks the prediction's numerical
settings before consulting its parameter dictionary, so an outer Fisher cache
cannot hide a changed integration grid. A custom prediction function without
a `cache_token` method must have fixed hidden inputs, or its caller must call
`cache.clear()` after changing them.

FFT geometry and survey rows are prepared objects. To change angular bins,
Fourier limits, source counts or an estimator definition, construct another
engine or provider in the same kernel. Public prepared arrays and dictionaries
are independent read-only copies. A partial edit raises an error, because
changing coordinates while keeping their old weights would define an
inconsistent observable. Parameters passed to each evaluation, including IA,
photo-z offsets and shear calibration, remain changeable.

For a direct C++ notebook binding, the same resolver supplies explicit counts:

```python
controls = Accuracy(chi_boost=2.0, kernel_boost=2.0)
arguments = dict(physical_tables)
arguments["los_z"] = controls.los_nodes(baseline=physical_tables["los_z"])
arguments.update(controls.model_settings(kernel_nz=101, table_boost=8, fit_nz=33))
model = compiled.notebook.PhysicsModel(**arguments)

settings = controls.projection_settings(baseline={"lmax": 20})
grid = model.projection_grid_notebook(bins=bins, parameters=ia, settings=settings)
```

The command-line configuration accepts the same field names in its top-level
`accuracy` mapping. Saved result metadata includes all factors, including
defaults. Numerical settings for the ordinary two-point CFASTPT module are a
separate explicit configuration; its integration selector has the meaning
defined by the ordinary core, not this project's multiplicative quadrature
factor.

## What remains fixed <a name="fixed"></a>

External Boltzmann `k` and `z` tables, source-distribution measurements,
integration limits, angular truncation orders `lmax`/`mmax`, FFT padding/window
choices, and measured bin edges keep their declared values. More interpolated
samples do not add information to an underresolved CAMB table. Refine its
producer grid separately when necessary.

Measured pair weights also remain physical input. Increasing the number of
within-bin quadrature nodes while retaining a catalogue weight array is
refused: those weights refer to the original nodes. Supply weights at the
new baseline count, or explicitly select the uniform-area approximation.

The [state tests](tests/README.md) compare reused calculations with independent
fresh processes. Those comparisons establish cache and ownership correctness;
the interpolation and quadrature comparisons described in the
[numerical guide](../../external_modules/code/cosmolike_core/cosmolike/3pcf/NUMERICS.md)
address accuracy relative to the declared physical model.
