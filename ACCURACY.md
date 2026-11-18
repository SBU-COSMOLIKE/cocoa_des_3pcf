# Sampling, integration and notebook changes

Increasing a numerical resolution answers a specific question: does the
calculated observable change when that approximation is made more accurate?
It does not test a missing physical contribution or an insufficient integration
range. Keep the range fixed for a resolution comparison, then vary the range
in a separate comparison. This separation is especially useful for a signed
IA contribution that crosses zero: compare its error with a declared component
amplitude, rather than dividing by each nearly zero entry.

The [executed interpolation study](tests/studies/interpolation_validation.md)
records why the first TATT demonstration Fisher result was rejected and
which corrections have been checked on the full aperture vector.

## Contents <a name="contents"></a>

- [Two global controls](#global)
- [Individual stages](#stages)
- [Source curves and photo-z derivatives](#source-curves)
- [Signed bispectra and triangle coordinates](#signed-projection)
- [CAMB solve and extrapolation limits](#camb-limits)
- [Choosing Fisher accuracy from a default-accuracy base](#fisher-noise)
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
| `source_boost` | Independent uniform source-density grid, when declared | Sampling |
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

## Source curves and photo-z derivatives <a name="source-curves"></a>

Three redshift grids have different jobs. CAMB's `physics.z` samples power
and background quantities. An optional `physics.source_nz` samples the
source density on its own uniform grid, without requesting more CAMB
transfer redshifts. `physics.kernel_nz` samples the prepared lensing and
alignment windows. Increasing one count cannot establish convergence of
the other two. `source_boost` multiplies the independent source intervals
and leaves the original measured samples unchanged; it is refused when
no independent source grid was declared.

The source reconstruction is also a physical input choice.
`linear_midpoint` joins the supplied midpoint samples by straight lines.
`pchip_midpoint` uses a shape-preserving cubic through those same samples.
The latter requires an explicit endpoint policy. `zero_at_measured_edges`
uses the supplied FITS `Z_LOW`/`Z_HIGH` support, intersected with nonnegative
redshift, and a zero plateau to the larger background upper bound.
`zero_at_observer_and_background_edge` instead extends the reconstruction
to the background boundary. These can produce different curves; neither
preserves each histogram cell's integrated probability. The added zero
endpoints and the reconstruction choice are recorded with the tables.
Neither choice clips negative samples. The demonstration's separately
authorized clipping is recorded when its source FITS file is created.

Why does this matter for Fisher derivatives? A photo-z shift evaluates
$`n(z-\Delta z)`$. At a knot of a piecewise-linear curve, its left and
right slopes can differ. Sampling a small shift exactly at that knot
can introduce an absolute-shift term, proportional to $`|\Delta z|`$,
in a finite-grid prediction. A fourth difference then measures this
coherent interpolation effect as well as any random numerical noise.
PCHIP makes the interior first derivative continuous, but it does not
make a nonzero endpoint slope continuous with a zero extension. Boundary
support, normalization and source resolution still need separate checks.

Each shift and resolution change reconstructs the immutable native
samples directly. It never shifts an already shifted curve. The native
axis and densities are retained as `source_native_z` and `source_native`;
`source_z` names the numerical sampling axis. Both C++ adapters copy
these physical inputs into the owned C model. The two-point bridge receives
the same shifted samples and applies no second photo-z shift. Nonnegative
source samples alone do not guarantee a nonnegative natural-cubic
interpolant of the later divided IA kernel; inspect and refine that stage.
The measured-edge policy also preserves `source_native_low/high` and
records `source_unshifted_support`. A photo-z shift moves that support;
the observer and finite background still limit the retained output.

## Signed bispectra and triangle coordinates <a name="signed-projection"></a>

A signed IA contribution can cross zero and vary by many orders of
magnitude across Fourier scale. Interpolating its raw amplitude with a
cubic can produce tails much larger than the physical value. The optional
`residual_scaling: 1` instead stores the signed ratio of the TATT residual
to the positive equilateral gravitational bispectrum at the same perimeter.
The positive scale is interpolated logarithmically and multiplied back
after reading the ratio. `signed_baseline: 1` similarly represents the NLA
baseline by positive GGG and signed NLA/GGG. These are numerical
representations of the declared components, not changes to IA coefficients
or a prescription to replace negative values by their absolute magnitude.

Resolution also depends on the coordinate. The PUV residual uses perimeter
$`p=\ell_1+\ell_2+\ell_3`$ and fractions $`u=\ell_1/p`$,
$`v=\ell_2/p`$. `shape_stretch` controls how a uniform bounded-tanh
coordinate concentrates samples near squeezed triangles. Increasing this
concentration reduces central resolution at fixed `nshape`; test the count
and coordinate together against direct projected values. The historical
stretch of three does not resolve every squeezed fraction allowed by the
wide demonstration Fourier range.

The prepared multipoles use scale $`E=\sqrt{\ell_1^2+\ell_2^2}`$ and
an angle $`\psi`$ that specifies the side ratio. With `psi_scale: 0`,
the angular coordinate is $`\log\psi`$. A positive value $`s`$ chooses
$`x=\log\psi+\psi/s`$: spacing is logarithmic for small angles and
approximately linear for angles much larger than $`s`$. The table is
uniform in $`x`$, so cubic preparation and arithmetic-index linear reads
remain possible. Both ordered-leg charts are retained. Compare the
prepared multipoles with direct angular integration of the same projected
grid to isolate this interpolation stage from the underlying physics.

These optional representations and coordinate choices are under numerical
validation. Their successful unit tests do not establish a converged survey
forecast. The first TATT demonstration Fisher result failed these checks
and must not be interpreted as evidence for stronger parameter constraints.

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

## Choosing Fisher accuracy from a default-accuracy base <a name="fisher-noise"></a>

A Fisher calculation can report artificially small errors when numerical
noise contaminates its derivatives. A difference of nearby predictions is
divided by a small parameter step. Squaring that derivative in
$`F=D^T C^{-1}D+F_{\rm prior}`$ turns some of the noise into positive
information. Agreement of the fiducial mean alone does not test this effect.

The expensive part of the screening calculation is performed once at
**default accuracy**: build the full derivative matrix and Fisher matrix,
including the stated priors. The shared core functions
`get_fom_noise_weights`, `get_stencil_noise`, and `predict_fom_noise` in
`cosmolike_notebook_utils/fisher.py` implement the estimate derived in the
LSST Y1×DESI Fisher note's section on derivative accuracy. The project uses
these functions, retaining the same meaning of each quantity.

Let $`S=F^{-1}`$ be the parameter covariance and $`S_w`$ its marginalized
$`(w_0,w_a)`$ block. For parameter $`p_j`$, the default-accuracy calculation
supplies a weight

$$
R_j^2\sigma_j^2=S_{j,w}S_w^{-1}S_{w,j},\qquad \sigma_j^2=S_{jj}.
$$

The weight measures how uncertainty in that parameter affects the dark-energy
ellipse. A nuisance parameter can matter even though it is neither $`w_0`$
nor $`w_a`$. Its covariance with the pair must be retained when marginalizing.
At each candidate accuracy, measure only the columns selected by these
weights and the baseline noise audit; rebuilding the entire derivative base
at every setting would lose the method's cost advantage.

Five equally spaced predictions give the fourth difference
$`\Delta^4m=m(-2h)-4m(-h)+6m(0)-4m(h)+m(2h)`$. If evaluation errors have
approximately independent, equal-size fluctuations along this stencil,

$$
\widehat\sigma_{\rm eval,j}^2=
\frac{\Delta^4m_j^T C^{-1}\Delta^4m_j}{70},\qquad
a_j=\frac{130}{144}\frac{\widehat\sigma_{\rm eval,j}^2}{h_j^2},\qquad
\delta\ln{\rm FoM}\simeq\frac12\sum_j R_j^2a_j\sigma_j^2.
$$

Here $`h_j`$ is an absolute step in the units of parameter $`p_j`$. The
project's adapter supplies parameter offsets about zero to the shared core
function, so the core's relative-step convention cannot accidentally rescale
a YAML step or reverse its interpretation at a negative fiducial value.
The factors 70 and 130 are sums of squared stencil coefficients. They
describe independent numerical errors, not a new physical covariance term.
For large predicted shifts, the shared function also evaluates the exact
ratio $`{\rm FoM}(F+\operatorname{diag}a)/{\rm FoM}(F)`$ instead of truncating
the matrix calculation at first order.

The full default-accuracy run saves the five vectors at both $`h`$ and
$`h/2`$ for every parameter. Its baseline noise audit therefore needs no
additional model evaluations. A candidate examining $`k`$ columns at one
spacing needs $`1+4k`$ distinct evaluations, including a shared center.
Omitted columns are listed, and their measured baseline contribution is
reported. A small baseline contribution is not a bound on how that column's
noise changes at another accuracy.

This procedure screens settings; it does not certify convergence. A smooth
fourth derivative also enters $`\Delta^4m`$, while an odd numerical error can
cancel from it and still bias the first derivative. Correlations between
nearby evaluations violate the independent-noise estimate. The final check
therefore compares full Fisher matrices at selected finer settings and at
both step sizes, with the covariance, priors, physical model and rows held
fixed. Separate range, angular-truncation and source-table checks remain
necessary. Report the actual FoMs from those full derivatives, never an
estimated corrected scientific FoM obtained by subtracting the noise model.

A concrete diagnostic is a numerical cusp, $`m(p)=v|p-p_0|`$. A centered
five-point fourth difference at $`p_0`$ is $`-4hv`$. Its inferred vector
noise amplitude therefore scales as $`h`$, while the corresponding false
information estimate can remain constant when the step is halved. That
behavior is not independent random noise. Smooth fourth-derivative
contamination instead scales as $`h^4`$; ideal independent evaluation noise
has roughly constant amplitude. Compare one-sided slopes, shifted stencil
centers and more than two step sizes when a table-cell crossing or source
boundary could be responsible. Shared stencil samples mean these scaling
checks are diagnostics, not independent statistical measurements.

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
