# Accuracy controls

1. [What a refinement comparison tests](#principle)
2. [Two global controls](#global)
3. [Individual stages](#stages)
4. [Source curves and photo-z derivatives](#source-curves)
5. [Signed bispectra and triangle coordinates](#signed-projection)
6. [CAMB solve and extrapolation limits](#camb-limits)
7. [Choosing Fisher accuracy from a default-accuracy base](#fisher-noise)
8. [Changing settings in a notebook](#notebook)
9. [What remains fixed](#fixed)
10. [Executed numerical studies](#studies)
11. [Shared smoothing controls](#shared-smoothing)

## What a refinement comparison tests <a name="principle"></a>

Increasing a numerical resolution answers a specific question: does the
calculated observable change when that approximation becomes more
accurate? It does not test a missing physical contribution or an
insufficient integration range. Keep the range fixed for a resolution
comparison, then vary the range in a separate comparison. For a signed
IA contribution that crosses zero, compare its error with the amplitude
of its component rather than dividing by each nearly zero entry.

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
distance. It subdivides its input redshift nodes interval by interval, and
the physical model evaluates the distance at every new node. The
subdivision is the next power of two above
`accuracy_boost * integration_accuracy_boost * chi_boost`. Thus all old nodes
and both limits remain present, including for a nonuniform baseline grid.
The resolver refuses a subdivision above 1024. A boost of three requests
four subintervals; the resolved node count records that distinction.

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
| `source_boost` | Independent uniform source-density grid, when the configuration declares one | Sampling |
| `fit_z_boost` | Matter-fit redshift anchors | Sampling |
| `bispectrum_boost` | Four coarse axes of the required supplied absolute bispectrum | Sampling |
| `feedback_table_boost` | Paired response dense interpolation ratio | Sampling |
| `feedback_z_boost` | BACCO response redshift intervals | Sampling |
| `feedback_k_boost` | BACCO response log-wavenumber intervals | Sampling |
| `feedback_u_boost` | BACCO response log shortest/middle side-ratio intervals | Sampling |
| `feedback_v_boost` | BACCO response triangle-opening intervals | Sampling |
| `power_table_boost` | Physical model's dense interpolation ratio | Sampling |
| `triangle_boost` | Coarse projected triangle coordinates | Sampling |
| `multipole_boost` | Coarse projected multipole coordinates | Sampling |
| `fft_boost` | Minimum FFT length before exact angular alignment | Sampling |
| `angular_boost` | Bispectrum angular quadrature nodes | Integration |
| `aperture_ell_boost` | Full-filter mean Fourier quadrature | Integration |
| `aperture_phi_boost` | Full-filter mean opening-angle quadrature | Integration |
| `xi_ell_boost` | Xi Fourier quadrature | Integration |
| `xi_theta_boost` | Xi within-bin angular quadrature | Integration |
| `covariance_ell_boost` | Short-mode Fourier quadrature | Integration |
| `covariance_window_boost` | Separate survey-window mode quadrature | Integration |
| `covariance_phi_boost` | Relative-angle quadrature | Integration |
| `covariance_radius_boost` | Center and halo offset quadratures | Integration |
| `covariance_k_boost` | Halo variance wavenumber quadrature | Integration |
| `covariance_mass_boost` | Halo mass quadrature | Integration |
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

For the [supplied BiHalofit route](BIHALOFIT_PYTHON.md), the coarse baseline
intervals grow by `accuracy_boost * bispectrum_boost`. Its explicit
`physics.bispectrum_table.boost` stays fixed. Doubling the coarse intervals
on all four axes therefore multiplies the stored table size by roughly
sixteen, even though the global factor enters each axis only once. The
49-by-193-by-97-by-33 example stores 30,271,857 double values, about 231 MiB
for one table; Python ownership, C ownership and temporary preparation need
additional memory. Change an individual YAML axis count when isolating that
axis's error. The Python fit's `fit_z_boost` and `power_table_boost` control
different approximations and remain independent of baseline interpolation.

The four BACCO input-axis controls apply to the dynamic producer. A loaded
fixed response file has no emulator from which to obtain new input values;
only its dense interpolation can be refined. For dynamic BACCO, a global
sampling boost increases all four input counts and the dense multiplier,
so its memory cost can grow rapidly. Refine one axis at a time. The
[BACCO notebook](EXAMPLE_FISHER_BACCO.ipynb) distinguishes these sampling
checks from changes to the physical continuation beyond the trained domain.

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
and leaves the original measured samples unchanged; the resolver refuses
it when the configuration declares no independent source grid.

The `core.model_from_core()` route derives $`dz/d\chi`$ numerically from
the stored distances. Its background node spacing therefore controls an
additional approximation. Increasing a dense-reader boost alone cannot
repair a derivative estimated from coarse distance samples. The
[independent bridge study](tests/studies/bridge_background_analytic.md)
checks the absolute units, endpoint/interior errors and their effect on
controlled IA projections. The direct CAMB preparation route instead
supplies this Jacobian from CAMB's Hubble rate; the bridge study's errors
do not apply automatically to that route or bound a survey forecast.

The source reconstruction is also a physical input choice.
`linear_midpoint` joins the input midpoint samples by straight lines.
`pchip_midpoint` uses a shape-preserving cubic through those same samples.
The latter requires an explicit endpoint policy. `zero_at_measured_edges`
uses the FITS `Z_LOW`/`Z_HIGH` support, intersected with nonnegative
redshift, and a zero plateau to the larger background upper bound.
`zero_at_observer_and_background_edge` instead extends the reconstruction
to the background boundary. These can produce different curves; neither
preserves each histogram cell's integrated probability. The saved tables
record the added zero endpoints and the reconstruction choice.
Neither choice clips negative samples. The demonstration's clipping is a
separate input choice: the preparation script records it when it creates
the processed source file, and the [Fisher guide](FISHER.md) explains
that assumption.

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
samples directly. It never shifts an already shifted curve. The provider
keeps the native axis and densities as `source_native_z` and
`source_native`; `source_z` names the numerical sampling axis. Both C++ adapters copy
these physical inputs into the owned C model. The two-point bridge receives
the same shifted samples and applies no second photo-z shift. Nonnegative
source samples alone do not guarantee a nonnegative natural-cubic
interpolant of the later divided IA kernel; inspect and refine that stage.
The measured-edge policy also preserves `source_native_low/high` and
records `source_unshifted_support`. A photo-z shift moves that support;
the observer and the finite background still limit the output.

The demonstration reconstructs the processed sources with measured-edge
PCHIP and samples the calculation on these grids:

    physics.source_nz: 9601    # independent uniform source-density nodes
    physics.kernel_nz: 401     # projected lensing/alignment kernel nodes
    physics.z count: 125       # CAMB power and background redshift nodes
    physics.los_z count: 129   # line-of-sight integration nodes

The [source study](tests/studies/source_policy_validation.md) records the
measured derivative changes between these grids, the smaller candidate
grids and both photo-z step sizes; those measurements select this
starting calculation and do not bound a marginalized FoM. The study also
keeps the original curves, the endpoint assumptions and the remaining
radial-refinement questions.

## Signed bispectra and triangle coordinates <a name="signed-projection"></a>

A signed IA contribution can cross zero and vary by many orders of
magnitude across Fourier scale. Interpolating its raw amplitude with a
cubic can produce tails much larger than the physical value. The optional
`residual_scaling: 1` instead stores the signed ratio of the TATT residual
to the positive equilateral gravitational bispectrum at the same perimeter.
The calculation interpolates the positive scale logarithmically and
multiplies it back after reading the ratio. `signed_baseline: 1` similarly represents the NLA
baseline by positive GGG and signed NLA/GGG. These are numerical
representations of the components, not changes to IA coefficients or a
prescription to replace negative values by their absolute magnitude.

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
remain possible. The preparation keeps both ordered-leg charts. Compare the
prepared multipoles with direct angular integration of the same projected
grid to isolate this interpolation stage from the underlying physics.
Successful unit tests of these representations and coordinate choices do
not establish a converged survey forecast.

## CAMB solve and extrapolation limits <a name="camb-limits"></a>

The physical input producer separates two wavenumber limits, following the
LSST Y1 likelihood. `physics.kmax_boltzmann` controls the transfer functions
CAMB actually evolves; its default is 5 in $`\mathrm{Mpc}^{-1}`$.
`physics.extrap_kmax` controls how far its power interpolator extends the
high-wavenumber tail, also in $`\mathrm{Mpc}^{-1}`$. If omitted, this second
limit covers at least 250 and 1.01 times the physical endpoint of the saved
table. CAMB supplies its log-power extrapolation; the producer requests no
additional transfer evolution merely to sample the tail.

The saved `physics.k` axis instead uses $`h\,\mathrm{Mpc}^{-1}`$. At $`h=0.7`$,
a saved node at 1000 therefore queries the interpolator at
$`700\,\mathrm{Mpc}^{-1}`$, while the Boltzmann solve can still end at
$`5\,\mathrm{Mpc}^{-1}`$. Increasing the saved table's upper endpoint must
not silently increase the transfer calculation. The producer refuses an
explicit extrapolation limit that does not cover the requested nodes.

The producer fixes the requested $`\sigma_8`$ by adjusting $`A_s`$ and
recalculating matter powers from the stored transfer functions. CAMB recomputes
the nonlinear correction at that amplitude; nonlinear power is not rescaled
by a constant. Metadata records both wavenumber limits, their units and this
amplitude procedure. Changing either limit invalidates the provider and
outer mean cache. Regenerate reference tables without this limit provenance
before varying cosmology in CAMB mode.

Two independent controls refine the transfer calculation:
`physics.camb_accuracy_boost` sets CAMB's `AccuracyBoost` (default 1), and
`physics.camb_k_per_logint` sets its transfer sampling per logarithmic
wavenumber interval (default 0, CAMB's automatic spacing). These controls
do not change the solved cutoff, the output `physics.k` grid, or any
CosmoLike sampling boost. All four choices must be recorded when comparing
power spectra. A dense output grid cannot recover structure missed by an
insufficiently resolved transfer calculation.

Both precision values enter saved metadata, the provider's exact cache
key and the timing-table reuse check. Changing them in a notebook rebuilds
CAMB tables; restoring the original values restores the original reference
arrays. Fixed-table providers refuse a changed precision request. Regenerate
the reference tables before declaring a new precision in a configuration.
Older tables without these fields retain their historical one/zero settings.

During shared-variance validation, the old three-point renormalization of
power was about 0.999686 at the fiducial point and changed nonmonotonically
under some parameter displacements. A nine-point diagnostic with CAMB
AccuracyBoost 2, transfer sampling 50, and 1025 uniform log-k output samples
reduced that multiplier's deviation from unity below 1.5e-7. This identifies
an input-accuracy requirement; it does not by itself establish converged
Fisher errors. The full joint comparison remains the acceptance test.

Vary the solved cutoff separately when testing sensitivity to the extrapolated
tail. A denser output grid only samples that same tail more closely; it does
not establish its physical accuracy. The C sampling and quadrature boosts
above change neither limit.

## Choosing Fisher accuracy from a default-accuracy base <a name="fisher-noise"></a>

A Fisher calculation can report artificially small errors when numerical
noise contaminates its derivatives. The derivative divides a difference of
nearby predictions by a small parameter step. Squaring that derivative in
$`F=D^T C^{-1}D+F_{\rm prior}`$ turns some of the noise into positive
information. Agreement of the fiducial mean alone does not test this effect.

The expensive part of the screening calculation runs once at **default
accuracy**: build the full derivative matrix and Fisher matrix, including
the configured priors. Three shared functions in the core's
`cosmolike_notebook_utils/fisher.py` compute the marginalized dark-energy
weights, the per-column stencil noise and the predicted FoM shift. They
implement the estimate derived in the derivative-accuracy section of the
Fisher machinery note in the LSST Y1 × DESI project
(`projects/lssty1xdesi/notes/fisher_machinery/`), and the project keeps
the same meaning of each quantity.

Let $`S=F^{-1}`$ be the parameter covariance and $`S_w`$ its marginalized
$`(w_0,w_a)`$ block. For parameter $`p_j`$, the default-accuracy calculation
supplies a weight

$$
R_j^2\sigma_j^2=S_{j,w}S_w^{-1}S_{w,j},\qquad \sigma_j^2=S_{jj}.
$$

The weight measures how uncertainty in that parameter affects the dark-energy
ellipse. A nuisance parameter can matter even though it is neither $`w_0`$
nor $`w_a`$; marginalization must keep its covariance with the pair.
At each candidate accuracy, measure only the columns that these weights
and the baseline noise audit select; rebuilding the entire derivative base
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
For large predicted shifts, the shared calculation also evaluates the exact
ratio $`{\rm FoM}(F+\operatorname{diag}a)/{\rm FoM}(F)`$ instead of truncating
the matrix calculation at first order.

The full default-accuracy run saves the five vectors at both $`h`$ and
$`h/2`$ for every parameter. Its baseline noise audit therefore needs no
additional model evaluations. A candidate examining $`k`$ columns at one
spacing needs $`1+4k`$ distinct evaluations, including a shared center.
The report lists omitted columns and their measured baseline contribution.
A small baseline contribution is not a bound on how that column's noise
changes at another accuracy.

This procedure screens settings; it does not certify convergence. A smooth
fourth derivative also enters $`\Delta^4m`$, while an odd numerical error can
cancel from it and still bias the first derivative. Correlations between
nearby evaluations violate the independent-noise estimate. The final check
therefore compares full Fisher matrices at finer settings and at both step
sizes, with the covariance, priors, physical model and rows held fixed.
Separate range, angular-truncation and source-table checks remain
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
changed control constructs a new owned C model. An old model the notebook still
holds remains valid and unchanged. `ExactMeanCache` also checks the prediction's numerical
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
arguments["variance"] = fit_arrays  # z/rnl/neff/sigma8 from this same power
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
choices, and measured bin edges keep their configured values. More interpolated
samples do not add information to an underresolved CAMB table. Refine its
producer grid separately when necessary.

Measured pair weights also remain physical input. The resolver refuses more
within-bin quadrature nodes while a catalogue weight array remains: those
weights refer to the original nodes. Supply weights at the new baseline
count, or select the uniform-area approximation explicitly.

The [state tests](tests/README.md) compare reused calculations with independent
fresh processes. Those comparisons establish cache and ownership correctness;
the interpolation and quadrature comparisons described in the
[numerical guide](../../external_modules/code/cosmolike_core/cosmolike/3pcf/NUMERICS.md)
address accuracy relative to the physical model.

## Executed numerical studies <a name="studies"></a>

| Study | What it measures |
|---|---|
| [Interpolation validation](tests/studies/interpolation_validation.md) | Projection-table comparisons against direct projected values and the full 80-row aperture-mean refinement |
| [Source and photo-z policy](tests/studies/source_policy_validation.md) | Source-model choices separated from source, kernel and radial resolution |
| [Fisher accuracy record](tests/studies/fisher_accuracy_validation.md) | Full NLA/TATT derivative bases, individually refined derivative columns, and coherent changes that the fourth-difference noise estimate misses |
| [Finite-window E/B study](tests/studies/finite_window_eb.md) | Why refinement of the E-only mean cannot measure leakage from omitted single-B bispectra after triangle cuts |
| [Fiducial angular-order study](tests/studies/angular_order_results.md) | Legendre and Fourier truncation orders separated from angular sampling and coupling interpolation; ten NLA/TATT fiducial mean comparisons and their distinction from derivative convergence |
| [BACCO response study](tests/studies/bacco_validation.md) | Direct emulator values, native interpolation, response-grid refinement and finite-difference steps, separated from neutrino and continuation choices |

## Shared smoothing controls <a name="shared-smoothing"></a>

`physics.variance_source: shared` asks the ordinary CosmoLike smoothing
engine for the fit's nonlinear radius, effective index and sigma8(z).
The default `local` retains the reference calculation while the full joint
mean and Fisher acceptance checks are completed. This selector is separate
from the matter prescription: the Python theory producer consumes those
copied variance arrays and always supplies the complete B table to C.

The project prepares the `local` fit through the independent Python
`ReferenceVariance` calculation and passes its owned arrays to the theory producer. Its
finite log-k integration, radius solve and scalar sigma8 normalization
retain the old reference convention. The normalization multiplies only
the three-point model's linear power; the ordinary two-point calculation
receives the original supplied spectrum. The saved metadata records the
multiplier and all inputs. Neither mode can silently normalize the other's
spectrum. Native constructors require the complete B table and never fit
it from smoothing arrays. Original-library equation comparisons use the
Python BiHalofit implementation against frozen pre-removal C and fastnc
outputs. Projection comparisons use `reference_physics_model`, which now
prepares an explicit, small Python B table; its coarse sampling is not a
survey accuracy result.

`physics.variance_growth: factorized` installs P(k,0)D(z)^2 and retains the
existing three-point effective-growth approximation. `evolving` instead
requires the complete linear P(k,z) table. With massive neutrinos those are
different physical choices. The shared route trusts the supplied power
amplitude; it does not rescale it to a second sigma8 scalar. The metadata
records that choice and the Boltzmann provider's exact neutrino density.

Four controls refine the shared tables: `variance_mass_boost` for top-hat
mass sampling, `variance_a_boost` for both scale-factor tables,
`variance_radius_boost` for Gaussian radii, and `variance_inverse_boost`
for the inverse Gaussian-variance table. Each multiplies `accuracy_boost`
and rounds upward to a power-of-two subdivision of the original intervals,
up to a factor of sixteen. Counts always come from the saved unboosted
baseline. Repeated calls therefore cannot square a boost. The analytic
FFTLog smoothing does not use `integration_accuracy_boost`; input power
sampling remains a separate producer-grid choice.

The provider retains owned snapshots keyed by the supplied power, cosmology,
output coordinates and resolved counts. Another provider may replace the
shared core state without changing an earlier snapshot or immutable model.
Changing a source distribution does not change its matter smoothing inputs.
The component tests check those ownership rules and compare the C and Python
fits using the same shared arrays.

The joint NLA and TATT comparisons pass on factorized-growth inputs prepared
with CAMB AccuracyBoost 2, k_per_logint 50 and 1025 uniform log-k samples.
At four-times shared sampling, the largest parameter-pair FoM differences
from the refined independent reference are 0.0541% for NLA and 0.0581% for
TATT, including both derivative steps. All two-point stencil rows remain
bitwise unchanged. NLA refinement from four to eight times sampling changes
any pair FoM by at most 0.0356%; there is no equivalent full TATT refinement
run. These are comparisons at specified inputs, not a universal forecast
accuracy guarantee. The older coarse-input calculation failed the same
comparison because the legacy scalar normalization changed the spectrum.
Some marginalized errors still change by about 2.8% under derivative-step
halving in both NLA methods. Defaults and native compatibility remain
separate from this acceptance record.

For analytic covariance, the shared route supplies halo variances and
logarithmic mass slopes at the exact requested mass and line-of-sight
nodes. It preserves the existing M200m density convention, Sheth-Tormen
mass function and Duffy concentration parameters. Shared smoothing does
not replace these physical prescriptions. The saved metadata includes
separate fit and halo variance records. A declared
`single_fluid_effective_growth` neutrino approximation requires factorized
growth; an evolving halo prescription needs its own physical validation.

The optional Cobaya `BiHalofitTheory` uses the same `variance_source` and
`variance_growth` selectors. Its `variance_accuracy` mapping supplies the
shared smoothing controls. The project setup must make the compiled 3PCF
interface importable when this route is selected. The local Python
reference does not require that compiled dependency.


## Background matter fraction in the native model

The theory producer and C halo covariance consume the provider's exact
`omega_m_z`. For pressureless matter the provider may derive it from
`Omega_m0*(1+z)^3/[H(z)/H0]^2`; massive neutrinos need their actual density
evolution. CAMB records its CDM+baryon+massive-neutrino fraction by default.
C never reconstructs another background or recomputes BiHalofit.

The required B table has its own interpolation error. Default Python axes
match the existing supplied example: 49 redshifts, 193 log-k nodes, 97
log-ratio nodes and 33 shape nodes, with dense-reader boost one. This is
30,271,857 values before any refinement, about 242 MB per double array.
Its accuracy and preparation cost must be measured in the actual forecast;
previous internal-C timings and Fisher results do not certify this route.
The global and bispectrum boosts refine all four coarse axes. Memory grows
with their product, so use individual sampling studies before large joint
boosts. Notebook and survey-refreeze runs remain paused during consolidation.


### Mandatory theory-B consolidation, 19 November 2026

The complete project suite passed 900 optimized tests and 900 UBSan tests.
Before/after comparisons of the existing supplied-B path preserved all
60 arrays bitwise: matter/tree queries, NLA/TATT alignment and projections,
radial outputs, the mean and the six supported covariance families. All
30 one-versus-six-worker pairs were bitwise identical. Frozen pre-removal C
and fastnc outputs remain independent equation references for Python
BiHalofit, including squeezed triangles and shared-variance inputs.
These are implementation and state checks. They do not establish full DES
Fisher convergence or transfer the retired internal-C timing to the theory
producer. Notebook execution and survey refreezes remain paused.
