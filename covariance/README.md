# Joint xi and aperture covariance

1. [Physical model and estimator](#model)
2. [Run the covariance calculation](#running)
3. [Inputs and saved terms](#inputs)
4. [Numerical refinement](#refinement)

## Physical model and estimator <a name="model"></a>

The compiled calculation returns the joint covariance of full-filter
aperture moments and the configured $`\xi_\pm`$ separation bins. It keeps
the full two-point vector and all calculated cross entries. The
[core physical guide](../../../external_modules/code/cosmolike_core/cosmolike/3pcf/COVARIANCE.md)
derives the Gaussian, $`BB`$, $`PT`$, $`PB`$, connected one-halo and
halo-abundance response families. Those are the six stored
terms; the last is not a complete super-sample covariance.

The physical field is gravitational lensing, with the Gaussian
shape-noise matrix from the configuration. Its connected halo model does not include nonlinear
intrinsic-alignment cumulants. Using it while varying an IA mean is an
explicit fixed-covariance approximation. The configuration records that
choice as `ia_covariance="gravitational_only"`. The driver refuses a
finite-SAS aperture mean with this analytic covariance, because selecting
measured triangles changes the estimator whose fluctuations the
covariance describes.

The halo prescription treats the input as one clustering matter fluid.
`neutrino_model="massless"` therefore refuses a positive neutrino mass.
The explicit `"single_fluid_effective_growth"` choice permits that
approximation with massive-neutrino power and records it. It does not
implement the ordinary core's cold-plus-baryon halo-formation prescription.
Fiducial multiplicative shear offsets must be zero for this covariance
driver. Fisher variations of those offsets change the mean while keeping
the saved covariance fixed.

## Run the covariance calculation <a name="running"></a>

We assume Cocoa is activated, the current directory is `Cocoa/`, and the
project extension, worker settings and physical tables are prepared using
the [project instructions](../README.md#evaluate). The project build
compiles the covariance files under `cosmolike/3pcf/`; it neither builds
nor changes the existing `cosmolike/covariances/` module.

The YAML or JSON file must supply catalogue area, noise and the intended numerical
grids. Those catalogue values are deliberately unset in the examples.
The CLI and notebook adapters call the same physical C functions. The
CLI additionally records row identity and the complete configuration;
the notebook user is responsible for preserving that information.

**Step :one:**: check the configuration and missing inputs.

```bash
python projects/des_3pcf/scripts/run_3pcf.py covariance \
  projects/des_3pcf/configs/y3_full_filter_example.yaml --validate-config
```

**Step :two:**: calculate the fiducial covariance after supplying those inputs.

```bash
python projects/des_3pcf/scripts/run_3pcf.py covariance \
  projects/des_3pcf/configs/y3_full_filter_example.yaml
```

The output path is `covariance.path` in the configuration. The driver
refuses to replace existing output unless the command adds `--overwrite`.
By default the calculation writes the summed covariance, its model mean
and row metadata, along with its smallest eigenvalue. Set
`covariance.save_components: true` to save the six terms too. It does not
clip eigenvalues or add a diagonal term to make an unresolved matrix
invertible.

## Inputs and saved terms <a name="inputs"></a>

| Configuration choice | Meaning |
|---|---|
| `physical.area` | Survey area in steradians |
| `noise.number_density` | Effective galaxies per square arcminute for every source bin |
| `noise.sigma_component` | RMS ellipticity for one component in every source bin |
| `noise.matrix` | Alternative explicit E/B noise cross-power matrix, including sample overlap if present |
| `physical.center_window` | 1 for a circle or 2 for a square with the same area |
| `physical.large_area` | Large-area short-mode integrals when 1; finite center-window integrals when 0 |
| `physical.noise_mode` | 1 for analytic continuous-annulus xi white noise; 0 for the finite-Fourier-band calculation |
| `pair_weight` | `uniform_area`, or explicit complete separation quadrature weights |
| `physical.profile_upsample` | Dense linear-read refinement after halo-profile cubic preparation |

With per-component dispersion the diagonal white-noise power is
$`N_i=\sigma_{e,i}^2/n_i`$, where $`n_i`$ is converted to inverse
steradians. No extra factor of two belongs in this definition. Both E
and B noise contribute to xi covariance even when the signal is pure E.

For `noise_mode=1`, the project passes the actual xi bin edges. If two
bins have angular annulus areas $`\Omega_b,\Omega_c`$ and overlap
$`\Omega_{bc}`$, their same-kind pure-noise covariance is

```math
C^{NN}_{(ij,b),(kl,c)}=
\frac{2(N_{ik}N_{jl}+N_{il}N_{jk})\Omega_{bc}}
{A\Omega_b\Omega_c}.
```

It vanishes for a xi-plus/xi-minus pair and for disjoint annuli. This
expression integrates the white-noise delta function over continuous
uniform-area bins. Only the signal-signal and signal-noise terms retain
the Fourier quadrature. The mode requires `large_area=1` and
`pair_weight="uniform_area"`. For other weighting or a finite window,
mode 0 keeps the configured band limit; it is not an exact unbandlimited
catalogue pair-count calculation.

With `save_components: true`, the saved `terms` array has shape
`[6,nrow,nrow]`. The file also stores each term under its own name:
`gaussian`, `bispectrum_squared`, `power_trispectrum_one_halo`,
`power_bispectrum`, `connected_one_halo` and `halo_abundance_response`.
`covariance` is their unmodified sum. `covariance_model_mean` is the
noise-subtracted gravitational mean of this covariance model, which can
differ from an IA mean saved by `evaluate`. The file carries row labels,
source indices, angles and all physical/numerical choices.

The saving option controls the output file. The C calculation still forms
all six matrices before summing them; omitting their saved copies does not
reduce that calculation's peak memory or integral count. The production
adapter borrows validated input arrays, whereas the notebook adapter owns
converted arrays for interactive use. Both evaluate the same equations.

## Numerical refinement <a name="refinement"></a>

Coarse halo-profile nodes determine the sampled function. A natural cubic
interpolator fills a dense uniform transformed-radius grid, and the hot
xi endpoint reads use prepared cell indices and linear fractions. Refining
only the dense grid cannot restore structure missed by coarse samples.
The example includes zero and positive logarithmic nodes. Resolving the
inner halo profile needs close spacing near zero; a uniform table over
the full radial range does not provide that inner resolution. Refine the
node spacing, especially where $`\ell_{\max}r`$ changes rapidly,
independently of `profile_upsample`. Logarithmic spacing covers a wide
radial range; it does not itself resolve large-radius Bessel
oscillations.

The Fourier, angular, halo-mass, halo-offset and line-of-sight integrals
each have separate fixed rules. The low-wave-number survey-window rule
must resolve the footprint scale; it is distinct from the short-mode
Fourier integration. The calculation shares source and aperture-radius
factors where their mathematical dependence permits it. Repeated angular-bin geometry
does not require a separate Bessel calculation for every source pair.

Check each returned family, their sum, its covariance modes and the final
marginalized errors under refinement. A positive matrix alone does not
establish accuracy, and a small negative eigenvalue is a diagnostic to
investigate. The [test guide](../tests/covariance/README.md) lists
independent physical checks; the [validation record](../tests/VALIDATION.md)
distinguishes them from survey-size convergence.
