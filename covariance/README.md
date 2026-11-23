# Joint xi and aperture covariance

1. [Physical model and estimator](#model)
2. [Run the covariance calculation](#running)
3. [Configuration and saved archive](#inputs)
4. [Numerical refinement](#refinement)
5. [Capabilities of the retired covariance command](#retired)

## Physical model and estimator <a name="model"></a>

The compiled calculation returns the joint covariance of full-filter
aperture moments and the configured $`\xi_\pm`$ separation bins. It keeps
the full two-point vector and all calculated cross entries. The
[core physical guide](../../../external_modules/code/cosmolike_core/cosmolike/3pcf/COVARIANCE.md)
derives the Gaussian, $`BB`$, $`PT`$, $`PB`$, connected one-halo and
halo-abundance response families. Those are the six stored
terms; the last is not a complete super-sample covariance.

The physical field is gravitational lensing, with the Gaussian
shape-noise matrix from the configuration. Its connected halo model does
not include nonlinear intrinsic-alignment cumulants. Using it while
varying an IA mean is an explicit fixed-covariance approximation, which
the configuration records as `ia_covariance="gravitational_only"`.

The driver computes only the $`\xi_\pm`$ rows (`--xi-only` is required).
The likelihood's Map3 mean uses the finite-SAS estimator, in which
measured triangles select the aperture integrand, while the C covariance
codes only the full-filter estimator; selecting measured triangles
changes the estimator whose fluctuations the covariance describes, so a
Map3 covariance is refused until a finite-SAS covariance exists. The
driver and the adapter both state this refusal before any numerical work.

The halo prescription treats the input as one clustering matter fluid.
`neutrino_model="massless"` therefore refuses a positive neutrino mass.
The explicit `"single_fluid_effective_growth"` choice permits that
approximation with massive-neutrino power and records it; it does not
implement the ordinary core's cold-plus-baryon halo-formation
prescription. Fiducial multiplicative shear offsets must be zero for this
driver; Fisher variations of those offsets change the mean while keeping
the saved covariance fixed.

## Run the covariance calculation <a name="running"></a>

We assume Cocoa is activated, the current directory is `Cocoa/`, and the
extension was compiled with covariance generation enabled
([project instructions](../README.md#computing_covariances)): unset
`IGNORE_COSMOLIKE_DES_3PCF_COVARIANCE`, then
`source ./projects/des_3pcf/scripts/compile_des_3pcf.sh`. The
covariance-off build exposes `has_covariance=False`, and the driver
refuses it with the recompile instruction.

[compute_covariance.py](compute_covariance.py) follows the LSST Y1
covariance layout: it reads one evaluate YAML, builds the Cobaya model
(CAMB, BiHalofit and `des_3pcf.joint_shear`), evaluates it once at
`sampler.evaluate.override`, and hands the evaluated likelihood to the
adapter [des_3pcf_covariance.py](des_3pcf_covariance.py), which calls the
C covariance kernels on the likelihood's own rows and physics model. No
sampler runs, and the likelihoods never read the matrix written here.

**Step :one:**: compute and save the fiducial covariance.

```bash
OMP_NUM_THREADS=8 python ./projects/des_3pcf/covariance/compute_covariance.py \
  ./projects/des_3pcf/EXAMPLE_EVALUATE_COVARIANCE.yaml --xi-only
```

[EXAMPLE_EVALUATE_COVARIANCE.yaml](../EXAMPLE_EVALUATE_COVARIANCE.yaml)
is the EXAMPLE_EVALUATE1 fiducial with the shear offsets $`m_i`$ set to
zero and `single_fluid_effective_growth` declared for its
$`m_\nu=0.06`$ eV. Command options, one per flag:

- `--output`: override the YAML's `output` key (a `.npz` path).
- `--overwrite`: replace an existing archive; without it the driver
  refuses.
- `--save-terms`: also store the six covariance terms.
- `--xi-only`: compute only the $`\xi_\pm`$ rows (required; see the
  refusal above).

The driver prints the matrix size, the smallest eigenvalue and the
likelihood and covariance times. It does not clip eigenvalues or add a
diagonal term to make an unresolved matrix invertible.

## Configuration and saved archive <a name="inputs"></a>

The YAML's `covariance` block accepts one key, `neutrino_model`. Every
other choice is written out in the adapter's configuration: the DES Y3
survey area, the four source densities and ellipticity dispersions, the
circular survey window, the large-area and closed-form noise modes, the
halo-model constants and the quadrature grids. The saved record therefore
states exactly what ran; changing a setting means editing the adapter,
whose comments give each field's units and meaning.

With per-component dispersion the diagonal white-noise power is
$`N_i=\sigma_{e,i}^2/n_i`$, where $`n_i`$ is converted to inverse
steradians. No extra factor of two belongs in this definition. Both E
and B noise contribute to xi covariance even when the signal is pure E.

The closed-form noise mode passes the actual xi bin edges. If two bins
have angular annulus areas $`\Omega_b,\Omega_c`$ and overlap
$`\Omega_{bc}`$, their same-kind pure-noise covariance is

```math
C^{NN}_{(ij,b),(kl,c)}=
\frac{2(N_{ik}N_{jl}+N_{il}N_{jk})\Omega_{bc}}
{A\Omega_b\Omega_c}.
```

It vanishes for a xi-plus/xi-minus pair and for disjoint annuli. This
expression integrates the white-noise delta function over continuous
uniform-area bins; only the signal-signal and signal-noise terms retain
the Fourier quadrature.

The archive holds `covariance` (the unmodified sum), `model_mean` (the
noise-subtracted gravitational mean of this covariance model, which can
differ from an IA mean saved by an evaluation), `row_labels`, and a
`metadata` JSON string with the resolved settings, the SHA-256 digest of
each input file named by its project-relative path, and the producing
script. With `--save-terms` it adds `terms` with shape `[6,nrow,nrow]`
and their names: `gaussian`, `bispectrum_squared`,
`power_trispectrum_one_halo`, `power_bispectrum`, `connected_one_halo`
and `halo_abundance_response`. It reads back with
`numpy.load(path, allow_pickle=False)`.

Omitting `--save-terms` controls only the output file: the C calculation
still forms all six matrices before summing them, so the option does not
reduce peak memory or integral count.

## Numerical refinement <a name="refinement"></a>

Coarse halo-profile nodes determine the sampled function. A natural cubic
interpolator fills a dense uniform transformed-radius grid, and the hot
xi endpoint reads use prepared cell indices and linear fractions. Refining
only the dense grid cannot restore structure missed by coarse samples.
The configuration includes zero and positive logarithmic nodes. Resolving
the inner halo profile needs close spacing near zero; a uniform table over
the full radial range does not provide that inner resolution. Refine the
node spacing, especially where $`\ell_{\max}r`$ changes rapidly,
independently of `profile_upsample`. Logarithmic spacing covers a wide
radial range; it does not itself resolve large-radius Bessel
oscillations.

The Fourier, angular, halo-mass, halo-offset and line-of-sight integrals
each have separate fixed rules. The low-wave-number survey-window rule
must resolve the footprint scale; it is distinct from the short-mode
Fourier integration. The calculation shares source and aperture-radius
factors where their mathematical dependence permits it; repeated
angular-bin geometry does not require a separate Bessel calculation for
every source pair.

Check each returned family, their sum, its covariance modes and the final
marginalized errors under refinement. A positive matrix alone does not
establish accuracy, and a small negative eigenvalue is a diagnostic to
investigate. The [test guide](../tests/covariance/README.md) lists
independent physical checks; the [validation record](../tests/VALIDATION.md)
distinguishes them from survey-size convergence.

## Capabilities of the retired covariance command <a name="retired"></a>

The former command-line covariance route (`scripts/run_3pcf.py covariance`
with its `configs/` files) was removed with the private prediction route;
it remains in Git history. Three of its capabilities were recorded as
lost, not migrated:

- A covariance for any row set or survey other than the Y3 likelihood's
  own rows, including the former DES Y6 forecast configuration with
  user-chosen apertures, bins, source curves, area and noise. A future
  forecast adds its own dataset and adapter configuration.
- The explicit cross-bin noise matrix input (`noise.matrix`), which could
  describe sample overlap between source bins.
- The forecast row layouts that the old command could assemble without a
  likelihood dataset.

The current driver always takes its rows, source distributions and
fiducial point from the evaluated `des_3pcf.joint_shear` likelihood.
