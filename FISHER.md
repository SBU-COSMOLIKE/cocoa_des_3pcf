# Joint forecasts without compressing the two-point vector

1. [Run a declared forecast](#run)
2. [Which physical calculation changes?](#parameters)
3. [What the saved derivatives mean](#derivatives)
4. [How the covariance weights information](#information)
5. [Correcting an analytic precision with simulations](#precision)
6. [Y3 and Y6 reproduction requirements](#reproduction)

## Run a declared forecast <a name="run"></a>

We assume Cocoa is active, the current directory is `Cocoa/`, and the
compiled library, thread environment and physical tables have been prepared
using the [project instructions](README.md#evaluate). The configuration
must name a covariance with exactly the same observable rows. For an
analytic matrix, first follow the [covariance guide](covariance/README.md).

**Step :one:**: copy a configuration and set the parameters to differentiate.

`fisher.steps` gives the positive displacement for each varied parameter;
its order defines the columns of the saved derivative matrix. Set
`fisher.priors` to the actual independent Gaussian widths to apply, or
leave it empty. `fisher.bounds` can reject a displacement outside a
physical domain before any prediction is evaluated. `fisher.fom_pairs`
names the two-parameter marginalized figures of merit to report.

**Step :two:**: check the selected inputs.

```bash
python projects/des_3pcf/scripts/run_3pcf.py fisher \
  projects/des_3pcf/configs/y3_full_filter_example.json --validate-config
```

**Step :three:**: run the derivatives and save the result.

```bash
python projects/des_3pcf/scripts/run_3pcf.py fisher \
  projects/des_3pcf/configs/y3_full_filter_example.json
```

The numerical file retains the configuration, row labels, fiducial mean,
two derivative estimates, two Fisher matrices and the fiducial covariance.
It also records a rank failure when parameters remain unconstrained. It
does not silently use a pseudoinverse to manufacture marginalized errors.
`--overwrite` explicitly permits replacing the configured output file.

## Which physical calculation changes? <a name="parameters"></a>

| Varied input | Calculation that changes |
|---|---|
| Cosmology, such as `omega_m`, `sigma8`, `w0`, `wa` | CAMB power/background tables and the physical C model |
| Source shift `dz1`, ..., `dzN` | Source densities, their normalization and projected kernels |
| IA coefficients `a1`, `eta1`, `a2`, `eta2`, `bta`, `ct` | Alignment contributions and their projected tables |
| Shear offsets `m1`, ..., `mN` | Final factors, one per observed shear field |
| Numerical controls or estimator geometry | A new preparation/plan; these are refinement choices, not cosmological parameters |

`cosmology_mode="fixed_tables"` accepts the stored cosmology and refuses a
different one. A cosmological Fisher forecast uses `"camb"`. The physical
provider retains the latest cosmology and source-weighted model with exact
parameter comparisons. A derivative displacement is never rounded away
because it is close to the reference value.

A positive photo-z shift uses $`n_i(z-\Delta z_i)`$, moving the population
to greater redshift. The curve is zero outside its supplied support, has
its observer endpoint set to zero, and is normalized after shifting.
Refine the source grid if appreciable density reaches its first cell.
Shear calibration is applied afterward: a $`(1,1,2)`$ aperture row receives
$`(1+m_1)^2(1+m_2)`$. Calibration-only displacements reuse the physical
mean and its geometry.

The isolated two-point route computes NLA E power and zero B signal.
It refuses nonlinear TATT, velocity-shear or a bispectrum-only baryon
choice. `mean.two_point.mode="core_tatt"` instead uses an existing compiled
ordinary CosmoLike module. That route installs the same physical tables,
uses its CFASTPT calculation and translates the IA normalization and
pivot. It changes no ordinary-core source file. Full linear power at
every redshift and cold-plus-baryon power for massive neutrinos are
required inputs to that bridge.

The ordinary core has one mutable model per process. The bridge owns its
use during a prediction; another project must not alter that model
concurrently. Independent worker calculations belong inside C. Running
simultaneous Python calls that change one shared cosmology is a different
operation and is not the threading model used here.

## What the saved derivatives mean <a name="derivatives"></a>

For one parameter $`p`$ and displacement $`h`$, the fourth-order centered
formula is

```math
\frac{\partial\boldsymbol\mu}{\partial p}\simeq
\frac{-\boldsymbol\mu(p+2h)+8\boldsymbol\mu(p+h)
-8\boldsymbol\mu(p-h)+\boldsymbol\mu(p-2h)}{12h}.
```

The calculation repeats this formula at $`h/2`$, reusing identical
parameter points. A smaller displacement reduces truncation error but
can magnify numerical noise in the mean. Agreement of these two columns
is therefore useful evidence, rather than a rule that the smaller step
must always be accepted.

| Saved array | Interpretation |
|---|---|
| `derivatives_h`, `derivatives_h2` | `[nrow,nparameter]` derivatives at the two step sizes |
| `derivative_relative_change` | Largest absolute column difference divided by the largest absolute refined derivative |
| `fisher_h`, `fisher_h2` | Information matrices with identical covariance and priors |
| `parameter_covariance_h2` | Inverse refined information, present only after the rank check passes |
| `parameter_names`, `row_labels` | Parameter-column and observable-row identity |

The column diagnostic is not a covariance-weighted acceptance test. Inspect
changes in marginalized errors and covariance-weighted derivative
differences as well. Repeat the prediction at refined coarse tables,
dense read tables, integration rules and transform boundaries. Derivative
step agreement on an unresolved numerical grid can reproduce the same
biased derivative twice.

## How the covariance weights information <a name="information"></a>

With Jacobian $`J_{i\alpha}=\partial\mu_i/\partial p_\alpha`$ and a
fixed fiducial covariance $`C`$, the mean-information forecast is

```math
F_{\alpha\beta}=J_{i\alpha}(C^{-1})_{ij}J_{j\beta}
+F^{\rm prior}_{\alpha\beta}.
```

The implementation solves with a Cholesky factor of $`C`$ rather than
forming its inverse. Every retained xi and aperture row participates,
including their cross-covariance. A Gaussian prior with width
$`\sigma_p`$ adds $`1/\sigma_p^2`$ to its parameter diagonal. The example
files contain no unverified survey nuisance priors.

The covariance is held fixed during differentiation. No parameter-dependent
covariance information term is included. Inverting the full parameter
matrix marginalizes over the other varied parameters. For a selected
pair, the reported figure of merit is
$`1/\sqrt{\det C_{\rm parameter,pair}}`$. This convention omits any
additional confidence-contour area constant.

## Correcting an analytic precision with simulations <a name="precision"></a>

The precision-matrix expansion uses an analytic model as the baseline and
simulations to estimate its residual covariance. It does not require
compressing the two-point rows. It does require the simulated vectors to
represent exactly the same estimator and row order as the analytic model.
See [Friedrich and Eifler](https://arxiv.org/abs/1703.07786) for the method.

For the CLI, `fisher.precision` declares `mode="pme"`, `order=1` or `2`,
the full joint `sample_covariance` file and its matrix `key`, and the
integer `independent_count`. It must also explicitly declare
`sample_assumption="independent_gaussian_vectors"`. Correlated rotations
of one simulation are not independent realizations merely because they
produce several stored vectors.

The second-order product has a finite-sample correction derived for a
full Wishart sample covariance. Applying that correction to an arbitrary
subtracted noisy covariance does not preserve its derivation. The Python
module also provides an explicitly independent two-group construction.
The CLI records the whitened residual size and the precision polynomial's
smallest eigenvalue. An indefinite precision is refused for Fisher
weighting; there is no eigenvalue repair. A positive result still needs
control of the expansion truncation and the assumed simulation statistics.

## Y3 and Y6 reproduction requirements <a name="reproduction"></a>

The supplied Y3 reference vector is a specific public simulated-data
configuration, with provenance in the download manifest. Matching its
rows and covariance is a file-consistency check. Reproducing the paper
also needs the final triangle estimator, scale selection, nuisance priors,
physical conventions and likelihood treatment. The full-filter analytic
forecast is a separate observable definition from the finite-SAS vector.

The Y6 example needs the external source/catalogue inputs and final priors.
It must not inherit Y3 source curves, area or noise by default. Absolute
forecast errors and joint gains should be reported only after those
choices and the numerical refinements are recorded for that release.
