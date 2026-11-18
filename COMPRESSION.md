# Optional compression and finite-simulation likelihoods

1. [What a compression preserves](#preservation)
2. [Retaining the joint information while compressing xi](#conditional)
3. [Fixed matrices and notebook use](#notebook)
4. [The finite-simulation prescription](#finite)
5. [Validation and reproduction limits](#limits)
6. [Acknowledgements and sources](#credits)

## What a compression preserves <a name="preservation"></a>

The default likelihood retains every selected physical measurement. A user
may instead choose a fixed matrix $`T`$ and define a measured statistic
$`y=T d`$. Its mean is $`T\mu(p)`$ and its covariance is $`TCT^T`$.
All three transformations are necessary. Keeping the old covariance after
compressing the mean would assign the wrong errors to the measurements.
The matrix is fitted once at a declared training point and held fixed
throughout inference. Rebuilding it at each sampled point would change
what was measured.

For a fixed Gaussian covariance, the local Fisher matrix is
$`F=D^TC^{-1}D`$, where column $`D_\alpha`$ is the derivative of the mean
with respect to parameter $`p_\alpha`$. Whitening means solving
$`L X=D`$ with $`C=LL^T`$. In these coordinates the noise has unit
covariance. The left singular vectors of $`X`$ identify the measured
combinations that respond to the trained parameter changes. Keeping their
span preserves that local Fisher matrix up to the declared numerical rank
threshold. The code scales columns before measuring rank, so changing a
parameter's units cannot erase a small but nonzero response by underflow.

This statement is local and concerns the supplied derivatives. It does not
establish their physical accuracy, preserve arbitrary nonlinear posterior
shapes, or retain goodness-of-fit information in discarded directions.
Every varied nuisance parameter needs a training column. A matrix trained
without `feedback_amplitude` has no information-preservation guarantee for
a later analysis that varies that amplitude. The inference record lists
sampled parameters absent from the recorded training names.

## Retaining the joint information while compressing xi <a name="conditional"></a>

Write the two-point vector as $`x`$ and the retained aperture vector as
$`z`$. Ordinary xi-only MOPED uses $`C_{xx}`$ and $`D_x`$. The joint
likelihood also knows $`C_{xz}`$, so a xi combination with no direct
parameter response can still be useful: it traces noise shared with Map3.
For example, let $`D_x=(1,0)^T`$, $`D_z=1`$, $`C_{xx}=I`$,
$`C_{xz}=(0,r)^T`$, and $`C_{zz}=1`$. The second xi row has zero mean
response, yet retaining it increases the amplitude information from 2 to
$`2+r^2/(1-r^2)`$. Discarding it loses that noise subtraction.

The `conditional_xi` option constructs the conditional covariance and
response:

$$
S=C_{xx}-C_{xz}C_{zz}^{-1}C_{zx},\qquad
J=D_x-C_{xz}C_{zz}^{-1}D_z.
$$

If $`S=LL^T`$ and the columns of $`Q`$ span $`L^{-1}J`$, the weights
are $`W=Q^TL^{-1}`$. Apply $`T=\operatorname{diag}(W,I)`$ to the
joint vector. The retained Map3 entries and their cross-covariance with
$`Wx`$ recover the local joint information. Neither the data nor the
cross-covariance is dropped. The `xi_moped` option is also available for
comparisons with analyses that deliberately use the ordinary xi basis;
`joint` compresses both probes together.

## Fixed matrices and notebook use <a name="notebook"></a>

[EXAMPLE_COMPRESSION.ipynb](EXAMPLE_COMPRESSION.ipynb) first evaluates the
three-row example, then compares the alternatives using the saved TATT
Fisher derivatives. It treats those saved values as historical numerical
arrays, with no claim that they describe the current implementation or
satisfy the physical accuracy requirements. The notebook saves the input
hash, fiducial, all parameter names, derivative field, training covariance
and numerical rank with each transform. Existing files with different
contents are refused instead of silently redefined.

The NPZ file stores `matrix[noutput,ninput]`, `input_labels[ninput]`,
`output_labels[noutput]` and a JSON `metadata` scalar. It uses no pickle.
`LinearCompression` also accepts an externally supplied matrix with these
labels and provenance, allowing a published matrix to be imported without
retraining. Selection occurs before compression. Input labels must match
exactly, including order; a matching row count alone is insufficient.
Compressed modes carry labels and values, without fictitious physical
angles or tomographic-bin assignments.

An inference YAML enables a matrix explicitly:

```yaml
compression:
  mode: file
  path: runs/conditional_xi_compression.npz
likelihood:
  mode: gaussian
```

Omitting these sections resolves to `compression.mode=none` and
`likelihood.mode=gaussian`; the run record retains these resolved choices.
The source matrix's bytes and metadata are included in the inference
identity checked before resuming a chain.

The notebook also prepares a basis using a fixed diagonal teaching
covariance independent of the supplied sample covariance. Its arbitrary
noise scales are recorded. This is a statistical demonstration, not an
analytical DES covariance calibration. A basis trained on the production
sample covariance does not have this independence property.

## The finite-simulation prescription <a name="finite"></a>

`FiniteSimulationLikelihood` implements the prescription of
[Percival et al., equations 52 and 54–55](https://arxiv.org/html/2108.10402v2#S7).
Let $`N_s`$ count independent Gaussian mock vectors, $`N_d`$ count
retained entries after selection and compression, and $`N_p`$ count
varied mean parameters, including nuisance parameters. The input $`S`$
is the ordinary unbiased sample covariance. No Hartlap rescaling is made.

$$
B=\frac{N_s-N_d-2}{(N_s-N_d-1)(N_s-N_d-4)},\qquad
m=N_p+2+\frac{N_s-1+B(N_d-N_p)}{1+B(N_d-N_p)},
$$

$$
\log L=\log\mathcal N-\frac m2
\log\left[1+\frac{(d-\mu)^TS^{-1}(d-\mu)}{N_s-1}\right].
$$

Gaussian mock vectors and independent realizations give the Wishart
sampling law assumed for $`S`$. The observed datum must be independent
of that estimate. When compression is used, its fixed matrix must also be
independent of the production sample covariance. The API requires named
assumption declarations and records them; it cannot verify those statements
without the underlying ensemble. Analytical, hybrid and PME covariance
estimates are refused as raw sample inputs.

The code requires $`N_s>N_d+4`$, $`1\leq N_p\leq N_d`$ and
$`m>N_d`$. The last condition is necessary for a normalized data density.
For example, $`N_s=796,N_d=480,N_p=20`$ gives $`m\simeq344.5`$,
which fails that condition even though the first inequality holds. This
implementation refuses that normalized-density request. It does not
silently compress the data or report evidence from an unspecified
posterior-kernel normalization. The full-vector Gaussian likelihood
remains available as a separate explicit statistical choice.

[EXAMPLE_NAUTILUS4.yaml](EXAMPLE_NAUTILUS4.yaml) uses the notebook's
independent pilot matrix and an explicit finite-simulation model. Run the
following from the activated `Cocoa/` directory after the notebook has
written its matrices.

**Step :one:**: inspect inputs and statistical declarations.

```bash
python projects/des_3pcf/scripts/run_nautilus.py projects/des_3pcf/EXAMPLE_NAUTILUS4.yaml --validate-config
```

**Step :two:**: evaluate a physical fiducial, an IA displacement and the
restored fiducial without sampling.

```bash
python projects/des_3pcf/scripts/run_nautilus.py projects/des_3pcf/EXAMPLE_NAUTILUS4.yaml --check
```

The example's 796-footprint independence and Gaussian-mock declarations
are working assumptions. Footprints sharing a full-sky realization need
an independence assessment. Informative priors and nonlinear posteriors
also limit the frequentist coverage argument. Recomputing $`m`$ after
compression defines that compressed analysis; it is not the marginal of
the original full-data t density.

## Validation and reproduction limits <a name="limits"></a>

The tests compare conditional compression with direct joint Fisher solves,
including unequal units, reordered rows, dependent derivative columns,
nonzero measured data, caller mutation and saved-file restoration. The
normalized likelihood is compared with SciPy's multivariate-t density,
an independently specified exponent, and its large-mock Gaussian limit.
A nonintegrable density and incompatible covariance provenance are refused.

For the archived 480-row TATT arrays, conditional xi compression retains
100 rows and agrees with the full local Fisher matrix to a maximum
normalized element difference of $`1.6\times10^{-15}`$. Full joint
compression retains 20 modes. These are algebra checks of that archive;
they are not evidence of posterior reproduction or numerical convergence
of the physical forecast.

Published reproduction additionally needs the exact cuts, source treatment,
priors, compression matrix and estimator conventions. The original
within-bin triangle weights are absent from the inspected public packed
FITS products. The shared-sky mock correlations and covariance sign
provenance remain scientific checks. See
[the estimator review](tests/studies/estimator_conventions.md).

## Acknowledgements and sources <a name="credits"></a>

**We thank Sunao Sugiyama and Rafael Gomes for making their theory and
analysis implementations available.** This work builds on
[fastnc](https://github.com/git-sunao/fastnc/tree/tatt),
[3pcf_integrator](https://github.com/rchgomes/3pcf_integrator), and
[desy6-3pcf](https://github.com/rchgomes/desy6-3pcf), with the reproduction
targets given by their [methods paper](https://arxiv.org/abs/2503.03964)
and [TATT paper](https://arxiv.org/abs/2601.09133).

The response compression follows
[Heavens, Jimenez and Lahav](https://arxiv.org/abs/astro-ph/9911102).
The finite-simulation prescription follows
[Percival, Friedrich, Sellentin and Heavens](https://arxiv.org/abs/2108.10402).
The distinct simulation correction method discussed elsewhere in the
project is **Oliver Friedrich and Tim Eifler, “Precision matrix expansion
— efficient use of numerical simulations in estimating errors on
cosmological parameters”** ([paper](https://arxiv.org/abs/1703.07786),
[PDF](https://arxiv.org/pdf/1703.07786)). It expands the precision matrix
about an analytical model; it is not the raw-sample likelihood above.
