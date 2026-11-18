# Optional compression and finite-simulation likelihoods

1. [What a compression preserves](#preservation)
2. [Retaining the joint information while compressing xi](#conditional)
3. [Fixed matrices and notebook use](#notebook)
4. [The finite-simulation prescription](#finite)
5. [Validation and reproduction limits](#limits)
6. [Acknowledgements and sources](#credits)

## What a compression preserves <a name="preservation"></a>

The default likelihood keeps every chosen measurement row. A user
may instead choose a fixed matrix $`T`$ and define a measured statistic
$`y=T d`$. Its mean is $`T\mu(p)`$ and its covariance is $`TCT^T`$.
All three transformations are necessary: keeping the old covariance after
compressing the mean would assign the wrong errors to the measurements.
The notebook fits the matrix once at a training point that the saved file
records, and inference holds it fixed. Rebuilding it at each sampled
point would change what was measured.

For a fixed Gaussian covariance, the local Fisher matrix is
$`F=D^TC^{-1}D`$, where column $`D_\alpha`$ is the derivative of the mean
with respect to parameter $`p_\alpha`$. Whitening means solving
$`L X=D`$ with $`C=LL^T`$. In these coordinates the noise has unit
covariance. The left singular vectors of $`X`$ identify the measured
combinations that respond to the trained parameter changes. Keeping their
span preserves that local Fisher matrix up to the stated numerical rank
threshold. The code scales columns before measuring rank, so changing a
parameter's units cannot erase a small but nonzero response by underflow.

This guarantee is local and concerns the training derivatives; it does
not preserve nonlinear posterior shapes or goodness-of-fit information in
discarded directions. Every varied nuisance parameter therefore needs a
training column: a matrix trained without `feedback_amplitude` carries no
information-preservation guarantee for a later analysis that varies that
amplitude. The inference record lists sampled parameters absent from the
recorded training names.

## Retaining the joint information while compressing xi <a name="conditional"></a>

Write the two-point vector as $`x`$ and the kept aperture vector as
$`z`$. MOPED, Massively Optimised Parameter Estimation and Data
compression ([Heavens, Jimenez and Lahav](https://arxiv.org/abs/astro-ph/9911102)),
projects the data onto its parameter-response directions. Ordinary
xi-only MOPED uses $`C_{xx}`$ and $`D_x`$. The joint likelihood also
knows $`C_{xz}`$, so a xi combination with no direct parameter response
can still be useful: it traces noise shared with Map3.
For example, let $`D_x=(1,0)^T`$, $`D_z=1`$, $`C_{xx}=I`$,
$`C_{xz}=(0,r)^T`$, and $`C_{zz}=1`$. The second xi row has zero mean
response, yet keeping it increases the amplitude information from 2 to
$`2+r^2/(1-r^2)`$. Discarding it loses that noise subtraction.

The `conditional_xi` option constructs the conditional covariance and
response:

```math
S=C_{xx}-C_{xz}C_{zz}^{-1}C_{zx},\qquad
J=D_x-C_{xz}C_{zz}^{-1}D_z.
```

If $`S=LL^T`$ and the columns of $`Q`$ span $`L^{-1}J`$, the weights
are $`W=Q^TL^{-1}`$. Apply $`T=\operatorname{diag}(W,I)`$ to the
joint vector. The kept Map3 entries and their cross-covariance with
$`Wx`$ recover the local joint information. Neither the data nor the
cross-covariance is dropped. The `xi_moped` option remains available for
comparisons with analyses that deliberately use the ordinary xi basis;
`joint` compresses both probes together.

## Fixed matrices and notebook use <a name="notebook"></a>

[EXAMPLE_COMPRESSION.ipynb](EXAMPLE_COMPRESSION.ipynb) first evaluates the
three-row example, then compares the alternatives using the saved TATT
Fisher derivatives. It treats those saved values as numerical inputs
only, without a claim that they satisfy the physical accuracy
requirements. The notebook saves the input hash, fiducial, all parameter
names, derivative field, training covariance and numerical rank with each
transform, and it refuses an existing file with different contents
instead of silently redefining it.

The NPZ file stores `matrix[noutput,ninput]`, `input_labels[ninput]`,
`output_labels[noutput]` and a JSON `metadata` scalar. It uses no pickle.
`LinearCompression` also accepts an externally built matrix with these
labels and provenance, so a published matrix can enter without
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
`likelihood.mode=gaussian`; the run record keeps these resolved choices.
The inference identity checked before resuming a chain includes the
source matrix's bytes and metadata.

The notebook also prepares a basis using a fixed diagonal teaching
covariance independent of the sample covariance, and the saved file
records its arbitrary noise scales. This is a statistical demonstration,
not an analytical DES covariance calibration. A basis trained on the
production sample covariance does not have this independence property.

## The finite-simulation prescription <a name="finite"></a>

`FiniteSimulationLikelihood` implements the prescription of
[Percival et al., equations 52 and 54–55](https://arxiv.org/html/2108.10402v2#S7).
Let $`N_s`$ count independent Gaussian mock vectors, $`N_d`$ count
kept entries after selection and compression, and $`N_p`$ count
varied mean parameters, including nuisance parameters. The input $`S`$
is the ordinary unbiased sample covariance, and the likelihood applies
no Hartlap rescaling. The intermediate factor $`B`$ and the exponent
$`m`$ are

```math
B=\frac{N_s-N_d-2}{(N_s-N_d-1)(N_s-N_d-4)},\qquad
m=N_p+2+\frac{N_s-1+B(N_d-N_p)}{1+B(N_d-N_p)},
```

```math
\log L=\log\mathcal N-\frac m2
\log\left[1+\frac{(d-\mu)^TS^{-1}(d-\mu)}{N_s-1}\right].
```

Gaussian mock vectors and independent realizations give the Wishart
sampling law assumed for $`S`$. The observed datum must be independent
of that estimate, and a fixed compression matrix must be independent of
the production sample covariance. The API requires the caller to state
these assumptions explicitly and records them; it cannot verify them
without the underlying ensemble. The likelihood refuses analytical,
hybrid and PME covariance estimates as raw sample inputs.

The code requires $`N_s>N_d+4`$, $`1\leq N_p\leq N_d`$ and
$`m>N_d`$. The last condition is necessary for a normalized data density.
For example, $`N_s=796,N_d=480,N_p=20`$ gives $`m\simeq344.5`$,
which fails that condition even though the first inequality holds. The
implementation refuses that request instead of silently compressing the
data; the full-vector Gaussian likelihood remains available as a separate
explicit statistical choice.

[EXAMPLE_NAUTILUS4.yaml](EXAMPLE_NAUTILUS4.yaml) uses the notebook's
independent pilot matrix and an explicit finite-simulation model. The
[inference guide](INFERENCE.md) gives the inspection, physical check and
sampling workflow; this YAML has no wrapper script, so each command there
takes `projects/des_3pcf/EXAMPLE_NAUTILUS4.yaml` as the argument of
`scripts/run_nautilus.py`. Run it after the notebook has written its
matrices.

The example's 796-footprint independence and Gaussian-mock statements are
working assumptions; footprints that share a full-sky realization need an
independence assessment. Informative priors and nonlinear posteriors also
limit the frequentist coverage argument. Recomputing $`m`$ after
compression defines that compressed analysis rather than the marginal of
the original full-data t density.

## Validation and reproduction limits <a name="limits"></a>

The tests compare conditional compression with direct joint Fisher solves,
including unequal units, reordered rows, dependent derivative columns,
nonzero measured data, caller mutation and saved-file restoration. They
compare the normalized likelihood with SciPy's multivariate-t density,
an independently specified exponent, and its large-mock Gaussian limit.
The code refuses a nonintegrable density and incompatible covariance
provenance.

For the archived 480-row TATT arrays, conditional xi compression keeps
100 rows and agrees with the full local Fisher matrix to a maximum
normalized element difference of $`1.6\times10^{-15}`$. Full joint
compression keeps 20 modes. These are algebra checks of that archive;
they are not evidence of posterior reproduction or numerical convergence
of the physical forecast.

Published reproduction additionally needs the exact cuts, source
treatment, priors, compression matrix and estimator conventions. The
[paper-target record](PAPER_TARGETS.md#gaps) lists those gaps, and its
[observed-measurements section](PAPER_TARGETS.md#observed) records the
provenance of the Map3 sign treatment inside the joint covariance.

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
The separate precision-matrix expansion used by the
[Fisher guide](FISHER.md#precision) is
[Friedrich and Eifler](https://arxiv.org/abs/1703.07786), “Precision
matrix expansion — efficient use of numerical simulations in estimating
errors on cosmological parameters”. It expands the precision matrix
about an analytical model; it is not the raw-sample likelihood above.
