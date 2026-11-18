# E and B contributions after finite triangle cuts

The predicted mean follows the paper's all-E TATT bispectrum. That is
sufficient for an ideal third moment of E-mode aperture mass. A finite
side–angle–side (SAS) integral is a different observable: truncating the
aperture weights can remove their exact E/B separation. The omitted
single-B bispectra therefore need a separate response test. No nonzero
correction has been established for the project's particular weights.

This note derives the distinction and specifies that test. It reports
source inspection and analytic reasoning, with no physical calculation
of the omitted contribution. Numerical convergence of the E-only mean
cannot measure an absent physical term.

## Which physical components enter the mean?

E and B describe the two components of a spin-two shape field relative
to its Fourier wavevector. Under a reflection E behaves as a scalar,
whereas B changes sign. The low-level IA routine returns nineteen
components. The density field is denoted by `d`; the E and B fields in
this list are intrinsic galaxy shapes:

| Group | Components | Used in the projected mean? |
|---|---|---|
| Density and E only | `ddE, dEd, Edd, dEE, EEd, EdE, EEE` | Yes, with the corresponding ordered radial kernels |
| One B field | `ddB, dBd, Bdd, dEB, dBE, EBd, BEd, BdE, EdB, EEB, EBE, BEE` | No; available through the low-level API |

The native `ia_prepared` routine evaluates the latter group when requested.
The scalar projected bispectrum instead selects the first seven. The
gravitational and linear-alignment baseline is also all-E. Inspect
[`physics_3pcf.c`](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/physics_3pcf.c)
and its [component declarations](https://github.com/SBU-COSMOLIKE/cocoa-cosmolike-core/blob/lowrank_fftlog/cosmolike/3pcf/physics_3pcf.h).
The pinned upstream [TATT projection](https://github.com/git-sunao/fastnc/blob/7b2336ce37a909bed51f07f32a2a4f096e71b6ce/fastnc/bispectrum.py#L1324)+makes the same selection.

Complex natural shear components do not change that statement. Rotating
an E-only shear into triangle-dependent coordinate frames already
produces complex correlations. Their imaginary parts do not supply the
omitted B-field bispectra. The paper explicitly restricts its numerical
three-point transformation to E modes after deriving the more general
IA expressions. [Gomes et al., Section II.6](https://arxiv.org/html/2601.09133v1#S2.SS6)

## Why single-B and double-B terms have different orders

Expand the fields in powers of the initial linear density:

```math
E=E^{(1)}+E^{(2)}+\cdots,\qquad
B=B^{(2)}+B^{(3)}+\cdots.
```

The superscript counts powers of that density. A linear tidal response
has no B component in the wavevector frame, while quadratic shape
operators can generate one. For a Gaussian initial density, an odd
number of linear fields has zero expectation. Every surviving pair
contributes one linear power spectrum, denoted by $`P_L`$.

| Correlation | Leading contraction | Formal order |
|---|---|---|
| EEE bispectrum | $`\langle E^{(1)}E^{(1)}E^{(2)}\rangle`$ and permutations | $`P_L^2`$ |
| EEB bispectrum | $`\langle E^{(1)}E^{(1)}B^{(2)}\rangle`$ | $`P_L^2`$ |
| EBB bispectrum | The five-field $`\langle E^{(1)}B^{(2)}B^{(2)}\rangle`$ vanishes; six-field contractions can survive | $`P_L^3`$ |
| BB power spectrum | $`\langle B^{(2)}B^{(2)}\rangle`$ | $`P_L^2`$ |

Thus the two-point TATT BB spectrum is not evidence that a double-B
bispectrum was omitted at the retained tree order. The immediate
finite-estimator question concerns the single-B terms, which have the
same formal order as the retained EEE terms. Double-B bispectra belong
to a higher-order assessment that must also consider EEE loop terms.

The nonlinear power replacement selected by `power_mode: 2` changes
the scale dependence of the retained expressions. It does not create
the missing contractions with two quadratic B legs. These counting
statements concern the perturbative operators underlying that declared
nonlinear prescription.

## What an exact aperture removes

An unrestricted compensated E aperture has the Fourier representation

```math
M_E(R)=\int\frac{d^2\ell}{(2\pi)^2}
       \widetilde U_R(\ell)E(\boldsymbol\ell).
```

Here $`R`$ is the angular aperture radius and $`\widetilde U_R`$ is its
circular filter. The field-level relation contains only E. Multiplying
three such relations shows that the ideal E-aperture third moment needs
only the EEE bispectrum, even when the physical shear contains B modes.
This property is stronger than cancellation of an ensemble average by
parity symmetry.

The finite calculation instead contracts natural components with weights
only over retained triangles. Schematically it evaluates

```math
X_S=\frac14\operatorname{Re}\sum_{\mu=0}^{3}
     \int d\Pi\,S(\triangle)T_\mu(\triangle)
                         \Gamma_\mu(\triangle).
```

The index $`\mu`$ labels four complex natural shear components;
$`T_\mu`$ are aperture weights and $`d\Pi`$ is the triangle integration
measure. The selection $`S`$ specifies finite side support and sampling.
The C weights already contain the factor one quarter, the quadrature
measure and the mirrored-radius prescription; none is applied twice.
Taking the real part does not establish exact E/B separation.

The methods analysis cuts measured triangle sides near 8 arcmin before
reconstructing the aperture statistic. That operation differs from
discarding a small aperture radius. The project uses a declared 8–80
arcmin theory quadrature; its remaining measurement differences are
listed in [the estimator study](estimator_conventions.md).
[Gomes et al., methods paper, Section III](https://arxiv.org/html/2503.03964v2#S3)

Exact finite-interval E/B separation requires weights constructed for
that purpose. A truncation of the Gaussian-aperture weights does not
automatically satisfy those conditions.
[Krause, Schneider and Eifler (2012)](https://academic.oup.com/mnras/article/423/4/3011/987345)

## Why parity alone does not settle the finite response

A parity-invariant ensemble permits a single-B bispectrum that is odd
under reflection of an oriented Fourier triangle:

```math
B_{EEB}(\mathcal R\boldsymbol\ell_1,
        \mathcal R\boldsymbol\ell_2,
        \mathcal R\boldsymbol\ell_3)
=-B_{EEB}(\boldsymbol\ell_1,\boldsymbol\ell_2,\boldsymbol\ell_3).
```

The symbol $`\mathcal R`$ denotes a reflection. Oddness does not require
the bispectrum to vanish at every orientation. For example, the inspected
quadratic single-B vertex contains a sum of terms proportional to
$`\hat k_x\hat k_y`$, which changes sign under reflection of one axis.
Integrating that bispectrum against an even scalar aperture filter
cancels the two orientations. This explains the parity null of the
coincident circular-aperture EEB moment.

This distinction also appears explicitly in Bakx et al.: their Appendix D
corrects an odd-B null argument by retaining the reflected triangle,
then explains why additional angular averaging can still give zero.
That result concerns oriented bispectra and is not a calculation of the
project's finite-aperture response.
[Bakx et al., Appendix D](https://arxiv.org/html/2504.10009v2#A4)

Let $`K^S_{EEB}`$ instead denote the response of the finite, real statistic
to an oriented EEB bispectrum. A parity-even statistic can have a
reflection-odd response kernel multiplying a reflection-odd bispectrum:

```math
\Delta X_S^{EEB}=\int_{\triangle_\ell}d\Pi_\ell\,
                  K^S_{EEB}(\triangle_\ell)
                  B_{EEB}(\triangle_\ell).
```

Their product is even, so parity supplies no general zero theorem. The
ideal aperture has $`K_{EEB}=0`$ by its field-level projection identity.
A cut on galaxy separations couples angular integrals that were
independent in the aperture construction; that identity need not survive.
This is an inference about the missing test, not a measured correction.
Special equal-radius combinations or permutation symmetries can still
cancel a particular response and must be included explicitly.

## A test that distinguishes the possibilities

1. Construct one retained-order single-B template and its ordered permutations.
2. Transform it with an independently checked parity-odd angular calculation.
3. Verify its zero response in the ideal E-aperture limit.
4. Apply the exact finite-SAS weights used by the forecast.
5. Refine angular integration separately from widening the triangle domain.

The first template can use the available density–density–B components.
Unequal source bins and scalene triangles help expose a response that
symmetry could conceal. The transformation must preserve orientation,
spin phases and all required source/field placements. Inserting the
template into the existing parity-even Legendre route is not this test.

The implemented mirror prescription is a sum of the weights with aperture
radii two and three exchanged. For equal radii it doubles the same weight;
it is not an antisymmetric subtraction. This observation does not prove
nonzero leakage, since the integrated response could still cancel. It
does show why the mirror sum alone cannot be cited as the required null
test. Likewise, exchange of identical E legs constrains the single-B
bispectrum and its response together, rather than forcing each to zero.

One proposed independent construction pairs Fourier triads with their
reflections while reversing B signs and transporting the spin phases.
That construction still requires derivation and validation before it
can serve as a reference. Rotating an E-only shear field by 45 degrees
does not by itself reproduce the physical TATT joint field statistics.
A separate controlled EBB template could test a higher-order estimator
response, without claiming that its amplitude is predicted by this model.

For a measured aperture correction, embed the residual in the joint
vector as $`\Delta d=(0_\xi,\Delta d_{\rm aperture})`$ and compute

```math
\Delta\chi^2=\Delta d^{\mathsf T}C^{-1}\Delta d
             =\lVert L^{-1}\Delta d\rVert^2,\qquad C=LL^{\mathsf T}.
```

Here $`C`$ is the full fixed joint covariance and $`L`$ its Cholesky
factor. Keeping the cross block matters even though xi is unchanged.
This distance measures the omitted mean in data space; parameter bias
and posterior coverage require additional calculations. A measured null
should state its precision and tested configurations.

The present claim is a **finite-SAS aperture prediction from the paper's
E-only TATT bispectrum, with unquantified leakage from omitted B
bispectra**. Matching that approximation is not an implementation error.
The absent finite-response test prevents treating it as a verified
prediction of every TATT contribution to the cut shear statistic.
