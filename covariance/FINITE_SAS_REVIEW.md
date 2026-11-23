# Reply: covariance for the finite-SAS aperture estimator

23 November 2026. Project source checked at `73c4337`.

I agree with the refusal in the current covariance driver and with the
proposed scientific direction. The covariance must describe the same
measurement as the mean. Matching aperture radii and source bins is not
enough. The additional explanation about loss of filter factorization is
also correct: this is a new estimator calculation, not a replacement of
one radial filter.

There are two qualifications to the proposed implementation and validation:

- We need the covariance of the finite-SAS measurement **mathematically**,
  but need not construct or store its entire binned three-point covariance.
  We can contract the desired aperture weights during the integration.
- The existing 15,997-map comparison is **not a matching finite-SAS
  covariance measurement**. It contains coarse aperture moments. That
  validation still has to be performed.

## 1. Which terms change?

Every covariance contribution involving a Map3 row must be evaluated with
the finite-SAS estimator. The underlying power spectra, bispectra and
connected correlations can remain the same physical inputs; their
integration against the estimator changes.

One terminology correction matters: `BB` is **not Gaussian covariance**.
It denotes products of two connected three-field correlations, each related
to a bispectrum. It is also unrelated to the shear B mode. The Gaussian
six-field contribution consists of Wick contractions into three two-field
correlations. The current code distinguishes these families in its
[physics guide](../../../external_modules/code/cosmolike_core/cosmolike/3pcf/COVARIANCE.md).

| Contribution | Required change for finite SAS |
|---|---|
| Gaussian Map3–Map3 pairings | Integrate the two selected triangle estimators, including spin phases, bin averaging and noise contractions. |
| Bispectrum products, `BB` | Apply the finite triangle selection to both estimators. |
| Power times connected trispectrum, `PT` | Apply the same selection and keep the appropriate contractions. |
| Xi–Map3 power–bispectrum terms, `PB` | Change the Map3-side estimator; retain the actual xi bin weights. |
| Connected one-halo terms | Recompute the one-halo estimator response for finite triangles, for both the five-field cross term and six-field Map3 term. |
| Halo-abundance/long-mode response | Differentiate the finite-SAS observable under the adopted response model, then form its response covariance. |

The pure Gaussian xi–Map3 cross covariance vanishes for zero-mean Gaussian
fields: it contains an odd number of fields. Nonzero cross covariance comes
from non-Gaussian correlations and their allowed signal/noise contractions.

For example, a one-halo contribution currently evaluates an aperture
observable on a halo using products of independently filtered profiles.
With finite SAS, the same halo must instead be passed through the selected
triangle measurement. Keeping the old one-halo covariance would keep the
old estimator even if the Gaussian part were replaced correctly.

Similarly, for fixed measurement geometry, the long-mode response should be
the aperture projection of the response of the binned natural components.
If selection or normalization itself responds to the long mode, that is an
additional response assumption to specify. Reusing the halo-abundance model
does not make it a complete response model for every physical effect.

The existing xi–xi block is unaffected by changing the Map3 estimator. We
can keep that block fixed as requested. Its own survey-window and weighting
assumptions remain those of the existing xi calculation.

This distinction is consistent with the literature. Linke et al. derive
their covariance from a particular aperture estimator and explicitly
distinguish it from the correlation-function estimator used for survey
data. Their finite-field discussion is especially relevant here.
[Linke et al., 2023](https://arxiv.org/abs/2212.04485)

## 2. What can be shared with TransformPlan?

Share the existing prepared aperture projection and its geometric data.
Do not implement a second copy of the finite-SAS weights in the covariance
code.

Let `g` collect the real and imaginary parts of the binned natural
components in a fixed order. This real representation avoids ambiguity
between complex covariance and complex conjugation. Define the projection
matrix `T` so that the measured aperture vector is

```math
y=Tg.
```

For fixed weights, its covariance is therefore

```math
C_{yy}=T\,C_{gg}\,T^{\mathsf T}.
```

The cross covariance requires a separate five-field calculation:

```math
C_{xy}=C_{xg}\,T^{\mathsf T},
```

where `x` is the binned xi vector. Building `C_gg` alone does not provide
`C_xg`. The other team's expression with `W` is equivalent if its
convention is `T = W^T`.

There are two parts to this sharing:

1. Reuse the final aperture contraction: component ordering, complex
   phases, source permutations, radii and integration weights.
2. Match the measurement of each input correlation bin: the selected
   sides, opening angles, distinct-object rules, triangle weights and
   normalization.

The current mean's preparation includes lower radial edges, logarithmic
widths and angular averaging. We should audit exactly which sides are
restricted in LogSAS; “all three sides between 8 and 80 arcmin” must not
be inferred from a two-side binning specification. The third side follows
from the opening angle and any additional selection actually imposed.

Sharing `TransformPlan` makes the projection consistent by construction.
It does **not** by itself prove that theoretical bin averages equal
TreeCorr's weighted catalogue averages. TreeCorr defines natural components
with explicit shear-projection conventions, and its aperture conversion
uses particular component combinations. Those conventions belong in the
comparison test.
[TreeCorr GGG documentation](https://rmjarvis.github.io/TreeCorr/_build/html/ggg.html)

If weights or normalization are estimated from a changing catalogue, the
fixed-matrix formula is conditional on that geometry. An unconditional
covariance may also need the variation of the conditional mean and
normalization. State whether galaxy positions and weights are held fixed;
do not silently interchange conditional and unconditional estimators.

## 3. The factorization roadblock is real, but a huge matrix is optional

The full-filter expression separates into one Fourier filter per field.
A restriction on triangle sides generally couples those fields and breaks
that separation. There is no universal replacement radial filter that
restores it.

Nevertheless, the desired answer is a covariance of aperture rows, not a
stored covariance of every intermediate triangle bin. In components,

```math
(C_{yy})_{ab}
=\sum_{u,v}T_{au}\,(C_{gg})_{uv}\,T_{bv}.
```

Here `u` and `v` include triangle bins, source combinations and natural
components. We may apply those weights while accumulating each covariance
integral. Equivalently, construct the combined finite-triangle estimator
kernel and integrate the cumulant partitions against that kernel. This is
the same linear contraction, with a different order of computation.

The implementation should therefore:

- Reuse the prepared projection data in compiled code.
- Keep the existing physical spectra, halo profiles and response machinery
  where their assumptions still apply.
- Accumulate the requested Map3–Map3 and xi–Map3 entries directly, or use
  bounded batches of intermediate triangle bins.
- Build an explicit small `C_gg` and `C_xg` only for tests against the direct
  contraction.

This avoids requiring a potentially enormous dense intermediate matrix.
It does not eliminate the new derivation or guarantee a particular runtime.
Any compression or factorization of the new kernels needs an accuracy test.
The covariance calculation must also retain the necessary survey geometry:
a translation-invariant bispectrum covariance projected through aperture
filters is not automatically the covariance of a finite-survey real-space
estimator. Linke et al. explain this finite-field distinction.
[Estimator and finite-field derivation](https://arxiv.org/abs/2212.04485)

## 4. Shape noise needs a catalogue-estimator treatment

The continuous-annulus xi expression cannot simply be reused for a
finite-SAS three-point bin.

For illustration, a weighted correlation bin has the schematic form

```math
\widehat\Gamma_u
=\frac{\sum_{t\in u} w_t\,q_t}{\sum_{t\in u} w_t},
```

where `t` is a triangle, `q_t` is the appropriately projected product of
three shears, and `w_t` is its weight. Even with fixed positions, the
covariance contains a double sum over triangles. Noise survives according
to which galaxy or pixel indices are shared, which source catalogues
overlap, and which shear components are conjugated.

Consequently, a raw triangle count alone is generally insufficient.
Weight-squared sums, permutations and overlaps can matter. Mixed
signal–noise terms need more information than the pure-noise variance of
one bin. TreeCorr's default shot-noise variances should not be treated as
the full cross-bin, cross-component covariance.

There are two defensible levels of implementation:

- **Uniform continuous approximation:** derive expected weighted counts
  and overlap integrals for an explicitly stated density and window.
  Validate this limit on uniform mock catalogues.
- **Catalogue-conditioned calculation:** use the actual geometry,
  weighted normalizations and required overlap information, or estimate
  the noise contribution through independent shape rotations of a fixed
  catalogue. If maps replace galaxies, use the map pixel weights and
  noise convention instead.

Not every non-Gaussian signal term requires a measured triplet count as an
input. The requirement is that its estimator normalization and sampling
approximation match the intended measurement. The noise calculation is
where object sharing makes that distinction especially visible.

## 5. Validation: correct the proposed simulation reference first

The claim that a matching covariance already exists from 15,997 finite-SAS
measurements is incorrect. The current
[comparison note](../notes/covariance_comparison/covariance_comparison.tex)
states that those files were measured as sixteen coarse squared/cubed
aperture moments. It explicitly distinguishes them from published
finite-triangle Map3. The four-bin xi comparison is a different, smaller
measurement exercise.

Moreover, the file inventory contains repeated cuts and realizations under
parent labels. It does not provide 15,997 independent gravitational skies.
The mapping between local labels and the published parent ensemble still
needs confirmation. The methods paper's covariance construction also has
specific pixelization and catalogue-noise choices.
[Gomes et al., covariance construction](https://arxiv.org/html/2503.03964v2#S3)

The existing GPTG full-vector checkpoints are another ensemble. Their
legacy mixed-source rows have the documented coincident-pixel issue; they
must not be used unchanged to certify a new finite-SAS covariance.

I recommend the following validation sequence before an expensive full run:

1. **Projection identity.** Feed synthetic binned natural components into
   the shared mean projection and covariance projection. On a small grid,
   compare explicit matrix contraction with the direct accumulation. Include
   unequal source bins, permutations and real/imaginary components.
2. **Pure-noise catalogues.** Test fixed positions and known shape-noise
   variance, first with uniform sampling and then with a mask. Vary density,
   weights and source overlap. Check distinct-object exclusions explicitly.
3. **Gaussian signal with and without noise.** Use matched band limits and
   pixel windows. This isolates the Wick terms; the zero-mean Gaussian
   xi–Map3 cross covariance must be consistent with zero within ensemble
   uncertainty. Test E and B noise components rather than assuming the
   finite-range statistic has ideal full-filter E/B separation.
4. **Connected-term tests.** Use a controlled halo ensemble or deterministic
   single-halo estimator calculation to check the finite-SAS one-halo
   response. Compare the long-mode response with finite differences of the
   same finite-SAS mean under the specified response prescription.
5. **Range and resolution limits.** Independently refine radial/angular
   binning and widen the triangle range. Compare with full-filter only when
   center windows, catalogue normalization, noise, object exclusions and
   boundary treatment also approach the same estimator. Widening sides
   alone on a masked finite catalogue is not a sufficient identity test.
6. **Matched non-Gaussian ensembles.** Measure the actual chosen estimator
   on GPTG and, separately, Rafael maps. Report parent-aware uncertainty.
   Use supported angular scales for each resolution. Compare the cross
   block, conditional Map3 modes and inference results, not only variances.

Agreement with an ideal Gaussian test checks geometry and contractions.
Agreement with non-Gaussian simulations additionally tests the adopted
physical covariance approximations. Neither test replaces the other.

## 6. What should run before the analytic finite-SAS covariance exists?

Keep the current refusal for requests that would silently combine a
finite-SAS mean with the full-filter analytical covariance. I would not
remove it merely by recording an approximation in a comment.

That does **not** require all Map3 forecasts to stop. A simulation covariance
measured with the same finite-SAS estimator is a valid route, subject to
its finite-simulation uncertainty and physical assumptions. A full-filter
mean together with a full-filter covariance is also a distinct legitimate
idealized forecast, if both are actually implemented and selected together.

A full-filter covariance paired with the finite-SAS mean can be studied as
an approximation only after a matched comparison establishes its impact on
the intended scales and inference. At present that test has not been done.
This study should not be presented as an estimator-matched scientific
forecast. A citation showing that a related aperture covariance worked
well is not a substitute for checking these cuts and weights.

The proposed hybrid is worth testing:

```math
C_{\rm hybrid}=
\begin{pmatrix}
A_{\rm analytic}&\widehat B_{\rm finite\ SAS}\\
\widehat B_{\rm finite\ SAS}^{\mathsf T}&\widehat D_{\rm finite\ SAS}
\end{pmatrix}.
```

However, replacing a sample covariance's two-point block by exact `A`
does not preserve positive definiteness automatically. It requires

```math
\widehat D-\widehat B^{\mathsf T}A^{-1}\widehat B>0.
```

With a large cross block and a limited number of independent parents,
sampling noise can violate this condition. Do not repair it with hidden
eigenvalue clipping. Test a stated constrained estimator, known-`A` control
or shrinkage procedure and account for its uncertainty. The
[two-stage PME study](../notes/covariance_from_simulations/covariance_from_simulations.tex)
derives this problem and possible alternatives. A standard sample-covariance
finite-mock likelihood is not automatically valid for the resulting hybrid.

Finally, distinguish a forecast covariance from a **reference for PME**.
A positive full-filter model could be investigated as an approximate
reference for correcting to the finite-SAS target. That is allowed
mathematically; it does not certify the reference's accuracy or convergence.
It does not remove the need for matched finite-SAS simulation measurements.

## Recommended implementation decision

Keep the refusal, share the existing finite-SAS projection, and first
derive the Gaussian/noise calculation plus explicit small-grid contraction
tests. Design the production integration to contract the desired aperture
rows without storing the entire triangle covariance. Then extend every
non-Gaussian and response family with the same estimator definition.

In parallel, prepare a genuinely matched simulation measurement for the
hybrid and PME studies. Do not describe the existing 15,997-map moment
comparison as that validation, and do not launch the full new measurement
ensemble until the finite-SAS measurement itself passes the small tests.
