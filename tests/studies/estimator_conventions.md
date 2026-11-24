# Matching the aperture estimator and covariance conventions

Three questions have different answers: what the publication specifies,
what the pinned theory code computes, and what the downloaded measurement
files can recover. This page records each answer from the pinned upstream
sources; it evaluates no physical model.

1. [What the publication specifies](#publication)
2. [What the pinned theory code computes](#pinned)
3. [What the measurement files contain](#measurements)
4. [Checking agreement with the published estimator](#checks)
5. [What the saved IA demonstration supports](#notebook)
6. [The joint covariance's Map3 sign](#sign)

## What the publication specifies <a name="publication"></a>

The methods paper specifies an original 20-bin logarithmic side grid from
0.5 to 80 arcmin, an opening-angle multipole limit of 100, conversion to
63 SAS angular bins, and a later cut near 8 arcmin on both side
parameters. Its wording gives an angular interval from 0.016 to 3.126
radians.
[Gomes et al., 2503.03964v2, Section III](https://arxiv.org/html/2503.03964v2#S3)

The corresponding radial construction is

```math
\Delta=\frac{\ln(80/0.5)}{20},\qquad
t_j=0.5\,e^{j\Delta},\quad j=0,\ldots,20,
```

where the final value is the upper edge and the preceding values are
lower edges. A low-side cut selects a subset of this bank; it does not
build ten new logarithmic bins beginning at exactly 8 arcmin. The
likelihood integrates its Map3 prediction over the measured bank:
`likelihood/joint_shear.yaml` declares the 20 bins from 0.5 to 80 arcmin
(`theta_min_arcmin`, `theta_max_arcmin`, `ntheta`) and the cut
`theta_lower_cut_arcmin: 8`, which keeps the 9 bins with lower edges from
8.151 to 62.070 arcmin at log width 0.25376, matching the pinned
`config/3pt.ini`. The former ten-bin demonstration bank and the retired
TATT configuration that labels it an approximation
(`configs/des_y3_tatt_demo_fisher.yaml`) remain in Git history. The cut's
selection rule for the surviving bins (lower edge, representative
separation, or a rule on the raw triangles) remains a question for the
producer.

One angular detail stays unresolved. Sixty-three equal bins covering the
interval from 0 to $`\pi`$ have centers $`(j+1/2)\pi/63`$, and those are
not the two endpoint numbers printed in the paper. The archived theory
configuration instead activates 100 angular samples between 0.01570796
and 3.12588469, with its former 63-sample line commented out. The two
alternatives keep their own names in any comparison.
[Pinned Y3 configuration, lines 16–19](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/3pt.ini#L16)

## What the pinned theory code computes <a name="pinned"></a>

The Y3 configuration uses 20 theta samples from 0.5 to 62.0703205 arcmin
and `dlnt=0.253758`, alongside `Lmax=50`, `Mmax=100`, `nfft=150` and a
pixel window with `nside=1024`. The shortened final theta value matches a
lower-edge bank whose final upper edge is near 80 arcmin; it is not
evidence for a physical upper cut at 62 arcmin. The interface builds
these samples with `np.logspace` and passes the stated width to fastnc.
[Configuration, lines 6–24](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/3pt.ini#L6),
[interface, lines 101–121](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/fastnc_interface.py#L101)

Before the aperture contraction, the interface replaces each lower edge
by that edge times

```math
f_{\rm Y3}=\frac{2}{3}
\frac{e^{3\Delta}-1}{e^{2\Delta}-1}-0.015226.
```

The code labels the subtraction an empirical CosmoGrid correction. The
first term is the mean separation for the area measure $`t\,dt`$ within a
logarithmic interval; the last term is neither that analytic result nor
the geometric midpoint, so it stays an explicit archived convention.
[Y3 interface, lines 265–276](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/fastnc_interface.py#L265)

The project's default uses the geometric midpoint $`e^{\Delta/2}`$ for
both side arguments of its aperture weights, and its radial transform
separately receives the bin width; replacing only the side-bin edges
therefore does not make the final aperture contraction identical to the
archived Y3 route. The area-weighted and empirical-offset choices, the
original-bank selection and the pinned scalar pixel window are documented
in [the estimator guide](../../README.md#estimator), which `Map3Engine`
implements.

The aperture measure and component conventions are recoverable: the Y3
integrator sends $`-[\Gamma_0,\Gamma_1,\Gamma_3,\Gamma_2]`$, then
contracts the complex kernels with measure proportional to
$`d_2^2\,d_3^2\,\Delta^2\,d\phi/(2\pi)`$, including the reflected
orientation and a final real part divided by four.
[Component conversion, line 237](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/fastnc_interface.py#L237),
[aperture integration, lines 118–172](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/fast_map3.py#L118)

These lines specify a theory quadrature; they do not specify how the
galaxy catalogue weights distribute triangles within each measurement
bin. A Fourier-mode average of the x-projected correlation is likewise
not automatically a catalogue-weighted average after a
triangle-dependent centroid rotation, so the `angular_average=point` and
optional `x_modes` routes keep those names during any comparison. The
geometry construction inspected here lived in the retired
`scripts/run_3pcf.py`, kept in Git history.

## What the measurement files contain <a name="measurements"></a>

The pinned Y6 source specifies 25 logarithmic bins over 1 to 150 arcmin,
100 angle bins centered at $`(j+1/2)\pi/100`$, and representative side
lengths using the analytic mean-separation factor above plus an optional
offset whose default is zero. The module passes lower edges into fastnc,
then passes the representative lengths into Map3Transform. This is an
internally stated theory convention and an independent comparison; it
does not replace the Y3 settings in a Y3 reproduction.
[Y6 binning, lines 19–71](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/desy6_3pcf/binning.py#L19),
[Y6 interface, lines 58–115](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/modules/fastnc_interface.py#L58)

The upstream test comparing Map3Transform to TreeCorr fills synthetic
Gamma arrays and manually fills `meand2`, `meand3` and `meanphi`, so it
checks the aperture operator for those values without validating them
against a catalogue.
[Y6 test, lines 45–64](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/tests/test_map3.py#L45)

The Y6 data guide says the measurement files are external. Its Map3
format stores tomographic labels, aperture radii, values and covariance;
it has no field for the original triangle averages or triangle weights.
[Y6 data guide, lines 3–5 and 28–55](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/data/README.md#L28)

Read-only FITS header inspection shows the same limitation for both
downloaded Y3 Map3 products: their 80-row tables contain only BIN1–BIN3,
THETA1–THETA3 and VALUE, followed by COVMAT. `BIN_TYPE=SSS` describes the
three aperture radii in that packed file; the earlier SAS triangle
binning, raw Gamma, triangle counts, mean sides, mean angles and bin
windows appear in neither product.

The two pinned measurement-packing scripts add a provenance check. The
unblinded script reads an already processed, externally stored 80-entry
Map3 NumPy vector and copies it into the FITS VALUE column; its opening
comment records that a prior version omitted the triangle cut, and the
script neither applies that cut nor defines its bins. The CosmoGrid
script extracts the final 80 entries of an already averaged joint vector
and its covariance block, changing the Map3 sign when writing that older
simulated product. Neither script contains the original triangle
measurements, cut mask or within-bin weights.
[Unblinded packing, lines 4–17 and 25–32](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/script/make_unblind_map3_scalecut3pcf.py#L4),
[CosmoGrid packing, lines 30–47](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/script/make_cosmogrid_meas_dv.py#L30)

The sign conversion belongs to the product packed by that script alone;
the unblinded packing script applies no sign change.

The Y3 measurement used TreeCorr 5.1.1's LogMultipole method on the
20-bin side bank from 0.5 to 80 arcmin, with multipole cutoff 100 (201
multipole values), converted the result to SAS form, and then kept the
9 side bins whose lower edge is at least 8 arcmin; the measurement
scripts are stored at NERSC and are not in the pinned repositories. The
measurement's author confirmed the conventions the pinned sources leave
open (private communication, 2026-11-23):

- The SAS conversion used TreeCorr's `toSAS()` with
  `phi_bin_size = 0.05`, which covers $`[0, \pi]`$ with 63 opening-angle
  bins. The measurement's SAS binning is therefore the 63-bin layout
  with centres $`(j+1/2)\pi/63`$, matching the likelihood's
  `phi_bins: 63`.
- Each side enters at its geometric bin centre, the lower edge times
  $`e^{\Delta/2}`$. At the measured $`\Delta = 0.253758`$ the archived
  empirical factor $`f_{\rm Y3}`$ displayed above is 1.1352797 while
  $`e^{\Delta/2} = 1.1352796`$, so the two rules give essentially the
  same sides. The aperture plan's default side rule is `geometric` (core
  `transform_interface.cpp`) and `likelihood/joint_shear.yaml` does not
  override it.
- The aperture integral is fastnc's, which builds on TreeCorr's
  calculation, summed over both orderings of the two sides,
  $`r_1 \le r_2`$ and $`r_1 \ge r_2`$. The likelihood's transform plan
  integrates over all ordered side pairs: its triangle grid is the full
  product of the kept side bank with itself.
- The theory kept three-point multipoles up to $`n = 100`$, and
  $`n = 50`$ would likely give much the same. The likelihood default
  `map_mmax: 20` is checked against 100 by
  `tests/test_map_multipole_truncation.py`, which evaluates the
  EXAMPLE_EVALUATE1 fiducial at both cutoffs, prints the measured
  differences and bounds the relative Map3 change below $`10^{-8}`$ and
  the covariance-weighted distance $`d^{\mathsf T} C^{-1} d`$ of the
  change $`d`$ below $`10^{-12}`$.
- The NSIDE 1024 HEALPix pixel window entered only the map-based
  simulations; the theory for the catalogue measurement has none, and
  the likelihood applies none. The pixel-window comparison is in
  `EXAMPLE_COMPARE_CODES.ipynb`.
- All released covariances used 796 simulations; the smaller NSIM header
  values are typos recorded in `data/README.md`. The four cutouts of one
  sky are independent; noise realizations of one cutout are not.
- The 400 local vectors of `chains/rafael/datavector_2pt3pt` are stored
  in the released row order and may be the first simulation input of the
  covariance tests.

The emulator training geometry and the original TreeCorr output and
configuration remain the missing sources for the catalogue's within-bin
weighting conventions.

## Checking agreement with the published estimator <a name="checks"></a>

The comparisons build on each other in this order:

1. The source, projection and derivative numerical checks run without
   changing the estimator; they answer whether the stated calculation is
   stable, not whether it is the published estimator.
2. The likelihood's integration uses the original radial bank and its
   kept subset; the former ten-bin approximation remains in Git history
   for a fixed-point comparison with the angle convention and
   representative-side rule unchanged.
3. At fixed edges, the geometric midpoint, the analytic area mean and
   the named Y3 empirical correction are compared; the angular choices
   and pixel smoothing follow separately, so one residual never carries
   several causes.
4. Once the archived measurement and training settings are obtained, one
   identical Gamma array goes through both aperture operators, then a
   physical Gamma prediction with ordered source legs. This separates
   aperture normalization from projection and IA differences.
5. All 80 residuals are recorded with their covariance-weighted size.
   The block $`C_{33}`$ gives a Map3-only comparison; with the xi vector
   held fixed, the conditional covariance is
   $`S=C_{33}-C_{32}C_{22}^{-1}C_{23}`$.

The strongest claim this sequence supports is agreement with a named,
published-geometry theory quadrature. Exact measured-estimator agreement
additionally requires the archived bin definitions, within-bin averages,
catalogue weighting and mask and pixel conventions.

## What the saved IA demonstration supports <a name="notebook"></a>

The [IA companion notebook](../../notes/intrinsic_alignments/EXAMPLE_FISHER_IA.ipynb)
evaluates the numbered Cobaya model directly. Its mean comparisons hold
cosmology, sources, row selection and covariance fixed while changing the
IA parameters. The accompanying [teaching note](../../notes/intrinsic_alignments/intrinsic_alignments.pdf)
explains normalized responses and conditional information. The notebook
has no workflow or historical-result loader; rerun its visible cells to
obtain predictions from the current implementation.

A common no-IA prediction, two NLA counterparts and two TATT sign choices
show the response to the signs discussed in
[the TATT paper, Figure 9 and Section III.2](https://arxiv.org/html/2601.09133v1#S3.SS2).
Current plots divide changes by each row's measurement error. That differs
from normalizing a curve by its maximum amplitude; matching IA signs alone
also does not reproduce the paper's source and estimator assumptions.

Once those means have been evaluated, they support another calculation
without further physical evaluations: the finite conditional separation of the two sign cases.
With $`\Delta\mu`$ the difference between the two cases,

```math
r_{3|2}=\Delta\mu_3-C_{32}C_{22}^{-1}\Delta\mu_2,
\qquad
\Delta\chi^2_{\rm joint}
=\Delta\mu_2^{\mathsf T} C_{22}^{-1}\Delta\mu_2
+r_{3|2}^{\mathsf T} S^{-1}r_{3|2}.
```

This tests every angular bin and source combination, including
$`C_{23}`$, using only saved predictions and linear solves. It answers
whether the visibly different Map3 curves distinguish the two fixed
parameter choices after covariance weighting; it is an expected Gaussian
distance for a fixed model, not a detection significance. The later
Fisher comparison tests local nuisance marginalization separately.

The port can therefore examine the paper's IA-response mechanism and
produce a local forecast with stated assumptions. The fixed covariance
and uncalibrated demonstration response support no claim of full TATT
covariance accuracy or recovery of the published posterior gains. The
[paired response API](../../../../external_modules/code/cosmolike_core/cosmolike/3pcf/README.md#paired-feedback) accepts independent matter power
and bispectrum tables; its synthetic example is not a calibrated
hydrodynamic family.

## The joint covariance's Map3 sign <a name="sign"></a>

An overall Map3 sign change leaves $`C_{33}`$ invariant but changes
$`C_{23}`$ and $`C_{32}`$, so agreement between the Map3-only
covariance block and the joint matrix's lower block cannot settle the joint
sign convention. Under the row transformation
$`T=\mathrm{diag}(I_2,-I_3)`$, the consistent transformation is
$`\mu\to T\mu`$ and $`C\to TCT`$; reversing only the mean or only the
cross block changes the experiment's Gaussian distance.

The pinned notebook
`plot-notebooks/create_fits_files_796_cov.ipynb`, code cell 4, loads
`data/dv/average_map3_796sims/`
`xip_xim_map3_covariance_8ARCMINCUT_24Feb25.npy` and assigns that entire
array to the FITS image without a sign operation. The next line names the
February 25 FITS destination but is commented in the archived
cell, so it is evidence of the intended packing route rather than proof
that this saved execution wrote the file. Cell 33 takes the same NPY's
$`C_{33}`$ block for the blind Map3 base used by the later unblinded
script.
[Pinned notebook, JSON lines 92–95](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/plot-notebooks/create_fits_files_796_cov.ipynb#L92),
[cell 33, lines 521–523](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/plot-notebooks/create_fits_files_796_cov.ipynb#L521)

The inspected likelihood path contains no downstream sign conversion:
ThreePointDataClass reads VALUE directly and returns its stored signal,
the Map3 likelihood compares that signal with its prediction, and the
joint likelihood reads COVMAT directly, selects rows and applies the
stated compression matrix to data, theory and covariance.
[Three-point reader, lines 244–283](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/threepoint.py#L244),
[signal return, lines 516–529](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/threepoint.py#L516),
[Map3 likelihood, lines 18–42](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/map3_likelihood.py#L18),
[joint covariance selection and compression, lines 91–205](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/joint_likelihood.py#L91)

The unblinded configuration pairs the March 10 xi file, the March 11
Map3 file and the February 25 joint covariance, which establishes their
intended use together without reconstructing the sign preprocessing of
the simulations behind the February 24 NPY; the `covariance_796sim.ipynb`
notebook reads and plots that NPY. The current status is unresolved
provenance, not a demonstrated wrong $`C_{23}`$ sign, so the covariance
in use stays unchanged. The decisive next evidence is the producer of the
exact February 24 NPY, showing that xi and Map3 for each realization use
the same final sign convention as the packed data and theory vectors; a
reproducible covariance from those paired vectors would strengthen the
check.
