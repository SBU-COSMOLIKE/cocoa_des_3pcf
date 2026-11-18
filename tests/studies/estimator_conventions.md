# Matching the aperture estimator and covariance conventions

This review separates three questions: what the publication specifies,
what the pinned theory code computes, and what can be recovered from the
downloaded measurement files. They have different answers. The source inspection below uses pinned upstream files. It does not
evaluate a physical model.

## 1. A published-geometry comparison is possible

The methods paper specifies an original 20-bin logarithmic side grid from
0.5 to 80 arcmin, an opening-angle multipole limit of 100, conversion to
63 SAS angular bins, and a later cut near 8 arcmin on both side parameters.
Its wording gives an angular interval from 0.016 to 3.126 radians.
[Gomes et al., 2503.03964v2, Section III](https://arxiv.org/html/2503.03964v2#S3)

The corresponding radial construction is

$$
\Delta=\frac{\ln(80/0.5)}{20},\qquad
t_j=0.5\exp(j\Delta),\quad j=0,\ldots,20.
$$

Here the final value is the upper edge; the preceding values are lower
edges. Applying a low-side cut should select a subset of this bank. It
should not replace it with ten new logarithmic bins beginning at exactly
8 arcmin. The current demonstration does the latter explicitly in
[the TATT YAML](../../configs/des_y3_tatt_demo_fisher.yaml).
The YAML notes label that choice as an approximation.

This gives a useful next comparison: keep a numerically checked physical
model fixed, replace only the radial bin bank by the original 20-bin
construction and its retained subset, and record the resulting 80-row
change. The rule selecting the surviving bins must be stated: lower edge,
representative separation, or a rule applied to the raw triangles. The
paper's approximate cut alone does not settle that choice.

There is also an unresolved angular detail. Sixty-three equal bins covering
the whole interval [0, pi] have centers (j+1/2)pi/63; those are not the two
endpoint numbers printed in the paper. The archived theory configuration
instead activates 100 angular samples between 0.01570796 and 3.12588469,
with its former 63-sample line commented out. These alternatives must not
be silently identified.
[Pinned Y3 configuration, lines 16–19](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/3pt.ini#L16)

## 2. The pinned theory contains additional, recoverable conventions

The Y3 configuration uses 20 theta samples from 0.5 to 62.0703205 arcmin
and `dlnt=0.253758`, alongside Lmax=50, Mmax=100, nfft=150 and a pixel
window with nside=1024. The shortened final theta value is consistent
with a lower-edge bank whose final upper edge is near 80 arcmin; it is
not evidence for a physical upper cut at 62 arcmin. The interface builds
these samples with `np.logspace` and passes the stated width to fastnc.
[Configuration, lines 6–24](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/config/3pt.ini#L6),
[interface, lines 101–121](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/fastnc_interface.py#L101)

Before the aperture contraction, the interface replaces each lower edge
by that edge times

$$
f_{\rm Y3}=\frac{2}{3}
\frac{e^{3\Delta}-1}{e^{2\Delta}-1}-0.015226.
$$

The code labels the subtraction an empirical CosmoGrid correction. The
first term is the mean separation for the area measure t dt within a
logarithmic interval. The last term is neither that analytic result nor
the geometric midpoint. It must remain an explicit archived convention.
[Y3 interface, lines 265–276](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/fastnc_interface.py#L265)

The default uses the geometric midpoint exp(Delta/2) for both
side arguments of its aperture weights. The explicit area-weighted and
empirical-offset choices, original-bank selection and pinned scalar
pixel window are described in [the estimator guide](../../ESTIMATOR.md). Its radial transform is separately
given the bin width. Therefore replacing only the side-bin edges does not
make the final aperture contraction identical to the archived Y3 route.
[Map3Engine](../../python/des3pcf/mean.py)

The explicit aperture measure and component conventions are recoverable:
the Y3 integrator sends `-[Gamma0,Gamma1,Gamma3,Gamma2]`, then contracts
the complex kernels with measure proportional to d2 squared times d3
squared times Delta squared times dphi/(2 pi), including the reflected
orientation and a final real part divided by four.
[Component conversion, line 237](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/fastnc_interface.py#L237),
[aperture integration, lines 118–172](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/fast_map3.py#L118)

These specify a theory quadrature. They do not specify how the galaxy
catalogue weights distribute triangles within each measurement bin.
Similarly, a Fourier-mode average of the x-projected correlation is not
automatically a catalogue-weighted average after a triangle-dependent
centroid rotation. The current `angular_average=point` and optional
`x_modes` routes should retain those names during the comparison.
[CLI geometry construction](../../scripts/run_3pcf.py)

## 3. Y6 supplies a useful explicit convention, not the missing Y3 weights

The pinned Y6 source specifies 25 logarithmic bins over 1–150 arcmin,
100 angle bins centered at (j+1/2)pi/100, and representative side lengths
using the analytic mean-separation factor above plus an optional offset.
Its default offset is zero. The module passes lower edges into fastnc,
then passes the representative lengths into Map3Transform. This is an
internally stated theory convention and a useful independent comparison.
It must not replace the Y3 settings in a purported Y3 reproduction.
[Y6 binning, lines 19–71](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/desy6_3pcf/binning.py#L19),
[Y6 interface, lines 58–115](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/modules/fastnc_interface.py#L58)

The upstream test comparing Map3Transform to TreeCorr fills synthetic Gamma
arrays and manually fills `meand2`, `meand3`, and `meanphi`. This checks the
aperture operator for the supplied values. It does not validate those values
against an actual catalogue.
[Y6 test, lines 45–64](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/tests/test_map3.py#L45)

The Y6 data guide says the measurement files are external. Its Map3 format
stores tomographic labels, aperture radii, values, and covariance; it has
no field for the original triangle averages or triangle weights.
[Y6 data guide, lines 3–5 and 28–55](https://github.com/rchgomes/desy6-3pcf/blob/fd59fe4fee694d359453d5f573d6a4a55a47a812/data/README.md#L28)

Read-only FITS header inspection confirms the same limitation for both
downloaded Y3 Map3 products: their 80-row tables contain only BIN1–BIN3,
THETA1–THETA3, and VALUE, followed by COVMAT. `BIN_TYPE=SSS` describes
the three aperture radii in that packed file. It does not recover the
earlier SAS triangle binning. No raw Gamma, triangle-count, mean-side,
mean-angle, or bin-window table is present in either inspected product.

The two pinned measurement-packing scripts provide a further provenance
check. The unblinded script reads an
already processed, externally stored 80-entry Map3 NumPy vector and copies
it into the FITS VALUE column. Its opening comment records that a prior
version omitted the triangle cut, but the script does not apply that cut
or define its bins. The CosmoGrid script extracts the final 80 entries of
an already averaged joint vector and its covariance block. It changes the
Map3 sign when writing that older simulated product. Neither script
contains the original triangle measurements, cut mask or within-bin weights.
[Unblinded packing, lines 4–17 and 25–32](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/script/make_unblind_map3_scalecut3pcf.py#L4),
[CosmoGrid packing, lines 30–47](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/script/make_cosmogrid_meas_dv.py#L30)

The sign conversion is specific to the product packed by that script; it
is not permission to negate every downloaded Map3 file. The unblinded
packing script applies no such sign change. The external measurement
producer, the emulator training geometry, and the original TreeCorr
output/configuration remain the specific missing sources needed to settle
the retained-bin and weighting conventions.

## 4. Recommended sequence and interpretation

1. Finish the present source/projection/derivative numerical checks without
   changing the estimator. They answer whether the declared calculation is
   stable, not whether it is the published estimator.
2. Compare the original radial bank with the ten-bin approximation at one
   fixed physical point. Keep the angle convention and representative-side
   rule unchanged in this first comparison.
3. At fixed edges, compare geometric midpoint, analytic area mean, and the
   named Y3 empirical correction. Then compare the distinct angular choices
   and pixel smoothing separately. Do not combine changes into one residual
   and attribute it to a single cause.
4. Once the archived measurement/training settings are obtained, compare
   one identical Gamma array through both aperture operators. Then compare
   a physical Gamma prediction, retaining the ordered source legs. This
   separates aperture normalization from projection and IA differences.
5. Record all 80 residuals and their covariance-weighted size. Using C33
   gives a Map3-only comparison. If the xi vector is held fixed and the
   question concerns the joint data, the relevant conditional covariance is
   S=C33-C32 C22^-1 C23. Neither norm establishes agreement with unknown
   catalogue weights.

The strongest claim supported by this sequence is initially agreement with
a named, published-geometry theory quadrature. Exact measured-estimator
agreement additionally requires the archived bin definitions, within-bin
averages, catalogue weighting and mask/pixel conventions. Matching 80
aperture labels is insufficient.

## 5. Consequences for the IA teaching notebook

The notebook calls its full-480-row, fixed-covariance calculation a
demonstration. Its execute-or-reuse method checks configuration, rows,
covariance and the saved physical input/implementation identities. An
explicit historical loader can inspect an older archive without claiming
that it represents the current code.
[Result loading](../../interface/cosmolike_des_3pcf_fisher_wrappers.py)

The `ia_figure9_workflows` helper prepares a common no-IA prediction,
two NLA counterparts and two TATT sign choices. Its grouped plots use
the no-IA case as their reference, as in Figure 9. The displayed residual
is normalized by the maximum absolute reference within each bin, not by
each angular sample. The complete IA settings are returned with the
workflows; matching amplitude choices alone is not an absolute reproduction
of the paper's source and estimator assumptions.
[TATT paper, Figure 9 and Section III.2](https://arxiv.org/html/2601.09133v1#S3.SS2),
[current sign cases](../../interface/cosmolike_des_3pcf_fisher_wrappers.py)

After numerical validation, the highest-value calculation using those
saved vectors is their finite conditional separation. Define Delta mu as
the difference between the two sign cases and compute

$$
r_{3|2}=\Delta\mu_3-C_{32}C_{22}^{-1}\Delta\mu_2,
\qquad
\Delta\chi^2_{\rm joint}
=\Delta\mu_2^T C_{22}^{-1}\Delta\mu_2
+r_{3|2}^T S^{-1}r_{3|2}.
$$

This tests every angular bin and source combination, including C23, using
only saved predictions and linear solves. It answers whether the visibly
different Map3 curves distinguish the two fixed parameter choices after
covariance weighting. It is an expected Gaussian distance for a fixed
model, not a detection significance, a nuisance-profiled likelihood result,
or evidence that a baryonic feedback parameter can be measured. The later
Fisher comparison can separately test local nuisance marginalization.

The port can therefore examine the paper's IA-response mechanism and
produce an additional local forecast with stated assumptions. The fixed covariance and uncalibrated demonstration response do not
support a claim of full TATT covariance accuracy, baryon robustness, a
joint feedback constraint, or recovery of the published posterior gains.
The [paired response API](../../FEEDBACK.md) accepts independent matter
power and bispectrum tables; its synthetic example is not a calibrated
hydrodynamic family.

## 6. The joint covariance's Map3 sign still needs its producer

An overall Map3 sign change leaves C33 invariant but changes C23 and C32.
Consequently, agreement between the selected Map3-only covariance and the
joint matrix's lower block cannot settle the joint sign convention. Under
the row transformation T=diag(I2,-I3), the consistent transformation is
mu -> T mu and C -> T C T. Reversing only the mean or only the cross block
changes the experiment's Gaussian distance.

The pinned notebook
`plot-notebooks/create_fits_files_796_cov.ipynb`, code cell 4, loads
`data/dv/average_map3_796sims/`
`xip_xim_map3_covariance_8ARCMINCUT_24Feb25.npy` and assigns that entire
array to the FITS image without a sign operation. The next line names the
selected February 25 FITS destination, but is commented in the archived
cell; it is evidence of the intended packing route, not proof that this
particular saved notebook execution wrote the file. Cell 33 takes the same
NPY's C33 block for the blind Map3 base used by the later unblinded script.
[Pinned notebook, JSON lines 92–95](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/plot-notebooks/create_fits_files_796_cov.ipynb#L92),
[cell 33, lines 521–523](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/plot-notebooks/create_fits_files_796_cov.ipynb#L521)

There is no downstream sign conversion in the inspected likelihood path:
ThreePointDataClass reads VALUE directly and returns its stored signal;
the Map3 likelihood compares that signal with its supplied prediction;
the joint likelihood reads COVMAT directly, selects rows, and applies
the declared compression matrix to data, theory and covariance.
[Three-point reader, lines 244–283](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/threepoint.py#L244),
[signal return, lines 516–529](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/threepoint.py#L516),
[Map3 likelihood, lines 18–42](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/map3_likelihood.py#L18),
[joint covariance selection and compression, lines 91–205](https://github.com/rchgomes/3pcf_integrator/blob/d6098d326dcb03bc750adcc9627119f52d928944/python/joint_likelihood.py#L91)

The unblinded configuration explicitly pairs the March 10 xi file, March
11 Map3 file and February 25 joint covariance. This establishes their
intended use together. It does not reconstruct the sign preprocessing of
the individual simulations that produced the February 24 NPY. The
`covariance_796sim.ipynb` notebook merely reads and plots that NPY.

Therefore the current status is unresolved provenance, not a demonstrated
wrong C23 sign. Keep the selected matrix unchanged. The decisive next
evidence is the producer of the exact February 24 NPY, showing that xi and
Map3 for each realization use the same final sign convention as the packed
data/theory vectors. A reproducible covariance from those paired vectors
would provide a stronger check. An observed-likelihood response check or a
positive Cholesky factorization cannot settle this convention by itself.
