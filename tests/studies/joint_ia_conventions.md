# Joint intrinsic-alignment coefficients and their numerical reconstruction

The Python and C sources use consistent physical coefficients for
`a1`, `a2`, `eta1`, `eta2`, and `bta`. The adapter accounts for the
normalization and pivot differences; the projection accounts for the
stored signs and factor of five. The two routes assign the same physical
meaning to their common IA parameters. This conclusion concerns the coefficient definitions. The
numerical source kernels and growth interpolation differ between the two
calculations, and their nonlinear IA approximations remain different.
Those differences need their own convergence and model checks.

This was a source review. It included no physical evaluation, build,
timing, or repository modification. The references below are local files
relative to the Cocoa root, with line numbers from the inspected files.
`P` denotes `projects/des_3pcf`; `C` denotes
`external_modules/code/cosmolike_core/cosmolike`.

## 1. The physical coefficients and the adapter

Let $D(z)$ be the supplied linear growth normalized to $D(0)=1$, and let
$c_\rho=\bar C_1\rho_{\rm crit}$. The three-point calculation uses

$$
C_1(z)=-a_1 c_\rho\frac{\Omega_m}{D(z)}
       \left(\frac{1+z}{1+z_p}\right)^{\eta_1},
\qquad
C_2(z)=5a_2 c_\rho\frac{\Omega_m}{D(z)^2}
       \left(\frac{1+z}{1+z_p}\right)^{\eta_2},
\qquad
C_{1\delta}(z)=b_{\rm TA}C_1(z).
$$

These are executable expressions in `C/3pcf/physics_3pcf.c:1523`.
The common parameter dictionary is assembled once in
`P/python/des3pcf/pipeline.py:585`; the same dictionary goes to the aperture
calculation and `CoreTwoPoint.spectra` at line 615. The C dictionary parser
assigns `eta2` to `parameters.eta2` independently of `a2`
(`C/3pcf/physics_interface.cpp:164`). It does not reproduce the separate
upstream integrator's documented assignment of its second redshift slope
from the second amplitude.

The ordinary core fixes its normalization to 0.01389 and its pivot to
$1+z_p=1.62$ whenever `set_nuisance_IA` installs the redshift-law model
(`C/generic_interface.cpp:4389,4443`). The project explicitly converts the
incoming amplitudes:

$$
A_i^{\rm core}=a_i\frac{c_\rho}{0.01389}
  \left(\frac{1.62}{1+z_p}\right)^{\eta_i},
\quad i=1,2.
$$

Substitution into the core's redshift law cancels both normalization and
pivot differences. This is the implemented expression in
`P/python/des3pcf/core_two_point.py:63`. The bridge leaves `bta` unchanged,
as required for a dimensionless multiplier of the physical $C_1$.
The current YAML supplies $c_\rho=0.0134$, so this conversion is necessary;
passing its amplitudes unmodified to the ordinary core would change both
physical alignment coefficients.

The ordinary core's `IA_A1_Z1Z2` returns the *positive* coefficient
$-C_1$, and `IA_A2_Z1Z2` returns $C_2/5$
(`C/IA.c:302,349,392,438`). Their names therefore cannot be compared
directly with the three-point coefficient names. The shear integrand
subtracts the first coefficient and multiplies the second by five in
terms linear in $C_2$, or by 25 in its square
(`C/cosmo2D.c:2933`). For example, its single-intrinsic-shape contribution
contains

$$
-W_s W_\kappa
 \{(-C_1)P+(-C_1)b_{\rm TA}P_{\delta,\delta s}
                 -C_2P_{\delta,s^2}\}
 =W_s W_\kappa
 \{C_1P+C_{1\delta}P_{\delta,\delta s}+C_2P_{\delta,s^2}\}.
$$

The three-point NLA kernel instead writes the signed coefficient directly
as $g+C_1W_s$ (`C/3pcf/physics_3pcf.c:1836`). These expressions give the same
physical sign convention. The ordinary EE and BB mixed terms also contain
the corresponding sign of $C_{1\delta}C_2$; the minus sign in front of the
stored positive core coefficient is required, rather than a discrepancy.
Both models use one power of $\Omega_m$ in $C_2$. A different convention
with $\Omega_m^2/\Omega_{m,\rm fid}$ is mentioned in the core comments but
is not the executable formula in either route.

The existing array test
`P/tests/test_pipeline.py:127` checks the coefficient conversion at several
redshifts with a nondefault pivot and unequal redshift slopes. It tests the
algebra of the adapter. The native restoration tests in
`P/tests/test_core_two_point_sequences.py` check repeated ordinary-core
state changes. Neither test, by itself, establishes that the two native
projection routines reconstruct identical functions between input nodes.

## 2. The density coefficient in the three-point tensor basis is not bta

The three-point tensor calculation stores

$$
c_{g,1}=2C_1,\qquad c_{g,2,2}=C_2,\qquad
c_{g,2,3}=C_{1\delta}-\frac23 C_2.
$$

The last expression at `C/3pcf/physics_3pcf.c:1542` is a change of operator
basis. It is not an extra shift of the physical density-weighting
parameter. Writing a tidal Hessian as $K_{ij}=s_{ij}+\delta_{ij}\delta/3$
shows why: the trace-free part of $K^2$ contains the trace-free part of
$s^2$ plus $(2/3)\delta s$. Its coefficient must be removed from the
separately stored density operator to recover
$C_2(s^2)^{\rm TF}+C_{1\delta}\delta s$.

There is a useful independent check specific to the transverse wavevectors
used by Limber projection. In the observed two-dimensional plane the
linear trace-free tidal matrix has the form $I/6+Q/2$, where

$$
Q(\varphi)=\begin{pmatrix}
\cos 2\varphi & \sin 2\varphi\\
\sin 2\varphi & -\cos 2\varphi
\end{pmatrix}.
$$

Here $\varphi$ is the direction of the transverse wavevector. The doubled
angle is the meaning of spin two: an ellipse's orientation is unchanged by
a half turn. The identity part has no shear orientation, while $Q$ carries
the trace-free shape. The spin-two part of the symmetrized product of two such
matrices is $(Q_1+Q_2)/12$, whereas the density-weighted linear operator has
spin-two part $(Q_1+Q_2)/4$. Their ratio is $1/3$. Thus the quadratic
three-point E vertex depends on $C_{1\delta}+C_2/3$ at a fixed redshift.
The implemented helicity conversion (`C/3pcf/physics_3pcf.c:1698`) has this
property, and its pure linear pieces are added as $C_1 B$, $C_1^2B$, and
$C_1^3B$ at line 1733. The internal factor two multiplying $C_1$ is removed
by the projection coefficients; it is not an extra observable amplitude.

This transverse identity does not make the two physical parameters
identical in the joint likelihood. Their imposed growth and redshift laws
differ. The ordinary two-point loops also integrate three-dimensional
internal wavevectors, for which this restricted transverse identity does
not hold. Stronger separation of `a2` and `bta` in a joint forecast can
therefore be physical. It should not be explained by claiming independent
triangle shapes for these two tree-level transverse vertices at a fixed
redshift.

## 3. The growth convention matches at supplied nodes, but not exactly between them

The table producer defines
$D(z)=\sigma_{8,\rm linear}(z)/\sigma_{8,\rm linear}(0)$
(`P/python/des3pcf/inputs.py:576,586,612`). The three-point constructor
requires its first value to be one
(`C/3pcf/physics_3pcf.c:913`). The ordinary bridge supplies
$G(z)=(1+z)D(z)$ together with the same redshift axis
(`P/python/des3pcf/core_two_point.py:254`). The core returns
$D(z)=aG(z)/G(0)$ (`C/cosmo3D.c:466`). Thus no extra scale-factor or
normalization factor is missing at the input nodes.

There is a real finite-grid difference. The ordinary core interpolates
$G$ linearly on the supplied redshift axis. The three-point preparation
constructs a cubic interpolant of $D$, samples it onto a denser uniform
grid, and then uses linear reads (`C/3pcf/physics_3pcf.c:474`). In general,

$$
\frac{(1-t)(1+z_j)D_j+t(1+z_{j+1})D_{j+1}}{1+z}
\ne D_{\rm cubic}(z),
\qquad z=(1-t)z_j+t z_{j+1}.
$$

This is a numerical difference, not a different intended definition of
`a1` or `a2`. A fractional growth error changes the two coefficients by
approximately $-\delta D/D$ and $-2\delta D/D$. No magnitude was measured
in this review. The ordinary-core setter's own comment explains that its
IA amplitudes inherit the supplied growth-grid error
(`projects/des_y3/interface/interface.cpp:375`). A controlled check should
compare the reconstructed growth and IA coefficients between nodes, then
refine the input redshift grid while holding the physical tables and
parameter definitions fixed. Merely increasing the three-point LOS count
does not refine the ordinary core's supplied growth axis.

With massive neutrinos this common $D$ is an effective scale-independent
growth convention. Sharing it between the observables establishes
consistency of that convention, not an exact treatment of scale-dependent
IA growth.

## 4. Source normalization and perturbative prescriptions need separate interpretation

The three-point lensing efficiency normalizes the input density by
trapezoidal integration in redshift. Its NLA source kernel instead
normalizes sampled $n(z)\,dz/d\chi$ on the lensing-kernel distance grid; the
quadratic IA source kernel normalizes that quantity on the independent
source distance grid (`C/3pcf/physics_3pcf.c:847,861,871`). These converge to
the same continuum normalization for consistent distances and Jacobians,
but their finite-grid errors differ. The ordinary core reconstructs its
source distribution separately and uses
$W_s=n_i(z)\,dz/d\chi$ (`C/radial_weights.c:155`). Identical input source
samples therefore do not prove equality of the reconstructed IA weights.
The measured-edge source, kernel and LOS refinement studies address this
issue; it is not repaired by changing the IA amplitudes.

The two nonlinear prescriptions also differ by design. The ordinary core
constructs its ten TATT loop spectra from $P_{\rm linear}(k,0)$
(`C/pt_cfastpt.c:788,900`) and scales them by $D^4$
(`C/cosmo2D.c:3181,3215`). The three-point `power_mode=2` evaluates its
quadratic tree vertices with the supplied nonlinear power at the queried
redshift. The common NLA pieces use their declared nonlinear power or
bispectrum model. These calculations share operator coefficients while
using different perturbative approximations for their correlation
functions. No recommendation is made to change `power_mode=2` to the
historical alternative.

One documentation correction is warranted:
`P/python/des3pcf/core_two_point.py:188` says the full redshift-dependent
linear power is required “for the ordinary core's loops.” This overstates
what the IA loops use. The bridge supplies a full physical table because
that is the ordinary-core interface, but the inspected IA loop producer
uses its zero-redshift slice and the separate $D^4$ factor. State that
explicitly, especially when discussing massive neutrinos.

A finite-grid native cross-check remains useful: set `a2=bta=ct=0`, supply
one common positive source model and power table, and compare the two
NLA angular-power routes after resolving their known Limber-wavenumber
choice and interpolation. Then vary `a1`, `eta1`, pivot and normalization.
Separately, check the physical quadratic operator amplitudes against
three-dimensional tensor formulas, including mixed-sign amplitudes. This
would test the bridge into the actual native calculations, beyond its
existing coefficient algebra and state-restoration checks. It is a
proposed diagnostic, not evidence of a detected coefficient bug.

The project rejects nonzero velocity-shear `ct` and bispectrum-only
feedback `fb` when requesting ordinary two-point TATT
(`P/python/des3pcf/core_two_point.py:45`). It retains both ordinary-core E
and B spectra (`P/python/des3pcf/mean.py:480`) when forming xi±. These
behaviors prevent two other apparent joint-parameter matches that would
otherwise omit an entire physical contribution on the two-point side.
