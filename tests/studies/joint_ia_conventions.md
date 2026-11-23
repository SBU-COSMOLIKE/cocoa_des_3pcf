# Joint intrinsic-alignment coefficients and their numerical reconstruction

The Python and C sources use consistent physical coefficients for `a1`,
`a2`, `eta1`, `eta2` and `bta`. The adapter accounts for the
normalization and pivot differences, and the projection accounts for the
stored signs and factor of five, so the two routes assign the same
physical meaning to their common IA parameters. This conclusion concerns
the coefficient definitions; the numerical source kernels, growth
interpolation and nonlinear IA approximations differ between the two
calculations and keep their own convergence and model checks.

This page records a source inspection; it ran no build, timing or
physical evaluation. File names below belong to the project's retired
`scripts/des3pcf/` package (removed with the private prediction route
and kept in Git history), to `tests/`, and to the CosmoLike core's
`cosmolike/` sources, including its `3pcf/` subdirectory. The C route is
current; the Python route inspected here is preserved evidence only.

1. [The physical coefficients and the adapter](#adapter)
2. [The density coefficient in the tensor basis](#basis)
3. [The growth convention](#growth)
4. [Source normalization and perturbative prescriptions](#kernels)
5. [Remaining native cross-checks](#remaining)

## The physical coefficients and the adapter <a name="adapter"></a>

Let $`D(z)`$ be the input linear growth normalized to $`D(0)=1`$, and let
$`c_\rho=\bar C_1\rho_{\rm crit}`$. The three-point calculation uses

```math
C_1(z)=-a_1 c_\rho\frac{\Omega_m}{D(z)}
       \left(\frac{1+z}{1+z_p}\right)^{\eta_1},
\qquad
C_2(z)=5a_2 c_\rho\frac{\Omega_m}{D(z)^2}
       \left(\frac{1+z}{1+z_p}\right)^{\eta_2},
\qquad
C_{1\delta}(z)=b_{\rm TA}C_1(z).
```

These are the executable expressions of the IA coefficient constructor in
`physics_3pcf.c`. The pipeline assembles the common parameter dictionary
once in `pipeline.py` and sends the same dictionary to the aperture
calculation and to `CoreTwoPoint.spectra`. The C dictionary parser in
`physics_interface.cpp` assigns `eta2` independently of `a2`; it does not
reproduce the separate upstream integrator's documented assignment of its
second redshift slope from the second amplitude.

The ordinary core fixes its normalization to 0.01389 and its pivot to
$`1+z_p=1.62`$ whenever `set_nuisance_IA` installs the redshift-law model
(`generic_interface.cpp`). The project converts the incoming amplitudes:

```math
A_i^{\rm core}=a_i\frac{c_\rho}{0.01389}
  \left(\frac{1.62}{1+z_p}\right)^{\eta_i},
\quad i=1,2.
```

Substitution into the core's redshift law cancels both normalization and
pivot differences; `core_two_point.py` implements this expression and
leaves `bta` unchanged, as a dimensionless multiplier of the physical
$`C_1`$ requires. The current YAML supplies $`c_\rho=0.0134`$, so this
conversion is necessary: passing the amplitudes unmodified to the
ordinary core would change both physical alignment coefficients.

The ordinary core's `IA_A1_Z1Z2` returns the *positive* coefficient
$`-C_1`$, and `IA_A2_Z1Z2` returns $`C_2/5`$ (`IA.c`), so their names
cannot be compared directly with the three-point coefficient names. The
shear integrand in `cosmo2D.c` subtracts the first coefficient and
multiplies the second by five in terms linear in $`C_2`$, or by 25 in its
square. For example, its single-intrinsic-shape contribution contains

```math
-W_s W_\kappa
 \{(-C_1)P+(-C_1)b_{\rm TA}P_{\delta,\delta s}
                 -C_2P_{\delta,s^2}\}
 =W_s W_\kappa
 \{C_1P+C_{1\delta}P_{\delta,\delta s}+C_2P_{\delta,s^2}\}.
```

The three-point NLA kernel instead writes the signed coefficient directly
as $`g+C_1W_s`$ in `physics_3pcf.c`. These expressions give the same
physical sign convention, and the ordinary EE and BB mixed terms carry
the corresponding sign of $`C_{1\delta}C_2`$: the minus sign in front of
the stored positive core coefficient is required rather than a
discrepancy. Both models use one power of $`\Omega_m`$ in $`C_2`$; a
different convention with $`\Omega_m^2/\Omega_{m,\rm fid}`$ appears in
core comments but is not the executable formula in either route.

The array test in the retired `test_pipeline.py` (kept in Git history)
checked the coefficient conversion at several redshifts with a nondefault
pivot and unequal redshift slopes, so it tested the algebra of the
retired adapter. The native restoration tests in
`test_core_two_point_sequences.py` check repeated ordinary-core state
changes. Whether the two native projection routines reconstruct identical
functions between input nodes is a separate question, treated below.

## The density coefficient in the tensor basis <a name="basis"></a>

The three-point tensor calculation stores

```math
c_{g,1}=2C_1,\qquad c_{g,2,2}=C_2,\qquad
c_{g,2,3}=C_{1\delta}-\frac23 C_2.
```

The last expression, in `physics_3pcf.c`, is a change of operator basis,
not an extra shift of the physical density-weighting parameter. Writing a
tidal Hessian as $`K_{ij}=s_{ij}+\delta_{ij}\delta/3`$ shows why: the
trace-free part of $`K^2`$ contains the trace-free part of $`s^2`$ plus
$`(2/3)\delta s`$, and that coefficient must be removed from the
separately stored density operator to recover
$`C_2(s^2)^{\rm TF}+C_{1\delta}\delta s`$.

An independent check applies to the transverse wavevectors of Limber
projection. In the observed two-dimensional plane the linear trace-free
tidal matrix has the form $`I/6+Q/2`$, where

```math
Q(\varphi)=\begin{pmatrix}
\cos 2\varphi & \sin 2\varphi\\
\sin 2\varphi & -\cos 2\varphi
\end{pmatrix}
```

and $`\varphi`$ is the direction of the transverse wavevector. The
doubled angle is the meaning of spin two: an ellipse's orientation is
unchanged by a half turn. The identity part has no shear orientation,
while $`Q`$ carries the trace-free shape. The spin-two part of the
symmetrized product of two such matrices is $`(Q_1+Q_2)/12`$, whereas the
density-weighted linear operator has spin-two part $`(Q_1+Q_2)/4`$; their
ratio is $`1/3`$. The quadratic three-point E vertex therefore depends on
$`C_{1\delta}+C_2/3`$ at a fixed redshift. The implemented helicity
conversion in `physics_3pcf.c` has this property, and its pure linear
pieces enter as $`C_1 B`$, $`C_1^2B`$ and $`C_1^3B`$. The internal factor
two multiplying $`C_1`$ is removed by the projection coefficients; it is
not an extra observable amplitude.

This transverse identity does not make the two physical parameters
identical in the joint likelihood: their imposed growth and redshift laws
differ, and the ordinary two-point loops integrate three-dimensional
internal wavevectors, for which the restricted transverse identity does
not hold. Stronger separation of `a2` and `bta` in a joint forecast can
therefore be physical.

## The growth convention <a name="growth"></a>

The table producer in `inputs.py` defines
$`D(z)=\sigma_{8,\rm linear}(z)/\sigma_{8,\rm linear}(0)`$, and the
three-point constructor requires its first value to be one. The ordinary
bridge in `core_two_point.py` supplies $`G(z)=(1+z)D(z)`$ together with
the same redshift axis, and the core (`cosmo3D.c`) returns
$`D(z)=aG(z)/G(0)`$. No extra scale-factor or normalization factor is
missing at the input nodes.

A real finite-grid difference remains. The ordinary core interpolates
$`G`$ linearly on the input redshift axis, while the three-point
preparation constructs a cubic interpolant of $`D`$, samples it onto a
denser uniform grid, and then uses linear reads (`physics_3pcf.c`). In
general,

```math
\frac{(1-t)(1+z_j)D_j+t(1+z_{j+1})D_{j+1}}{1+z}
\ne D_{\rm cubic}(z),
\qquad z=(1-t)z_j+t z_{j+1}.
```

This is a numerical difference, not a different intended definition of
`a1` or `a2`. A fractional growth error changes the two coefficients by
approximately $`-\delta D/D`$ and $`-2\delta D/D`$; this inspection
measured no magnitude. A controlled check compares the reconstructed
growth and IA coefficients between nodes, then refines the input redshift
grid while holding the physical tables and parameter definitions fixed.
Increasing the three-point line-of-sight count does not refine the
ordinary core's input growth axis.

With massive neutrinos this common $`D`$ is an effective
scale-independent growth convention. Sharing it between the observables
establishes consistency of that convention, not an exact treatment of
scale-dependent IA growth.

## Source normalization and perturbative prescriptions <a name="kernels"></a>

The three-point lensing efficiency normalizes the input density by
trapezoidal integration in redshift. Its NLA source kernel instead
normalizes sampled $`n(z)\,dz/d\chi`$ on the lensing-kernel distance
grid, and the quadratic IA source kernel normalizes that quantity on the
independent source distance grid (`physics_3pcf.c`). These converge to
the same continuum normalization for consistent distances and Jacobians,
but their finite-grid errors differ. The ordinary core reconstructs its
source distribution separately and uses $`W_s=n_i(z)\,dz/d\chi`$
(`radial_weights.c`), so identical input source samples do not prove
equality of the reconstructed IA weights. The measured-edge source,
kernel and line-of-sight refinement studies address this difference.

The two nonlinear prescriptions also differ by design. The ordinary core
constructs its ten TATT loop spectra from $`P_{\rm linear}(k,0)`$
(`pt_cfastpt.c`) and scales them by $`D^4`$ (`cosmo2D.c`); the bridge
supplies a full redshift-dependent linear table because that is the
ordinary-core interface, and the `spectra` docstring in
`core_two_point.py` states that the loops read the table's zero-redshift
slice and evolve it with $`D^4`$. The three-point `power_mode=2`
evaluates its quadratic tree vertices with the input nonlinear power at
the queried redshift, and the common NLA pieces use their stated
nonlinear power or bispectrum model. The two calculations share operator
coefficients while using different perturbative approximations for their
correlation functions.

The project rejects nonzero velocity-shear `ct` and bispectrum-only
feedback `fb` when requesting ordinary two-point TATT
(`core_two_point.py`), and it keeps both ordinary-core E and B spectra
when forming $`\xi_\pm`$ (`mean.py`). These behaviors prevent two
apparent joint-parameter matches that would otherwise omit an entire
physical contribution on the two-point side.

## Remaining native cross-checks <a name="remaining"></a>

A finite-grid native cross-check remains open: set `a2=bta=ct=0`, supply
one common positive source model and power table, and compare the two NLA
angular-power routes after resolving their known Limber-wavenumber choice
and interpolation; then vary `a1`, `eta1`, pivot and normalization.
Separately, the physical quadratic operator amplitudes can be checked
against three-dimensional tensor formulas, including mixed-sign
amplitudes. These checks would test the bridge inside the actual native
calculations, beyond its existing coefficient algebra and
state-restoration tests.
