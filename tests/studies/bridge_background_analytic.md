# The distance derivative at the core-to-3pcf boundary

The core supplies sampled comoving distances. The bridge converts their
finite-difference derivative into $`J(z)=dz/d\chi`$, which converts a source
density per redshift into a density per comoving distance. Using this
derivative is a numerical-method choice. Copying a distance table correctly
does not establish that the derivative is accurate, and reproducing the
same stencil in Python would check agreement between implementations rather
than agreement with the underlying derivative.

The permanent test is
[test_bridge_background_analytic.py](../test_bridge_background_analytic.py).
It uses an independently differentiable distance and observes the actual
bridge input. Production C/C++ source and the installed extension are not
modified by the test.

## An independent derivative

The controlled geometry is

```math
\chi(z)=\frac{Lz}{1+qz},\qquad
\frac{d\chi}{dz}=\frac{L}{(1+qz)^2},\qquad
J(z)=\frac{(1+qz)^2}{L},
```

with $`L=3000\,\mathrm{Mpc}/h`$ and $`q=0.35`$. The parameter $`q`$ describes
the shape of this curve; it is not spatial curvature. This geometry is not
a Friedmann solution for the model's density parameters. It provides a
known mathematical answer for testing a numerical conversion. The separate
$`q=0`$ case gives the constant inverse derivative $`1/L`$ and checks the
absolute unit scale at every node.

A small C function compiled in pytest's temporary directory copies the
public input's redshift, distance and Jacobian arrays. It includes the
actual public header, so the compiler determines the struct layout. Python
does not declare a duplicate struct or calculate memory offsets. The
observer is the first consumer of a fresh bridge input; a repeated refresh
would return no new input. It checks sizes and finite values and immediately
returns NumPy-owned copies. The bridge is then reset before any Python
model factory is called. The observer computes no derivative and replaces
no native function.

The curved-distance check uses 41, 81, 161 and 321 background nodes between
redshifts zero and two. It reports three errors separately: the maximum
interior error, the observer endpoint and the upper endpoint. The current
interior centered derivative has an error proportional to the squared
node spacing. Halving that spacing should therefore reduce its error by
about four. At the observer the bridge uses a forward slope, whose leading
error is proportional to the spacing. The upper endpoint continues the
interior derivative and has a different error from the observer. A single
maximum over all nodes would obscure these differences.

For this rational distance, a centered derivative on a uniform grid can
also be analyzed exactly. If $`h`$ is the redshift spacing,

```math
\frac{\chi(z+h)-\chi(z-h)}{2h}
=\frac{L}{(1+qz)^2-(qh)^2},\qquad
\frac{J_{\rm FD}(z)}{J(z)}-1
=-\left(\frac{qh}{1+qz}\right)^2.
```

This explains the expected rate. The numerical test still compares the
actual C input with the analytic derivative $`J(z)`$; the expression above
is not used to manufacture its reference array.

## Why intrinsic alignments care about this conversion

Conservation of the source count gives

```math
n_\chi(\chi)\,d\chi=n_z(z)\,dz,
\qquad n_\chi(\chi)=n_z(z)J(z).
```

The NLA model's intrinsic-shape kernel contains $`F_I(z)n_\chi(\chi)`$,
where $`F_I`$ is the dimensionless alignment coefficient. For example,

```math
C_\ell^{II}=\int d\chi\,
\frac{[F_I n_\chi]^2}{\chi^2}
P\!\left(\frac{\ell}{\chi},z\right),
```

and the linear-alignment part of the three-intrinsic-shape bispectrum is

```math
B_{III}(\ell_1,\ell_2,\ell_3)=\int d\chi\,
\frac{[F_I n_\chi]^3}{\chi^4}
B_m\!\left(\frac{\ell_1}{\chi},
           \frac{\ell_2}{\chi},
           \frac{\ell_3}{\chi};z\right).
```

Thus a redshift-dependent Jacobian error changes both the weighting and
the normalization of the intrinsic-shape contributions. The implementation
renormalizes its sampled source kernels. That normalization can hide a
constant multiplicative Jacobian error, which is another reason to inspect
the unnormalized bridge input separately.

The observable check compares a bridge model with a direct model given
exact analytic Jacobian samples. Both receive the same sampled distance,
growth, power and source functions. Their interpolation and integration
rules are identical. Source and line-of-sight grids remain fixed while only
the background grid is refined. This comparison isolates the derivative
choice; neither model is a continuum projection reference.

The source is the smooth normalized polynomial
$`n_z(z)=30z^2(z_{\max}-z)^2/z_{\max}^5`$, which vanishes at both endpoints.
The recorded quantities are the two-point II contribution, projected GGI
and III linear-alignment terms, and the quadratic TATT residual. The last
quantity reads the separate source-grid shape kernel and checks another
consumer of the derivative. Pure gravitational shear serves as a control:
its prediction should not change when only the Jacobian changes. For each
quantity, the reported error is the maximum absolute difference divided
by that quantity's maximum reference magnitude, avoiding an unstable ratio
at a single zero crossing.

## Sampling used by the examples

Two additional checks read the background count and upper redshift from
the Y3 example YAML files. These are 101 nodes to redshift three for the
TATT example, and 125 nodes to redshift
3.1 for the BACCO example. Each check doubles the number of background
intervals twice. Its source, kernel and line-of-sight counts remain fixed
at the example's settings; when there is no separate source count, the
initial background count defines a fixed independent source grid.

These are production-like sampling checks on a controlled polynomial
source. They do not use the measured DES source densities, a survey data
vector or its covariance. Their errors cannot be quoted as bounds on DES
IA errors or on a Fisher figure of merit. Those claims require the actual
survey prediction and a fixed full covariance.

## Measured results

The final checks passed all five cases in each of the strict and debug
builds. The existing debug extension checks undefined behavior and floating
division by zero. Each of the 15 input-archive pairs matches byte for byte
between builds, and all saved numerical measurements agree exactly.
[The combined record](results/bridge_background_analytic.json) retains the
measurements, input identities and executed commands.

Errors below are dimensionless relative errors. The IA column is the largest
error among II, GGI, III and the quadratic TATT residual.

| Background nodes, to $`z=2`$ | Interior Jacobian | Observer Jacobian | Upper-end Jacobian | Largest IA error |
|---|---:|---:|---:|---:|
| 41 | $`2.958\times10^{-4}`$ | $`1.750\times10^{-2}`$ | $`5.574\times10^{-4}`$ | $`8.638\times10^{-5}`$ |
| 81 | $`7.524\times10^{-5}`$ | $`8.750\times10^{-3}`$ | $`1.358\times10^{-4}`$ | $`2.153\times10^{-5}`$ |
| 161 | $`1.897\times10^{-5}`$ | $`4.375\times10^{-3}`$ | $`3.353\times10^{-5}`$ | $`5.379\times10^{-6}`$ |
| 321 | $`4.764\times10^{-6}`$ | $`2.188\times10^{-3}`$ | $`8.330\times10^{-6}`$ | $`1.345\times10^{-6}`$ |

The observer has zero source density in this test, so its relatively large
local derivative error receives no direct source weight. Nearby samples
still affect interpolation and normalization. The projection checks measure
those combined effects. Pure gravitational shear was unchanged in both
builds. With a linear distance, the largest IA discrepancy was
$`1.24\times10^{-15}`$, establishing the much smaller difference from the
other input conversions.

| Example sampling | Fixed source/kernel/LOS counts | Initial largest IA error | Largest error after four times as many background intervals |
|---|---|---:|---:|
| TATT: 101 nodes to $`z=3`$ | 101 / 201 / 65 | $`3.931\times10^{-5}`$ | $`2.127\times10^{-6}`$ at 401 nodes |
| BACCO: 125 nodes to $`z=3.1`$ | 9601 / 401 / 129 | $`2.643\times10^{-5}`$ | $`1.602\times10^{-6}`$ at 497 nodes |

The permanent tests separate two types of bound. The derivative-rate checks
follow the analytic first-/second-order predictions above. The absolute IA
limits are measured allowances for these fixed toy inputs: $`5\times10^{-6}`$
at 321 nodes, and $`1.2\times10^{-4}`$/$`7\times10^{-6}`$ for the initial/refined
example sampling. These retain about three to five times the measured
maxima. The IA rate checks allow departures from the asymptotic factor of
four on coarse grids; the GGI error's first halving gave a factor of about
3.1. These allowances are regression criteria for this experiment, not a
physical uncertainty assigned to the IA model.

All derivative probes here use uniform background grids. Changes in spacing
at joins between uniform segments are outside this check. The reference
also retains the same finite interpolation and projection rules, so these
results isolate the derivative choice rather than the total projection
error.

Run from the project directory in the existing Cocoa environment:

```bash
env OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m pytest -q tests/test_bridge_background_analytic.py
```

Every physical input array, scalar and numerical control is saved to an
NPZ file and reloaded before either model runs. The same archive contains
the ordinary-core scalars, angular queries and IA controls. JSON records
identify each archive by SHA256 and retain the extension, both public
headers, test source and observer-source hashes. No data vector, covariance
or mask is used in these raw-observable comparisons. The combined record
also describes the initial observation-order error: a cached refresh returns
no new input. The corrected observer copies the first refresh and resets
before a model factory is called. The original failed log remains identified
by path and hash; no production source or binary was changed to fix the test.
