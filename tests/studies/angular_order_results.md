# Fiducial angular-order checks for NLA and TATT

All ten calculations in the [five-case protocol](angular_order_validation.md)
completed. Each starts from its model's full `triangle_1p5` Fisher
configuration and evaluates one mean at the same physical parameter point.
No derivatives are evaluated in this study. The 400 two-point rows remain
bitwise unchanged in every case. The 80 aperture rows, their ordering,
source inputs and supplied joint covariance stay the same scientific
comparison throughout.

The distinction between the controls matters. The Legendre degree L
describes the angular dependence of the projected bispectrum. The Fourier
order M describes the opening-angle dependence of the transformed shear
correlation. Increasing either order does not add samples of the input
bispectrum. The prepared angular samples therefore receive a separate
test. The geometric coupling has its own interpolation, which can also
be removed without changing the physical bispectrum samples.

## Distance between predictions

For two calculated vectors, this study measures

```math
D^2=(d_a-d_b)^{\mathsf T}C^{-1}(d_a-d_b),
```

using the complete fixed 480-row covariance, including the two-point–
three-point cross block. This is a distance between predictions at one
parameter point, not a fit to observed data. Although the two-point
difference is zero, replacing C by its aperture block would change the
metric. The full inverse weights the remaining aperture difference with
the aperture covariance conditional on the two-point data.

All cases retain triangle boost 1.5 and the same 576-node aligned FFT
grid. The baseline has L=M=20, prepared angular counts 32+32 and 200
coupling interpolation nodes. The phrase “direct coupling” below means
evaluation of the analytic finite sum at the actual Fourier side ratio;
it removes coupling interpolation, not floating-point roundoff.

| Candidate relative to inherited baseline | NLA D² | TATT D² |
|---|---:|---:|
| L=50, M=20; other settings retained | 0.00143060 | 0.00156042 |
| L=50, M=50; other settings retained | 0.00143060 | 0.00156042 |
| L=M=50; angular samples 64+64 | 0.00491712 | 0.00527268 |
| L=M=50; direct coupling | 0.000612707 | 0.000691225 |
| L=M=50; angular samples 64+64 and direct coupling | 0.00338021 | 0.00365968 |

The first two rows look equal at the printed precision, but the saved
vectors are not identical. Comparing L=50, M=50 directly with L=50,
M=20 gives D²=5.37e-18 for NLA and 5.72e-18 for TATT. Thus the added
M=21–50 contributions have negligible effect on these particular
aperture means. This is a statement about the integrated observable at
the tested parameter point; it does not bound every natural component
or every displaced parameter vector.

To isolate angular sampling from coupling interpolation, the next table
uses the same L=M=50, 32+32-sample, interpolated-coupling reference for
all rows. Adjacent rows of the first table would otherwise change two
controls at once.

| Change from the shared L=M=50 reference | NLA D² | TATT D² |
|---|---:|---:|
| Angular samples 32+32 to 64+64 only | 0.00118991 | 0.00125743 |
| Remove coupling interpolation only | 0.000206343 | 0.000205111 |
| Make both changes | 0.000675263 | 0.000716676 |

The combined distance is smaller than the angular-only distance.
The two changes partly cancel in the covariance metric; their squared
distances must not be added as independent error variances. This is why
the protocol retained separate angular-only and coupling-only cases.

## Independent check of the larger-order geometric coupling

The finite Fourier sum was also compared with a direct fixed
Gauss-Legendre integral of its defining angular expression. This tests
the geometric routine separately from any cosmology or aperture mean.
The new [component test](../test_coupling_high_orders.py) covers every
L from 0 through 50 and every M from 0 through 100 at seven side-ratio
angles, including equal sides and both angular endpoints. Endpoint
phases are supplied in their analytic limiting forms.

Both the 512-node and 1024-node tests pass. Their largest absolute
differences from the native sum are 8.27e-13 and 1.46e-12, respectively;
on the natural coupling normalization 4*pi² these are 2.09e-14 and
3.71e-14. Zero coefficients are included through this absolute metric,
without division by a vanishing reference. The change between the two
quadrature calculations is about 2.01e-12, so the slightly larger
1024-node residual is consistent with the scale of their numerical
agreement rather than evidence of monotonic improvement from more nodes.
The numerical record is `runs/high_order_coupling_check.json`.

These checks do not test every angle arbitrarily close to equal sides,
the subsequent Bessel transform, or the full aperture estimator at
M=100. They establish a separate component check at the larger orders
without assuming that a small final aperture change validates every
intermediate quantity. Both OpenMP and BLAS used one worker.

## What this establishes and what it leaves open

These fiducial distances are small in the supplied covariance metric,
but no acceptance tolerance was imposed after observing them. The full
[triangle Fisher comparison](fisher_accuracy_validation.md) already
shows why a mean distance alone cannot bound a marginalized ellipse.
Derivatives divide differences by parameter steps, and nuisance
marginalization can leave a much smaller response than the original
derivative columns. The angular study therefore does not certify the
Fisher calculations at L=M=20 or at L=M=50.

For a subsequent derivative check, retain L=M=50, 64+64 angular samples
and direct coupling as an explicit candidate, rather than silently
changing the default configuration or replacing its saved arrays.
Compare selected derivatives first and follow a material change with
all nuisance columns. The independent parameter-step investigation can
continue on the existing triangle-refined configuration because it asks
a different question. Keep its grid identity explicit.

The unchanged 63 point-angle centers avoid pairwise aliasing between
retained modes in the signed range -50 to 50. They do not establish
convergence of the complete angle-dependent aperture kernel. Tests of
M=100, the number of angle centers, FFT density, Fourier limits and
the catalogue's exact finite-bin weights remain separate. The
[protocol](angular_order_validation.md) derives these distinctions and
lists the memory required before increasing further counts.

## Saved evidence and reproduction

The driver is [angular_order_study.py](angular_order_study.py). It
records the full configurations, input and implementation identities,
the protocol bytes, every 480-row vector and a receipt for each completed
case. Results are in `runs/nla_angular_orders/summary.json` and
`runs/tatt_angular_orders/summary.json`. These local run directories are
ignored by Git; the protocol, executable study and this numerical record
are versioned. Resuming verifies the matching completed prefix and
refuses conflicting files. No numerical result is relabeled with a
different physical implementation identity.

Run the commands from the project directory in the activated Cocoa
environment with serial BLAS and at most four OpenMP workers, as shown
in the [protocol's execution steps](angular_order_validation.md).
One model process ran at a time. This study did not measure runtime.

**Step :one:** inspect the NLA configuration without launching predictions.

```bash
python tests/studies/angular_order_study.py \
  runs/nla_accuracy/triangle_1p5.json --output runs/nla_angular_orders
```

**Step :two:** execute or verify the same NLA study.

```bash
python tests/studies/angular_order_study.py \
  runs/nla_accuracy/triangle_1p5.json --output runs/nla_angular_orders --execute
```

**Step :three:** execute or verify TATT separately.

```bash
python tests/studies/angular_order_study.py \
  runs/tatt_accuracy/triangle_1p5.json --output runs/tatt_angular_orders --execute
```
