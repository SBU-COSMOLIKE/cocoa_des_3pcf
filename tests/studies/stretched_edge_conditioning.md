# Stretched PUV coordinate: extreme-edge diagnostic

The single-core diagnostic used the controlled spectra from
`tests/data_vector/test_physics_model.py`, the alignment parameters from
`test_residual_scaling.py`, `residual_scaling=1`, `shape_stretch=10.106`,
17 coarse shape nodes, 7 perimeter nodes, a dense-table factor of two, and
Fourier bounds `[0.001,300000]`. The source triplet was `(0,1,0)`. No CAMB
calculation or survey prediction was run. BLAS and OpenMP each used one worker.

The six diagnostic triangles use shape
indices `(0,-1)`, `(1,-2)`, `(-2,1)`, `(-1,0)`, `(2,-2)`, `(-2,2)` from the
original coarse axis. Negative indices count backward from the final node.
Their sides are `p*(u,v,1-u-v)` at the middle original perimeter node.
All triangles obey physical closure. Some are nearly flattened at the
precision of their represented side lengths. In the table, GGG is the
positive gravity-only projection at each actual triangle, used only as an
error scale; it is distinct from the equilateral perimeter scale used
internally when preparing the residual table.

| Ordered shape indices | Prepared residual / GGG | Direct residual / GGG |
|---|---:|---:|
| `(0,-1)` | 1240.283544996715 | 1240.5854941706896 |
| `(1,-2)` | 47.40981933601365 | 47.41008064181727 |
| `(-2,1)` | 14.307124277603807 | 14.287607603221405 |
| `(-1,0)` | 373.86558721828277 | 373.865587336744 |
| `(2,-2)` | 4.643750628290998 | 4.643750956384265 |
| `(-2,2)` | 1.3995942130359955 | 1.3994559132448112 |

For stretch `s`, width `w=0.5-u_min`, and
`x=2*(u-u_min)/w-1`, the inverse shape coordinate has derivative

```math
\frac{dt}{du}=
\frac{2\tanh s}{s\,w\,[1-\tanh^2(s)x^2]}.
```

Near the upper endpoint at this stretch, a one-ulp error in the ratio can
therefore become a few times $`10^{-9}`$ in the transformed coordinate. In
the first diagnostic triangle, reconstructing `ell2/(ell1+ell2+ell3)` gives
`0.5-5.551115123125783e-17`; its inverse coordinate differs from the intended
endpoint by `-2.742949978262743e-9`. For the third triangle, corresponding
coordinate differences are `3.202802467683341e-10` and
`-5.694067439776518e-11`. The interpolated residual is sensitive to these
small coordinate changes. Replacing only the inverse with a cancellation-
resistant exponential/logarithmic expression reduces some differences but
cannot remove uncertainty in the represented side ratios themselves.

As a separate check, changing the first side by one ulp changed the direct
residual by at most about $`7\times10^{-7}`$ times GGG across these six
triangles, which is much smaller than the largest prepared/direct
discrepancies above. This check shows why a uniform tight knot tolerance
is inappropriate without examining coordinate conditioning and the
interpolated table's local variation; it does not bound every direction
of triangle perturbation.

The safety test keeps exact-endpoint and strict-axis-order checks, and
compares direct values only at physical interior knots with
`|dt/du|<5` and all three fractions between `0.01` and `0.49`. No query is
snapped, quantized, or clipped to make that test pass. The extreme values
above remain a numerical-study limitation: their effect on an actual mean
requires component-resolved triangle, multipole, and finite-estimator
refinement.
