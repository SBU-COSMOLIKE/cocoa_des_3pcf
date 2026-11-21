"""Experimental scalar-GPU map generation with CPU spin-two shear synthesis.

The GPU keeps the Gaussian sky, nonlinear point transform and scalar inverse
solve on device. Only the resulting harmonic coefficients return to ducc0
for spin-two shear. Random draws and frozen calibration retain the original
CPU convention. This file is exercised by the comparison test before a
production MPI worker is permitted to select it.
"""
import time
import numpy as np
import ducc0
import mlx.core as mx
from generate_maps import transform


def generate_gpu_map(sht, seed, record, data, roots, variance, ell, factor,
                     indices, geometry, nodes, weights, threads):
    """Generate one four-bin shear realization using the scalar GPU path.

    roots[ell] correlates the four Gaussian source-bin coefficients. The
    fixed calibration supplies the point-function parameters and ensemble
    variance. geometry describes full-sky RING pixels; indices selects the
    DES footprint only after spin-two synthesis. factor contains the signed
    convergence-to-shear harmonic conversion, including zero at ell<2.

    Returns float32 pixels shaped (4, 2, number_of_footprint_pixels), plus
    per-bin timing and true normal-residual diagnostics. The random stream
    and noise ordering match the CPU generator for paired comparisons.
    """
    # Draw independent complex Gaussian coefficients, then correlate bins.
    # Real and imaginary parts each have variance 1/2. The m=0 block must
    # instead be real with variance one to represent a real full-sky field.
    rng = np.random.default_rng(seed)
    white = rng.normal(size=(len(ell),4))
    white = (white+1j*rng.normal(size=white.shape))/np.sqrt(2.)
    white[:record['lmax']+1] = np.sqrt(2.)*white[:record['lmax']+1].real
    # roots has one 4x4 covariance square root per degree. Multiplying it
    # by white coefficients creates the chosen cross-bin angular spectra.
    # Only after this double-precision draw do we round to complex64.
    alms = np.einsum('nij,nj->ni',roots[ell],white).T.astype(np.complex64)
    result = np.empty((4,2,len(indices)),dtype=np.float32)
    diagnostics = []
    for source in range(4):
        start = time.monotonic()
        gaussian = sht.synthesis(mx.array(alms[source]))
        mx.eval(gaussian)
        synthesized = time.monotonic()
        # Ensemble normalization is fixed by Gaussian quadrature, not by
        # the particular sky's mean or variance.
        values = transform(gaussian=nodes,parameters=data['parameters'][source])
        mean = weights @ values
        norm = np.sqrt(weights @ (values-mean)**2)
        # Evaluate the same fitted point function on GPU. logaddexp is a
        # stable log(1+exp(x)); it avoids overflowing an intermediate exp.
        # The mean and norm above are ensemble quantities, not this map's
        # measured mean or RMS. Per-map renormalization would erase part of
        # the fluctuation we are trying to measure with a covariance.
        a1,a2,b,x0,transition = data['parameters'][source].astype(np.float32)
        g = gaussian
        value = (mx.exp(float(a1)*g-float(a1*a1/2.))+float(b)*g)*mx.exp(
            float((a2-a1)/transition)*mx.logaddexp(
                mx.array(0.,dtype=mx.float32),float(transition)*(g-float(x0)),
            )
        )
        kappa = float(np.float32(np.sqrt(variance[source])))*(
            value-float(np.float32(mean)))/float(np.float32(norm))
        mx.eval(kappa)
        transformed = time.monotonic()
        # The nonlinear mapping generates modes above lmax. Solve for its
        # least-squares scalar coefficients at the retained resolution;
        # simply applying the pixel adjoint would leave discretization error.
        coefficients, quality = sht.solve(kappa)
        solved = time.monotonic()
        # Scalar GPU harmonics do not implement spin-two shear harmonics.
        # Transfer only coefficients back to CPU, apply the existing signed
        # ell factor, synthesize both components, and then select footprint
        # pixels. Keep the original noise draw order for reproducibility.
        shear = ducc0.sht.synthesis(
            alm=np.array(coefficients)[None,:]*factor.astype(np.float32),
            spin=2,
            mode='GRAD_ONLY',
            lmax=record['lmax'],
            nthreads=threads,
            **geometry,
        )[:,indices]
        shear[1] *= record['second_component_sign_for_ducc']
        shear += data['noise_sigma'][source]*rng.normal(size=shear.shape)
        result[source] = shear
        diagnostics.append([source,synthesized-start,transformed-synthesized,
                            solved-transformed,time.monotonic()-solved,
                            quality['iterations'],quality['normal_residual']])
    return result, diagnostics
