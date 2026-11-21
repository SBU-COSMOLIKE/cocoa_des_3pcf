"""Fit and generate a bounded GPTG shear-map pilot on Rafael's DES footprint.

This is a research implementation of Zhong et al. (arXiv:2411.04759),
not their released code or fitted model. Its first calibration uses a
masked convergence reconstruction and an approximate E-minus-B spectrum.
Forward comparisons expose the errors from those approximations. Generated
maps must not be called validated DES simulations on the basis of this fit.

Dependencies: numpy, scipy, ducc0, PyYAML, astropy and astropy-healpix.
Use BLAS=1 and at most the currently authorized eight native transform workers. Run from this folder:
  python generate_maps.py fit --runs 32
  python generate_maps.py generate --count 64
The two commands are separate operations. Generated maps and fit products
stay under the project's ignored chains/gptg_method directory.
"""

import argparse
import os
from pathlib import Path
import sys
import time

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'

import ducc0
import numpy as np
from scipy.optimize import least_squares
from scipy.special import roots_hermitenorm, roots_legendre, ndtri, expit
import yaml

PROJECT = Path(__file__).resolve().parents[2]
OUTPUT = PROJECT / 'chains' / 'gptg_method'
# Reuse the identical aperture estimator already applied to Rafael's maps.
sys.path.insert(0, str(PROJECT / 'scripts'))
from measure_covariance_maps import aperture_weights


def transform(gaussian, parameters):
    """Evaluate Table 1's five-parameter unnormalized point transformation.

    The input is a unit-variance Gaussian field. logaddexp evaluates
    log(1+exp(x)) without overflowing. Normalization and physical amplitude
    are applied separately, with their Gaussian expectations held fixed.
    """
    a1, a2, b, x0, transition = parameters
    return (np.exp(a1*gaussian-a1*a1/2.)+b*gaussian) * np.exp(
        (a2-a1)/transition*np.logaddexp(0., transition*(gaussian-x0))
    )


def normalized_transform(gaussian, parameters, nodes, weights):
    """Subtract the ensemble mean and divide by its ensemble standard deviation.

    This is not a per-map rescaling: normalizing each simulated map by its
    measured scatter would remove fluctuations needed by a covariance test.
    A physical convergence amplitude is restored after this unit conversion.
    """
    values = transform(
        gaussian=nodes,
        parameters=parameters,
    )
    mean = weights @ values
    variance = weights @ (values-mean)**2
    return (transform(
        gaussian=gaussian,
        parameters=parameters,
    )-mean)/np.sqrt(variance)


def transform_slope_sign(gaussian, parameters):
    """Return the derivative divided by its positive transition factor.

    A good fit to central quantiles can still turn over in a tail. Such a
    fit is not the monotone inverse-Gaussianization assumed by the method.
    This check covers the fitted quantiles and the Gaussian quadrature tails.
    """
    a1, a2, b, x0, transition = parameters
    exponential = np.exp(a1*gaussian-a1*a1/2.)
    return (a1*exponential+b + (exponential+b*gaussian)*(a2-a1)
            * expit(transition*(gaussian-x0)))


def harmonic_indices(lmax):
    """Name ell for the m-major triangular harmonic storage used by ducc0."""
    return np.concatenate([np.arange(m, lmax+1) for m in range(lmax+1)])


def spectra(first, second, ell, lmax):
    """Average a_lm b_lm* over m, including the unstored negative m modes."""
    products = (first*second.conj()).real.copy()
    products[lmax+1:] *= 2.
    return np.bincount(
        ell,
        weights=products,
        minlength=lmax+1,
    )/(2*np.arange(lmax+1)+1)


def setup(lmax):
    """Set fixed pixel geometry, quadrature and the convergence-to-shear factor."""
    indices = np.load(file=PROJECT/'maps_cov_512_v2/indices_512.npy')
    geometry = ducc0.healpix.Healpix_Base(512, 'RING').sht_info()
    ell = harmonic_indices(lmax=lmax)
    factor = np.zeros(len(ell))
    valid = ell >= 2
    factor[valid] = -np.sqrt((ell[valid]+2.)*(ell[valid]-1.) /
                            (ell[valid]*(ell[valid]+1.)))
    nodes, weights = roots_hermitenorm(160)
    weights /= np.sqrt(2*np.pi)
    return indices, geometry, ell, factor, nodes, weights


def pseudo_modes(shear, indices, geometry, lmax, workers, second_sign=1.):
    """Measure masked E/B harmonics with a fixed equal-area pixel quadrature.

    Zeros encode the known window inside a pseudo-spectrum transform.
    They are not additional observed zero-shear galaxies. No deconvolution
    of the mask is implied by this operation.
    """
    maps = np.zeros((2, 12*512**2))
    maps[:, indices] = shear
    maps[1] *= second_sign
    return ducc0.sht.adjoint_synthesis(
        map=maps,
        spin=2,
        lmax=lmax,
        nthreads=workers,
        **geometry,
    ) * (4*np.pi/maps.shape[-1])


def fit(args):
    """Fit an initial generator, then leave its accuracy to forward tests.

    One cut and one draw are used from each selected run. The approximate
    E-minus-B power removes isotropic noise and some mask leakage; it is
    not an exact pseudo-Cl deconvolution. Quantile fits use a filtered,
    masked E reconstruction. Their parameters are initial estimates of
    a forward model, not measurements of a noiseless convergence PDF.
    """
    OUTPUT.mkdir(
        parents=True,
        exist_ok=True,
    )
    indices, geometry, ell, factor, nodes, weights = setup(lmax=args.lmax)
    first_weight, second_weight, centers = aperture_weights(indices=indices)
    paths = sorted((PROJECT/'maps_cov_512_v2').glob(
        'ShearMaps_DESY3_run_*_ncut0_nreal0.npy'
    ))[:args.runs]
    if len(paths) != args.runs:
        raise ValueError('The requested number of complete primary maps is unavailable.')
    spectrum = np.zeros((2, 4, 4, args.lmax+1))
    reconstruction = []
    apertures = []
    timing = time.monotonic()
    # Sample geometry, not map amplitudes, to choose reconstruction pixels.
    sample_pixels = centers[:, 0].astype(int)
    sign = args.second_sign
    for run, path in enumerate(paths):
        shear = np.load(
            file=path,
            mmap_mode='r',
        )
        modes = []
        proxy = []
        for source in range(4):
            result = pseudo_modes(
                shear=shear[source],
                indices=indices,
                geometry=geometry,
                lmax=args.lmax,
                workers=args.workers,
                second_sign=sign,
            )
            modes.append(result)
            inverse_factor = np.zeros_like(factor)
            inverse_factor[ell >= 2] = 1./factor[ell >= 2]
            # The same smoothing and mask must be applied to generated maps
            # when testing this reconstructed distribution.
            smooth = np.exp(-ell*(ell+1)*np.deg2rad(10./60.)**2/2.)
            convergence = ducc0.sht.synthesis(
                alm=result[:1]*inverse_factor*smooth,
                spin=0,
                lmax=args.lmax,
                nthreads=args.workers,
                **geometry,
            )
            proxy.append(convergence[0, sample_pixels])
        modes = np.asarray(modes)
        for i in range(4):
            for j in range(i, 4):
                for parity in range(2):
                    value = spectra(modes[i, parity], modes[j, parity], ell, args.lmax)
                    spectrum[parity, i, j] += value
                    if i != j:
                        spectrum[parity, j, i] += value
        reconstruction.append(proxy)
        aperture = first_weight @ shear[:, 0].T + second_weight @ shear[:, 1].T
        apertures.append(aperture.T.reshape(4, 2, 64))
        print('fit map', run+1, '/', len(paths), flush=True)
    spectrum /= len(paths)
    np.savez_compressed(
        OUTPUT/'fit_measurements.npz',
        pseudo_cl=spectrum,
        reconstruction=reconstruction,
        aperture=apertures,
        centers=centers,
    )
    # The mask approximation below is only an initial spectrum estimate.
    # The later forward measurement uses the actual mask again and reports
    # its mismatch instead of declaring this deconvolution exact.
    fsky = len(indices)/(12*512**2)
    multipoles = np.arange(args.lmax+1)
    band_edges = np.unique(np.geomspace(10, args.lmax+1, 18).astype(int))
    cl = np.zeros((args.lmax+1, 4, 4))
    sigma_noise = np.zeros(4)
    high = multipoles >= int(.75*args.lmax)
    pixel_area = 4*np.pi/(12*512**2)
    for i in range(4):
        white_noise = np.average(spectrum[1, i, i, high], weights=2*multipoles[high]+1)
        sigma_noise[i] = np.sqrt(max(white_noise/fsky/pixel_area, 0.))
        for j in range(i, 4):
            centers_l = []
            values = []
            for lower, upper in zip(band_edges[:-1], band_edges[1:]):
                take = (multipoles >= lower) & (multipoles < upper)
                weight = 2*multipoles[take]+1
                signal = (spectrum[0, i, j, take]-spectrum[1, i, j, take])/fsky
                # Gamma_E power differs from convergence power on the sphere.
                ratio = ((multipoles[take]+2.)*(multipoles[take]-1.) /
                         (multipoles[take]*(multipoles[take]+1.)))
                centers_l.append(np.average(multipoles[take], weights=weight))
                values.append(np.average(signal/ratio, weights=weight))
            cl[:, i, j] = np.interp(multipoles, centers_l, values)
            cl[:, j, i] = cl[:, i, j]
    cl[:2] = 0.
    # Sampling and the approximate mask correction can make cross-spectra
    # incompatible. Record the PSD projection instead of hiding its size.
    eigenvalues, eigenvectors = np.linalg.eigh(cl)
    repaired = (eigenvectors*np.maximum(
        eigenvalues,
        0.,
    )[:, None, :]) @ eigenvectors.swapaxes(
        -1,
        -2,
    )
    change = np.linalg.norm(repaired-cl)/np.linalg.norm(cl)
    cl = repaired
    # Quantiles rank the reconstructed convergence values. Because a
    # monotone transform preserves ranks, its value at a Gaussian quantile
    # predicts the corresponding output quantile. The fitted distribution
    # is a noisy/smoothed proxy, not a recovered noiseless convergence PDF.
    probabilities = np.linspace(.02, .98, 49)
    gaussian_quantiles = ndtri(probabilities)
    parameters = []
    fit_residuals = []
    fitted_family = []
    for source in range(4):
        observed = np.asarray(reconstruction)[:, source].ravel()
        observed = (observed-observed.mean())/observed.std()
        quantiles = np.quantile(observed, probabilities)
        def residual(values):
            predicted = normalized_transform(gaussian_quantiles, values, nodes, weights)
            tail_slope = transform_slope_sign(np.linspace(-40., 40., 321), values)
            return np.concatenate([predicted-quantiles,
                                   20.*np.minimum(tail_slope, 0.)])
        result = least_squares(
            fun=residual,
            x0=[.2, .15, .05, 0., 1.],
            bounds=([.005, .005, 0., -3., .2], [1.5, 1.5, 1., 3., 4.]),
            max_nfev=300,
        )
        chosen = result.x
        family = 'five-parameter'
        if np.min(transform_slope_sign(np.linspace(-40., 40., 4001), chosen)) < -1.e-8:
            # Table 1 also supplies a four-parameter family with b=0.
            # Its logarithmic derivative interpolates between positive a1
            # and a2, making it monotone for every real Gaussian input.
            # Record this model restriction; do not hide a turnover by
            # clipping generated tail values after drawing a map.
            def four_residual(values):
                full = np.array([values[0], values[1], 0., values[2], values[3]])
                return normalized_transform(
                    gaussian_quantiles,
                    full,
                    nodes,
                    weights,
                )-quantiles
            four = least_squares(
                fun=four_residual,
                x0=[.15, .15, 0., 1.],
                bounds=([.005, .005, -3., .2], [1.5, 1.5, 3., 4.]),
                max_nfev=300,
            )
            chosen = np.array([four.x[0], four.x[1], 0., four.x[2], four.x[3]])
            family = 'four-parameter; b fixed to zero for global monotonicity'
        parameters.append(chosen)
        fitted_family.append(family)
        error = normalized_transform(
            gaussian_quantiles,
            chosen,
            nodes,
            weights,
        )-quantiles
        fit_residuals.append(float(np.sqrt(np.mean(error**2))))
    np.savez_compressed(
        OUTPUT/'calibration.npz',
        cl_target=cl,
        parameters=parameters,
        noise_sigma=sigma_noise,
    )
    record = {
        'method': 'GPTG initial masked-field fit; not validated DES simulations',
        'paper': 'https://arxiv.org/abs/2411.04759',
        'nside': 512,
        'lmax': args.lmax,
        'workers': args.workers,
        'second_component_sign_for_ducc': sign,
        'source_files': [path.name for path in paths],
        'noise_model': 'constant independent Gaussian per pixel and source bin',
        'noise_sigma_per_component': sigma_noise.tolist(),
        'point_parameters_a1_a2_b_x0_t': np.asarray(parameters).tolist(),
        'fitted_family': fitted_family,
        'reconstructed_pdf_quantile_rms': fit_residuals,
        'target_cl_psd_relative_change': float(change),
        'elapsed_seconds': time.monotonic()-timing,
    }
    (OUTPUT/'calibration.yaml').write_text(yaml.safe_dump(record, sort_keys=False))
    print(yaml.safe_dump(record, sort_keys=False), flush=True)


def gaussian_spectra(cl, parameters, nodes, weights):
    """Invert the point-transform correlation relation before drawing fields.

    Hermite coefficients turn the Gaussian two-point integral into a power
    series in its correlation. Legendre transforms connect that correlation
    to angular spectra. Positive spectra and normalized Gaussian variances
    are checked after the finite transform; changes are reported.
    """
    lmax = len(cl)-1
    variance = np.einsum('l,lii->i', (2*np.arange(lmax+1)+1)/(4*np.pi), cl)
    if np.any(variance <= 0.):
        raise ValueError('The initial signal fit has a nonpositive variance.')
    # Orthonormal Hermite coefficients convert a correlated Gaussian
    # expectation into sum_n a_in*a_jn*rho**n. This avoids reevaluating a
    # two-dimensional integral for every angular separation and bin pair.
    hermite = np.empty((65, len(nodes)))
    hermite[0] = 1.
    hermite[1] = nodes
    for degree in range(1, 64):
        hermite[degree+1] = (nodes*hermite[degree]-np.sqrt(
            degree,
        )*hermite[degree-1])/np.sqrt(
            degree+1,
        )
    coefficients = []
    for values in parameters:
        if np.min(transform_slope_sign(np.linspace(-40., 40., 4001), values)) < -1.e-8:
            raise ValueError('Nonmonotone transformation: refit before generating maps.')
        field = normalized_transform(nodes, values, nodes, weights)
        coefficients.append(hermite @ (weights*field))
    coefficients = np.asarray(coefficients)
    # Legendre quadrature connects angular correlations and spectra.
    # Pairwise invertibility is weaker than matrix positivity: the final
    # spectral matrices must also permit a joint four-bin Gaussian draw.
    mu, quadrature = roots_legendre(2*lmax+2)
    legendre = np.polynomial.legendre.legvander(mu, lmax)
    rho = np.linspace(-1., 1., 8193)
    latent = np.zeros_like(cl)
    for i in range(4):
        for j in range(i, 4):
            series = coefficients[i]*coefficients[j]
            series[0] = 0.
            relation = np.polynomial.polynomial.polyval(rho, series)
            if np.any(np.diff(relation) <= 0.):
                raise ValueError(
                    'The fitted point-transform correlation is not monotone.',
                )
            target = legendre @ ((2*np.arange(lmax+1)+1)*cl[:, i, j]/(4*np.pi))
            target /= np.sqrt(variance[i]*variance[j])
            if target.min() < relation.min()-1.e-5 or target.max() > relation.max()+1.e-5:
                raise ValueError(
                    'Target cross-correlation is not realizable by these fitted PDFs.',
                )
            gaussian = np.interp(target, relation, rho)
            latent[:, i, j] = 2*np.pi*(legendre.T @ (quadrature*gaussian))
            latent[:, j, i] = latent[:, i, j]
    eigenvalues, eigenvectors = np.linalg.eigh(latent)
    repaired = (eigenvectors*np.maximum(
        eigenvalues,
        0.,
    )[:, None, :]) @ eigenvectors.swapaxes(
        -1,
        -2,
    )
    correction = np.linalg.norm(repaired-latent)/np.linalg.norm(latent)
    latent = repaired
    diagonal = np.einsum('l,lii->i', (2*np.arange(lmax+1)+1)/(4*np.pi), latent)
    latent /= np.sqrt(diagonal[None, :, None]*diagonal[None, None, :])
    return latent, variance, correction, diagonal


def generate(args):
    """Save independent fitted maps and measure the same raw aperture moments.

    Each saved file has four bins, two components and the original footprint
    pixel order. No generated map is thrown away because its covariance
    correction looks bad. The seeds and generation approximations are saved.
    """
    record = yaml.safe_load((OUTPUT/'calibration.yaml').read_text())
    lmax = record['lmax']
    indices, geometry, ell, factor, nodes, weights = setup(lmax=lmax)
    first_weight, second_weight, centers = aperture_weights(indices=indices)
    data = np.load(file=OUTPUT/'calibration.npz')
    latent, variance, correction, diagonal = gaussian_spectra(
        cl=data['cl_target'],
        parameters=data['parameters'],
        nodes=nodes,
        weights=weights,
    )
    eigenvalues, eigenvectors = np.linalg.eigh(latent)
    roots = eigenvectors*np.sqrt(np.maximum(eigenvalues, 0.))[:, None, :]
    maps_folder = OUTPUT/'maps'
    maps_folder.mkdir(exist_ok=True)
    np.save(
        file=maps_folder/'indices_512.npy',
        arr=indices,
    )
    moments = []
    aperture_all = []
    stages = []
    # Float64 is an explicit small-reference option for the storage test.
    # Production maps use float32; all numerical work remains float64.
    storage_dtype = np.float32
    if args.float64_reference:
        storage_dtype = np.float64
    sign = record['second_component_sign_for_ducc']
    for number in range(args.count):
        path = maps_folder/f'GPTG_DES_{number:06d}.npy'
        if path.exists():
            raise FileExistsError(
                f'{path} exists; keep the previous experiment separate.',
            )
        start = time.monotonic()
        rng = np.random.default_rng(seed=args.seed+number)
        white_real = rng.normal(size=(len(ell), 4))
        white_imaginary = rng.normal(size=(len(ell), 4))
        white = (white_real+1j*white_imaginary)/np.sqrt(2.)
        # m=0 has real coefficients of full variance. Every stored m>0
        # coefficient has two components with half that variance; reality
        # supplies its unstored negative-m partner.
        white[:lmax+1] = np.sqrt(2.)*white[:lmax+1].real
        alms = np.einsum('nij,nj->ni', roots[ell], white).T
        shears = []
        for source in range(4):
            gaussian = ducc0.sht.synthesis(
                alm=alms[source:source+1],
                spin=0,
                lmax=lmax,
                nthreads=args.workers,
                **geometry,
            )
            kappa = np.sqrt(variance[source])*normalized_transform(
                gaussian=gaussian,
                parameters=data['parameters'][source],
                nodes=nodes,
                weights=weights,
            )
            # A nonlinear point function adds angular frequencies beyond
            # lmax. The iterative transform projects onto the retained
            # modes; its map residual includes discarded high frequencies.
            # This projection is a physical-resolution approximation even
            # when the least-squares solver has converged.
            result = ducc0.sht.pseudo_analysis(
                map=kappa,
                spin=0,
                lmax=lmax,
                nthreads=args.workers,
                maxiter=30,
                epsilon=1.e-6,
                **geometry,
            )
            if result[1] not in (1, 2):
                raise RuntimeError(f'Transform did not converge: {result[1:]}')
            shear = ducc0.sht.synthesis(
                alm=result[0]*factor,
                spin=2,
                mode='GRAD_ONLY',
                lmax=lmax,
                nthreads=args.workers,
                **geometry,
            )[:, indices]
            shear[1] *= sign
            # Noise is drawn independently in every retained pixel and
            # source/component. It is not generated by transforming a
            # scalar convergence field, which would force an E-only noise.
            shear += data['noise_sigma'][source]*rng.normal(size=shear.shape)
            shears.append(shear)
        shears = np.asarray(shears)
        made = time.monotonic()
        stored = shears.astype(storage_dtype)
        np.save(
            file=path,
            arr=stored,
        )
        shears = stored.astype(np.float64)
        saved = time.monotonic()
        values = first_weight @ shears[:, 0].T + second_weight @ shears[:, 1].T
        values = values.T.reshape(4, 2, 64)
        aperture_all.append(values)
        moments.append(np.concatenate([(values**2).mean(axis=2).ravel(),
                                       (values**3).mean(axis=2).ravel()]))
        stages.append([made-start, saved-made, time.monotonic()-saved])
        print('saved map', number+1, '/', args.count, 'seconds', stages[-1], flush=True)
    np.savetxt(
        fname=OUTPUT/'generated_moments.txt',
        X=moments,
    )
    np.savez_compressed(OUTPUT/'generated_measurements.npz', aperture=aperture_all)
    np.savetxt(
        fname=OUTPUT/'generation_timing.txt',
        X=stages,
        header='generation_with_noise save_npy aperture_measurement_seconds',
    )
    (OUTPUT/'generation.yaml').write_text(yaml.safe_dump({
        'count': args.count,
        'seed_start': args.seed,
        'workers': args.workers,
        'latent_cl_psd_relative_change': float(correction),
        'latent_variance_before_normalization': diagonal.tolist(),
        'dtype': np.dtype(storage_dtype).name,
        'shape': [4, 2, len(indices)],
        'noise': record['noise_model'],
        'status': 'initial calibrated pilot; forward validation required',
    }, sort_keys=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['fit', 'generate'])
    parser.add_argument(
        '--runs',
        type=int,
        default=32,
    )
    parser.add_argument(
        '--count',
        type=int,
        default=64,
    )
    parser.add_argument(
        '--lmax',
        type=int,
        default=1024,
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=4,
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=241104759,
    )
    parser.add_argument(
        '--second-sign',
        type=float,
        choices=[-1., 1.],
        default=1.,
    )
    parser.add_argument(
        '--float64-reference',
        action='store_true',
        help='Keep a small float64 reference for storage tests; default maps are float32.',
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error('Use one to eight workers; BLAS remains one.')
    if args.operation == 'fit':
        fit(args)
    else:
        generate(args)
