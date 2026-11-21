"""Measure float32 storage error using all 64 saved float64 pilot maps.

Generation and aperture accumulation stay in float64. Only the saved shear
values are rounded to float32 and then decoded to float64, isolating storage
precision from arithmetic precision. Original maps are never modified.
"""

import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '4'

import numpy as np
from generate_maps import OUTPUT, PROJECT, aperture_weights


def main():
    """Compare raw moments, covariance and precision on the same realizations."""
    indices = np.load(file=PROJECT / 'maps_cov_512_v2/indices_512.npy')
    first, second, centers = aperture_weights(indices=indices)
    paths = sorted((OUTPUT / 'maps').glob('GPTG_DES_*.npy'))
    if len(paths) != 64:
        raise ValueError(f'Expected the 64-map pilot, found {len(paths)} maps.')
    original_vectors = []
    rounded_vectors = []
    maximum_shear_error = 0.
    for path in paths:
        original = np.load(
            file=path,
            mmap_mode='r',
        )
        if original.dtype != np.float64:
            raise ValueError(f'The storage reference must remain float64: {path}')
        # This is the full write/read rounding operation. Casting back to
        # float64 cannot recover lost digits, but lets subsequent arithmetic
        # use the same precision and summation path as the reference.
        rounded = original.astype(np.float32).astype(np.float64)
        maximum_shear_error = max(
            maximum_shear_error,
            np.max(np.abs(rounded-original)),
        )
        for shear, collection in (
            (original, original_vectors),
            (rounded, rounded_vectors),
        ):
            values = first @ shear[:, 0].T + second @ shear[:, 1].T
            values = values.T.reshape(4, 2, 64)
            collection.append(np.concatenate([
                (values**2).mean(axis=2).ravel(),
                (values**3).mean(axis=2).ravel(),
            ]))
    original_vectors = np.asarray(original_vectors)
    rounded_vectors = np.asarray(rounded_vectors)
    covariance = np.cov(
        m=original_vectors,
        rowvar=False,
        ddof=1,
    )
    rounded_covariance = np.cov(
        m=rounded_vectors,
        rowvar=False,
        ddof=1,
    )
    scale = np.sqrt(np.diag(covariance))
    difference = rounded_vectors-original_vectors
    inverse = np.linalg.solve(
        a=np.linalg.cholesky(covariance),
        b=np.eye(16),
    )
    white_difference = difference @ inverse.T
    matrix_change = inverse @ (rounded_covariance-covariance) @ inverse.T
    matrix_eigenvalues = np.linalg.eigvalsh(matrix_change)
    # Relative precision eigenvalues are 1/(1+lambda) in the covariance's
    # own coordinates. This checks all directions, including weak ones.
    inverse_change = 1./(1.+matrix_eigenvalues)-1.
    results = np.array([
        maximum_shear_error,
        np.max(np.abs(difference/scale)),
        np.max(np.linalg.norm(white_difference, axis=1)),
        np.max(np.abs(np.diag(rounded_covariance)/np.diag(covariance)-1.)),
        np.max(np.abs(matrix_eigenvalues)),
        np.max(np.abs(inverse_change)),
    ])
    labels = [
        'max_absolute_shear_rounding',
        'max_moment_change_in_its_single_map_standard_deviation',
        'max_joint_covariance_weighted_vector_change',
        'max_fractional_variance_change',
        'max_relative_covariance_eigendirection_change',
        'max_relative_precision_eigendirection_change',
    ]
    output = OUTPUT / 'float32_storage_check.txt'
    output.write_text('\n'.join(f'{label} {value:.17e}'
                                for label, value in zip(labels, results))+'\n')
    np.savetxt(
        fname=OUTPUT / 'float32_moment_differences.txt',
        X=difference,
        header='64 maps x 16 raw aperture moments; float32 decoded to float64 minus float64',
    )
    print(output.read_text(), end='')
    print('No per-row relative cubic errors: cubes near zero make that ratio misleading.')


if __name__ == '__main__':
    main()
