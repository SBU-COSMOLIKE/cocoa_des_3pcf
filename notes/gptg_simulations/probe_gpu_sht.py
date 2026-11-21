"""Probe Apple GPU primitives and S2FFT without permitting CPU fallback.

This is a compatibility test, not a production transform implementation.
A successful CPU S2FFT transform does not establish GPU support. Explicit
MPS tensors and the reported output device prevent that confusion.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'

import platform
import numpy as np
import torch
import mlx.core as mx


def main():
    torch.set_num_threads(1)
    print('macOS', platform.mac_ver()[0])
    print('torch', torch.__version__, 'MPS available', torch.backends.mps.is_available())
    print('Metal', mx.metal.device_info())
    for dtype in (torch.float32, torch.complex64):
        try:
            values = torch.ones((16,), dtype=dtype, device='mps')
            transformed = torch.fft.fft(values)
            torch.mps.synchronize()
            print('torch FFT', dtype, transformed.device)
        except Exception as error:
            print('torch FFT', dtype, type(error).__name__, str(error))
    # MLX has its own Metal FFT implementation. Failure of Torch's MPS FFT
    # must not be generalized to every Apple GPU programming route.
    with mx.stream(mx.gpu):
        values = mx.array(np.arange(16, dtype=np.float32))
        transformed = mx.fft.fft(values)
        mx.eval(transformed)
        np.testing.assert_allclose(np.array(transformed), np.fft.fft(np.arange(16)),
                                   rtol=1e-5, atol=1e-5)
        print('MLX Metal FFT passed against NumPy')
    try:
        import s2fft
        from s2fft.precompute_transforms import spherical
        # Test an actual HEALPix inverse at a deliberately tiny band limit.
        # Production requires lmax=1024 and spin 0 AND spin 2 transforms.
        coefficients = torch.zeros((4, 7), dtype=torch.complex64, device='mps')
        coefficients[0, 3] = 1.
        result = spherical.inverse(
            flm=coefficients,
            L=4,
            nside=2,
            sampling='healpix',
            spin=0,
            reality=True,
            method='torch',
        )
        print('S2FFT result device', result.device, 'shape', result.shape)
    except Exception as error:
        print('S2FFT GPU inverse', type(error).__name__, str(error))


if __name__ == '__main__':
    main()
