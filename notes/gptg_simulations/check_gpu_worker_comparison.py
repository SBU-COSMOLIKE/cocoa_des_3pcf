"""Check worker-count experiments against the original saved CPU maps.

Run after benchmark_gpu_sht_mpi.py --compare-gpu-workers. The comparison
uses identical map seeds and frozen calibration. It checks pixels and the
small aperture journal, not the full TreeCorr data vector. All generated
arrays remain under chains/. No production map is changed.
"""
import argparse
from pathlib import Path
import numpy as np
from generate_maps import PROJECT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--pipeline', action='store_true')
    args = parser.parse_args()
    reference_folder = PROJECT / 'chains/gptg_ensemble_15000'
    reference = np.loadtxt(reference_folder / 'comparison_measurements.txt')
    reports = []
    modes = ('sequential', 'pipeline') if args.pipeline else ('gpu1', 'gpu2')
    for mode in modes:
        folder = args.folder / mode
        measured = np.loadtxt(folder / 'comparison_measurements.txt')
        numbers = measured[:, 0].astype(int)
        np.testing.assert_array_equal(numbers, np.arange(len(numbers)))
        np.testing.assert_array_equal(measured[:, :2], reference[numbers, :2])
        pixel_errors = []
        for number in numbers:
            filename = f'GPTG_DES_{number:06d}.npy'
            original = np.load(reference_folder / 'comparison' / filename).astype(float)
            candidate = np.load(folder / 'comparison' / filename).astype(float)
            assert np.isfinite(candidate).all()
            # Compare each source and component separately: averaging over
            # bins could conceal an error concentrated in one source bin.
            error = np.sqrt(np.mean((candidate-original)**2, axis=-1)
                            / np.mean(original**2, axis=-1))
            pixel_errors.append(np.max(error))
        # Scale moment errors by the realization-to-realization scatter.
        # Fractional errors divided by an almost-zero cubic moment would
        # misrepresent an otherwise tiny numerical difference.
        scatter = np.std(reference[numbers, 4:], axis=0, ddof=1)
        moment_error = np.max(np.abs(measured[:, 4:]-reference[numbers, 4:])
                              / scatter)
        maximum = max(pixel_errors)
        assert np.isfinite(moment_error)
        assert max(maximum, moment_error) < 1.e-4
        reports.append(f'{mode}: {len(numbers)} maps; max component pixel RMS '
                       f'{maximum:.10e}; max moment/scatter {moment_error:.10e}')
    # Scheduling changes which rank receives a seed. Compare only seeds
    # evaluated on GPU in both cases to isolate scheduling from CPU/GPU
    # rounding differences. Pixel identity also checks the noise stream.
    gpu_ids = []
    for mode in modes:
        lines = (args.folder / mode / 'arithmetic_workers.txt').read_text().splitlines()
        gpu_ids.append({int(line.split()[1]) for line in lines
                        if line and not line.startswith('#')
                        and line.split()[-1].startswith('metal_sht')})
    common = sorted(gpu_ids[0] & gpu_ids[1])
    if not common:
        raise AssertionError('No common GPU seeds; increase the comparison count')
    for number in common:
        filename = f'GPTG_DES_{number:06d}.npy'
        left = np.load(args.folder / modes[0] / 'comparison' / filename)
        right = np.load(args.folder / modes[1] / 'comparison' / filename)
        np.testing.assert_array_equal(left, right)
    reports.append(f'{len(common)} common GPU maps are pixel-identical across schedules')
    text = '\n'.join(reports) + '\n' 
    (args.folder / 'numerical_checks.txt').write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
