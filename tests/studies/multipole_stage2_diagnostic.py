"""Isolate b_L interpolation while holding the projected bispectrum fixed.

Run from the DES 3pcf project directory after compiling the optional residual
scaling mode. For example, use the Cocoa Python executable with
  /tmp/des3pcf_residual_scaling/multipole_stage2_diagnostic.py
  --output chains/multipole_stage2_diagnostic.npz
The existing demonstration configuration selects the stored physical tables;
this script does not request CAMB or change any source-preparation policy.

Each angular reference integrates the piecewise-linear grid.projected(mu)
against every retained Legendre polynomial. Gauss quadrature on each segment
is exact for that polynomial product. Thus the reference shares the declared
projected-B approximation but does not use the prepared b_L table. Differences
from grid.multipoles isolate its two interpolation axes and dense upsampling.
This is not an independent test of the physical bispectrum or its mu cutoff.
The fixed projected grid uses residual_scaling=1, nshape=160, and the optional
shape stretch matched to its smallest supported fraction. Every saved case
records those choices; only the b_L interpolation controls change in the scan.
The default case retains psi_scale=0. The mixed case uses psi_scale=0.02,
with each refinement then changing one count relative to that mixed chart.
"""

import argparse
import json
import os
from pathlib import Path
import sys

# Set limits before NumPy, SciPy, the project driver, or native libraries load.
for environment_name in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
):
    os.environ[environment_name] = "1"
os.environ["OMP_THREAD_LIMIT"] = "4"

import numpy as np
from scipy.special import eval_legendre
from threadpoolctl import threadpool_info, threadpool_limits


def arguments():
    """Require one new output and select independent interpolation scans."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/des_y3_tatt_demo_fisher.yaml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nshape", type=int, default=160)
    parser.add_argument("--shape-stretch", type=float, default=None)
    parser.add_argument("--psi-scale", type=float, default=0.02)
    parser.add_argument(
        "--cases",
        nargs="+",
        choices=(
            "default",
            "mixed",
            "ell_refined",
            "psi_refined",
            "dense_refined",
        ),
        default=[
            "default",
            "mixed",
            "ell_refined",
            "psi_refined",
            "dense_refined",
        ],
    )
    return parser.parse_args()


def query_pairs():
    """Return fixed Fourier pairs in both ordered angular charts.

    ELL=sqrt(ell1 squared+ell2 squared) is a scale, and PSI=atan2(ell2,ell1)
    fixes their ratio. Each pair is repeated with the sides exchanged while
    the source labels remain [0,1,0]. Unequal-source TATT cannot infer that
    second chart by transposing the bispectrum from the first one.
    """
    coordinates = []
    sides = []
    for magnitude in (100.0, 1000.0, 10000.0, 100000.0):
        for angle in (0.17, 0.43, 0.68):
            first = magnitude * np.cos(angle)
            second = magnitude * np.sin(angle)
            coordinates.append([magnitude, angle, 0.0])
            sides.append([first, second])
            coordinates.append([magnitude, angle, 1.0])
            sides.append([second, first])
    return np.asarray(coordinates), np.asarray(sides)


def exact_linear_reference(mu, sampled, lmax):
    """Integrate each linear angular segment and supply a zero-safe scale.

    Arguments:
      mu = increasing angular knots used by the projected-grid preparation.
      sampled = signed projected B at those knots, shape [nmu].
      lmax = highest Legendre degree, including degree zero.
    Returns:
      (coefficients[lmax+1], angular_scale), where angular_scale is exactly
      one half of the integral of abs(linear B). It is at least abs(b_0),
      so a small signed multipole does not produce a misleading relative
      error. This is an amplitude scale, not a data-covariance weighting.
    """
    degree = (lmax + 3) // 2
    nodes, weights = np.polynomial.legendre.leggauss(deg=degree)
    result = np.zeros(shape=lmax + 1, dtype=np.float64)
    absolute_integral = 0.0
    for interval in range(mu.size - 1):
        width = mu[interval + 1] - mu[interval]
        fraction = 0.5 * (1.0 + nodes)
        coordinate = mu[interval] + fraction * width
        first = sampled[interval]
        second = sampled[interval + 1]
        values = first + fraction * (second - first)
        for order in range(lmax + 1):
            polynomial = eval_legendre(order, coordinate)
            result[order] += (
                (order + 0.5) * 0.5 * width
                * np.sum(weights * values * polynomial)
            )
        amplitude = abs(first) + abs(second)
        if first * second < 0.0:
            # Two triangular areas meet at the one linear zero crossing.
            absolute_integral += 0.5 * width * (
                abs(first) * (abs(first) / amplitude)
                + abs(second) * (abs(second) / amplitude)
            )
        else:
            absolute_integral += 0.5 * width * amplitude
    return result, 0.5 * absolute_integral


def evaluate_grid(grid, pairs, lmax):
    """Compare prepared coefficients with direct integration of their B grid."""
    mu = grid.axes()["mu"]
    count = pairs.shape[0]
    actual = np.zeros(shape=(2, count, lmax + 1), dtype=np.float64)
    reference = np.zeros_like(actual)
    angular_scale = np.zeros(shape=(2, count), dtype=np.float64)
    samples = np.zeros(shape=(2, count, mu.size), dtype=np.float64)
    for pair_index in range(count):
        first, second = pairs[pair_index]
        ell = np.asarray([first, second], dtype=np.float64)
        closing = np.sqrt(
            (first - second)**2 + 2.0 * first * second * (1.0 - mu)
        )
        triangles = np.column_stack(
            (
                np.full(shape=mu.size, fill_value=first),
                np.full(shape=mu.size, fill_value=second),
                closing,
            )
        )
        for component_index, component in enumerate((1, 2)):
            prepared = grid.multipoles(ell=ell, component=component)
            actual[component_index, pair_index] = prepared[:, 0, 1]
            sampled = grid.projected(sides=triangles, component=component)
            samples[component_index, pair_index] = sampled
            coefficients, scale = exact_linear_reference(
                mu=mu,
                sampled=sampled,
                lmax=lmax,
            )
            reference[component_index, pair_index] = coefficients
            angular_scale[component_index, pair_index] = scale
    return actual, reference, angular_scale, samples, mu


def main():
    """Prepare fixed physical tables and save every interpolation comparison."""
    options = arguments()
    destination = options.output.resolve()
    report_path = destination.with_suffix(".json")
    if destination.suffix != ".npz":
        raise ValueError("The diagnostic output must have the .npz suffix.")
    if destination.exists() or report_path.exists():
        raise FileExistsError(
            "Choose new diagnostic NPZ and JSON output paths."
        )
    project = Path.cwd().resolve()
    if not (project / "scripts/run_3pcf.py").is_file():
        raise ValueError("Run this script from Cocoa/projects/des_3pcf.")
    sys.path.insert(0, str(project / "scripts"))
    sys.path.insert(0, str(project.parents[1] / "external_modules/code/cosmolike_core"))
    import run_3pcf as driver

    threadpool_limits(limits=1, user_api="blas")
    config, path, digest = driver.load_configuration(
        path=options.config.resolve(),
        command="evaluate",
    )
    # This scan never changes cosmology. Explicitly forbid a producer
    # solve even if the input Fisher configuration normally permits CAMB.
    config["physics"] = dict(config["physics"])
    config["physics"]["cosmology_mode"] = "fixed_tables"
    config["threads"] = 1
    interpolation = config["physics"].get(
        "source_interpolation", "linear_midpoint",
    )
    if interpolation != "linear_midpoint":
        raise ValueError("This comparison requires the legacy source curves.")
    compiled = driver.compiled_extension(config=config)
    compiled.set_omp_threads(nthreads=1)
    for pool in threadpool_info():
        if pool["user_api"] == "blas" and pool["num_threads"] != 1:
            raise RuntimeError("BLAS must remain serial for this diagnostic.")
    provider, _ = driver.provider_for_run(
        compiled=compiled,
        config=config,
        path=path,
    )
    parameters = dict(config["parameters"])
    parameters.update(config["ia"])
    model, _, _ = provider.get(parameters=parameters)
    bins = np.asarray([0, 1, 0], dtype=np.int32)
    fixed_settings = dict(
        config["mean"]["aperture_settings"]["projection_settings"]
    )
    fixed_settings["residual_scaling"] = 1
    fixed_settings["prepare_multipoles"] = 1
    fixed_settings["nshape"] = options.nshape
    shape_stretch = options.shape_stretch
    if shape_stretch is None:
        minimum = min(
            1.0e-6,
            0.25 * fixed_settings["ell_min"] / fixed_settings["ell_max"],
        )
        shape_stretch = 0.5 * np.log(0.5 / minimum)
    fixed_settings["shape_stretch"] = shape_stretch
    lmax = fixed_settings["lmax"]
    cases = {
        "default": {"psi_scale": 0.0},
        "mixed": {"psi_scale": options.psi_scale},
        "ell_refined": {
            "psi_scale": options.psi_scale,
            "nell": 2 * fixed_settings["nell"] - 1,
        },
        "psi_refined": {
            "psi_scale": options.psi_scale,
            "npsi": 2 * fixed_settings["npsi"] - 1,
        },
        "dense_refined": {
            "psi_scale": options.psi_scale,
            "multipole_boost": 2 * fixed_settings["multipole_boost"],
        },
    }
    coordinates, pairs = query_pairs()
    arrays = {"coordinates_ELL_PSI_chart": coordinates, "ell_pairs": pairs}
    summaries = []
    retained_reference = None
    retained_samples = None
    retained_mu = None
    for name in options.cases:
        settings = dict(fixed_settings)
        settings.update(cases[name])
        grid = model.projection_grid(
            bins=bins,
            parameters=config["ia"],
            settings=settings,
        )
        actual, reference, scale, sampled, mu = evaluate_grid(
            grid=grid,
            pairs=pairs,
            lmax=lmax,
        )
        if retained_reference is None:
            retained_reference = reference.copy()
            retained_samples = sampled.copy()
            retained_mu = mu.copy()
        else:
            # A changed reference would confound b_L interpolation with
            # a changed physical projected grid or angular integration.
            np.testing.assert_array_equal(
                reference.view(dtype=np.uint64),
                retained_reference.view(dtype=np.uint64),
            )
            np.testing.assert_array_equal(
                sampled.view(dtype=np.uint64),
                retained_samples.view(dtype=np.uint64),
            )
            np.testing.assert_array_equal(
                mu.view(dtype=np.uint64),
                retained_mu.view(dtype=np.uint64),
            )
        if np.any(scale <= 0.0):
            raise ValueError("A diagnostic component has zero angular norm.")
        difference = actual - reference
        normalized = difference / scale[:, :, None]
        arrays[f"{name}_actual"] = actual
        arrays[f"{name}_difference"] = difference
        arrays[f"{name}_normalized_difference"] = normalized
        summary = {
            "case": name,
            "settings": settings,
            "maximum_absolute_error_by_component": np.max(
                np.abs(difference), axis=(1, 2),
            ).tolist(),
            "maximum_angular_scaled_error_by_component": np.max(
                np.abs(normalized), axis=(1, 2),
            ).tolist(),
            "angular_scaled_error_per_pair": np.max(
                np.abs(normalized), axis=2,
            ).tolist(),
        }
        summaries.append(summary)
        print(json.dumps(summary, sort_keys=True), flush=True)
    arrays["reference"] = retained_reference
    arrays["projected_samples"] = retained_samples
    arrays["mu"] = retained_mu
    arrays["angular_scale"] = scale
    arrays["absolute_reference_b0"] = np.abs(retained_reference[:, :, 0])
    metadata = {
        "config_path": str(path),
        "config_sha256": digest,
        "execution_overrides": {
            "cosmology_mode": "fixed_tables",
            "threads": 1,
            "residual_scaling": 1,
            "nshape": options.nshape,
            "shape_stretch": shape_stretch,
        },
        "bins": bins.tolist(),
        "components": ["NLA baseline", "signed TATT residual"],
        "scale_definition": "0.5 * integral(abs(piecewise-linear B)) dmu",
        "reference_definition": "exact segmentwise B_linear times P_L integral",
        "scope": "b_L interpolation relative to a fixed projected B grid",
        "summaries": summaries,
        "threadpools": threadpool_info(),
    }
    arrays["metadata_json"] = np.asarray(json.dumps(metadata, sort_keys=True))
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open(mode="xb") as stream:
        np.savez_compressed(file=stream, **arrays)
    with report_path.open(mode="x") as stream:
        json.dump(obj=metadata, fp=stream, indent=2, sort_keys=True)
        stream.write("\n")


if __name__ == "__main__":
    main()
