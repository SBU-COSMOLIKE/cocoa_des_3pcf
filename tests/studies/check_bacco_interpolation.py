"""Measure interpolation error in the paired BACCO matter responses.

The power response is P_hydro/P_DMO; the bispectrum response is B_hydro/B_DMO.
This study compares native table reads between their sample nodes with direct
emulator evaluations at the same physical coordinates. Both calculations use
the declared neutrino conversion and continuation beyond emulator bounds.
Agreement therefore tests interpolation, not those physical assumptions.

Eight triangle shapes at six redshifts give 48 deterministic samples. Each
triangle also supplies three power queries, one at each side in h/Mpc.
Separate cases refine redshift, scale, two shape coordinates, and the dense
native table. The last case multiplies its boost by 1.5: the example's boost
changes from two to three, limiting the memory of a four-dimensional table.

Run from the project directory after activating Cocoa:
  python tests/studies/check_bacco_interpolation.py

The default configuration is configs/des_y3_bacco_fisher.yaml and its saved
CAMB input is runs/y3_bacco_physics.npz. No CAMB solve, projection, Fisher
calculation or timing runs here. Cases execute sequentially with one OpenMP
worker and one BLAS worker. Arrays and a JSON summary are written under
runs/bacco_interpolation; an existing result is refused. Relative response
errors do not by themselves establish the accuracy of a projected observable.
"""

import argparse
import json
import os
from pathlib import Path
import sys


# These limits apply before numerical libraries load. This is a response
# interpolation check, so it needs neither an OpenMP team nor threaded BLAS.
for variable in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
    "OMP_THREAD_LIMIT",
):
    os.environ[variable] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

import numpy as np
from scipy.interpolate import RectBivariateSpline


PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / "scripts"))
import run_3pcf as driver

from des3pcf.accuracy import accuracy_settings
from des3pcf.bacco_feedback import (
    BaccoFeedback,
    BaccoSettings,
    _cold_weights,
    continued_triangles,
    smooth_turn_on,
    total_bispectrum_response,
    total_power_response,
)
from des3pcf.baryon_scenarios import BaccoColdResponses
from des3pcf.inputs import load_tables
from des3pcf.mean import physics_model
from des3pcf.provenance import bacco_identity, file_digest, implementation_identity


def sample_points(settings, maximum_z):
    """Choose triangles on both sides of the response continuation boundaries.

    For sorted sides a<=b<=c, the native shape coordinates are u=a/b and
    v=(c-b)/a. Thus a=b*u and c=b+a*v generate valid triangles directly.
    The squeezed example has one side below the saved CAMB k range; the
    all-soft example has every side below the emulator's low anchor.

    Arguments:
      settings = BaccoSettings defining the Fourier anchors and table bounds.
      maximum_z = largest redshift in the supplied CAMB table.
    Returns:
      sides[48,3] in h/Mpc, redshifts[48], and string regime labels[48].
    Raises:
      ValueError if a chosen query lies outside the native response table.
    """
    # Fractions avoid the regular redshift nodes. The last query also tests
    # the z>3 continuation when the example table ends at z=3.1.
    fractions = np.asarray(a=[.023, .137, .297, .529, .877, .983])
    shapes = (
        ("all_soft", .235 * settings.low_k_anchor, .47, .31),
        ("low_anchor", .965 * settings.low_k_anchor, .83, .07),
        ("moderate_scale", .173, .29, .61),
        ("nonlinear_scale", 1.73, .61, .37),
        ("high_anchor", .831 * settings.high_k_anchor, .53, .41),
        ("high_scale", 3.71 * settings.high_k_anchor, .79, .67),
        ("squeezed", 3.17, 3.7e-7, .43),
        ("nearly_flattened", .713, .37, .9973),
    )
    sides = []
    redshifts = []
    labels = []
    for fraction in fractions:
        for label, middle, ratio, opening in shapes:
            shortest = middle * ratio
            sides.append([shortest, middle, middle + shortest * opening])
            redshifts.append(float(fraction * maximum_z))
            labels.append(label)
    sides = np.asarray(a=sides, dtype=float)
    redshifts = np.asarray(a=redshifts, dtype=float)
    ratios = sides[:, 0] / sides[:, 1]
    if (np.any(np.log(sides) < settings.logk_min)
            or np.any(np.log(sides) > settings.logk_max)
            or np.any(np.log(ratios) < settings.logu_min)):
        raise ValueError(
            "Study queries exceed the native response bounds; widen those bounds "
            "before comparing interpolation instead of endpoint clamping."
        )
    return sides, redshifts, np.asarray(a=labels)


def direct_responses(settings, tables, response_record, sides, redshifts):
    """Evaluate continued, converted emulator ratios without a response table.

    Only the input CAMB spectra are interpolated here. The emulator sees
    each physical query after its declared domain mapping, rather than a
    regular grid of response samples. Shared conversion helpers keep this
    a test of interpolation; independent component algebra has unit tests.

    Arguments:
      settings = selected BaccoSettings; tables = saved DMO/CAMB arrays.
      response_record = validated producer coordinates and neutrino fraction.
      sides = [n,3] physical triangle sides in h/Mpc.
      redshifts = [n] physical query redshifts within the CAMB interval.
    Returns:
      total-matter power ratios[n,3] and bispectrum ratios[n].
    Side effects:
      Loads installed BACCO networks and evaluates one batch per redshift.
    """
    emulator = BaccoColdResponses(coordinates=response_record["coordinates"])
    fraction = response_record["neutrino_fraction"]
    source_logk = np.log(tables["k"])
    nonlinear = RectBivariateSpline(
        x=tables["z"], y=source_logk, z=np.log(tables["p_nonlinear"]),
    )
    neutrino = None
    if fraction > 0.0:
        neutrino = RectBivariateSpline(
            x=tables["z"], y=source_logk, z=np.log(tables["p_linear_nu"]),
        )
    power = np.empty(shape=sides.shape)
    bispectrum = np.empty(shape=redshifts.shape)
    for redshift in np.unique(ar=redshifts):
        indices = np.flatnonzero(a=redshifts == redshift)
        physical = sides[indices]
        mapped, taper = continued_triangles(sides=physical, settings=settings)
        clipped = np.clip(
            a=physical.ravel(), a_min=settings.low_k_anchor,
            a_max=settings.high_k_anchor,
        )
        wave, inverse = np.unique(ar=clipped, return_inverse=True)
        cold_power, cold_bispectrum = emulator.evaluate(
            z=min(float(redshift), 3.0), k=wave, sides=mapped,
        )
        # P is continued separately on each physical leg. Using the three
        # rescaled B sides here would change the chosen mixed-neutrino model.
        cold_power = cold_power[inverse].reshape(physical.shape)
        fade = 1.0
        if settings.high_z_mode == "fade_to_unity":
            fade = 1.0 - smooth_turn_on(
                value=(redshift - 3.0) / settings.high_z_width,
            )
        power_taper = smooth_turn_on(value=physical / settings.low_k_anchor)
        cold_power = np.exp(np.log(cold_power) * power_taper * fade)
        cold_bispectrum = np.exp(np.log(cold_bispectrum) * taper * fade)
        weights = _cold_weights(
            k=physical, z=redshift, fraction=fraction, nonlinear=nonlinear,
            neutrino=neutrino, source_logk=source_logk,
        )
        power[indices] = total_power_response(
            cold_response=cold_power, cold_weight=weights,
            mode=settings.neutrino_model,
        )
        bispectrum[indices] = total_bispectrum_response(
            cold_response=cold_bispectrum, leg_power=cold_power,
            cold_weight=weights, mode=settings.neutrino_model,
        )
    return power, bispectrum


def error_summary(relative_error):
    """Summarize signed native/direct minus one residuals without a pass threshold.

    Arguments:
      relative_error = [sample,leg] power errors or [sample] bispectrum errors.
    Returns:
      maximum absolute error, root mean square, and the worst array index.
    """
    magnitude = np.abs(relative_error)
    worst = np.unravel_index(
        indices=np.argmax(a=magnitude), shape=magnitude.shape,
    )
    index = []
    for value in worst:
        index.append(int(value))
    return {
        "maximum_absolute": float(np.max(a=magnitude)),
        "root_mean_square": float(np.sqrt(np.mean(a=relative_error**2))),
        "worst_index": index,
    }


def run(configuration, output):
    """Save single-axis interpolation comparisons at one declared cosmology.

    Arguments:
      configuration = BACCO YAML with an existing saved CAMB table.
      output = directory receiving samples.npz and summary.json.
    Returns:
      JSON-compatible record of the settings, dependencies, and response errors.
    Raises:
      ValueError for a non-BACCO configuration or changed input dependencies.
      FileExistsError if either output already exists.
    Side effects:
      Runs emulator preparation and native response reads sequentially; writes
      only the two study outputs. No native model survives into the next case.
    """
    config, path, configuration_digest = driver.load_configuration(
        path=configuration, command="evaluate",
    )
    if config.get("feedback", {}).get("mode") != "bacco":
        raise ValueError("This study requires feedback.mode=bacco; choose its YAML.")
    output = Path(output).resolve()
    for name in ("samples.npz", "summary.json"):
        if (output / name).exists():
            raise FileExistsError(f"{output / name} exists; choose another output.")
    table_path = driver.resolve_path(
        configuration_path=path, value=config["physics"]["tables"],
    )
    tables, metadata = load_tables(path=table_path)
    settings = BaccoSettings(**config["feedback"]["settings"])
    baseline = accuracy_settings(value=config["accuracy"])
    sides, redshifts, regimes = sample_points(
        settings=settings, maximum_z=float(tables["z"][-1]),
    )
    # A global boost changes several grids together. These cases change
    # exactly one factor relative to the same baseline configuration.
    cases = {"baseline": baseline}
    for coordinate in ("z", "k", "u", "v", "table"):
        changed = baseline.record()
        name = f"feedback_{coordinate}_boost"
        multiplier = 2.0
        if coordinate == "table":
            multiplier = 1.5
        changed[name] *= multiplier
        cases[coordinate] = accuracy_settings(value=changed)
    config["threads"] = 1
    compiled = driver.compiled_extension(config=config)
    dependency_paths = {
        "configuration": path,
        "physical_tables": table_path,
        "study_script": Path(__file__).resolve(),
        "native_extension": Path(compiled.__file__).resolve(),
    }
    dependencies = {}
    for name, filename in dependency_paths.items():
        dependencies[name] = {
            "path": str(filename),
            "sha256": file_digest(path=filename),
        }
    bacco = bacco_identity()
    implementation = implementation_identity(
        project=PROJECT, three_point_path=compiled.__file__, use_camb=False,
    )
    producer = BaccoFeedback(settings=settings)
    reference_power = None
    reference_bispectrum = None
    power_values = []
    bispectrum_values = []
    records = []
    for name, controls in cases.items():
        print(f"Preparing BACCO interpolation case {name}", flush=True)
        pair = producer.prepare(
            parameters=config["parameters"], tables=tables, metadata=metadata,
            accuracy=controls,
        )
        response_record = pair.record()
        if reference_power is None:
            reference_power, reference_bispectrum = direct_responses(
                settings=settings, tables=tables, response_record=response_record,
                sides=sides, redshifts=redshifts,
            )
        physical = config["physics"]
        model = physics_model(
            compiled=compiled, tables=tables, feedback=pair, accuracy=controls,
            kernel_nz=physical["kernel_nz"], table_boost=physical["table_boost"],
            fit_nz=physical["fit_nz"], source_readers=physical["source_readers"],
        )
        power = model.power_response(
            k=sides.ravel(), z=np.repeat(a=redshifts, repeats=3), amplitude=1.0,
        ).reshape(sides.shape)
        bispectrum = model.bispectrum_response(
            sides=sides, redshifts=redshifts, amplitude=1.0,
        )
        # Keep only the small returned arrays, not every dense native table.
        power_values.append(power.copy())
        bispectrum_values.append(bispectrum.copy())
        del model
        del pair
        records.append({
            "name": name,
            "accuracy": controls.record(),
            "counts": response_record["counts"],
            "native_boost": controls.count(
                baseline=settings.boost, stage="feedback_table_boost",
            ),
            "power_relative_error": error_summary(
                relative_error=power / reference_power - 1.0,
            ),
            "bispectrum_relative_error": error_summary(
                relative_error=bispectrum / reference_bispectrum - 1.0,
            ),
        })
        print(json.dumps(obj=records[-1], sort_keys=True), flush=True)
    # File equality is checked again before saving a record that names them.
    # It does not attest to memory if somebody edits already imported code.
    for name, filename in dependency_paths.items():
        if file_digest(path=filename) != dependencies[name]["sha256"]:
            raise ValueError(f"{filename} changed during the study; rerun unchanged.")
    if bacco_identity() != bacco:
        raise ValueError("BACCO files changed during the study; restart and rerun.")
    completed_implementation = implementation_identity(
        project=PROJECT, three_point_path=compiled.__file__, use_camb=False,
    )
    if completed_implementation != implementation:
        raise ValueError("Response preparation code changed; restart and rerun.")
    power_values = np.asarray(a=power_values)
    bispectrum_values = np.asarray(a=bispectrum_values)
    report = {
        "configuration_sha256": configuration_digest,
        "dependencies": dependencies,
        "bacco_implementation": bacco,
        "implementation": implementation,
        "parameters": config["parameters"],
        "settings": config["feedback"]["settings"],
        "neutrino_fraction": response_record["neutrino_fraction"],
        "sample_count": int(redshifts.size),
        "power_queries_per_sample": 3,
        "errors": "native/direct minus one; array indices are zero-based",
        "cases": records,
        "interpretation": (
            "Response interpolation only; shared continuation and neutrino closure "
            "are not independently validated, and no survey accuracy is claimed."
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    # Exclusive creation protects earlier results even if another process
    # reaches the same output between preflight and this write.
    with (output / "samples.npz").open(mode="xb") as stream:
        np.savez_compressed(
            file=stream, case_names=np.asarray(a=list(cases)),
            sides=sides, redshifts=redshifts, regimes=regimes,
            reference_power=reference_power, reference_bispectrum=reference_bispectrum,
            native_power=power_values, native_bispectrum=bispectrum_values,
            power_relative_error=power_values / reference_power - 1.0,
            bispectrum_relative_error=bispectrum_values / reference_bispectrum - 1.0,
        )
    with (output / "summary.json").open(mode="x", encoding="utf-8") as stream:
        json.dump(obj=report, fp=stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return report


def main():
    """Read paths and launch the sequential interpolation study.

    Returns:
      nothing; reports the saved JSON path after both outputs are written.
    Side effects:
      Runs the physical response calls described in run; never changes inputs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configuration", type=Path,
        default=PROJECT / "configs/des_y3_bacco_fisher.yaml",
    )
    parser.add_argument(
        "--output", type=Path, default=PROJECT / "runs/bacco_interpolation",
    )
    arguments = parser.parse_args()
    run(configuration=arguments.configuration, output=arguments.output)
    print(arguments.output.resolve() / "summary.json")


if __name__ == "__main__":
    main()
