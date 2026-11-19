"""Check the bridge's distance derivative against an analytic geometry.

The chosen distance is chi=L*z/(1+q*z), so its exact inverse derivative
is dz/dchi=(1+q*z)^2/L. This is a controlled geometry, not a background
solution of the Friedmann equation for the model's density parameters.
No Python implementation of the native finite-difference stencil supplies
the expected answer. A tiny C observer, compiled against the public header,
copies the actual bridge input before its interpolation into a model.

The IA checks compare two models with identical sampled inputs and fixed
source/line-of-sight grids. Only the direct constructor receives the exact
Jacobian. This isolates the observable effect of adopting the bridge's
derivative. A separate exact-linear distance checks units and establishes
the error from the other input conversions. These toy errors do not bound
the errors of a DES prediction with a measured source distribution.
"""

from test_supplied_baseline import baseline_input
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pytest
import yaml

from studies.bihalofit_refinement import pin_inputs


PROJECT = Path(__file__).resolve().parents[1]
CORE = PROJECT.parents[1] / "external_modules/code/cosmolike_core"
DISTANCE_SCALE = 3000.0  # Mpc/h; chosen explicitly to test this unit boundary.
CURVATURE = 0.35  # Shape of the analytic distance, not spatial curvature.

# C owns the struct layout. Python receives only an opaque pointer and
# ordinary arrays: it never guesses field offsets, casts a hand-written
# struct, or replaces any native function. This observer links no numerical
# implementation; its only job is to copy checked public input fields.
OBSERVER_SOURCE = r"""
#include <math.h>
#include <stddef.h>
#include "cosmolike/3pcf/threepcf_core_bridge.h"

int copy_bridge_background(
  const physics_input_3pcf* input, // fresh bridge input; copy before any reset
  const int count, // capacity of each caller-owned output array
  double* z, // copied redshift nodes [count]
  double* chi, // copied distances [count], Mpc/h
  double* jacobian // copied inverse derivatives [count], h/Mpc
)
{
  if (NULL == input
      || NULL == z
      || NULL == chi
      || NULL == jacobian) {
    return 1;
  }
  if (count != input->nz
      || NULL == input->z
      || NULL == input->chi
      || NULL == input->dz_dchi) {
    return 2;
  }
  for (int node=0; node<count; node++) {
    if (!isfinite(input->z[node])
        || !isfinite(input->chi[node])
        || !isfinite(input->dz_dchi[node])) {
      return 3;
    }
    z[node] = input->z[node];
    chi[node] = input->chi[node];
    jacobian[node] = input->dz_dchi[node];
  }
  return 0;
}
"""


class BackgroundObserver:
    """Own the two shared-library handles used for a checked input copy.

    The extension still owns the bridge input. read() is the first consumer
    after fresh table installation; a repeated refresh returns NULL. The
    pointer is used only during this call, and every returned value belongs
    to NumPy. Later bridge resets cannot invalidate the copy.
    Setting ctypes argument/return types describes the C calling convention;
    it does not replace executable functions or modify the extension.
    """

    def __init__(self, *, compiled, library, command):
        """Retain libraries and source identities for the test report.

        Arguments:
          compiled = the already imported project extension.
          library = temporary observer library built from OBSERVER_SOURCE.
          command = exact compiler command, retained for reproduction.
        """
        self.extension = ctypes.CDLL(name=compiled.__file__)
        self.observer = ctypes.CDLL(name=str(library))
        self.refresh = self.extension.threepcf_bridge_refresh
        self.refresh.argtypes = []
        self.refresh.restype = ctypes.c_void_p
        self.copy = self.observer.copy_bridge_background
        pointer = ctypes.POINTER(ctypes.c_double)
        self.copy.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            pointer,
            pointer,
            pointer,
        ]
        self.copy.restype = ctypes.c_int
        header = CORE / "cosmolike/3pcf/physics_3pcf.h"
        bridge_header = CORE / "cosmolike/3pcf/threepcf_core_bridge.h"
        self.identity = {
            "extension": compiled.__file__,
            "extension_sha256": hashlib.sha256(
                Path(compiled.__file__).read_bytes()).hexdigest(),
            "physics_header_sha256": hashlib.sha256(
                header.read_bytes()).hexdigest(),
            "bridge_header_sha256": hashlib.sha256(
                bridge_header.read_bytes()).hexdigest(),
            "test_source_sha256": hashlib.sha256(
                Path(__file__).read_bytes()).hexdigest(),
            "observer_source_sha256": hashlib.sha256(
                OBSERVER_SOURCE.encode()).hexdigest(),
            "compiler_command": command,
        }

    def read(self, *, count):
        """Copy z, chi and dz/dchi on the first refresh after installation.

        Arguments:
          count = expected number of background nodes, checked again in C.
        Returns:
          three owned arrays in the order z, chi, dz/dchi.
        Raises:
          AssertionError if the public bridge input fails the C checks.
        """
        redshift = np.empty(shape=count, dtype=np.float64)
        distance = np.empty(shape=count, dtype=np.float64)
        jacobian = np.empty(shape=count, dtype=np.float64)
        pointer = ctypes.POINTER(ctypes.c_double)
        # A C function has positional arguments. The compiler, rather than
        # Python, interprets the returned public-struct pointer.
        status = self.copy(
            self.refresh(), count, redshift.ctypes.data_as(pointer),
            distance.ctypes.data_as(pointer), jacobian.ctypes.data_as(pointer),
        )
        reasons = {
            1: "missing input or output pointer",
            2: "node count mismatch or missing background array",
            3: "nonfinite background sample",
        }
        assert status == 0, (
            f"Bridge copy failed: {reasons.get(status, 'unknown C status')} "
            f"(status={status}, expected nodes={count}). Check the installed "
            "tables and that the extension was built from these public headers."
        )
        return redshift, distance, jacobian


@pytest.fixture(scope="module")
def background_observer(compiled, tmp_path_factory):
    """Compile the read-only observer in pytest's temporary directory.

    The helper needs a C compiler and the public project headers, but no
    numerical libraries, package installation or native module rebuild.
    Compiler failure is a test failure, never an omitted numerical check.
    """
    compiler = os.environ.get("C_COMPILER", os.environ.get("CC", "cc"))
    command = shlex.split(s=compiler)
    if not command or shutil.which(cmd=command[0]) is None:
        raise FileNotFoundError(
            f"C compiler {compiler!r} is unavailable; set C_COMPILER to an installed "
            "compiler before running the analytic bridge test."
        )
    header = CORE / "cosmolike/3pcf/threepcf_core_bridge.h"
    if not header.is_file():
        raise FileNotFoundError(f"The test observer requires the public header {header}.")
    directory = tmp_path_factory.mktemp(basename="bridge_background_observer")
    source = directory / "observer.c"
    library = directory / "observer.so"
    source.write_text(OBSERVER_SOURCE)
    command.extend([
        "-std=c99",
        "-O0",
        "-fPIC",
    ])
    if sys.platform == "darwin":
        command.append("-dynamiclib")
    else:
        command.append("-shared")
    command.extend([
        "-I",
        str(CORE),
        str(source),
        "-o",
        str(library),
    ])
    subprocess.run(args=command, check=True, capture_output=True, text=True)
    return BackgroundObserver(compiled=compiled, library=library, command=command)


@pytest.fixture(autouse=True)
def clear_bridge_after_test(compiled):
    """Keep these global-state checks from contaminating a later test."""
    yield
    compiled.core.initial_setup()


def analytic_inputs(*, count, maximum=2.0, curvature=CURVATURE,
                    source_count=801, kernel_count=401, los_count=129):
    """Sample analytic inputs while keeping source/LOS grids independent.

    Arguments:
      count, maximum = background redshift count and upper boundary.
      curvature = q in chi=L*z/(1+q*z); zero gives an exactly linear distance.
      source_count, kernel_count, los_count = fixed auxiliary resolutions.
    Returns:
      complete direct-model inputs with the exact analytic inverse derivative.

    The source density is a normalized polynomial, proportional to
    z^2*(zmax-z)^2. Both endpoints vanish, as ordinary source densities do
    at the observer and beyond their support. Its analytic definition is
    unchanged when the background grid is refined. P and D are positive
    toy functions used solely to give the IA kernels a finite projection.
    """
    redshift = np.linspace(start=0.0, stop=maximum, num=count)
    wave = np.geomspace(start=1.0e-5, stop=1.0e3, num=96)
    linear = 2.0e4 * wave / (1.0 + (wave / 0.15)**2)**1.8
    growth = 1.0 / (1.0 + redshift)
    nonlinear0 = linear * (1.0 + (wave / 0.4)**2)**0.2
    source_z = np.linspace(start=0.0, stop=maximum, num=source_count)
    density = 30.0 * source_z**2 * (maximum - source_z)**2 / maximum**5
    return {
        "k": wave,
        "p_linear0": linear,
        "z": redshift,
        "p_nonlinear": growth[:, None]**2 * nonlinear0[None, :],
        "chi": DISTANCE_SCALE * redshift / (1.0 + curvature * redshift),
        "dz_dchi": (1.0 + curvature * redshift)**2 / DISTANCE_SCALE,
        "growth": growth,
        "sources": density[None, :],
        "source_z": source_z,
        "omega_m": 0.3,
        "omega_de": 0.7,
        "w0": -1.0,
        "wa": 0.0,
        "ns": 0.965,
        "sigma8": 0.8,
        "los_z": np.linspace(start=0.005, stop=maximum, num=los_count),
        "kernel_nz": kernel_count,
        "source_readers": False,
        "table_boost": 8,
        "fit_nz": 33,
    }


def bridge_parameters(*, inputs):
    """Return the four ordinary-core scalars installed for this geometry test."""
    return {
        "omega_matter": inputs["omega_m"],
        "omega_baryon": 0.05,
        "hubble": 70.0,
        "omega_nu_h2": 0.0,
    }


def background_baseline(compiled, inputs):
    """Supply a physical B so the Jacobian test retains its radial weights.

    A constant B overweights the observer endpoint and defines a different
    accuracy experiment. Use the Python fit with shared variance, as the
    former internal bridge did, and compare identical B in both projectors.
    """
    from des3pcf.bihalofit import BiHalofitProducer, bispectrum_axes

    snapshot = compiled.core.variance_snapshot(
        z=np.linspace(start=0.0, stop=inputs["z"][-1], num=33),
    )
    fit = {name: snapshot[name] for name in ("z", "rnl", "neff", "sigma8")}
    tables = dict(inputs)
    tables["omega_m_z"] = np.full(inputs["z"].shape, 0.3)
    producer = BiHalofitProducer(**bispectrum_axes(
        zmax=inputs["z"][-1], counts=(9, 65, 33, 17),
    ))
    return dict(producer.prepare(tables=tables, variance=fit,
                                 source_readers=True).native)


def install_background(*, compiled, inputs, construct_model=True):
    """Build the bridge model using public setters and ordinary h units.

    Arguments:
      compiled = extension whose global core state is reset here.
      inputs = analytic_inputs output; its exact dz/dchi is not installed.
      construct_model = false reserves the first refresh for the C observer.
    Returns:
      the immutable bridge model when requested; otherwise None. A raw-input
      test must copy the first refresh and reset before any Python factory
      call, because each fresh input has only one construction consumer.

    The independent source override avoids changing the source discretization
    when the background changes. Core power storage is ln(P), ordered by
    wavenumber then redshift; the distance table already uses Mpc/h.
    """
    core = compiled.core
    core.initial_setup()
    core.set_cosmological_parameters(**bridge_parameters(inputs=inputs))
    logwave = np.log10(inputs["k"])
    linear = inputs["growth"][:, None]**2 * inputs["p_linear0"][None, :]
    core.set_linear_power_spectrum(
        log10k=logwave, z=inputs["z"], lnP=np.log(linear).ravel(order="F"),
    )
    core.set_non_linear_power_spectrum(
        log10k=logwave, z=inputs["z"],
        lnP=np.log(inputs["p_nonlinear"]).ravel(order="F"),
    )
    core.set_distances(z=inputs["z"], chi=inputs["chi"])
    core.set_growth(z=inputs["z"], G=(1.0 + inputs["z"]) * inputs["growth"])
    core.set_sources(source_z=inputs["source_z"], sources=inputs["sources"])
    core.set_los(z=inputs["los_z"])
    core.set_resolution(
        kernel_nz=inputs["kernel_nz"], source_readers=inputs["source_readers"],
        table_boost=inputs["table_boost"], fit_nz=inputs["fit_nz"],
    )
    core.set_scalars(ns=0.965, sigma8=0.8, w0=-1.0, wa=0.0)
    core.set_baseline_bispectrum(
        baseline=background_baseline(compiled=compiled, inputs=inputs),
        omega_m_z=np.full(inputs["z"].shape, 0.3), sigma8=0.8,
    )
    if construct_model:
        return core.model_from_core()
    return None


def observable_inputs():
    """Return every query array and IA control for the saved input record."""
    return {
        "ell": np.array([50.0, 200.0, 800.0]),
        "pair": np.array([0, 0], dtype=np.int32),
        "sides": np.array([[80.0, 120.0, 150.0], [200.0, 300.0, 400.0]]),
        "triplet": np.array([0, 0, 0], dtype=np.int32),
        "nla_amplitudes": np.array([0.0, 2.0, -2.0]),
        "placement_amplitude": 2.0,
        "placement_operator": 1,
        "placement_masks": np.array([4, 7], dtype=np.int32),
        "tatt_a1": 0.7,
        "tatt_a2": 0.6,
        "tatt_bta": 0.8,
        "tatt_ct": 0.2,
        "tatt_power_mode": 2,
    }


def freeze_case(*, path, inputs):
    """Save and reload every physical input and query before either model runs.

    Returns:
      reloaded physical inputs, reloaded query inputs and an NPZ identity.
      Arrays, scalar cosmology values and numerical controls are all saved.
      The input hash identifies the complete file, including dtype and shape.
    """
    dictionaries = {
        "physical": inputs,
        "query": observable_inputs(),
        "bridge": bridge_parameters(inputs=inputs),
    }
    restored = pin_inputs(path=path, dictionaries=dictionaries, numpy=np)
    record = {
        "file": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "physical_fields": sorted(inputs),
        "query_fields": sorted(restored["query"]),
        "bridge_scalars": restored["bridge"],
    }
    return restored["physical"], restored["query"], record


def ia_observables(*, model, query):
    """Evaluate small II, GGI, III and quadratic-TATT projections.

    Arguments:
      model = one owned physical model on the fixed source/LOS grids.
      query = reloaded output of observable_inputs, including all IA controls.
    Returns:
      named arrays; gg is a control that should not depend on the Jacobian.

    C(A1)=GG+A1*GI+A1^2*II in NLA, so the even combination of A1=+2
    and -2 isolates the II contribution at amplitude two. The native field
    selector directly isolates GGI and III in the projected bispectrum.
    Quadratic TATT reads its separate source-grid shape kernel; checking its
    residual therefore tests a second consumer of the distance derivative.
    """
    ell = query["ell"]
    pair = query["pair"]
    amplitudes = query["nla_amplitudes"]
    gg = model.nla_power(ell=ell, bins=pair, parameters={"a1": amplitudes[0]})
    plus = model.nla_power(ell=ell, bins=pair, parameters={"a1": amplitudes[1]})
    minus = model.nla_power(ell=ell, bins=pair, parameters={"a1": amplitudes[2]})
    result = {
        "gg": gg,
        "ii": 0.5 * (plus + minus) - gg,
    }
    assert np.all(result["ii"] > 0.0), "The controlled II signal must be positive."
    sides = query["sides"]
    triplet = query["triplet"]
    for name, legs in zip(("ggi", "iii"), query["placement_masks"]):
        parameters = {
            "a1": query["placement_amplitude"],
            "operator_component": query["placement_operator"],
            "intrinsic_legs": int(legs),
        }
        result[name] = model.projected(
            sides=sides, bins=triplet, parameters=parameters,
        )
    parameters = {
        "a1": query["tatt_a1"],
        "a2": query["tatt_a2"],
        "bta": query["tatt_bta"],
        "ct": query["tatt_ct"],
        "power_mode": query["tatt_power_mode"],
    }
    result["quadratic_tatt"] = model.projected(
        sides=sides, bins=triplet, parameters=parameters, component=2,
    )
    return result


def compare_ia(*, compiled, inputs, query):
    """Compare matched models; only one receives the analytic Jacobian.

    Returns:
      per-observable maximum absolute difference divided by that observable's
      maximum reference magnitude, plus both arrays for later inspection.
      This normalization avoids a spurious ratio at a single zero crossing.

    The reference uses exact Jacobian samples but retains the same finite
    interpolation and quadrature as the bridge model. This comparison
    isolates the derivative choice; it is not a continuum projection test.
    """
    bridged = install_background(compiled=compiled, inputs=inputs)
    # Match smoothing explicitly so this comparison isolates the distance
    # derivative, rather than comparing two variance prescriptions.
    snapshot = compiled.core.variance_snapshot(
        z=np.linspace(start=0.0, stop=inputs["z"][-1], num=33))
    fit = {}
    for name in ("z", "rnl", "neff", "sigma8"):
        fit[name] = snapshot[name]
    direct = compiled.production.PhysicsModel(
        **inputs, baseline=background_baseline(compiled=compiled, inputs=inputs),
        omega_m_z=np.full(inputs["z"].shape, 0.3),
    )
    actual = ia_observables(model=bridged, query=query)
    reference = ia_observables(model=direct, query=query)
    result = {}
    for name in reference:
        scale = float(np.max(np.abs(reference[name])))
        assert scale > 0.0, f"The controlled {name} signal must be nonzero."
        error = float(np.max(np.abs(actual[name] - reference[name])) / scale)
        result[name] = {
            "relative_error": error,
            "bridge": actual[name].tolist(),
            "analytic_jacobian": reference[name].tolist(),
        }
    assert result["gg"]["relative_error"] < 3.0e-11
    return result


def save_record(*, path, identity, measurements):
    """Save finite JSON measurements with their binary/header identities."""
    record = {
        "scope": "analytic toy geometry; no bound on DES survey IA errors",
        "metric": "raw observables; no survey data, covariance or mask is used",
        "identity": identity,
        "measurements": measurements,
    }
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def test_linear_distance_has_exact_inverse_units(compiled, background_observer,
                                               tmp_path):
    """A linear distance checks absolute scale, including both endpoints."""
    inputs = analytic_inputs(count=41, curvature=0.0)
    inputs, query, pinned = freeze_case(path=tmp_path / "linear_inputs.npz",
                                       inputs=inputs)
    install_background(compiled=compiled, inputs=inputs, construct_model=False)
    redshift, distance, jacobian = background_observer.read(count=41)
    compiled.core.initial_setup()
    np.testing.assert_array_equal(x=redshift, y=inputs["z"])
    np.testing.assert_array_equal(x=distance, y=inputs["chi"])
    np.testing.assert_allclose(actual=jacobian, desired=1.0 / DISTANCE_SCALE,
                               rtol=2.0e-12, atol=0.0)
    # The bridge has already been reset. These independent NumPy arrays
    # therefore cannot accidentally keep reading its released scratch space.
    observables = compare_ia(compiled=compiled, inputs=inputs, query=query)
    measurements = {"inputs": pinned, "observables": observables}
    save_record(path=tmp_path / "linear.json", identity=background_observer.identity,
                measurements=measurements)
    for values in observables.values():
        assert values["relative_error"] < 3.0e-11


def test_curved_distance_derivative_refines(compiled, background_observer, tmp_path):
    """Interior errors decrease quadratically; the observer is only first order.

    The last endpoint uses the core's continued derivative rather than the
    first endpoint's forward slope. Report it separately: collapsing all
    nodes into one maximum would hide interior convergence behind z=0.
    """
    measurements = []
    for count in (41, 81, 161, 321):
        inputs = analytic_inputs(count=count)
        inputs, unused_query, pinned = freeze_case(
            path=tmp_path / f"jacobian_inputs_{count}.npz", inputs=inputs,
        )
        install_background(compiled=compiled, inputs=inputs, construct_model=False)
        redshift, distance, jacobian = background_observer.read(count=count)
        compiled.core.initial_setup()
        np.testing.assert_array_equal(x=redshift, y=inputs["z"])
        np.testing.assert_array_equal(x=distance, y=inputs["chi"])
        exact = (1.0 + CURVATURE * redshift)**2 / DISTANCE_SCALE
        relative = np.abs(jacobian / exact - 1.0)
        measurements.append({
            "inputs": pinned,
            "background_count": count,
            "interior_maximum": float(np.max(relative[1:-1])),
            "observer": float(relative[0]),
            "upper_endpoint": float(relative[-1]),
        })
    save_record(path=tmp_path / "jacobian.json", identity=background_observer.identity,
                measurements=measurements)
    for previous, current in zip(measurements[:-1], measurements[1:]):
        assert current["interior_maximum"] < 0.3 * previous["interior_maximum"]
        assert current["observer"] < 0.55 * previous["observer"]
        assert current["upper_endpoint"] < 0.3 * previous["upper_endpoint"]
    assert measurements[-1]["interior_maximum"] < 5.0e-6
    assert measurements[-1]["upper_endpoint"] < 1.0e-5


def test_ia_projection_refines_with_the_background(compiled, background_observer,
                                                 tmp_path):
    """Fixed source and LOS grids isolate the Jacobian's IA effect."""
    measurements = []
    for count in (41, 81, 161, 321):
        inputs = analytic_inputs(count=count)
        inputs, query, pinned = freeze_case(
            path=tmp_path / f"ia_inputs_{count}.npz", inputs=inputs,
        )
        measurements.append({
            "inputs": pinned,
            "background_count": count,
            "observables": compare_ia(compiled=compiled, inputs=inputs, query=query),
        })
    save_record(path=tmp_path / "ia_refinement.json",
                identity=background_observer.identity, measurements=measurements)
    for name in ("ii", "ggi", "iii", "quadratic_tatt"):
        errors = []
        for measurement in measurements:
            errors.append(measurement["observables"][name]["relative_error"])
        for previous, current in zip(errors[:-1], errors[1:]):
            assert current < 0.4 * previous, (name, errors)
        # Measured maximum at 321 nodes: 1.345e-6. This separate absolute
        # allowance is about four times that value; it is not an IA theory
        # error budget or a bound for a different source distribution.
        assert errors[-1] < 5.0e-6, (name, errors)


@pytest.mark.parametrize("configuration", [
    "des_y3_tatt_fisher.yaml",
    "des_y3_bacco_fisher.yaml",
])
def test_production_like_background_sampling(compiled, background_observer,
                                             tmp_path, configuration):
    """Quantify each example's background sampling on the same analytic toy.

    The count and endpoint come from the example YAML. Source, LOS and kernel
    counts remain fixed while background intervals are doubled twice. The
    toy source is not the survey n(z), and no DES accuracy claim follows.
    """
    settings = yaml.safe_load(stream=(PROJECT / "configs" / configuration).read_text())
    physics = settings["physics"]
    initial = physics["z"]["count"]
    source_count = physics.get("source_nz", initial)
    measurements = []
    for factor in (1, 2, 4):
        count = factor * (initial - 1) + 1
        inputs = analytic_inputs(
            count=count, maximum=physics["z"]["upper"], source_count=source_count,
            kernel_count=physics["kernel_nz"], los_count=physics["los_z"]["count"],
        )
        inputs, query, pinned = freeze_case(
            path=tmp_path / f"production_inputs_{count}.npz", inputs=inputs,
        )
        measurements.append({
            "inputs": pinned,
            "configuration": configuration,
            "background_count": count,
            "background_upper": physics["z"]["upper"],
            "source_count": source_count,
            "kernel_count": physics["kernel_nz"],
            "los_count": physics["los_z"]["count"],
            "observables": compare_ia(compiled=compiled, inputs=inputs, query=query),
        })
    save_record(path=tmp_path / "production_sampling.json",
                identity=background_observer.identity, measurements=measurements)
    for name in ("ii", "ggi", "iii", "quadratic_tatt"):
        coarse = measurements[0]["observables"][name]["relative_error"]
        fine = measurements[-1]["observables"][name]["relative_error"]
        # Initial measured maxima were 3.932e-5 and 2.128e-6 across these
        # configurations. The absolute limits retain roughly three times
        # those values. Two halvings of the spacing predict a factor 1/16
        # asymptotically; 0.12 allows finite-grid departures from that rate.
        assert coarse < 1.2e-4, (name, measurements)
        assert fine < 0.12 * coarse, (name, measurements)
        assert fine < 7.0e-6, (name, measurements)
