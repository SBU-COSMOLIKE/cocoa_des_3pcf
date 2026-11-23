"""Prepare and predict explicitly defined DES three-point rows.

Example, from this project directory:
  python scripts/run_3pcf.py evaluate configs/y3_full_filter_example.yaml

Absolute paths also work from another directory. Every path in the
JSON or YAML file is relative to that file. ``--validate-config`` reads the
configuration and reports missing inputs. Native code resolves numerical
settings; no CAMB solve or physical integral runs.

The commands separate expensive physical-table preparation, repeated
mean predictions and one fiducial covariance. Fisher forecasts use the
numbered Cobaya model in the teaching notebooks. The
three-point and covariance integrals run in C; CAMB prepares background
and power tables. Output NPZ files retain
the full row labels, input choices and named arrays; no data compression or
unannounced scale selection occurs.
"""


import argparse
from dataclasses import asdict
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import yaml

# Each scientific process owns at most six C workers. Matrix operations
# must not create another worker team inside those workers or subprocesses.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1",
)

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT.parents[1] / "external_modules/code/cosmolike_core"))

from des3pcf.physical_provider import camb_tables, load_tables, save_tables, camb_accuracy_settings
from cobaya.likelihoods.des_3pcf.survey import shifted_sources
from cobaya.likelihoods.des_3pcf.survey import reconstructed_sources
from cobaya.likelihoods.des_3pcf.survey import ARCMIN_TO_RADIAN, forecast_rows, read_survey
from cobaya.likelihoods.des_3pcf.survey import read_joint_covariance, read_source_distributions
from des3pcf.physical_provider import PhysicalProvider
from des3pcf.pipeline import JointPrediction
from des3pcf.pipeline import COSMOLOGY_NAMES, IA_NAMES
from des3pcf.feedback import load_feedback
from theory.bacco_bispectrum.response import BaccoFeedback, BaccoSettings, BACCO_PARAMETERS
from theory.bacco_bispectrum.emulator import installed_bacco_weights
from des3pcf.pixel_window import load_healpix_window
from cosmolike_notebook_utils.fisher import symmetric_matrix
from des3pcf.provenance import prediction_identity
from des3pcf.provenance import verify_prediction_identity


FORMAT = "cosmolike-3pcf-run-v1"
TERM_NAMES = (
    "gaussian",
    "bispectrum_squared",
    "power_trispectrum_one_halo",
    "power_bispectrum",
    "connected_one_halo",
    "halo_abundance_response",
)


import cosmolike_des_3pcf_interface as ci


def checked_fields(mapping, allowed, required, name):
    """Reject misspelled or absent fields before physical preparation.

    Arguments:
      mapping = the configuration mapping to check.
      allowed, required = sets of recognized and mandatory names.
      name = location named in a diagnostic.
    Returns:
      nothing when the object follows its schema.
    """
    if not isinstance(mapping, dict):
        raise ValueError(f"{name} must be a configuration mapping.")
    unknown = set(mapping) - set(allowed)
    absent = set(required) - set(mapping)
    if unknown or absent:
        raise ValueError(
            f"{name}: unknown fields {sorted(unknown)}; "
            f"missing fields {sorted(absent)}."
        )


def resolve_path(configuration_path, value):
    """Resolve a declared file relative to its configuration directory."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A required file path is unset; supply its location.")
    return (configuration_path.parent / value).resolve()


def grid_values(settings):
    """Make stored table nodes; these are not integration weights.

    A list gives literal nodes. An object gives lower/upper/count and
    optional logarithmic spacing. Its endpoints are included. For a positive
    logarithmic grid, include_zero adds one initial zero node; count remains
    the number of positive nodes. Quadrature settings are passed separately
    to the native covariance quadrature.
    """
    if isinstance(settings, list):
        values = np.asarray(settings, dtype=np.float64)
    else:
        checked_fields(
            mapping=settings,
            allowed={"lower", "upper", "count", "logarithmic",
                     "include_zero"},
            required={"lower", "upper", "count"}, name="table grid",
        )
        lower = settings["lower"]
        upper = settings["upper"]
        count = settings["count"]
        if (not np.isfinite(lower) or not np.isfinite(upper)
                or lower >= upper or isinstance(count, bool)
                or int(count) != count or count < 2):
            raise ValueError("Grid needs finite lower<upper and count>=2.")
        logarithmic = settings.get("logarithmic", False)
        include_zero = settings.get("include_zero", False)
        if (not isinstance(logarithmic, bool)
                or not isinstance(include_zero, bool)):
            raise ValueError("Grid spacing flags must be boolean.")
        if include_zero and not logarithmic:
            raise ValueError(
                "include_zero requires a positive logarithmic grid."
            )
        if logarithmic:
            if lower <= 0:
                raise ValueError("Logarithmic grid lower bound must be >0.")
            # Cocoa supplies power on uniformly spaced log10(k) nodes.
            # Keep the coordinate explicit, as in the other projects;
            # the same construction also serves other positive log grids.
            log10_values = np.linspace(
                start=np.log10(lower), stop=np.log10(upper), num=int(count),
            )
            values = 10.0**log10_values
            if include_zero:
                values = np.concatenate((np.zeros(1), values))
        else:
            values = np.linspace(start=lower, stop=upper, num=int(count))
    if (values.ndim != 1 or values.size < 2
            or not np.all(np.isfinite(values))
            or np.any(np.diff(values) <= 0)):
        raise ValueError("Grid must be finite, increasing and one-dimensional.")
    return np.ascontiguousarray(values)


def load_configuration(path, command):
    """Read the v1 schema and check choices before preparing a physical model."""
    configuration_path = Path(path).resolve()
    content = configuration_path.read_text(encoding="utf-8")
    if configuration_path.suffix.lower() in (".yaml", ".yml"):
        import yaml

        try:
            config = yaml.safe_load(content)
        except yaml.YAMLError as error:
            raise ValueError(f"Invalid YAML configuration: {error}") from error
    else:
        config = json.loads(content)
    top_fields = {
        "format",
        "description",
        "notes",
        "threads",
        "source",
        "physics",
        "parameters",
        "ia",
        "rows",
        "mean",
        "covariance",
        "fisher",
        "outputs",
        "accuracy",
        "feedback",
        "pixel_window",
    }
    checked_fields(
        mapping=config, allowed=top_fields,
        required=top_fields - {"description", "notes", "fisher", "covariance",
                               "accuracy", "feedback", "pixel_window"},
        name="configuration",
    )
    # Save all factors, including defaults, so reopening a result cannot
    # substitute a newer code default for its numerical preparation.
    config["accuracy"] = ci.init_3pcf_accuracy(value=config.get("accuracy"))
    if "pixel_window" in config:
        checked_fields(mapping=config["pixel_window"],
                       allowed={"path", "nside", "sha256"},
                       required={"path", "nside", "sha256"}, name="pixel_window")
    if "feedback" in config:
        mode = config["feedback"].get("mode", "file")
        if mode == "bacco":
            checked_fields(mapping=config["feedback"], allowed={"mode", "settings"},
                           required={"mode", "settings"}, name="feedback")
            config["feedback"]["settings"] = asdict(
                BaccoSettings(**config["feedback"]["settings"]),
            )
        elif mode == "file":
            checked_fields(mapping=config["feedback"], allowed={"mode", "path", "boost"},
                           required={"path", "boost"}, name="feedback")
            boost = config["feedback"]["boost"]
            if isinstance(boost, bool) or int(boost) != boost or boost < 1:
                raise ValueError("feedback.boost must be a positive integer.")
        else:
            raise ValueError("feedback.mode must be file or bacco.")
    if config["format"] != FORMAT:
        raise ValueError(f"Configuration format must be {FORMAT}.")
    threads = config["threads"]
    if isinstance(threads, bool) or int(threads) != threads or not 1 <= threads <= 6:
        raise ValueError("threads must be an integer from one through six.")
    source = config["source"]
    checked_fields(
        mapping=source, allowed={"path", "extension", "nbin"},
        required={"path", "extension", "nbin"}, name="source",
    )
    nbin = source["nbin"]
    if isinstance(nbin, bool) or int(nbin) != nbin or nbin < 1:
        raise ValueError("source.nbin must be a positive integer.")
    physics = config["physics"]
    checked_fields(
        mapping=physics,
        allowed={"tables", "cosmology_mode", "kernel_nz", "k", "z",
                 "los_z", "halofit_version", "source_interpolation",
                 "source_readers", "table_boost", "fit_nz",
                 "source_nz", "source_padding",
                 "kmax_boltzmann", "extrap_kmax",
                 "camb_accuracy_boost", "camb_k_per_logint",
                 "bispectrum_source", "bispectrum_table",
                 "variance_source", "variance_growth"},
        required={"tables", "cosmology_mode", "kernel_nz"}, name="physics",
    )
    if physics["cosmology_mode"] not in ("fixed_tables", "camb"):
        raise ValueError("physics.cosmology_mode must be fixed_tables or camb.")
    physics.setdefault("bispectrum_source", "supplied")
    if physics.get("variance_source", "local") not in ("local", "shared"):
        raise ValueError("physics.variance_source must be local or shared.")
    if physics.get("variance_growth", "factorized") not in ("factorized", "evolving"):
        raise ValueError("physics.variance_growth must be factorized or evolving.")
    if (physics.get("variance_source", "local") == "local"
            and physics.get("variance_growth", "factorized") != "factorized"):
        raise ValueError("Evolving variance growth requires the shared variance source.")
    supplied_bispectrum_settings(physics=physics)
    for name, default in (("kernel_nz", 101), ("table_boost", 8),
                          ("fit_nz", 33)):
        value = physics.get(name, default)
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"physics.{name} must be a positive integer.")
    if physics.get("fit_nz", 33) < 4:
        raise ValueError("physics.fit_nz must be at least four.")
    if physics["kernel_nz"] < 4:
        raise ValueError("physics.kernel_nz must be at least four.")
    if not isinstance(physics.get("source_readers", False), bool):
        raise ValueError("physics.source_readers must be a boolean.")
    reconstruction = physics.get("source_interpolation", "linear_midpoint")
    if reconstruction not in ("linear_midpoint", "pchip_midpoint"):
        raise ValueError("Choose linear_midpoint or explicit pchip_midpoint sources.")
    source_nz = physics.get("source_nz")
    if source_nz is not None:
        if (isinstance(source_nz, bool) or not np.isscalar(source_nz)
                or not np.isfinite(source_nz)
                or int(source_nz) != source_nz or source_nz < 4
                or source_nz > np.iinfo(np.int32).max):
            raise ValueError("physics.source_nz must be an integer from four to INT_MAX.")
    padding = physics.get("source_padding", "zero_outside_native")
    if reconstruction == "pchip_midpoint":
        if source_nz is None or padding not in (
            "zero_at_observer_and_background_edge", "zero_at_measured_edges",
        ):
            raise ValueError(
                "PCHIP needs source_nz and explicit source_padding: "
                "zero_at_observer_and_background_edge or zero_at_measured_edges."
            )
    elif padding != "zero_outside_native":
        raise ValueError("Linear midpoint sources require zero_outside_native padding.")
    # These are physical wavenumbers in 1/Mpc, not the saved k grid's h/Mpc.
    # Separating them bounds CAMB work while retaining the requested read grid.
    physics.setdefault("kmax_boltzmann", 5.0)
    physics.setdefault("extrap_kmax", None)
    camb_accuracy_settings(
        accuracy_boost=physics.get("camb_accuracy_boost", 1.0),
        k_per_logint=physics.get("camb_k_per_logint", 0),
    )
    for name in ("kmax_boltzmann", "extrap_kmax"):
        value = physics[name]
        if value is None and name == "extrap_kmax":
            continue
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"physics.{name} must be positive and finite in 1/Mpc.")
    if command == "prepare":
        for name in ("k", "z", "los_z", "halofit_version"):
            if name not in physics:
                raise ValueError(f"prepare requires physics.{name}.")
        if "source_interpolation" not in physics:
            raise ValueError("prepare requires an explicit source_interpolation choice.")
        redshift = grid_values(settings=physics["z"])
        if redshift[0] != 0 or redshift.size > 150:
            raise ValueError("CAMB table z must start at 0 with <=150 nodes.")
    required_parameters = {"omega_m", "omega_b", "h", "ns", "sigma8",
                           "w0", "wa"}
    allowed_parameters = set(COSMOLOGY_NAMES + IA_NAMES)
    if config.get("feedback", {}).get("mode") == "bacco":
        allowed_parameters.update(BACCO_PARAMETERS)
        required_parameters.update(BACCO_PARAMETERS)
    for index in range(nbin):
        allowed_parameters.add(f"m{index + 1}")
        allowed_parameters.add(f"dz{index + 1}")
    checked_fields(
        mapping=config["parameters"], allowed=allowed_parameters,
        required=required_parameters, name="parameters",
    )
    for name, value in config["parameters"].items():
        if isinstance(value, bool) or not np.isfinite(value):
            raise ValueError(f"Parameter {name} must be a finite number.")
    if "k" in physics and physics["extrap_kmax"] is not None:
        maximum = grid_values(settings=physics["k"])[-1] * config["parameters"]["h"]
        if physics["extrap_kmax"] < maximum:
            raise ValueError(
                "physics.extrap_kmax must cover h*physics.k.upper in 1/Mpc."
            )
    checked_fields(mapping=config["ia"], allowed=set(IA_NAMES),
                   required=set(), name="ia")
    for name, value in config["ia"].items():
        if not np.isfinite(value):
            raise ValueError(f"IA parameter {name} must be finite.")
    amplitude = config["parameters"].get(
        "feedback_amplitude", config["ia"].get("feedback_amplitude", 0.0),
    )
    legacy_feedback = config["parameters"].get("fb", config["ia"].get("fb", 0.0))
    if amplitude != 0.0 and "feedback" not in config:
        raise ValueError("Nonzero feedback_amplitude requires paired feedback tables.")
    if amplitude != 0.0 and legacy_feedback != 0.0:
        raise ValueError("Use either paired feedback_amplitude or legacy fb, not both.")
    if config.get("feedback", {}).get("mode") == "bacco":
        if amplitude != 1.0:
            raise ValueError("Dynamic BACCO requires fixed feedback_amplitude=1.")
    rows = config["rows"]
    if rows.get("mode") == "forecast":
        checked_fields(
            mapping=rows,
            allowed={"mode", "aperture_radii_arcmin", "xi_edges_arcmin"},
            required={"mode", "aperture_radii_arcmin"}, name="rows",
        )
    elif rows.get("mode") == "measurement":
        checked_fields(
            mapping=rows,
            allowed={"mode", "two_point", "aperture", "assumed_angle_unit"},
            required={"mode", "two_point", "aperture"}, name="rows",
        )
        if rows["two_point"] is None and rows["aperture"] is None:
            raise ValueError("Measurement rows need at least one FITS file.")
    else:
        raise ValueError("rows.mode must be forecast or measurement.")
    mean = config["mean"]
    checked_fields(
        mapping=mean,
        allowed={"aperture_estimator", "aperture_settings", "xi_settings",
                 "two_point"},
        required={"aperture_settings", "two_point"}, name="mean",
    )
    estimator = mean.get("aperture_estimator", "full_filter")
    if estimator not in ("full_filter", "finite_sas"):
        raise ValueError("Mean estimator must be full_filter or finite_sas.")
    if estimator == "finite_sas":
        settings = mean["aperture_settings"]
        settings.setdefault("representative_rule", "geometric")
        settings.setdefault("representative_offset", 0.0)
        settings.setdefault("side_min_arcmin", None)
        minimum = settings["side_min_arcmin"]
        if minimum is not None:
            if isinstance(minimum, bool) or not np.isfinite(minimum) or minimum < 0.0:
                raise ValueError("side_min_arcmin must be finite and nonnegative.")
        if settings["representative_rule"] not in ("geometric", "area_weighted"):
            raise ValueError("representative_rule must be geometric or area_weighted.")
        if not np.isfinite(settings["representative_offset"]):
            raise ValueError("representative_offset must be finite.")
    if estimator == "full_filter":
        settings = mean["aperture_settings"]
        checked_fields(mapping=settings,
                       allowed={"ell", "phi", "projection_mode",
                                "projection_settings",
                                "projection_batch_size"},
                       required={"ell", "phi"}, name="aperture_settings")
        for name in ("ell", "phi"):
            grid_values(settings=settings[name])
        if (settings["phi"]["lower"] != 0
                or settings["phi"]["upper"] != 2 * np.pi):
            raise ValueError("Full-filter phi quadrature must span 0 to 2*pi.")
    two_point = mean["two_point"]
    checked_fields(
        mapping=two_point,
        allowed={"mode", "directory", "module", "interpolation_type",
                 "accuracy_boost", "integration_accuracy", "fpt_internal_boost",
                 "ell_internal"},
        required={"mode"}, name="mean.two_point",
    )
    if two_point["mode"] not in ("isolated_nla", "core_tatt"):
        raise ValueError("two_point.mode must be isolated_nla or core_tatt.")
    if two_point["mode"] == "core_tatt":
        for name in ("directory", "module"):
            if name not in two_point:
                raise ValueError(f"core_tatt requires two_point.{name}.")
    has_xi = rows.get("xi_edges_arcmin") is not None
    if rows["mode"] == "measurement":
        has_xi = rows.get("two_point") is not None
    if has_xi and command == "evaluate":
        unsupported = ("ct", "fb")
        if two_point["mode"] == "isolated_nla":
            unsupported = ("a2", "bta", "ct", "fb")
        for name in unsupported:
            value = config["parameters"].get(name, config["ia"].get(name, 0))
            if value != 0:
                raise ValueError(
                    f"The selected xi route cannot vary/use {name}; "
                    "choose a model supplying its E/B spectra."
                )
    if command == "covariance":
        if "covariance" not in config:
            raise ValueError(f"{command} requires covariance settings.")
        covariance = config["covariance"]
        if not isinstance(covariance.get("save_components", False), bool):
            raise ValueError("covariance.save_components must be boolean.")
        if covariance.get("mode") not in ("analytic", "file"):
            raise ValueError("covariance.mode must be analytic or file.")
        if covariance["mode"] == "analytic":
            for index in range(nbin):
                calibration = config["parameters"].get(f"m{index + 1}", 0.0)
                if calibration != 0:
                    raise ValueError(
                        "Analytic covariance requires fiducial m_i=0; its "
                        "lensing weights do not apply calibration factors. "
                        "Fisher displacements may vary m_i with that fixed "
                        "zero-calibration fiducial covariance."
                    )
            neutrino_model = covariance.get("neutrino_model", "massless")
            if neutrino_model not in (
                "massless", "single_fluid_effective_growth"
            ):
                raise ValueError("Unknown covariance.neutrino_model.")
            if (config["parameters"].get("mnu", 0.0) > 0
                    and neutrino_model == "massless"):
                raise ValueError(
                    "Massive-neutrino covariance needs an explicit "
                    "single_fluid_effective_growth approximation; "
                    "the P_cb halo treatment is not supplied here."
                )
            if (neutrino_model == "single_fluid_effective_growth"
                    and physics.get("variance_growth", "factorized") != "factorized"):
                raise ValueError(
                    "single_fluid_effective_growth requires factorized variance "
                    "growth; evolving halo variances need a separately declared "
                    "and validated neutrino prescription."
                )
            if estimator != "full_filter":
                raise ValueError(
                    "Analytic full-filter covariance requires a full_filter "
                    "mean. Supply a matched file for finite_sas forecasts."
                )
            if covariance.get("ia_covariance") != "gravitational_only":
                raise ValueError(
                    "Declare covariance.ia_covariance=gravitational_only; "
                    "connected nonlinear-IA covariance is not supplied."
                )
        elif covariance.get("estimator") != estimator:
            raise ValueError(
                "A supplied covariance must declare the same aperture "
                "estimator as the mean; do not identify them by row count."
            )
    digest = hashlib.sha256(content.encode()).hexdigest()
    return config, configuration_path, digest




def required_files(config, path, command):
    """List actual prerequisites; dry checks do not create or import them."""
    requirements = []
    if "feedback" in config:
        if config["feedback"].get("mode", "file") == "file":
            requirements.append(("paired feedback responses", config["feedback"]["path"]))
        else:
            for name, record in installed_bacco_weights().items():
                requirements.append((f"BACCO {name} weights", record["path"]))
    if "pixel_window" in config:
        requirements.append(("scalar pixel window", config["pixel_window"]["path"]))
    if command == "prepare":
        requirements.append(("source FITS", config["source"]["path"]))
    else:
        requirements.append(("physics tables", config["physics"]["tables"]))
        if config["rows"]["mode"] == "measurement":
            for name in ("two_point", "aperture"):
                value = config["rows"][name]
                if value is not None:
                    requirements.append((name + " FITS", value))
        requirements.append(("three-point extension",
                             str(PROJECT / "interface" /
                                 "cosmolike_des_3pcf_interface.so")))
        core = config["mean"]["two_point"]
        if core["mode"] == "core_tatt" and command != "covariance":
            directory = resolve_path(path, core["directory"])
            requirements.append(("ordinary-core extension",
                                 str(directory / (core["module"] + ".so"))))
    if command == "covariance":
        covariance = config["covariance"]
        if covariance["mode"] == "file":
            requirements.append(("fiducial covariance", covariance["path"]))
    missing = []
    for name, value in requirements:
        if value is None:
            missing.append({"input": name, "path": None})
            continue
        location = resolve_path(path, value)
        if not location.is_file():
            missing.append({"input": name, "path": str(location)})
    if command == "covariance" and config["covariance"]["mode"] == "analytic":
        settings = config["covariance"]
        if settings.get("physical", {}).get("area") is None:
            missing.append({"input": "survey area in steradians", "path": None})
        noise = settings.get("noise", {})
        if noise.get("matrix") is None:
            for name in ("number_density", "sigma_component"):
                if noise.get(name) is None:
                    missing.append({"input": "shape noise " + name,
                                    "path": None})
    return missing


def rows_for_run(config, path):
    """Retain the entire declared vector; measured and forecast rows differ."""
    settings = config["rows"]
    if settings["mode"] == "forecast":
        edges = None
        if settings.get("xi_edges_arcmin") is not None:
            edges = grid_values(settings=settings["xi_edges_arcmin"])
            edges *= ARCMIN_TO_RADIAN
        radii = np.asarray(settings["aperture_radii_arcmin"], dtype=np.float64)
        return forecast_rows(nbin=config["source"]["nbin"],
                             aperture_radii=radii * ARCMIN_TO_RADIAN,
                             xi_edges=edges)
    two_point = None
    aperture = None
    if settings["two_point"] is not None:
        two_point = resolve_path(path, settings["two_point"])
    if settings["aperture"] is not None:
        aperture = resolve_path(path, settings["aperture"])
    return read_survey(
        two_point_path=two_point, aperture_path=aperture,
        nbin=config["source"]["nbin"],
        assumed_angle_unit=settings.get("assumed_angle_unit", "arcmin"),
    )


def check_destination(path, overwrite):
    """Check an output before starting work; overwrite is never implicit."""
    if path.suffix != ".npz":
        raise ValueError("Every numerical output must have the .npz suffix.")
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Output exists: {path}; use --overwrite explicitly."
        )
    path.parent.mkdir(parents=True, exist_ok=True)


def save_result(path, arrays, metadata, rows, overwrite):
    """Write arrays and full row identity with no pickled Python objects."""
    result = dict(arrays)
    result["row_labels"] = rows.labels
    result["row_kind"] = rows.kind
    result["row_bins"] = rows.bins
    result["row_angles_radians"] = rows.angles
    result["observed_values"] = rows.values
    result["metadata_json"] = np.array(json.dumps(metadata, sort_keys=True))
    mode = "wb" if overwrite else "xb"
    with path.open(mode=mode) as stream:
        np.savez_compressed(stream, **result)


def prepare(config, path, metadata, overwrite):
    """Prepare the declared source reconstruction and independent CAMB tables.

    Midpoint samples of the selected source FITS are saved unchanged. That
    FITS may itself be the explicitly processed demonstration input; its
    content hash distinguishes it from the downloaded calibration file.
    The optional source_nz
    controls only source preparation; it never raises CAMB's redshift count.
    Legacy linear preparation without that control retains its old result.
    """
    settings = config["physics"]
    destination = resolve_path(path, settings["tables"])
    check_destination(path=destination, overwrite=overwrite)
    source_path = resolve_path(path, config["source"]["path"])
    with_edges = settings.get("source_padding") == "zero_at_measured_edges"
    source_data = read_source_distributions(
        path=source_path, nbin=config["source"]["nbin"],
        extension=config["source"]["extension"],
        with_edges=with_edges,
    )
    source_z, source_values = source_data[:2]
    native_low = None
    native_high = None
    if with_edges:
        native_low, native_high = source_data[2:]
    redshift = grid_values(settings=settings["z"])
    outside = source_z > redshift[-1]
    if np.any(source_values[:, outside] > 0):
        raise ValueError(
            "Extend physics.z: nonzero source support lies beyond it."
        )
    source_nodes = None
    source_record = {
        "source_reconstruction": settings["source_interpolation"],
        "source_padding": settings.get("source_padding", "zero_outside_native"),
    }
    if settings.get("source_nz") is None:
        sources = np.empty((source_values.shape[0], redshift.size))
        for index in range(source_values.shape[0]):
            sources[index] = np.interp(x=redshift, xp=source_z,
                                      fp=source_values[index], left=0, right=0)
        raw_integrals = np.trapz(sources, x=redshift, axis=1)
        sources = shifted_sources(z=redshift, sources=sources,
                                  shifts=np.zeros(source_values.shape[0]))
    else:
        source_nodes = np.linspace(start=0.0, stop=redshift[-1],
                                   num=int(settings["source_nz"]))
        sources, source_record = reconstructed_sources(
            native_z=source_z, native_sources=source_values, source_z=source_nodes,
            shifts=np.zeros(source_values.shape[0]),
            reconstruction=settings["source_interpolation"],
            padding=settings.get("source_padding", "zero_outside_native"),
            native_low=native_low, native_high=native_high,
        )
        raw_integrals = np.asarray(source_record["source_shift_normalizations"])
    tables, preparation = camb_tables(
        parameters=config["parameters"], k=grid_values(settings=settings["k"]),
        z=redshift, sources=sources,
        los_z=grid_values(settings=settings["los_z"]),
        halofit_version=settings["halofit_version"],
        kmax_boltzmann=settings["kmax_boltzmann"],
        extrap_kmax=settings["extrap_kmax"],
        camb_accuracy_boost=settings.get("camb_accuracy_boost", 1.0),
        camb_k_per_logint=settings.get("camb_k_per_logint", 0),
        source_z=source_nodes,
        include_bispectrum_background=True,
    )
    tables["source_native_z"] = source_z
    tables["source_native"] = source_values
    if with_edges:
        tables["source_native_low"] = native_low
        tables["source_native_high"] = native_high
    preparation.update(metadata)
    preparation.update(source_record)
    preparation["source_fits"] = str(source_path)
    preparation["source_fits_sha256"] = hashlib.sha256(
        source_path.read_bytes()
    ).hexdigest()
    if settings["source_interpolation"] == "pchip_midpoint":
        import scipy

        preparation["source_reconstruction_library"] = "scipy.interpolate.PchipInterpolator"
        preparation["source_reconstruction_library_version"] = scipy.__version__
    preparation["source_interpolation"] = settings["source_interpolation"]
    preparation["source_integrals_before_normalization"] = (
        raw_integrals.tolist()
    )
    save_tables(path=destination, tables=tables, metadata=preparation,
                overwrite=overwrite)
    return destination


def compiled_extension(config):
    """Import the existing project extension and set its worker count once."""
    sys.path.insert(0, str(PROJECT / "interface"))
    compiled = importlib.import_module(name="cosmolike_des_3pcf_interface")
    compiled.set_omp_threads(nthreads=config["threads"])
    return compiled


def supplied_bispectrum_settings(physics):
    """Resolve explicit table axes without evaluating a bispectrum.

    logk and logu are already logarithmic coordinates, so their normal
    YAML grids use linear spacing. The producer validates uniformity in
    each stored coordinate. Domain extension is explicit: this route does
    not clamp a baseline bispectrum beyond its table bounds.

    Arguments:
      physics = the configuration's physical-table and source choices.
    Returns:
      an axis dictionary for the required Python matter-bispectrum producer.
    Raises:
      ValueError for an unknown source, missing axes or invalid uniform grids.
    """
    source = physics.get("bispectrum_source", "supplied")
    if source != "supplied":
        raise ValueError("physics.bispectrum_source must be supplied; C fit removed.")
    if "bispectrum_table" not in physics:
        if "z" not in physics:
            # Fixed-table jobs resolve the endpoint from the loaded background.
            return None
        from theory.bihalofit.model import bispectrum_axes

        return bispectrum_axes(zmax=float(grid_values(settings=physics["z"])[-1]))
    table = physics["bispectrum_table"]
    checked_fields(
        mapping=table,
        allowed={"z", "logk", "logu", "v", "boost"},
        required={"z", "logk", "logu", "v"},
        name="physics.bispectrum_table",
    )
    axes = {}
    for name in ("z", "logk", "logu", "v"):
        axes[name] = grid_values(settings=table[name])
    axes["boost"] = table.get("boost", 2)
    from theory.bihalofit.model import BiHalofitProducer

    BiHalofitProducer(**axes)
    return axes


def provider_for_run(compiled, config, path):
    """Load one fiducial physical table without running CAMB on import.

    The provider owns the cosmological preparation policy. It reuses the
    fiducial tables for a nuisance-only displacement, and requests new CAMB
    tables when the explicit cosmology key changes in camb mode. The CLI
    only supplies the declared table resolutions and reference parameters.
    """
    table_path = resolve_path(path, config["physics"]["tables"])
    tables, provenance = load_tables(path=table_path)
    pixel_window = None
    if "pixel_window" in config:
        window_settings = config["pixel_window"]
        pixel_window, window_record = load_healpix_window(
            path=resolve_path(path, window_settings["path"]),
            nside=window_settings["nside"], sha256=window_settings["sha256"],
        )
        provenance = dict(provenance)
        provenance["pixel_window"] = window_record
    feedback = None
    if "feedback" in config:
        if config["feedback"].get("mode", "file") == "bacco":
            feedback = BaccoFeedback(settings=config["feedback"]["settings"])
        else:
            feedback = load_feedback(
                path=resolve_path(path, config["feedback"]["path"]),
                boost=config["feedback"]["boost"],
            )
    if tables["sources"].shape[0] != config["source"]["nbin"]:
        raise ValueError("Table source count differs from source.nbin.")
    provider = PhysicalProvider(
        compiled=compiled, reference_tables=tables,
        reference_metadata=provenance,
        reference_parameters=config["parameters"],
        cosmology_mode=config["physics"]["cosmology_mode"],
        kernel_nz=config["physics"]["kernel_nz"],
        halofit_version=config["physics"].get("halofit_version", "takahashi"),
        kmax_boltzmann=config["physics"].get("kmax_boltzmann", 5.0),
        extrap_kmax=config["physics"].get("extrap_kmax"),
        camb_accuracy_boost=config["physics"].get("camb_accuracy_boost", 1.0),
        camb_k_per_logint=config["physics"].get("camb_k_per_logint", 0),
        source_readers=config["physics"].get("source_readers", False),
        table_boost=config["physics"].get("table_boost", 8),
        fit_nz=config["physics"].get("fit_nz", 33),
        source_nz=config["physics"].get("source_nz"),
        source_interpolation=config["physics"].get("source_interpolation"),
        source_padding=config["physics"].get("source_padding"),
        accuracy=config["accuracy"],
        feedback=feedback,
        pixel_window=pixel_window,
        bispectrum_source=config["physics"].get("bispectrum_source", "supplied"),
        bispectrum_table=supplied_bispectrum_settings(physics=config["physics"]),
        variance_source=config["physics"].get("variance_source", "local"),
        variance_growth=config["physics"].get("variance_growth", "factorized"),
    )
    return provider, provenance


def prediction_for_run(compiled, config, path, rows, provider):
    """Connect the chosen aperture estimator and two-point IA calculation."""
    mean = config["mean"]
    estimator = mean.get("aperture_estimator", "full_filter")
    settings = dict(mean["aperture_settings"])
    if estimator == "finite_sas":
        edges = grid_values(settings=settings.pop("theta_edges_arcmin"))
        logarithmic_widths = np.diff(np.log(edges))
        if not np.allclose(logarithmic_widths, logarithmic_widths[0],
                           rtol=1.0e-12, atol=1.0e-14):
            raise ValueError("Finite-SAS side bins must have equal log widths.")
        count = settings.pop("phi_bins")
        if isinstance(count, bool) or int(count) != count or count < 1:
            raise ValueError("phi_bins must be a positive integer.")
        width = np.pi / count
        settings["theta_lower"] = edges[:-1] * ARCMIN_TO_RADIAN
        minimum = settings.pop("side_min_arcmin", None)
        if minimum is not None:
            if "side_keep" in settings:
                raise ValueError("Choose side_min_arcmin or side_keep, not both.")
            settings["side_keep"] = edges[:-1] >= minimum
        settings["dlnt"] = float(logarithmic_widths[0])
        settings["phi"] = (np.arange(count) + 0.5) * width
        settings["dphi"] = width
        averaging = settings.pop("angular_average", "point")
        if averaging not in ("point", "x_modes"):
            raise ValueError("angular_average must be point or x_modes.")
        options = dict(settings.get("transform_options", {}))
        if averaging == "x_modes":
            options["dphi"] = width
        settings["transform_options"] = options
    core_adapter = None
    core = mean["two_point"]
    if core["mode"] == "core_tatt" and np.any(rows.kind != "map3"):
        core_adapter = dict(
            core=compiled, nbin=config["source"]["nbin"],
            threads=config["threads"],
            interpolation_type=core.get("interpolation_type", 0),
            accuracy_boost=core.get("accuracy_boost", 1.0),
            integration_accuracy=core.get("integration_accuracy", 0),
            fpt_internal_boost=core.get("fpt_internal_boost", 0.5),
            ell_internal=core.get("ell_internal", 192),
        )
    return JointPrediction(
        compiled=compiled, rows=rows, provider=provider,
        ia_defaults=config["ia"], aperture_estimator=estimator,
        aperture_settings=settings, xi_settings=mean.get("xi_settings"),
        two_point_core=core_adapter,
        accuracy=config["accuracy"],
    )


def read_matrix(settings, path, rows, positive):
    """Load a joint matrix only after verifying the exact row association.

    NPZ requires row_labels and the named matrix. A rank-deficient sample
    covariance is allowed when positive=False, as needed by PME. The FITS
    reader checks its full COVMAT block order and positive definiteness.
    """
    source = resolve_path(path, settings["path"])
    if source.suffix.lower() in (".fits", ".fit"):
        return read_joint_covariance(path=source, rows=rows)
    with np.load(source, allow_pickle=False) as archive:
        if "row_labels" not in archive.files:
            raise ValueError("NPZ covariance needs exact row_labels.")
        if not np.array_equal(archive["row_labels"], rows.labels):
            raise ValueError(
                "Covariance row labels do not match the full vector."
            )
        matrix = archive[settings.get("key", "covariance")]
        matrix = symmetric_matrix(values=matrix, name="Joint covariance",
                                  positive=positive)
    if matrix.shape != (rows.values.size, rows.values.size):
        raise ValueError("Covariance shape differs from the full row count.")
    return matrix


def evaluate(compiled, config, path, metadata, rows, provider, overwrite):
    """Save the complete model mean and its difference from supplied data."""
    destination = resolve_path(path, config["outputs"]["evaluate"])
    check_destination(path=destination, overwrite=overwrite)
    prediction = prediction_for_run(compiled=compiled, config=config, path=path,
                                    rows=rows, provider=provider)
    second_extension = None
    if prediction.two_point_core is not None:
        second_extension = prediction.compiled.__file__
    metadata["prediction_identity"] = prediction_identity(
        config=config,
        configuration_path=path,
        project=PROJECT,
        three_point_path=compiled.__file__,
        two_point_path=second_extension,
    )
    vector = prediction(parameters=config["parameters"])
    if provider._active_baseline is not None:
        metadata["baseline_bispectrum"] = provider._active_baseline["metadata"]
    if provider._active_feedback is not None:
        metadata["feedback_reference"] = provider._active_feedback.record()
    arrays = {"mean": vector}
    if config["rows"]["mode"] == "measurement":
        arrays["mean_minus_data"] = vector - rows.values
    completed_identity = prediction_identity(
        config=config,
        configuration_path=path,
        project=PROJECT,
        three_point_path=compiled.__file__,
        two_point_path=second_extension,
    )
    verify_prediction_identity(
        recorded=metadata["prediction_identity"],
        current=completed_identity,
        description="Mean calculation during evaluation",
    )
    save_result(path=destination, arrays=arrays, metadata=metadata,
                rows=rows, overwrite=overwrite)
    return destination


def covariance(compiled, config, path, metadata, rows, provider, overwrite):
    """Save the fiducial sum and optionally its six physical components.

    Configuration grids become fixed nodes and weights in covariance_inputs;
    that helper also connects measured xi-bin edges to the selected noise
    treatment. The C routine supplies each physical matrix. We retain their
    sum even if its minimum eigenvalue is nonpositive, so a later precision
    calculation can refuse it without hiding which term caused the problem.
    """
    settings = config["covariance"]
    # Covariance is an optional build, as in LSST Y1. Mean-only runs
    # must not import bindings omitted by that build selection.
    from cosmolike_des_3pcf_interface import covariance_inputs, shape_noise

    if settings["mode"] != "analytic":
        raise ValueError("covariance command requires mode=analytic.")
    destination = resolve_path(path, settings["path"])
    check_destination(path=destination, overwrite=overwrite)
    metadata["prediction_identity"] = prediction_identity(
        config=config,
        configuration_path=path,
        project=PROJECT,
        three_point_path=compiled.__file__,
        covariance_output=True,
    )
    noise_settings = settings["noise"]
    if "matrix" in noise_settings:
        noise = np.asarray(noise_settings["matrix"], dtype=np.float64)
    else:
        noise = shape_noise(number_density=noise_settings["number_density"],
                            sigma_component=noise_settings["sigma_component"])
    nbin = config["source"]["nbin"]
    if noise.shape != (nbin, nbin):
        raise ValueError("Shape-noise matrix must contain every source bin.")
    grids = dict(settings["grids"])
    grids["profile_radius"] = grid_values(settings=grids["profile_radius"])
    inputs = covariance_inputs(
        kind=rows.kind.tolist(),
        bins=rows.bins,
        angles=rows.angles,
        noise=noise,
        grids=grids,
        physical=settings["physical"],
        theta_count=settings["theta_count"],
        pair_weight=settings["pair_weight"],
        accuracy=config["accuracy"],
    )
    model, inputs, physical_record = provider.covariance_model(
        parameters=config["parameters"], inputs=inputs,
    )
    for name in ("variance", "halo_variance"):
        if name in physical_record:
            copied = dict(physical_record[name])
            copied["sampling"] = dict(copied["sampling"])
            copied["input_sha256"] = dict(copied["input_sha256"])
            metadata[name] = copied
    if provider._active_baseline is not None:
        metadata["baseline_bispectrum"] = provider._active_baseline["metadata"]
    if provider._active_feedback is not None:
        metadata["feedback_reference"] = provider._active_feedback.record()
    model_mean, terms = compiled.production.physical_covariance(
        model=model, inputs=inputs,
    )
    total = np.sum(terms, axis=0)
    metadata["covariance_minimum_eigenvalue"] = float(
        np.linalg.eigvalsh(total)[0]
    )
    keep_components = settings.get("save_components", False)
    metadata["covariance_term_names"] = list(TERM_NAMES)
    metadata["covariance_components_saved"] = keep_components
    arrays = {
        "covariance": total,
        "covariance_model_mean": model_mean,
    }
    # The present C entry computes/returns all six matrices either way.
    # This choice removes duplicate saved diagnostics, not C integration
    # work or the temporary matrix storage required by that entry.
    if keep_components:
        arrays["terms"] = terms
        for index, name in enumerate(TERM_NAMES):
            arrays[name] = terms[index]
    completed_identity = prediction_identity(
        config=config,
        configuration_path=path,
        project=PROJECT,
        three_point_path=compiled.__file__,
        covariance_output=True,
    )
    verify_prediction_identity(
        recorded=metadata["prediction_identity"],
        current=completed_identity,
        description="Physical inputs during covariance construction",
    )
    save_result(path=destination, arrays=arrays, metadata=metadata,
                rows=rows, overwrite=overwrite)
    return destination




def main(argv=None):
    """Run one command, returning a shell status with a named failure."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate",
                                            "covariance"))
    parser.add_argument("config", type=Path)
    parser.add_argument("--validate-config", action="store_true",
                        help="Check schema and missing inputs; do no physics.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Permit replacing the declared NPZ output.")
    arguments = parser.parse_args(args=argv)
    try:
        config, path, digest = load_configuration(
            path=arguments.config, command=arguments.command,
        )
        missing = required_files(config=config, path=path,
                                 command=arguments.command)
        if arguments.validate_config:
            report = {
                "schema_valid": True,
                "command": arguments.command,
                "missing_inputs": missing,
                "description": config.get("description", ""),
            }
            print(yaml.safe_dump(report, sort_keys=False))
            return 0
        if missing:
            raise FileNotFoundError(json.dumps(missing, indent=2))
        metadata = {
            "format": "cosmolike-3pcf-result-v1",
            "command": arguments.command,
            "configuration": config,
            "configuration_path": str(path),
            "configuration_sha256": digest,
            "aperture_estimator": config["mean"].get(
                "aperture_estimator", "full_filter"),
            "compression": "none",
        }
        if config.get("covariance", {}).get("mode") == "analytic":
            metadata["covariance_neutrino_model"] = config["covariance"].get(
                "neutrino_model", "massless"
            )
        if arguments.command == "prepare":
            destination = prepare(config=config, path=path, metadata=metadata,
                                  overwrite=arguments.overwrite)
        else:
            rows = rows_for_run(config=config, path=path)
            compiled = compiled_extension(config=config)
            provider, provenance = provider_for_run(
                compiled=compiled, config=config, path=path,
            )
            metadata["physical_table_metadata"] = provenance
            commands = {
                "evaluate": evaluate,
                "covariance": covariance,
            }
            destination = commands[arguments.command](
                compiled=compiled, config=config, path=path, metadata=metadata,
                rows=rows, provider=provider, overwrite=arguments.overwrite,
            )
        print(destination)
        return 0
    except (ValueError, OSError, KeyError, ImportError, TypeError) as error:
        print(f"3pcf {arguments.command}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
