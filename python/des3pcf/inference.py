"""Prepare the joint-data likelihood and explicit independent sampling priors.

An inference YAML names an existing prediction YAML. Both the Fisher and
Nautilus examples therefore call the same JointPrediction and physical
tables. The inference file additionally chooses its data, row selection,
prior densities and sampler settings. None is inferred from a filename.

Preparation retains one physical provider and one fixed covariance factor.
The sampler evaluates different parameter dictionaries through that owner;
it does not store an unbounded collection of previously sampled means.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import stats
import yaml

from .compression import load_compression
from .likelihood import FixedGaussianLikelihood, FiniteSimulationLikelihood
from .pipeline import COSMOLOGY_NAMES
from .provenance import file_digest, implementation_identity


FORMAT = "cosmolike-3pcf-inference-v1"
DISCRETE_PARAMETERS = {
    "num_massive_neutrinos",
    "power_mode",
    "ct_mode",
    "suppress_only",
}


def fields(mapping, allowed, required, name):
    """Reject missing or misspelled configuration fields before preparation."""
    if not isinstance(mapping, dict):
        raise ValueError(f"{name} must be a YAML mapping, received {type(mapping)}.")
    unknown = set(mapping) - set(allowed)
    missing = set(required) - set(mapping)
    if unknown or missing:
        raise ValueError(
            f"{name}: unknown fields {sorted(unknown)}, missing {sorted(missing)}; "
            "use the fields in EXAMPLE_NAUTILUS1.yaml."
        )


def prior_distributions(priors):
    """Construct normalized independent densities in YAML parameter order.

    Uniform priors have explicit minimum/maximum. A normal prior has a mean
    and sigma; optional minimum/maximum choose a truncated, renormalized
    Gaussian instead. scipy.stats uses standardized limits for truncnorm,
    so physical limits are first converted to (limit-mean)/sigma.

    Arguments:
      priors = ordered mapping from parameter names to prior specifications.
    Returns:
      dictionary of SciPy distribution objects, exposing pdf, logpdf and ppf.
    Raises:
      ValueError for a malformed prior or a discrete parameter name.
    """
    if not isinstance(priors, dict) or not priors:
        raise ValueError("priors must name at least one sampled parameter.")
    distributions = {}
    for name, specification in priors.items():
        if name in DISCRETE_PARAMETERS:
            raise ValueError(f"{name} is a discrete model choice; keep it fixed.")
        fields(
            mapping=specification,
            allowed={"distribution", "minimum", "maximum", "mean", "sigma", "latex"},
            required={"distribution"}, name=f"prior {name}",
        )
        for key in ("minimum", "maximum", "mean", "sigma"):
            if key in specification:
                value = specification[key]
                if isinstance(value, bool) or not np.isfinite(value):
                    raise ValueError(f"Prior {name}.{key}={value} must be finite.")
        distribution = specification["distribution"]
        lower = specification.get("minimum")
        upper = specification.get("maximum")
        if (lower is None) != (upper is None):
            raise ValueError(f"Prior {name} must declare both limits or neither.")
        if lower is not None and lower >= upper:
            raise ValueError(f"Prior {name} requires minimum < maximum.")
        if distribution == "uniform":
            if lower is None or "mean" in specification or "sigma" in specification:
                raise ValueError(f"Uniform prior {name} requires only two limits.")
            distributions[name] = stats.uniform(loc=lower, scale=upper - lower)
        elif distribution == "normal":
            if "mean" not in specification or specification.get("sigma", 0) <= 0:
                raise ValueError(f"Normal prior {name} requires mean and sigma > 0.")
            mean = specification["mean"]
            sigma = specification["sigma"]
            if lower is None:
                distributions[name] = stats.norm(loc=mean, scale=sigma)
            else:
                distributions[name] = stats.truncnorm(
                    a=(lower - mean) / sigma, b=(upper - mean) / sigma,
                    loc=mean, scale=sigma,
                )
        else:
            raise ValueError(f"Prior {name} must choose uniform or normal.")
    return distributions


def projection_overrides(specification):
    """Validate the optional numerical chart without changing the IA model.

    Only the signed-baseline representation may differ from the referenced
    prediction file here. Changing integration grids or angular cuts requires
    another prediction configuration, where those choices remain visible.
    """
    overrides = specification.get("projection_overrides", {})
    fields(
        mapping=overrides,
        allowed={"signed_baseline"},
        required=set(),
        name="projection_overrides",
    )
    if "signed_baseline" in overrides:
        value = overrides["signed_baseline"]
        if isinstance(value, bool) or not isinstance(value, int) or value not in (0, 1):
            raise ValueError(
                "projection_overrides.signed_baseline must be integer 0 or 1."
            )
    return dict(overrides)


def read_dataset(data_file, path=None):
    """Read a local dataset beside the project's bundled data by default."""
    directory = Path(__file__).resolve().parents[2] / "data"
    if path is not None:
        directory = Path(path).resolve()
    location = (directory / data_file).resolve()
    settings = yaml.safe_load(location.read_text(encoding="utf-8"))
    return settings, location


def cobaya_specification(parameters, dataset):
    """Map standard Cobaya parameters onto the existing prediction preparation.

    Cobaya owns the actual prior evaluation. These equivalent independent
    distributions are used only to validate the fiducial and to count the
    sampled parameters for the finite-simulation likelihood. The legacy
    sampler fields below are unused by the Cobaya/Nautilus examples.
    """
    specification = deepcopy(dataset)
    specification["format"] = FORMAT
    specification["priors"] = {}
    for name, definition in parameters.items():
        if not isinstance(definition, dict) or "prior" not in definition:
            value = definition
            if isinstance(definition, dict):
                value = definition.get("value")
            if isinstance(value, (int, float)) and name in specification["initial"]:
                specification["initial"][name] = value
            continue
        prior = definition["prior"]
        kind = prior.get("dist", "uniform")
        if kind == "uniform":
            if "min" in prior:
                lower = prior["min"]
                upper = prior["max"]
            else:
                lower = prior["loc"]
                upper = lower + prior["scale"]
            entry = {"distribution": "uniform", "minimum": lower,
                     "maximum": upper}
        elif kind in ("norm", "truncnorm"):
            mean = prior.get("loc", 0.0)
            sigma = prior.get("scale", 1.0)
            entry = {"distribution": "normal", "mean": mean, "sigma": sigma}
            if kind == "truncnorm":
                entry["minimum"] = mean + sigma * prior["a"]
                entry["maximum"] = mean + sigma * prior["b"]
        else:
            raise ValueError(
                f"JointShear prior {name} uses {kind}; supported priors are "
                "uniform, norm and truncnorm."
            )
        entry["latex"] = definition.get("latex", name)
        specification["priors"][name] = entry
    specification["sampler"] = {
        "n_live": 1000,
        "n_networks": 4,
        "n_eff": 10000,
        "n_like_max": 100000,
        "f_live": 0.01,
        "seed": 42,
    }
    specification["output"] = "unused_by_cobaya"
    return specification


def load_inference(path):
    """Read inference choices without importing the compiled physical model.

    Arguments:
      path = inference YAML filename; relative paths are resolved beside it.
    Returns:
      parsed dictionary and absolute configuration path.
    """
    location = Path(path).resolve()
    specification = yaml.safe_load(location.read_text(encoding="utf-8"))
    if "format" not in specification:
        component = "des3pcf.cobaya_likelihood.JointShear"
        options = specification["likelihood"][component]
        dataset, location = read_dataset(
            data_file=options["data_file"], path=options.get("path"),
        )
        specification = cobaya_specification(
            parameters=specification["params"], dataset=dataset,
        )
    return validate_inference(specification=specification, location=location)


def validate_inference(specification, location):
    """Validate prepared inference choices before physical construction."""
    fields(
        mapping=specification,
        allowed={
            "format",
            "description",
            "notes",
            "prediction",
            "projection_overrides",
            "likelihood",
            "compression",
            "threads",
            "data",
            "selection",
            "initial",
            "priors",
            "check",
            "sampler",
            "output",
        },
        required={
            "format",
            "prediction",
            "threads",
            "data",
            "selection",
            "initial",
            "priors",
            "check",
            "sampler",
            "output",
        },
        name="inference",
    )
    if specification["format"] != FORMAT:
        raise ValueError(f"Inference format must be {FORMAT}.")
    projection_overrides(specification=specification)
    threads = specification["threads"]
    if isinstance(threads, bool) or int(threads) != threads or not 1 <= threads <= 4:
        raise ValueError("Inference threads must be an integer from one to four.")
    data = specification["data"]
    fields(
        mapping=data,
        allowed={"mode", "two_point", "aperture", "covariance"},
        required={"mode"}, name="data",
    )
    if data["mode"] not in ("synthetic_fiducial", "observed"):
        raise ValueError("Data mode must be synthetic_fiducial or observed.")
    if data["mode"] == "observed":
        for name in ("two_point", "aperture", "covariance"):
            if name not in data or not isinstance(data[name], str):
                raise ValueError(f"Observed data require a filename for {name}.")
    elif len(data) != 1:
        raise ValueError("Synthetic data come from the fiducial, not external vectors.")
    selection = specification["selection"]
    fields(
        mapping=selection, allowed={"mode", "indices"},
        required={"mode"}, name="selection",
    )
    if selection["mode"] == "all":
        if "indices" in selection:
            raise ValueError("Selection all must not also declare row indices.")
    elif selection["mode"] == "indices":
        if "indices" not in selection or not selection["indices"]:
            raise ValueError("Selection indices requires a nonempty integer list.")
    else:
        raise ValueError("Selection mode must be all or indices.")
    initial = specification["initial"]
    if not isinstance(initial, dict):
        raise ValueError("initial must be a parameter mapping, possibly empty.")
    for name, value in initial.items():
        if isinstance(value, bool) or not np.isfinite(value):
            raise ValueError(f"Initial {name}={value} must be finite.")
    prior_distributions(priors=specification["priors"])
    statistical_settings(specification=specification)
    check = specification["check"]
    fields(
        mapping=check, allowed={"parameter", "delta"},
        required={"parameter", "delta"}, name="check",
    )
    if (check["parameter"] not in specification["priors"]
            or isinstance(check["delta"], bool)
            or not np.isfinite(check["delta"]) or check["delta"] == 0.0):
        raise ValueError("check needs a sampled parameter and a finite nonzero delta.")
    sampler = specification["sampler"]
    fields(
        mapping=sampler,
        allowed={"n_live", "n_networks", "n_eff", "n_like_max", "f_live", "seed"},
        required={"n_live", "n_networks", "n_eff", "n_like_max", "f_live", "seed"},
        name="sampler",
    )
    for name in ("n_live", "n_networks", "n_eff", "n_like_max", "seed"):
        value = sampler[name]
        if isinstance(value, bool) or int(value) != value or value < 1:
            raise ValueError(f"sampler.{name} must be a positive integer.")
    if not 0.0 < sampler["f_live"] < 1.0:
        raise ValueError("sampler.f_live must lie strictly between zero and one.")
    return specification, location


def statistical_settings(specification):
    """Resolve explicit statistical choices before importing the physical model.

    Omitted sections retain the full selected vector and its fixed Gaussian
    likelihood. A compression must be supplied as a previously saved matrix;
    fitting a basis inside a likelihood evaluation would redefine the datum.
    Finite-simulation settings declare assumptions about the raw sample S.
    """
    compression = specification.get("compression", {"mode": "none"})
    fields(mapping=compression, allowed={"mode", "path"}, required={"mode"},
           name="compression")
    if compression["mode"] == "none":
        if "path" in compression:
            raise ValueError("compression.mode=none must not name a path.")
    elif compression["mode"] == "file":
        if not isinstance(compression.get("path"), str) or not compression["path"]:
            raise ValueError("compression.mode=file requires a saved matrix path.")
    else:
        raise ValueError("compression.mode must be none or file.")
    likelihood = specification.get("likelihood", {"mode": "gaussian"})
    fields(mapping=likelihood, allowed={"mode", "mock_count", "assumptions"},
           required={"mode"}, name="likelihood")
    if likelihood["mode"] == "gaussian":
        if len(likelihood) != 1:
            raise ValueError("Gaussian likelihood must not declare finite-mock fields.")
    elif likelihood["mode"] == "percival":
        count = likelihood.get("mock_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 6:
            raise ValueError("Percival likelihood.mock_count must be an integer >=6.")
        assumptions = likelihood.get("assumptions")
        fields(mapping=assumptions,
               allowed={"covariance_kind", "independent_mocks", "independent_data",
                        "gaussian_mocks", "compression_independent"},
               required={"covariance_kind", "independent_mocks", "independent_data",
                         "gaussian_mocks"},
               name="likelihood.assumptions")
        if assumptions["covariance_kind"] != "sample":
            raise ValueError("Percival requires covariance_kind=sample, without Hartlap.")
        for name in ("independent_mocks", "independent_data", "gaussian_mocks"):
            if assumptions[name] is not True:
                raise ValueError(f"Percival requires {name}=true as an assumption.")
        if compression["mode"] == "file":
            if assumptions.get("compression_independent") is not True:
                raise ValueError("Percival compression must be independent of sample S.")
    else:
        raise ValueError("likelihood.mode must be gaussian or percival.")
    return deepcopy(likelihood), deepcopy(compression)


def analysis_configuration(specification, location, driver):
    """Combine explicit inference settings with an unchanged prediction YAML.

    Arguments:
      specification, location = results from load_inference.
      driver = imported project run_3pcf module, shared with Fisher notebooks.
    Returns:
      prediction configuration/path/hash, baseline parameters and distributions.
    """
    prediction_path = (location.parent / specification["prediction"]).resolve()
    config, path, digest = driver.load_configuration(
        path=prediction_path, command="evaluate",
    )
    # The inference owner may choose a signed numerical chart while the
    # referenced Fisher file and any driver-owned dictionary remain intact.
    config = deepcopy(config)
    overrides = projection_overrides(specification=specification)
    if overrides:
        aperture = config["mean"]["aperture_settings"]
        prepared = aperture.setdefault("projection_settings", {})
        prepared.update(overrides)
    config["threads"] = specification["threads"]
    if config["covariance"]["mode"] != "file":
        raise ValueError("Inference requires a fixed covariance file in prediction.")
    if specification["data"]["mode"] == "observed":
        data = specification["data"]
        config["rows"]["mode"] = "measurement"
        for name in ("two_point", "aperture"):
            config["rows"][name] = str((location.parent / data[name]).resolve())
        config["covariance"]["path"] = str(
            (location.parent / data["covariance"]).resolve()
        )
    parameters = dict(config["ia"])
    parameters.update(config["parameters"])
    for name, value in specification["initial"].items():
        if name not in parameters:
            raise ValueError(f"Initial parameter {name} is absent from prediction.")
        parameters[name] = value
    distributions = prior_distributions(priors=specification["priors"])
    if config.get("feedback", {}).get("mode") == "bacco":
        if parameters.get("feedback_amplitude") != 1.0:
            raise ValueError("Dynamic BACCO requires fixed feedback_amplitude=1.")
        if "feedback_amplitude" in distributions:
            raise ValueError("Sample BACCO's physical parameters, not feedback_amplitude.")
    for name, distribution in distributions.items():
        if name not in parameters:
            raise ValueError(f"Sampled parameter {name} is absent from prediction.")
        if name in COSMOLOGY_NAMES and config["physics"]["cosmology_mode"] != "camb":
            raise ValueError(f"Sampling {name} requires physics.cosmology_mode=camb.")
        if not np.isfinite(distribution.logpdf(parameters[name])):
            raise ValueError(f"Initial {name} lies outside its normalized prior.")
    displaced = dict(parameters)
    name = specification["check"]["parameter"]
    displaced[name] += specification["check"]["delta"]
    if not np.isfinite(distributions[name].logpdf(displaced[name])):
        raise ValueError(f"The check displacement of {name} leaves its prior.")
    return config, path, digest, parameters, distributions


def inspect_inference(path, driver, specification=None):
    """List dependencies and choices without evaluating a model or covariance.

    The output is configuration inspection, not the --check calculation.
    Missing files are reported together so preparation can address them once.
    """
    if specification is None:
        specification, location = load_inference(path=path)
    else:
        specification, location = validate_inference(
            specification=specification, location=Path(path).resolve(),
        )
    config, prediction_path, digest, parameters, _ = analysis_configuration(
        specification=specification, location=location, driver=driver,
    )
    missing = driver.required_files(
        config=config, path=prediction_path, command="evaluate",
    )
    covariance_path = driver.resolve_path(
        configuration_path=prediction_path, value=config["covariance"]["path"],
    )
    if not covariance_path.is_file():
        missing.append({"input": "fixed joint covariance", "path": str(covariance_path)})
    statistics, compression = statistical_settings(specification=specification)
    if compression["mode"] == "file":
        compression_path = (location.parent / compression["path"]).resolve()
        if not compression_path.is_file():
            missing.append({"input": "fixed compression", "path": str(compression_path)})
    family = "normalized_fixed_gaussian"
    uncertainty = "not marginalized"
    if statistics["mode"] == "percival":
        family = "normalized_percival"
        uncertainty = "Percival finite-sample prescription under declared assumptions"
    return {
        "configuration": str(location),
        "configuration_sha256": file_digest(path=location),
        "prediction_configuration": str(prediction_path),
        "prediction_sha256": digest,
        "projection_overrides": projection_overrides(specification=specification),
        "data_mode": specification["data"]["mode"],
        "selection": specification["selection"],
        "sampled_parameters": list(specification["priors"]),
        "prior_specifications": specification["priors"],
        "fixed_and_initial_parameters": parameters,
        "likelihood": family,
        "likelihood_settings": statistics,
        "compression": compression,
        "covariance_uncertainty": uncertainty,
        "threads": specification["threads"],
        "missing_inputs": missing,
        "notes": specification.get("notes", []),
    }


def prepare_likelihood(path, driver, specification=None):
    """Build the same physical mean as Fisher, then fix data and covariance.

    Arguments:
      path = inference YAML.
      driver = project run_3pcf module supplying shared readers and constructors.
    Returns:
      likelihood owner, prior distributions, input specification and run record.
    Side effects:
      Imports compiled interfaces and constructs geometry; synthetic data
      additionally require one fiducial prediction. No sampling occurs.
    """
    record = inspect_inference(path=path, driver=driver,
                               specification=specification)
    if record["missing_inputs"]:
        raise FileNotFoundError(
            "Inference inputs are missing: " + json.dumps(record["missing_inputs"])
        )
    if specification is None:
        specification, location = load_inference(path=path)
    else:
        specification, location = validate_inference(
            specification=specification, location=Path(path).resolve(),
        )
    config, prediction_path, _, parameters, distributions = analysis_configuration(
        specification=specification, location=location, driver=driver,
    )
    rows = driver.rows_for_run(config=config, path=prediction_path)
    covariance = driver.read_matrix(
        settings=config["covariance"], path=prediction_path, rows=rows, positive=True,
    )
    selection = specification["selection"]
    indices = None
    if selection["mode"] == "indices":
        indices = np.asarray(selection["indices"])
        # This validation precedes physical model construction and applies
        # to the full, labeled vector rather than to an unlabeled count.
        rows.select(indices=indices)
    statistics, compression_settings = statistical_settings(specification=specification)
    compression = None
    if compression_settings["mode"] == "file":
        compression_path = (location.parent / compression_settings["path"]).resolve()
        compression = load_compression(path=compression_path)
        selected_rows = rows
        if indices is not None:
            selected_rows = rows.select(indices=indices)
        compression.check_labels(labels=selected_rows.labels)
    compiled = driver.compiled_extension(config=config)
    provider, table_metadata = driver.provider_for_run(
        compiled=compiled, config=config, path=prediction_path,
    )
    prediction = driver.prediction_for_run(
        compiled=compiled, config=config, path=prediction_path,
        rows=rows, provider=provider,
    )
    data = None
    if specification["data"]["mode"] == "observed":
        data = rows.values
    arguments = {
        "predict": prediction,
        "rows": rows,
        "covariance": covariance,
        "parameters": parameters,
        "data": data,
        "indices": indices,
        "compression": compression,
    }
    if statistics["mode"] == "gaussian":
        likelihood = FixedGaussianLikelihood(**arguments)
    else:
        likelihood = FiniteSimulationLikelihood(
            **arguments, mock_count=statistics["mock_count"],
            parameter_count=len(distributions), assumptions=statistics["assumptions"],
        )
        record["percival"] = {
            "mock_count": likelihood.mock_count,
            "parameter_count": likelihood.parameter_count,
            "dimension": likelihood.dimension,
            "exponent": likelihood.exponent,
        }
    paths = {
        "physical_tables": config["physics"]["tables"],
        "source_curves": config["source"]["path"],
        "joint_covariance": config["covariance"]["path"],
        "two_point_rows": config["rows"]["two_point"],
        "aperture_rows": config["rows"]["aperture"],
    }
    if config.get("feedback", {}).get("mode", "file") == "bacco":
        from .provenance import bacco_identity

        record["bacco_implementation"] = bacco_identity()
    elif "feedback" in config:
        paths["paired_feedback"] = config["feedback"]["path"]
    if "pixel_window" in config:
        paths["pixel_window"] = config["pixel_window"]["path"]
    if compression is not None:
        paths["compression"] = str(compression_path)
        record["compression_metadata"] = json.loads(compression.metadata_json)
        training_names = compression.metadata.get("parameter_names", ())
        outside_training = []
        for name in distributions:
            if name not in training_names:
                outside_training.append(name)
        record["compression_parameters_outside_training"] = outside_training
    record["input_files"] = {}
    for name, value in paths.items():
        filename = driver.resolve_path(configuration_path=prediction_path, value=value)
        record["input_files"][name] = {
            "path": str(filename),
            "sha256": file_digest(path=filename),
        }
    record["table_provenance"] = table_metadata
    record["effective_prediction_configuration"] = config
    two_point_path = None
    if prediction.two_point_core is not None:
        two_point_path = prediction.two_point_core.core.__file__
    project = Path(driver.__file__).resolve().parents[1]
    record.update(implementation_identity(
        project=project,
        three_point_path=compiled.__file__,
        two_point_path=two_point_path,
        use_camb=config["physics"]["cosmology_mode"] == "camb",
        extra_scripts=("EXAMPLE_NAUTILUS1.py", "EXAMPLE_NAUTILUS2.py",
                       "EXAMPLE_NAUTILUS3.py", "EXAMPLE_NAUTILUS4.py"),
    ))
    if specification["data"]["mode"] == "observed":
        manifest_path = project / "data" / "y3_observed_manifest.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        observed_root = project / "data" / "upstream_y3_observed"
        expected_paths = set()
        for item in manifest["files"]:
            if item["path"].endswith(".fits"):
                expected_paths.add(str(observed_root / item["path"]))
        actual_paths = set()
        for name in ("joint_covariance", "two_point_rows", "aperture_rows"):
            actual_paths.add(record["input_files"][name]["path"])
        if actual_paths == expected_paths:
            # Header inspection in the manifest applies only to its exact
            # FITS bytes. A locally edited file must not inherit that claim.
            for item in manifest["files"]:
                if not item["path"].endswith(".fits"):
                    continue
                content = (observed_root / item["path"]).read_bytes()
                header = f"blob {len(content)}\0".encode(encoding="ascii")
                digest = hashlib.sha1(header + content).hexdigest()
                if len(content) != item["bytes"] or digest != item["git_blob_sha1"]:
                    raise ValueError(
                        "Observed FITS differs from the pinned manifest: "
                        + item["path"]
                    )
            record["observed_file_provenance"] = manifest
    record["row_labels"] = likelihood.rows.labels.tolist()
    record["row_count"] = int(likelihood.rows.values.size)
    record["prediction_notes"] = config.get("notes", [])
    return likelihood, distributions, specification, record


def check_likelihood(likelihood, distributions, specification):
    """Evaluate fiducial, displaced and restored points before sampling.

    Returns:
      diagnostics of a changed physical mean and bitwise restored mean.
    Raises:
      ValueError if the displacement changes no retained observable or if
      restored state differs. An unavailable model raises its own diagnostic.
    """
    parameters = dict(likelihood.parameters)
    first = likelihood.mean(parameters=parameters)
    name = specification["check"]["parameter"]
    displaced = dict(parameters)
    displaced[name] += specification["check"]["delta"]
    second = likelihood.mean(parameters=displaced)
    restored = likelihood.mean(parameters=parameters)
    if not np.array_equal(first, restored):
        raise ValueError(
            "Restored fiducial differs bitwise after the check displacement."
        )
    if np.array_equal(first, second):
        raise ValueError(
            "Check displacement changes no selected mean; check model reach."
        )
    first_score, first_chi2 = likelihood.score_mean(mean=first)
    second_score, second_chi2 = likelihood.score_mean(mean=second)
    log_prior = 0.0
    for parameter, distribution in distributions.items():
        log_prior += float(distribution.logpdf(parameters[parameter]))
    return {
        "parameter": name,
        "delta": specification["check"]["delta"],
        "fiducial_log_prior": log_prior,
        "fiducial_log_likelihood": first_score,
        "fiducial_chi_squared": first_chi2,
        "displaced_log_likelihood": second_score,
        "displaced_chi_squared": second_chi2,
        "maximum_absolute_mean_change": float(np.max(np.abs(second - first))),
        "restored_bitwise": True,
    }
