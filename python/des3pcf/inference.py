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

from .likelihood import FixedGaussianLikelihood
from .pipeline import COSMOLOGY_NAMES


FORMAT = "cosmolike-3pcf-inference-v1"
DISCRETE_PARAMETERS = {
    "num_massive_neutrinos",
    "power_mode",
    "ct_mode",
    "suppress_only",
}


def file_digest(path):
    """Read a file in bounded chunks and return its SHA-256 identifier.

    Arguments:
      path = local input file.
    Returns:
      hexadecimal digest of its exact bytes, without interpreting the file.
    """
    digest = hashlib.sha256()
    with Path(path).open(mode="rb") as stream:
        while True:
            content = stream.read(1024 * 1024)
            if not content:
                break
            digest.update(content)
    return digest.hexdigest()


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


def load_inference(path):
    """Read inference choices without importing the compiled physical model.

    Arguments:
      path = inference YAML filename; relative paths are resolved beside it.
    Returns:
      parsed dictionary and absolute configuration path.
    """
    location = Path(path).resolve()
    specification = yaml.safe_load(location.read_text(encoding="utf-8"))
    fields(
        mapping=specification,
        allowed={
            "format",
            "description",
            "notes",
            "prediction",
            "projection_overrides",
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


def inspect_inference(path, driver):
    """List dependencies and choices without evaluating a model or covariance.

    The output is configuration inspection, not the --check calculation.
    Missing files are reported together so preparation can address them once.
    """
    specification, location = load_inference(path=path)
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
        "likelihood": "normalized_fixed_gaussian",
        "covariance_uncertainty": "not marginalized",
        "threads": specification["threads"],
        "missing_inputs": missing,
        "notes": specification.get("notes", []),
    }


def prepare_likelihood(path, driver):
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
    record = inspect_inference(path=path, driver=driver)
    if record["missing_inputs"]:
        raise FileNotFoundError(
            "Inference inputs are missing: " + json.dumps(record["missing_inputs"])
        )
    specification, location = load_inference(path=path)
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
    likelihood = FixedGaussianLikelihood(
        predict=prediction, rows=rows, covariance=covariance,
        parameters=parameters, data=data, indices=indices,
    )
    paths = {
        "physical_tables": config["physics"]["tables"],
        "source_curves": config["source"]["path"],
        "joint_covariance": config["covariance"]["path"],
        "two_point_rows": config["rows"]["two_point"],
        "aperture_rows": config["rows"]["aperture"],
    }
    record["input_files"] = {}
    for name, value in paths.items():
        filename = driver.resolve_path(configuration_path=prediction_path, value=value)
        record["input_files"][name] = {
            "path": str(filename),
            "sha256": file_digest(path=filename),
        }
    record["table_provenance"] = table_metadata
    record["effective_prediction_configuration"] = config
    record["three_point_extension"] = {
        "path": compiled.__file__,
        "sha256": file_digest(path=compiled.__file__),
    }
    if prediction.two_point_core is not None:
        extension_path = prediction.two_point_core.core.__file__
        record["two_point_extension"] = {
            "path": extension_path,
            "sha256": file_digest(path=extension_path),
        }
    # Physical input preparation and sampling also depend on Python source.
    # Recording these small files prevents a changed response or prior rule
    # from continuing a checkpoint merely because both C binaries match.
    project = Path(driver.__file__).resolve().parents[1]
    implementation = list((project / "python" / "des3pcf").glob("*.py"))
    implementation.append(Path(driver.__file__))
    implementation.append(project / "scripts" / "run_nautilus.py")
    record["python_implementation_sha256"] = {}
    for filename in sorted(implementation):
        relative = filename.relative_to(project).as_posix()
        record["python_implementation_sha256"][relative] = file_digest(path=filename)
    if config["physics"]["cosmology_mode"] == "camb":
        import camb
        from camb import baseconfig

        # Sampled cosmologies ask CAMB for additional power tables. Its
        # version string alone does not detect a locally rebuilt Fortran
        # library, so that library and its Python wrappers enter the identity.
        package = Path(camb.__file__).resolve().parent
        library = Path(baseconfig.CAMBL).resolve()
        record["camb_implementation"] = {
            "version": camb.__version__,
            "library_path": str(library),
            "library_sha256": file_digest(path=library),
            "python_sha256": {},
        }
        for filename in sorted(package.rglob("*.py")):
            relative = filename.relative_to(package).as_posix()
            record["camb_implementation"]["python_sha256"][relative] = file_digest(
                path=filename,
            )
    if specification["data"]["mode"] == "observed":
        manifest_path = project / "data" / "y3_observed_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
