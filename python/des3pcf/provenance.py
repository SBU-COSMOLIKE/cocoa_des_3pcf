"""Record the files that define one prepared prediction and its Fisher base.

A configuration filename does not identify its contents or the files it
references. Exact hashes distinguish a numerical-resolution comparison from
an accidental change of source curves, power tables, or physical code.
Fingerprinting reads files and, for a CAMB target, imports its package; it
never asks CAMB or a compiled shear model to calculate a prediction.

Beginning/end checks detect changes to these files during a long run. They
cannot inspect already-loaded Python code or native memory, detect a file
changed and then restored between checks, or make concurrent writes atomic.
Start from a fresh process with stable source and binary files; the hashes
are a consistency check on disk dependencies, not a memory attestation.
"""

import hashlib
from pathlib import Path


def file_digest(path):
    """Hash exact file bytes using bounded chunks."""
    digest = hashlib.sha256()
    with Path(path).open(mode="rb") as stream:
        while True:
            content = stream.read(1024 * 1024)
            if not content:
                break
            digest.update(content)
    return digest.hexdigest()


def implementation_identity(
    project, three_point_path, two_point_path=None, use_camb=False, extra_scripts=(),
):
    """Reuse the same physical-code fingerprint for inference and forecasts.

    Arguments:
      project = project root containing python/, interface/, and scripts/.
      three_point_path, two_point_path = actual extension paths; the second
          is omitted when the declared calculation uses no ordinary core.
      use_camb = whether displaced cosmologies can produce new CAMB tables.
      extra_scripts = additional project-relative script names, such as the
          sampler driver. run_3pcf.py is always part of the physical workflow.
    Returns:
      JSON-compatible hashes. No physical model is constructed.
    """
    project = Path(project).resolve()
    third = Path(three_point_path).resolve()
    result = {
        "three_point_extension": {
            "path": str(third),
            "sha256": file_digest(path=third),
        },
    }
    if two_point_path is not None:
        second = Path(two_point_path).resolve()
        result["two_point_extension"] = {
            "path": str(second),
            "sha256": file_digest(path=second),
        }
    implementation = list((project / "python" / "des3pcf").glob("*.py"))
    implementation.append(project / "scripts" / "run_3pcf.py")
    for name in extra_scripts:
        implementation.append(project / name)
    result["python_implementation_sha256"] = {}
    for filename in sorted(set(implementation)):
        relative = filename.relative_to(project).as_posix()
        result["python_implementation_sha256"][relative] = file_digest(path=filename)
    if use_camb:
        import camb
        from camb import baseconfig

        # A local Fortran rebuild may keep the same version label. Both
        # that library and its Python callers define newly produced tables.
        package = Path(camb.__file__).resolve().parent
        library = Path(baseconfig.CAMBL).resolve()
        result["camb_implementation"] = {
            "version": camb.__version__,
            "library_path": str(library),
            "library_sha256": file_digest(path=library),
            "python_sha256": {},
        }
        for filename in sorted(package.rglob("*.py")):
            relative = filename.relative_to(package).as_posix()
            result["camb_implementation"]["python_sha256"][relative] = file_digest(
                path=filename,
            )
    return result


def prediction_identity(
    config, configuration_path, project, three_point_path, two_point_path=None,
):
    """Bind a saved Fisher base to its physical files and implementation.

    The caller separately records its complete resolved configuration and
    checks labeled data rows and covariance values. This identity supplies
    the missing exact-byte dependency check. None-valued input paths are
    omitted only when the configuration generates those rows directly.
    """
    origin = Path(configuration_path).resolve().parent
    paths = {
        "physical_tables": config["physics"]["tables"],
        "source_curves": config["source"]["path"],
        "joint_covariance": config.get("covariance", {}).get("path"),
        "two_point_rows": config["rows"].get("two_point"),
        "aperture_rows": config["rows"].get("aperture"),
        "paired_feedback": config.get("feedback", {}).get("path"),
    }
    precision = config.get("fisher", {}).get("precision", {})
    if precision.get("mode") == "pme":
        paths["sample_covariance"] = precision["sample_covariance"]["path"]
    files = {}
    for name, value in paths.items():
        if value is None:
            continue
        filename = Path(value)
        if not filename.is_absolute():
            filename = origin / filename
        filename = filename.resolve()
        files[name] = {
            "path": str(filename),
            "sha256": file_digest(path=filename),
        }
    return {
        "format": "des3pcf-prediction-identity-v1",
        "input_files": files,
        "implementation": implementation_identity(
            project=project,
            three_point_path=three_point_path,
            two_point_path=two_point_path,
            use_camb=config["physics"]["cosmology_mode"] == "camb",
        ),
    }


def verify_prediction_identity(recorded, current, description):
    """Refuse missing provenance or changed files before accepting a result.

    This comparison makes no physical prediction and never fabricates a
    missing legacy identity from the files currently on disk.
    """
    if recorded is None:
        raise ValueError(
            f"{description} has no recorded physical dependency identity; "
            "rebuild the Fisher baseline before screening numerical settings."
        )
    if recorded != current:
        raise ValueError(
            f"{description}: physical input or implementation files changed; "
            "no result may be saved or reused under the former identity."
        )
