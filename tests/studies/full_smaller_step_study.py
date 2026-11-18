"""Complete all h/4 Fisher derivatives using saved h/2 stencil points.

Here h denotes each parameter's original absolute step. A five-point
fourth-order derivative at spacing h/4 needs the center and offsets
-h/2, -h/4, +h/4, +h/2. The center and outer points are already stored in
the full Fisher archive. Only the two inner points per parameter need a
new physical prediction: 34 means for the 17-parameter NLA example and
40 for the 20-parameter TATT example. An exactly matching two-column
step study can supply four of those means, reducing these counts to
30 and 36. No recorded vector enters a physical engine's cache.

The saved-vector lookup is passed to forecast.differentiate with spacing
h/2. That shared routine reconstructs both D_h2 and D_h4 in its original
arithmetic order. D_h2 must match the original archive bit for bit before
D_h4 is published. A smaller step can amplify numerical error; agreement
or disagreement between these derivatives does not establish integration,
angular-order, source-model or physical-model convergence.

From the project directory, inspect the proposed coordinates first:
  python tests/studies/full_smaller_step_study.py \
    runs/nla_accuracy/triangle_1p5.json --output runs/nla_full_h4 \
    --reuse-step-study runs/nla_triangle_step_study

Repeat with --execute only when physical work is authorized and the
numerical queue is available. Each calculated point receives a separate
archive and immutable receipt. The final full_h4.npz has its own study
format and records the original configuration plus an explicit step
divisor; it is not relabeled as an ordinary baseline Fisher result.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

if __name__ == "__main__":
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    os.environ["OMP_THREAD_LIMIT"] = "4"

import numpy as np


# Integer labels count quarters of the original absolute step. The full
# h/2 stencil includes these labels; the new calculation adds -1 and +1.
HALF_UNITS = (
    -4,
    -2,
    0,
    2,
    4,
)
QUARTER_UNITS = (
    -2,
    -1,
    0,
    1,
    2,
)
FORMAT = "des3pcf-full-smaller-step-v1"


def project_directory(requested):
    """Locate the installed project, or validate an explicit project directory."""
    if requested is not None:
        candidates = [Path(requested)]
    else:
        candidates = [
            Path(__file__).resolve().parent.parent.parent,
            Path.cwd(),
            Path.cwd() / "Cocoa/projects/des_3pcf",
        ]
    for candidate in candidates:
        if (candidate / "python/des3pcf/forecast.py").is_file():
            return candidate.resolve()
    raise ValueError("Pass --project with the des_3pcf project directory.")


def point_key(parameters):
    """Represent every scalar parameter exactly, including signed zero.

    Arguments:
      parameters = complete finite physical-parameter dictionary.
    Returns:
      immutable ordered pairs of names and hexadecimal float representations.
    """
    entries = []
    for name in sorted(parameters):
        value = float(parameters[name])
        if not np.isfinite(value):
            raise ValueError(f"Parameter {name} is nonfinite; supply a finite reference.")
        entries.append((name, value.hex()))
    return tuple(entries)


def quarter_point_plan(reference, steps, bounds):
    """Validate every h/4 coordinate before any physical prediction or file write.

    The complete union also includes the saved h/2 stencil's outer points.
    All coordinates must remain distinct, finite and within declared bounds.
    Steps that cannot be divided by four without changing their binary value
    on reconstruction are refused, rather than matching near-equal nodes.

    Returns:
      records for exactly -h/4 and +h/4 per parameter, in step-dictionary order.
    """
    point_key(parameters=reference)
    if not steps:
        raise ValueError("At least one absolute parameter step is required.")
    records = []
    for column, name in enumerate(steps):
        if name not in reference:
            raise ValueError(f"Missing reference value for varied parameter {name}.")
        step = float(steps[name])
        quarter = step / 4.0
        if (not np.isfinite(step) or step <= 0.0 or quarter <= 0.0
                or quarter * 4.0 != step):
            raise ValueError(f"Step {name} cannot define exact positive h/4 spacing.")
        lower = -np.inf
        upper = np.inf
        if name in bounds:
            lower, upper = bounds[name]
            if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
                raise ValueError(f"Bounds for {name} must be finite and ordered.")
        previous = None
        for unit in (
            -4,
            -2,
            -1,
            0,
            1,
            2,
            4,
        ):
            coordinate = float(reference[name]) + unit * quarter
            if not np.isfinite(coordinate) or not lower <= coordinate <= upper:
                raise ValueError(
                    f"The h/4 stencil for {name} leaves its declared bounds."
                )
            if previous is not None and coordinate <= previous:
                raise ValueError(f"The h/4 coordinates for {name} collapse in float64.")
            previous = coordinate
        for unit in (-1, 1):
            parameters = dict(reference)
            parameters[name] = float(reference[name]) + unit * quarter
            suffix = "minus" if unit < 0 else "plus"
            records.append({
                "id": f"column_{column:03d}_{suffix}",
                "column": column,
                "parameter": name,
                "quarter_step_unit": unit,
                "parameters": parameters,
                "parameter_key": point_key(parameters=parameters),
            })
    return records


def recorded_vector(bank, parameters, values, nrow):
    """Store an owned vector in a separate archive lookup, never an engine cache.

    Duplicate coordinates must contain identical float64 bits. Read-only
    storage and returned copies prevent one later array operation from
    changing an already verified archived value.
    """
    vector = np.array(object=values, dtype=np.float64, copy=True)
    if vector.shape != (nrow,) or not np.all(a=np.isfinite(vector)):
        raise ValueError("Recorded mean must be a finite vector in the full row order.")
    key = point_key(parameters=parameters)
    if key in bank:
        if not np.array_equal(a1=bank[key].view(np.uint64), a2=vector.view(np.uint64)):
            raise ValueError("The same physical coordinate has different recorded bits.")
    else:
        vector.flags.writeable = False
        bank[key] = vector


def baseline_bank(arrays, reference, steps):
    """Check shared archived nodes and collect the saved h/2 mean vectors.

    The original h stencil is checked against h/2 through the existing
    reuse_nodes helper. Only the half-step stencil is needed by the new
    calculation. No provider or physical predictor is constructed here.
    """
    from cosmology_step_study import reuse_nodes

    names = arrays["parameter_names"].tolist()
    if names != list(steps):
        raise ValueError("Saved parameter order differs from the configured steps.")
    nrow = arrays["mean"].size
    expected_shape = (
        len(names),
        5,
        nrow,
    )
    for key in ("stencil_values_h", "stencil_values_h2"):
        values = arrays[key]
        if (values.dtype != np.float64 or values.shape != expected_shape
                or not np.all(a=np.isfinite(values))):
            raise ValueError(f"Saved {key} must contain finite float64 full stencils.")
    bank = {}
    recorded_vector(bank=bank, parameters=reference, values=arrays["mean"], nrow=nrow)
    for column, name in enumerate(names):
        shared = reuse_nodes(
            stencil_h=arrays["stencil_values_h"][column],
            stencil_h2=arrays["stencil_values_h2"][column],
        )
        if not np.array_equal(
            a1=shared[0].view(np.uint64), a2=arrays["mean"].view(np.uint64),
        ):
            raise ValueError("A saved stencil center differs from the baseline mean.")
        quarter = float(steps[name]) / 4.0
        for node, unit in enumerate(HALF_UNITS):
            parameters = dict(reference)
            if unit != 0:
                parameters[name] = float(reference[name]) + unit * quarter
            recorded_vector(
                bank=bank, parameters=parameters,
                values=arrays["stencil_values_h2"][column, node], nrow=nrow,
            )
    return bank


def reconstruct_derivatives(bank, reference, steps, bounds, arrays):
    """Use the existing five-point derivative and require exact archived D_h2.

    Returns:
      D_h2, D_h4, h/4 stencil values and the varied parameter coordinates.
      The lookup refuses any point absent from the recorded bank; this
      function cannot make a physical prediction.
    """
    from des3pcf.forecast import differentiate

    def lookup(parameters):
        """Return only an exactly recorded physical coordinate, as an owned array."""
        key = point_key(parameters=parameters)
        if key not in bank:
            raise ValueError("A derivative requested an unrecorded physical coordinate.")
        return bank[key].copy()

    half_steps = {}
    for name, value in steps.items():
        half_steps[name] = float(value) / 2.0
    names, mean, derivative_half, derivative_quarter = differentiate(
        predict=lookup, parameters=reference, steps=half_steps, bounds=bounds,
    )
    saved_half = arrays["derivatives_h2"]
    if (saved_half.dtype != np.float64 or saved_half.shape != derivative_half.shape
            or not np.array_equal(
                a1=derivative_half.view(np.uint64), a2=saved_half.view(np.uint64),
            )):
        raise ValueError("Reconstructed D_h2 differs bitwise; do not publish D_h4.")
    if not np.array_equal(a1=mean.view(np.uint64), a2=arrays["mean"].view(np.uint64)):
        raise ValueError("Reconstructed center differs bitwise from the baseline.")
    stencil_shape = (
        len(names),
        5,
        mean.size,
    )
    stencil_values = np.empty(shape=stencil_shape, dtype=np.float64)
    coordinates = np.empty(shape=(len(names), 5), dtype=np.float64)
    for column, name in enumerate(names):
        quarter = float(steps[name]) / 4.0
        for node, unit in enumerate(QUARTER_UNITS):
            parameters = dict(reference)
            if unit != 0:
                parameters[name] = float(reference[name]) + unit * quarter
            coordinates[column, node] = parameters[name]
            stencil_values[column, node] = lookup(parameters=parameters)
    return (
        derivative_half,
        derivative_quarter,
        stencil_values,
        coordinates,
    )


def load_step_reuse(directory, identity, arrays, config, reference, steps):
    """Accept only a two-column study tied exactly to this Fisher baseline.

    The existing step-reader validator checks the stored five-point arrays,
    every shared coordinate and each declared newly evaluated physical point.
    Its noise scores and derivative arrays are not used to replace means.
    """
    from cosmolike_des_3pcf_accuracy_wrappers import _validate_step_vectors
    from des3pcf.provenance import verify_prediction_identity

    summary_path = directory / "summary.json"
    archive_path = directory / "stencils.npz"
    summary_text = summary_path.read_text(encoding="utf-8")
    summary = json.loads(s=summary_text)
    with np.load(file=archive_path, allow_pickle=False) as archive:
        saved = {}
        for key in archive.files:
            saved[key] = archive[key].copy()
    metadata_text = str(saved.pop("metadata_json").item())
    metadata = json.loads(s=metadata_text)
    if (summary != metadata or summary.get("format") != "des3pcf-cosmology-step-study-v1"
            or summary.get("executed") is not True):
        raise ValueError("Step reuse requires matching completed summary/NPZ metadata.")
    recorded = summary["identity"]
    for key in ("baseline_sha256", "configuration_sha256"):
        if recorded[key] != identity[key]:
            raise ValueError(f"Step reuse changes {key}; choose the matching baseline.")
    producer_digests = []
    for helper_path, helper_digest in identity["helper_sha256"].items():
        if Path(helper_path).name == "cosmology_step_study.py":
            producer_digests.append(helper_digest)
    if len(producer_digests) != 1:
        raise ValueError("Reuse identity must name exactly one point-generation script.")
    producer_digest = producer_digests[0]
    if recorded.get("study_sha256") != producer_digest:
        raise ValueError("Step reuse has a different point-generation script hash.")
    labels = []
    for case in summary["cases"]:
        labels.append(case["step_label"])
    required_labels = {
        "h",
        "h2",
        "h4",
    }
    allowed_labels = set(required_labels)
    allowed_labels.add("two_h")
    if (len(labels) != len(set(labels)) or not required_labels.issubset(labels)
            or not set(labels).issubset(allowed_labels)):
        raise ValueError("Step reuse requires unique h, h2, h4 and optional two_h cases.")
    verify_prediction_identity(
        recorded=recorded.get("prediction_identity"),
        current=identity["prediction_identity"], description="Optional step-study reuse",
    )
    names = arrays["parameter_names"].tolist()
    selected_names = summary["selected_parameters"]
    if selected_names != ["omega_m", "h"]:
        raise ValueError("Reuse requires the declared omega_m/h two-column study.")
    columns = []
    absolute_steps = []
    for name in selected_names:
        columns.append(names.index(name))
        absolute_steps.append(float(steps[name]))
    if not np.array_equal(a1=saved["selected_columns"], a2=np.asarray(a=columns)):
        raise ValueError("Step-reuse parameter indices differ from the full baseline.")
    if (not np.array_equal(a1=saved["parameter_names"], a2=arrays["parameter_names"])
            or not np.array_equal(a1=saved["row_labels"], a2=arrays["row_labels"])
            or not np.array_equal(a1=saved["absolute_base_steps"], a2=absolute_steps)):
        raise ValueError("Step-reuse names, row order or absolute steps differ.")
    _validate_step_vectors(
        summary=summary, arrays=saved, baseline=arrays, configuration=config,
        columns=columns, steps=np.asarray(a=absolute_steps),
    )
    reused = {}
    nrow = arrays["mean"].size
    for position, name in enumerate(selected_names):
        quarter = float(steps[name]) / 4.0
        for node, unit in ((1, -1), (3, 1)):
            parameters = dict(reference)
            parameters[name] = float(reference[name]) + unit * quarter
            recorded_vector(
                bank=reused, parameters=parameters,
                values=saved["stencil_values_h4"][position, node], nrow=nrow,
            )
    return reused, summary


def json_text(record):
    """Serialize an immutable record with stable key order and finite numbers."""
    text = json.dumps(obj=record, indent=2, allow_nan=False)
    return text + "\n"


def checked_text(path, content, write=False):
    """Refuse conflicting existing bytes; create only explicitly authorized records."""
    encoded = content.encode("utf-8")
    if path.exists():
        if not path.is_file() or path.read_bytes() != encoded:
            raise FileExistsError(f"Conflicting immutable study record: {path}")
    elif write:
        with path.open(mode="xb") as stream:
            stream.write(encoded)


def read_point(path, expected, nrow):
    """Validate one saved prediction against its exact coordinate and study identity."""
    with np.load(file=path, allow_pickle=False) as archive:
        metadata_text = str(archive["metadata_json"].item())
        metadata = json.loads(s=metadata_text)
        mean = archive["mean"].copy()
    # JSON turns tuples into arrays. Compare the serialized representation
    # consistently, retaining every hexadecimal coordinate value.
    expected_json = json.loads(s=json_text(record=expected))
    if metadata != expected_json:
        raise ValueError(f"Saved point metadata differs from this study: {path}")
    if (mean.dtype != np.float64 or mean.shape != (nrow,)
            or not np.all(a=np.isfinite(mean))):
        raise ValueError(f"Saved point is not a finite full-row float64 vector: {path}")
    return mean


def verified_point(path, receipt_path, expected, nrow, reused):
    """Verify an existing point's receipt and any independently known reused mean.

    An archive without its receipt is refused, including an interrupted write
    between these files. Inspect and remove that incomplete pair explicitly
    before resuming. A receipt is never invented for an existing archive.
    Hashing before and after the read also detects changes during loading.
    """
    from des3pcf.provenance import file_digest

    if not receipt_path.is_file():
        raise FileNotFoundError(
            f"Existing point lacks a receipt: {path}. Inspect and remove the "
            "incomplete pair before resuming."
        )
    digest_before = file_digest(path=path)
    record = expected["point"]
    receipt = {"point": record, "archive_sha256": digest_before}
    checked_text(path=receipt_path, content=json_text(record=receipt))
    mean = read_point(path=path, expected=expected, nrow=nrow)
    if file_digest(path=path) != digest_before:
        raise ValueError("A saved point changed while being loaded.")
    if record["origin"] == "two_column_step_study":
        key = point_key(parameters=record["parameters"])
        known = reused[key]
        if not np.array_equal(a1=mean.view(np.uint64), a2=known.view(np.uint64)):
            raise ValueError(
                "Saved reused mean differs bitwise from its verified source."
            )
    return mean


def run(configuration, output, project, reuse_step_study=None, execute=False):
    """Validate a full baseline and optionally calculate only missing h/4 means.

    Output means and receipts are immutable. Resumption rechecks every point
    before reusing it; recorded vectors remain separate from the live engine.
    The physical calculation uses all original numerical settings and priors.
    """
    project = project.resolve()
    output = output.resolve()
    for directory in (
        project / "tests/studies",
        project / "interface",
        project / "scripts",
        project / "python",
    ):
        sys.path.insert(0, str(directory))
    import run_3pcf as driver
    from des3pcf.provenance import file_digest, prediction_identity
    from des3pcf.provenance import verify_prediction_identity
    from des3pcf.fisher_comparison import compare_information, marginalized_errors
    from full_triangle_report import information_summary, derivative_changes, json_default
    from threadpoolctl import threadpool_limits

    blas_limit = threadpool_limits(limits=1, user_api="blas")
    config, path, config_digest = driver.load_configuration(
        path=configuration, command="fisher",
    )
    if config["fisher"].get("precision", {}).get("mode", "inverse") != "inverse":
        raise ValueError("This study requires the unchanged supplied covariance inverse.")
    if (isinstance(config["threads"], bool) or not isinstance(config["threads"], int)
            or not 1 <= config["threads"] <= 4):
        raise ValueError("Declare an integer from one through four OpenMP workers.")
    baseline_path = driver.resolve_path(
        configuration_path=path, value=config["outputs"]["fisher"],
    )
    baseline_digest = file_digest(path=baseline_path)
    with np.load(file=baseline_path, allow_pickle=False) as archive:
        arrays = {}
        for key in archive.files:
            arrays[key] = archive[key].copy()
    metadata_text = str(arrays.pop("metadata_json").item())
    metadata = json.loads(s=metadata_text)
    if (metadata["configuration_sha256"] != config_digest
            or metadata["configuration"] != config
            or metadata.get("command") != "fisher"):
        raise ValueError("Baseline configuration bytes or saved declaration differ.")
    offsets = [
        -2.0,
        -1.0,
        0.0,
        1.0,
        2.0,
    ]
    if metadata.get("stencil_offsets") != offsets:
        raise ValueError("Baseline must declare stencil offsets [-2, -1, 0, 1, 2].")
    rows = driver.rows_for_run(config=config, path=path)
    for key, expected in (
        ("row_labels", rows.labels),
        ("row_kind", rows.kind),
        ("row_bins", rows.bins),
        ("row_angles_radians", rows.angles),
        ("observed_values", rows.values),
    ):
        if not np.array_equal(a1=arrays[key], a2=expected):
            raise ValueError(f"Current {key} differs from the full Fisher baseline.")
    covariance = driver.read_matrix(
        settings=config["covariance"], path=path, rows=rows, positive=True,
    )
    if not np.array_equal(a1=covariance, a2=arrays["fiducial_covariance"]):
        raise ValueError("Current covariance differs from the full Fisher baseline.")
    reference = dict(config["ia"])
    reference.update(config["parameters"])
    steps = config["fisher"]["steps"]
    bounds = config["fisher"].get("bounds", {})
    plan = quarter_point_plan(reference=reference, steps=steps, bounds=bounds)
    bank = baseline_bank(arrays=arrays, reference=reference, steps=steps)
    third_extension = project / "interface/cosmolike_des_3pcf_interface.so"
    second_extension = None
    core_settings = config["mean"]["two_point"]
    if core_settings["mode"] == "core_tatt" and np.any(a=rows.kind != "map3"):
        core_directory = driver.resolve_path(
            configuration_path=path, value=core_settings["directory"],
        )
        second_extension = core_directory / (core_settings["module"] + ".so")
    reuse_directory = None
    if reuse_step_study is not None:
        reuse_directory = reuse_step_study.resolve()
    helper_files = (
        project / "tests/studies/cosmology_step_study.py",
        project / "tests/studies/full_triangle_report.py",
        project / "interface/cosmolike_des_3pcf_accuracy_wrappers.py",
        project / "interface/cosmolike_des_3pcf_fisher_wrappers.py",
    )

    def snapshot():
        """Hash the exact baseline, optional reused study and calculation dependencies."""
        helpers = {}
        for helper in helper_files:
            helpers[str(helper)] = file_digest(path=helper)
        reuse = None
        if reuse_directory is not None:
            reuse = {
                "directory": str(reuse_directory),
                "summary_sha256": file_digest(path=reuse_directory / "summary.json"),
                "archive_sha256": file_digest(path=reuse_directory / "stencils.npz"),
            }
        return {
            "baseline_sha256": file_digest(path=baseline_path),
            "configuration_sha256": file_digest(path=path),
            "study_sha256": file_digest(path=Path(__file__)),
            "helper_sha256": helpers,
            "reuse": reuse,
            "prediction_identity": prediction_identity(
                config=config, configuration_path=path, project=project,
                three_point_path=third_extension, two_point_path=second_extension,
            ),
        }

    beginning = snapshot()
    if (beginning["configuration_sha256"] != config_digest
            or beginning["baseline_sha256"] != baseline_digest):
        raise ValueError("Configuration or baseline changed while loading the study.")
    verify_prediction_identity(
        recorded=metadata.get("prediction_identity"),
        current=beginning["prediction_identity"],
        description="Full smaller-step baseline",
    )
    reused = {}
    reuse_summary = None
    if reuse_directory is not None:
        reused, reuse_summary = load_step_reuse(
            directory=reuse_directory, identity=beginning, arrays=arrays,
            config=config, reference=reference, steps=steps,
        )
    reused_count = 0
    point_records = []
    for record in plan:
        record = dict(record)
        record["origin"] = "prediction"
        if record["parameter_key"] in reused:
            record["origin"] = "two_column_step_study"
            reused_count += 1
        point_records.append(record)
    if reused_count != len(reused):
        raise ValueError("The reused archive contains coordinates outside the h/4 plan.")
    manifest = {
        "format": FORMAT,
        "baseline_configuration": str(path),
        "baseline_fisher": str(baseline_path),
        "identity": beginning,
        "parameter_names": list(steps),
        "absolute_base_steps": dict(steps),
        "step_divisor": 4,
        "points": point_records,
        "optional_reused_point_count": reused_count,
        "planned_physical_mean_count": len(plan) - reused_count,
        "threads": config["threads"],
        "blas_threads": 1,
        "scope": (
            "Full smaller-step derivatives at unchanged numerical and physical settings."
        ),
    }
    manifest_text = json_text(record=manifest)
    manifest_bytes = manifest_text.encode("utf-8")
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    allowed = {
        "manifest.json",
        "points",
        "full_h4.npz",
        "summary.json",
        "result_receipt.json",
    }
    if output.exists():
        if not output.is_dir():
            raise FileExistsError("Choose a directory for smaller-step outputs.")
        has_entries = False
        for entry in output.iterdir():
            has_entries = True
            if entry.name not in allowed:
                raise FileExistsError(f"Unexpected study output: {entry}")
        if has_entries and not (output / "manifest.json").is_file():
            raise FileExistsError("Existing outputs have no verified study manifest.")
    checked_text(path=output / "manifest.json", content=manifest_text)
    expected_points = set()
    pending = []
    for record in point_records:
        archive_name = record["id"] + ".npz"
        receipt_name = record["id"] + ".json"
        expected_points.add(archive_name)
        expected_points.add(receipt_name)
        archive_path = output / "points" / archive_name
        receipt_path = output / "points" / receipt_name
        expected = {"manifest_sha256": manifest_sha, "point": record}
        if archive_path.exists():
            mean = verified_point(
                path=archive_path, receipt_path=receipt_path, expected=expected,
                nrow=rows.values.size, reused=reused,
            )
            recorded_vector(
                bank=bank, parameters=record["parameters"], values=mean,
                nrow=rows.values.size,
            )
        else:
            if receipt_path.exists():
                raise FileNotFoundError(
                    f"Point receipt has no corresponding archive: {receipt_path}"
                )
            pending.append(record)
    point_directory = output / "points"
    if point_directory.exists():
        if not point_directory.is_dir():
            raise FileExistsError("The points output must be a directory.")
        for entry in point_directory.iterdir():
            if entry.name not in expected_points:
                raise FileExistsError(f"Unexpected point record: {entry}")
    final_path = output / "full_h4.npz"
    summary_path = output / "summary.json"
    final_receipt_path = output / "result_receipt.json"
    if final_path.exists() and pending:
        raise ValueError(
            "A completed h/4 archive has missing point records; restore them."
        )
    if final_receipt_path.exists():
        if not final_path.is_file() or not summary_path.is_file():
            raise FileNotFoundError("The final receipt requires both result and summary.")
        expected_receipt = {
            "manifest_sha256": manifest_sha,
            "result_sha256": file_digest(path=final_path),
            "summary_sha256": file_digest(path=summary_path),
        }
        checked_text(path=final_receipt_path,
                     content=json_text(record=expected_receipt))
    new_count = 0
    for record in pending:
        if record["origin"] == "prediction":
            new_count += 1
    preflight = {
        "manifest": manifest,
        "pending_point_archives": len(pending),
        "physical_means_needed_this_invocation": new_count,
        "execute_requested": execute,
    }
    verify_prediction_identity(recorded=beginning, current=snapshot(),
                               description="Smaller-step preflight")
    if not execute:
        print(json_text(record=preflight), end="")
        return preflight
    output.mkdir(parents=True, exist_ok=True)
    checked_text(path=output / "manifest.json", content=manifest_text, write=True)
    point_directory.mkdir(exist_ok=True)
    prediction = None
    if new_count > 0:
        compiled = driver.compiled_extension(config=config)
        provider, _ = driver.provider_for_run(compiled=compiled, config=config, path=path)
        prediction = driver.prediction_for_run(
            compiled=compiled, config=config, path=path, rows=rows, provider=provider,
        )
        third_extension = Path(compiled.__file__)
        if prediction.two_point_core is not None:
            second_extension = Path(prediction.two_point_core.core.__file__)
        verify_prediction_identity(recorded=beginning, current=snapshot(),
                                   description="Loaded smaller-step engine")
    evaluated = 0
    receipts = []
    for record in point_records:
        archive_path = point_directory / (record["id"] + ".npz")
        receipt_path = point_directory / (record["id"] + ".json")
        expected = {"manifest_sha256": manifest_sha, "point": record}
        if not archive_path.exists():
            verify_prediction_identity(recorded=beginning, current=snapshot(),
                                       description="Before smaller-step prediction")
            if record["origin"] == "two_column_step_study":
                mean = reused[record["parameter_key"]].copy()
            else:
                physical_parameters = dict(record["parameters"])
                predicted = prediction(parameters=physical_parameters)
                mean = np.array(object=predicted, dtype=np.float64, copy=True)
                evaluated += 1
            recorded_vector(bank=bank, parameters=record["parameters"], values=mean,
                            nrow=rows.values.size)
            verify_prediction_identity(recorded=beginning, current=snapshot(),
                                       description="After smaller-step prediction")
            with archive_path.open(mode="xb") as stream:
                np.savez_compressed(
                    file=stream, mean=mean,
                    metadata_json=np.asarray(a=json_text(record=expected)),
                )
        else:
            mean = verified_point(
                path=archive_path, receipt_path=receipt_path, expected=expected,
                nrow=rows.values.size, reused=reused,
            )
            recorded_vector(bank=bank, parameters=record["parameters"], values=mean,
                            nrow=rows.values.size)
        receipt = {"point": record, "archive_sha256": file_digest(path=archive_path)}
        checked_text(path=receipt_path, content=json_text(record=receipt), write=True)
        receipts.append(receipt)
    if evaluated != new_count:
        raise RuntimeError(
            "Physical mean count differs from the validated missing-point plan."
        )
    half, quarter, stencil_values, coordinates = reconstruct_derivatives(
        bank=bank, reference=reference, steps=steps, bounds=bounds, arrays=arrays,
    )
    baseline_statistics = information_summary(
        derivatives=half, arrays=arrays, metadata=metadata,
        saved_fisher=arrays["fisher_h2"], step="h2",
    )
    names = list(steps)
    comparison = compare_information(
        derivatives=quarter, names=names, kind=rows.kind, covariance=covariance,
        priors=config["fisher"]["priors"],
    )
    errors, ranks = marginalized_errors(comparison=comparison, names=names)
    quarter_steps = {}
    base_step_values = []
    quarter_step_values = []
    for name in names:
        quarter_steps[name] = float(steps[name]) / 4.0
        base_step_values.append(float(steps[name]))
        quarter_step_values.append(quarter_steps[name])
    statistics = information_summary(
        derivatives=quarter, arrays=arrays, metadata=metadata,
    )
    changes = derivative_changes(
        baseline=half, refined=quarter, covariance=covariance,
        names=names, steps=quarter_steps, label="h",
    )
    summary = {
        "format": FORMAT,
        "manifest_sha256": manifest_sha,
        "identity": beginning,
        "baseline_configuration": config,
        "step_divisor": 4,
        "reconstructed_h2_bitwise_equal": True,
        "physical_mean_count": manifest["planned_physical_mean_count"],
        "optional_reused_mean_count": reused_count,
        "rank_status_h4": ranks,
        "point_receipts": receipts,
        "reuse_study_provenance": reuse_summary,
        "fisher_summary_h4": statistics,
        "fisher_summary_h2": baseline_statistics,
        "column_changes_h2_to_h4": changes,
        "scope": (
            "Full h/4 derivative matrix with unchanged rows, covariance, priors and "
            "numerical model. Smaller steps can amplify numerical errors; this does "
            "not certify integration or angular-order convergence."
        ),
    }
    summary_text = json.dumps(
        obj=summary, indent=2, allow_nan=False, default=json_default,
    )
    summary_text += "\n"
    output_arrays = {
        "mean": arrays["mean"],
        "parameter_names": arrays["parameter_names"],
        "absolute_base_steps": np.asarray(a=base_step_values),
        "absolute_h4_steps": np.asarray(a=quarter_step_values),
        "stencil_values_h4": stencil_values,
        "stencil_coordinates_h4": coordinates,
        "derivatives_h2_reconstructed": half,
        "derivatives_h4": quarter,
        "fisher_h2_baseline": arrays["fisher_h2"],
        "fisher_h4": comparison["fisher_joint"],
        "fiducial_covariance": covariance,
        "row_labels": rows.labels,
        "row_kind": rows.kind,
        "row_bins": rows.bins,
        "row_angles_radians": rows.angles,
        "observed_values": rows.values,
    }
    output_arrays.update(comparison)
    output_arrays.update(errors)
    output_arrays["metadata_json"] = np.asarray(a=summary_text)
    for receipt in receipts:
        point_path = point_directory / (receipt["point"]["id"] + ".npz")
        if file_digest(path=point_path) != receipt["archive_sha256"]:
            raise ValueError(
                "A saved point changed before the full result was published."
            )
        receipt_path = point_directory / (receipt["point"]["id"] + ".json")
        checked_text(path=receipt_path, content=json_text(record=receipt))
    verify_prediction_identity(recorded=beginning, current=snapshot(),
                               description="Completed full smaller-step calculation")
    if final_path.exists():
        with np.load(file=final_path, allow_pickle=False) as archive:
            if set(archive.files) != set(output_arrays):
                raise ValueError("Existing full h/4 result has different array fields.")
            for key, value in output_arrays.items():
                saved = archive[key]
                if (saved.dtype != value.dtype or saved.shape != value.shape
                        or saved.tobytes() != value.tobytes()):
                    raise ValueError(f"Existing full h/4 result changes {key}.")
    else:
        with final_path.open(mode="xb") as stream:
            np.savez_compressed(file=stream, **output_arrays)
    checked_text(path=summary_path, content=summary_text, write=True)
    final_receipt = {
        "manifest_sha256": manifest_sha,
        "result_sha256": file_digest(path=final_path),
        "summary_sha256": file_digest(path=summary_path),
    }
    checked_text(path=final_receipt_path,
                 content=json_text(record=final_receipt), write=True)
    print(json_text(record={
        "result": str(final_path),
        "physical_means_this_invocation": evaluated,
        "physical_mean_count_total": summary["physical_mean_count"],
        "optional_reused_mean_count": reused_count,
        "reconstructed_h2_bitwise_equal": True,
    }), end="")
    del blas_limit
    return summary


def main():
    """Default to preflight; require explicit execution for new physical means."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("configuration", type=Path)
    parser.add_argument("--project", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reuse-step-study", type=Path)
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    project = project_directory(requested=arguments.project)
    run(configuration=arguments.configuration, output=arguments.output, project=project,
        reuse_step_study=arguments.reuse_step_study, execute=arguments.execute)


if __name__ == "__main__":
    main()
