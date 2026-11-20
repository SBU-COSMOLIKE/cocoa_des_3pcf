"""Notebook entry points for the same YAML-driven joint Fisher calculation.

The notebook reads a named configuration rather than installing mutable
module defaults. Each run calls the command-line driver in one separate
process, with bounded C workers and one BLAS worker. Reopening a notebook
therefore does not require restarting its kernel to change a configuration.
The small controlled IA demonstration below runs directly in the kernel;
it owns a fresh immutable physical model on every call.

These wrappers contain input preparation, file checks and orchestration.
Physical projection and intrinsic-alignment kernels remain in C. Statistical
matrix operations use the same des3pcf helpers as the command-line driver.
"""

import hashlib
import importlib
import json
import yaml
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
from scipy.linalg import solve_triangular


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT.parents[1] / "external_modules/code/cosmolike_core"))

import run_3pcf as driver
from cosmolike_notebook_utils.fisher import compare_information
from cosmolike_notebook_utils.fisher import parameter_covariance, symmetric_matrix
from des3pcf.provenance import prediction_identity
from des3pcf.provenance import verify_prediction_identity
from des3pcf.survey import ARCMIN_TO_RADIAN, read_source_distributions


class FisherWorkflow:
    """Retain a configuration path; reread its settings for every operation.

    Arguments:
      configuration = JSON/YAML path, relative to the current directory.
    Side effects:
      Construction writes nothing and imports no compiled extension. An
      explicit run launches the requested preparation or prediction command.
    """

    def __init__(self, configuration):
        self.configuration = Path(configuration).resolve()

    def inspect(self, command="fisher"):
        """Check the configuration and list missing inputs without physics.

        The reported displacement count includes six distinct off-center
        evaluations per varied parameter plus one shared fiducial mean.
        It is a work count, not a time or memory estimate.
        """
        config, path, digest = driver.load_configuration(
            path=self.configuration, command=command,
        )
        missing = driver.required_files(
            config=config, path=path, command=command,
        )
        report = {
            "description": config.get("description", ""),
            "configuration_sha256": digest,
            "missing_inputs": missing,
            "notes": config.get("notes", []),
            "threads": config["threads"],
        }
        if command == "fisher":
            names = list(config["fisher"]["steps"])
            report["varied_parameters"] = names
            report["distinct_mean_evaluations"] = 1 + 6 * len(names)
            report["gaussian_prior_sigmas"] = config["fisher"]["priors"]
        return report

    def public_covariance(self):
        """Load labeled rows and the declared full matrix without a mean run.

        Returns:
          rows, covariance and block counts in xip/xim/map3 order.
        Raises:
          ValueError for mismatched labels, estimator choices or positivity.
        """
        config, path, _ = driver.load_configuration(
            path=self.configuration, command="fisher",
        )
        rows = driver.rows_for_run(config=config, path=path)
        matrix = driver.read_matrix(
            settings=config["covariance"], path=path, rows=rows, positive=True,
        )
        counts = {}
        for name in ("xip", "xim", "map3"):
            counts[name] = int(np.count_nonzero(rows.kind == name))
        return rows, matrix, counts

    def run(self, command, overwrite=False):
        """Execute one explicit CLI command and return the saved result path.

        Production preparation and a full Fisher stencil can be expensive.
        This method is never called by inspect or by result loading. Its
        child process inherits the Cocoa environment, uses BLAS=1 and leaves
        the notebook's ordinary CosmoLike module state unchanged.
        """
        if command not in ("prepare", "evaluate", "covariance", "fisher"):
            raise ValueError("Choose prepare, evaluate, covariance or fisher.")
        self.inspect(command=command)
        arguments = [
            sys.executable,
            "-B",
            str(PROJECT / "scripts" / "run_3pcf.py"),
            command,
            str(self.configuration),
        ]
        if overwrite:
            arguments.append("--overwrite")
        environment = dict(os.environ)
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
            environment[name] = "1"
        result = subprocess.run(
            args=arguments, env=environment, capture_output=True, text=True,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"3pcf {command} failed with status {result.returncode}:\n"
                f"{result.stderr}\n{result.stdout}"
            )
        # Native libraries may print progress before the driver's final
        # output-path line. Do not interpret those diagnostics as a filename.
        lines = result.stdout.strip().splitlines()
        if not lines:
            raise RuntimeError("The driver returned no saved-output path.")
        return Path(lines[-1])

    def load_result(self, command="fisher"):
        """Read an archived result with its original recorded provenance.

        This historical reader checks YAML identity but does not establish
        that the current implementation or source files match the archive.
        Use ensure_result for an execute-or-reuse workflow with that check.
        The file digest includes comments. Editing a configuration after a
        run requires a new result or explicit inspection of the old file;
        it cannot silently relabel an old forecast as the new calculation.
        """
        config, path, digest = driver.load_configuration(
            path=self.configuration, command=command,
        )
        if command == "covariance":
            filename = config["covariance"]["path"]
        else:
            filename = config["outputs"][command]
        output = driver.resolve_path(configuration_path=path, value=filename)
        arrays = {}
        with np.load(output, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            if metadata["configuration_sha256"] != digest:
                raise ValueError(
                    "Saved result belongs to a different configuration."
                )
            for name in archive.files:
                if name != "metadata_json":
                    arrays[name] = archive[name].copy()
        return arrays, metadata

    def ensure_result(self, command="fisher"):
        """Read a matching saved result, or explicitly run the missing work.

        Arguments:
          command = evaluate or fisher; physical preparation is shared.
        Returns:
          Owned arrays and metadata, checked against the current YAML and
          measurement row order. Fisher files also retain the exact supplied
          covariance. Current physical input and implementation fingerprints
          must match; a legacy archive without an identity is refused.
        Raises:
          ValueError for an existing stale result; it is never overwritten.
          RuntimeError when a requested child calculation fails.
        """
        if command not in ("evaluate", "fisher"):
            raise ValueError("Choose evaluate or fisher for a saved mean run.")
        config, path, _ = driver.load_configuration(
            path=self.configuration, command=command,
        )
        output = driver.resolve_path(
            configuration_path=path, value=config["outputs"][command],
        )
        tables = driver.resolve_path(
            configuration_path=path, value=config["physics"]["tables"],
        )
        if not output.is_file():
            if not tables.is_file():
                print(f"Preparing physical tables: {tables}")
                self.run(command="prepare")
            print(f"Running {command}: {self.configuration.name}")
            self.run(command=command)
        else:
            print(f"Reading saved {command}: {output}")
        arrays, metadata = self.load_result(command=command)
        second_extension = None
        core = config["mean"]["two_point"]
        if core["mode"] == "core_tatt" and np.any(arrays["row_kind"] != "map3"):
            directory = driver.resolve_path(path, core["directory"])
            second_extension = directory / (core["module"] + ".so")
        current_identity = prediction_identity(
            config=config,
            configuration_path=path,
            project=PROJECT,
            three_point_path=PROJECT / "interface/cosmolike_des_3pcf_interface.so",
            two_point_path=second_extension,
        )
        verify_prediction_identity(
            recorded=metadata.get("prediction_identity"),
            current=current_identity,
            description=f"Saved {command} selected for reuse",
        )
        rows = driver.rows_for_run(config=config, path=path)
        for key, expected in (
            ("row_labels", rows.labels),
            ("row_kind", rows.kind),
            ("row_bins", rows.bins),
            ("row_angles_radians", rows.angles),
        ):
            if not np.array_equal(arrays[key], expected):
                raise ValueError(f"Saved result has different {key}.")
        if command == "fisher":
            matrix = driver.read_matrix(
                settings=config["covariance"], path=path,
                rows=rows, positive=True,
            )
            if not np.array_equal(arrays["fiducial_covariance"], matrix):
                raise ValueError("Saved Fisher uses a different covariance.")
        return arrays, metadata


def verify_public_inputs():
    """Compare downloaded Y3 bytes with the pinned Git blob identifiers.

    Git's blob identifier hashes its byte-count header followed by the
    file bytes. A matching filename alone does not establish provenance.
    Missing files are reported; this helper never downloads replacements.
    """
    manifest_path = PROJECT / "data" / "y3_reference_manifest.yaml"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    records = []
    for record in manifest["files"]:
        path = PROJECT / "data" / "upstream_y3" / record["path"]
        status = "missing"
        if path.is_file():
            content = path.read_bytes()
            header = f"blob {len(content)}\0".encode("ascii")
            digest = hashlib.sha1(header + content).hexdigest()
            status = "matches_pinned_blob"
            if digest != record["git_blob_sha1"]:
                status = "different_bytes"
        records.append({"file": record["path"], "status": status})
    return {
        "repository": manifest["repository"],
        "commit": manifest["commit"],
        "files": records,
    }


def controlled_tables():
    """Construct small smooth inputs for teaching IA responses, not DES data.

    Units are k[h/Mpc], P[(Mpc/h)^3], chi[Mpc/h] and dimensionless growth.
    The nonlinear redshift law deliberately differs from D squared so that
    power_mode1 and2 represent distinguishable choices. Source curves have
    two broad peaks. These analytic shapes are controlled inputs, not a CAMB
    cosmology, a measurement, or a calibration of BiHalofit/TATT accuracy.
    """
    wave = np.geomspace(start=1.0e-5, stop=1.0e3, num=96)
    redshift = np.linspace(start=0.0, stop=2.0, num=41)
    linear = 2.0e4 * wave / (1.0 + (wave / 0.15)**2)**1.8
    nonlinear0 = linear * (1.0 + (wave / 0.35)**2)**0.22
    sources = np.empty((2, redshift.size), dtype=np.float64)
    sources[0] = redshift**2 * np.exp(-(redshift / 0.6)**1.5)
    sources[1] = redshift**2 * np.exp(-(redshift / 0.95)**1.5)
    return {
        "k": wave,
        "p_linear0": linear,
        "z": redshift,
        "p_nonlinear": nonlinear0[None, :] / (1.0 + redshift[:, None])**2.4,
        "chi": 3000.0 * redshift / (1.0 + 0.35 * redshift),
        "dz_dchi": (1.0 + 0.35 * redshift)**2 / 3000.0,
        "growth": 1.0 / (1.0 + redshift),
        "sources": sources,
        "los_z": np.array([0.15, 0.5, 1.0, 1.8]),
        "omega_m": 0.3,
        "omega_de": 0.7,
        "w0": -1.0,
        "wa": 0.0,
        "ns": 0.965,
        "sigma8": 0.8,
    }


def controlled_ia_response(power_mode=2, threads=1):
    """Evaluate a small projected-bispectrum IA sign comparison in C.

    The two nonzero choices mirror the sign reversal used to discuss IA
    degeneracies: (A1,A2)=(-0.2,0.4) and (0.2,-0.4). Their actual projected
    responses depend on cosmology, source curves and the chosen statistic.
    This demonstration uses twelve equilateral angular triangles and two
    auto-source triplets; it does not reproduce a published Map3 figure.

    Returns:
      dictionary with ell, signed total projected bispectra, case labels,
      source labels and the complete controlled input/model declaration.
    """
    if (isinstance(threads, bool) or int(threads) != threads
            or not 1 <= threads <= 4 or power_mode not in (0, 1, 2)):
        raise ValueError("Use threads1..4 and power_mode0,1 or2.")
    compiled = importlib.import_module(name="cosmolike_des_3pcf_interface")
    compiled.set_omp_threads(nthreads=int(threads))
    from theory.bihalofit.model import bispectrum_axes, prepare_reference_bispectrum

    tables = controlled_tables()
    # The toy provider derives its pressureless fraction from its own H(z).
    # These arrays teach IA responses; they are not a calibrated cosmology.
    expansion = 2997.92458 * tables["dz_dchi"]
    tables["omega_m_z"] = tables["omega_m"] * (1.0 + tables["z"])**3
    tables["omega_m_z"] /= expansion**2
    tables, baseline = prepare_reference_bispectrum(
        tables=tables,
        axes=bispectrum_axes(zmax=tables["z"][-1], counts=(9, 65, 33, 17)),
    )
    resolved_model = compiled.accuracy_model_settings(
        accuracy={},
        kernel_nz=101,
        table_boost=8,
        fit_nz=33,
    )
    model = compiled.production.PhysicsModelWithBispectrum(
        k=tables["k"],
        p_linear0=tables["p_linear0"],
        z=tables["z"],
        p_nonlinear=tables["p_nonlinear"],
        chi=tables["chi"],
        dz_dchi=tables["dz_dchi"],
        growth=tables["growth"],
        sources=tables["sources"],
        omega_m=tables["omega_m"],
        sigma8=tables["sigma8"],
        omega_m_z=tables["omega_m_z"],
        source_z=tables.get("source_z"),
        los_z=compiled.accuracy_los_nodes(accuracy={}, baseline=tables["los_z"]),
        baseline=baseline["native"],
        kernel_nz=resolved_model["kernel_nz"],
        table_boost=resolved_model["table_boost"],
        fit_nz=resolved_model["fit_nz"],
        source_readers=False,
    )
    ell = np.geomspace(start=30.0, stop=3000.0, num=12)
    sides = np.ascontiguousarray(np.repeat(ell[:, None], repeats=3, axis=1))
    triplets = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)
    parameters = [
        {
            "a1": 0.0,
            "a2": 0.0,
            "bta": 0.0,
            "ct": 0.0,
        },
        {
            "a1": -0.2,
            "a2": 0.4,
            "bta": 0.91,
            "ct": 0.0,
        },
        {
            "a1": 0.2,
            "a2": -0.4,
            "bta": 0.91,
            "ct": 0.0,
        },
    ]
    signals = np.empty((len(parameters), len(triplets), len(ell)))
    for case, ia in enumerate(parameters):
        ia["power_mode"] = int(power_mode)
        for index, bins in enumerate(triplets):
            signals[case, index] = model.projected(
                sides=sides, bins=bins, parameters=ia,
            )
    return {
        "ell": ell,
        "signals": signals,
        "triplets": triplets,
        "labels": ["No IA", "A1=-0.2, A2=0.4", "A1=0.2, A2=-0.4"],
        "parameters": parameters,
        "interpretation": "Controlled inputs; no DES accuracy claim.",
    }


def demonstration_workflow(model):
    """Select one explicitly labeled DES Y3 demonstration configuration.

    Arguments:
      model = nla or tatt; both vary w0 and wa and keep all public rows.
    Returns:
      FisherWorkflow without launching CAMB or any native calculation.
    Raises:
      ValueError for an unknown model; FileNotFoundError if YAML is absent.
    """
    if model not in ("nla", "tatt"):
        raise ValueError("The demonstration model must be nla or tatt.")
    path = PROJECT / "configs" / f"des_y3_{model}_demo_fisher.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    return FisherWorkflow(configuration=path)


def source_report(workflow):
    """Read the declared midpoint curves and their preprocessing record.

    Arguments:
      workflow = FisherWorkflow containing the demonstration source path.
    Returns:
      Redshift, source density[bin,z], midpoint integrals and YAML sidecar.
      Source FITS SHA256 must match that sidecar; original bytes are untouched.
    Raises:
      ValueError for an inconsistent sidecar or an invalid source curve.
    """
    config, path, _ = driver.load_configuration(
        path=workflow.configuration, command="prepare",
    )
    source = driver.resolve_path(
        configuration_path=path, value=config["source"]["path"],
    )
    redshift, density = read_source_distributions(
        path=source, nbin=config["source"]["nbin"],
        extension=config["source"]["extension"],
    )
    sidecar = source.with_suffix(".yaml")
    record = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if record["output"]["sha256"] != digest:
        raise ValueError("Source FITS differs from its preprocessing record.")
    return {
        "redshift": redshift,
        "density": density,
        "midpoint_integrals": np.trapz(density, x=redshift, axis=1),
        "preprocessing": record,
    }


def plot_sources(report):
    """Draw one panel per declared source bin through the shared plot package.

    Arguments:
      report = source_report output, with midpoint redshift and densities.
    Returns:
      Shared Matplotlib figure and axes; no model or normalization is applied.
    """
    plots = importlib.import_module(name="cosmolike_notebook_utils.3pcf")
    labels = []
    for index in range(report["density"].shape[0]):
        labels.append(f"Source bin {index + 1}")
    return plots.plot_ia_components(
        components=[(report["redshift"], report["density"])],
        component_labels=labels, coordinate_label=r"Redshift $z$",
        quantity_label=r"$n_i(z)$", legend=["Clipped and normalized"],
        xscale="linear", ncols=4, figsize=(15, 3.8),
        bintextpos=None, colorbar=None, show=None,
    )


def ia_sign_workflows(workflow):
    """Write two explicit mean configurations with opposite A1/A2 signs.

    The cosmology, source tables, all other IA parameters, estimators and
    numerical grids are copied from the TATT demonstration. Absolute input
    paths allow the generated JSON files to live beside their saved outputs.
    Existing files must match byte for byte and are never overwritten.

    Arguments:
      workflow = base TATT FisherWorkflow; its fiducial remains a third case.
    Returns:
      Three workflows and mathematical labels in fiducial/negative/positive
      A1 order. Their evaluate commands compute actual xi and Map3 rows.
    Raises:
      ValueError if the base is not TATT or generated files have changed.
    """
    config, path, _ = driver.load_configuration(
        path=workflow.configuration, command="evaluate",
    )
    if config["mean"]["two_point"]["mode"] != "core_tatt":
        raise ValueError("The sign comparison requires the core TATT model.")
    directory = PROJECT / "chains" / "ia_response"
    cases = (
        ("negative_a1", -0.2, 0.4),
        ("positive_a1", 0.2, -0.4),
    )
    prepared = []
    for tag, amplitude1, amplitude2 in cases:
        derived = json.loads(s=json.dumps(obj=config))
        path_fields = (
            ("source", "path"),
            ("physics", "tables"),
            ("rows", "two_point"),
            ("rows", "aperture"),
            ("covariance", "path"),
        )
        for section, key in path_fields:
            resolved = driver.resolve_path(
                configuration_path=path, value=derived[section][key],
            )
            derived[section][key] = str(resolved)
        two_point = derived["mean"]["two_point"]
        two_point["directory"] = str(driver.resolve_path(
            configuration_path=path, value=two_point["directory"],
        ))
        derived["ia"]["a1"] = amplitude1
        derived["ia"]["a2"] = amplitude2
        derived["description"] = (
            f"Demonstration IA sign response: A1={amplitude1}, "
            f"A2={amplitude2}; all other inputs fixed."
        )
        for command in ("evaluate", "fisher"):
            derived["outputs"][command] = str(
                directory / f"{tag}_{command}.npz"
            )
        target = directory / f"{tag}.yaml"
        text = yaml.safe_dump(derived, sort_keys=False)
        if target.exists() and target.read_text(encoding="utf-8") != text:
            raise ValueError(f"Existing sign configuration differs: {target}")
        prepared.append((target, text))
    # Validate every existing destination before creating either case.
    directory.mkdir(parents=True, exist_ok=True)
    workflows = [workflow]
    for target, text in prepared:
        if not target.exists():
            with target.open(mode="x", encoding="utf-8") as stream:
                stream.write(text)
        workflows.append(FisherWorkflow(configuration=target))
    labels = [
        "TATT fiducial",
        r"$A_1=-0.2,\ A_2=0.4$",
        r"$A_1=0.2,\ A_2=-0.4$",
    ]
    return workflows, labels


def plot_mean_responses(results, labels, kind, residual=False):
    """Arrange actual saved auto-bin predictions without changing row weights.

    Arguments:
      results = ordered result dictionaries from ensure_result(evaluate).
      labels = one label per case; first case is the comparison reference.
      kind = xip, xim or map3; plots select four auto-bin groups for legibility.
      residual = True draws 100*(prediction-reference)/max|reference| in
          each bin. It never divides by a zero-crossing individual sample.
    Returns:
      Shared figure/axes; all 480 rows remain in the Fisher calculations.
    Raises:
      ValueError for mismatched labels, unequal-radius Map3 or angular grids.
    """
    if kind not in ("xip", "xim", "map3") or len(results) != len(labels):
        raise ValueError("Supply matching cases and xip, xim or map3.")
    plots = importlib.import_module(name="cosmolike_notebook_utils.3pcf")
    first = results[0]
    panels = []
    panel_labels = []
    common = None
    for source in range(4):
        selected = first["row_kind"] == kind
        selected &= first["row_bins"][:, 0] == source
        selected &= first["row_bins"][:, 1] == source
        if kind == "map3":
            selected &= first["row_bins"][:, 2] == source
        indices = np.flatnonzero(selected)
        angles = first["row_angles_radians"][indices]
        column = 2
        if kind == "map3":
            if not np.all(angles == angles[:, :1]):
                raise ValueError("This plot requires equal aperture radii.")
            column = 0
        coordinate = angles[:, column] / ARCMIN_TO_RADIAN
        order = np.argsort(coordinate)
        coordinate = coordinate[order]
        if common is None:
            common = coordinate
        elif not np.array_equal(common, coordinate):
            raise ValueError("Auto-bin panels have different angular grids.")
        panels.append(indices[order])
        panel_labels.append(f"Source bin {source + 1}")
    curves = []
    for result in results:
        if not np.array_equal(result["row_labels"], first["row_labels"]):
            raise ValueError("Mean cases use different measurement rows.")
        curves.append((common, result["mean"][np.asarray(panels)]))
    reference = None
    if residual:
        reference = curves[0]
        curves = curves[1:]
        labels = labels[1:]
    quantity = r"$\xi_+(\theta)$"
    coordinate_label = r"$\theta$ [arcmin]"
    if kind == "xim":
        quantity = r"$\xi_-(\theta)$"
    elif kind == "map3":
        quantity = r"$\langle M_{\rm ap}^3(R)\rangle$"
        coordinate_label = r"$R$ [arcmin]"
    return plots.plot_ia_components(
        components=curves, components_ref=reference,
        component_labels=panel_labels, coordinate_label=coordinate_label,
        quantity_label=quantity, residual="scaled", legend=labels,
        ncols=4, figsize=(16, 4.3), bintextpos=None,
        colorbar=None, show=None,
    )


def fisher_summary(result, metadata):
    """Summarize rank checks and exact accounting of conditional information.

    Arguments:
      result, metadata = saved Fisher arrays and matching run declaration.
    Returns:
      Names, available marginalized errors, rank status, prior declaration,
      and the relative norm of Fjoint-F2-F3|2 before adding priors. Each
      derivative change also has a covariance-weighted norm, divided by
      its h/2 norm using the saved fiducial covariance. A zero reference
      gives zero for identical columns and infinity for a nonzero change.
      Missing covariances are reported rather than replaced by a pseudoinverse.
    """
    joint = result["fisher_joint_likelihood"]
    difference = joint - result["fisher_two_point_likelihood"]
    difference -= result["fisher_conditional_three_point"]
    scale = np.linalg.norm(joint)
    # Xi and aperture moments have very different numerical amplitudes.
    # Whitening [ndata,nparameter] by the full covariance gives every row
    # its measured error scale and retains cross correlations. The ratio
    # is unchanged by rescaling the units of either observable.
    cholesky = np.linalg.cholesky(a=result["fiducial_covariance"])
    derivative_difference = result["derivatives_h"] - result["derivatives_h2"]
    whitened_difference = solve_triangular(
        a=cholesky, b=derivative_difference, lower=True,
    )
    whitened_reference = solve_triangular(
        a=cholesky, b=result["derivatives_h2"], lower=True,
    )
    difference_norm = np.linalg.norm(x=whitened_difference, axis=0)
    reference_norm = np.linalg.norm(x=whitened_reference, axis=0)
    derivative_change = np.full(shape=reference_norm.shape, fill_value=np.inf)
    np.divide(difference_norm, reference_norm, out=derivative_change,
              where=reference_norm > 0.0)
    derivative_change[(reference_norm == 0.0) & (difference_norm == 0.0)] = 0.0
    report = {
        "names": result["parameter_names"].tolist(),
        "rank_status": metadata["comparison_rank_status"],
        "gaussian_prior_sigmas": metadata["configuration"]["fisher"]["priors"],
        "information_identity_relative_error": float(
            np.linalg.norm(difference) / scale
        ),
        "derivative_relative_change": result["derivative_relative_change"],
        "derivative_covariance_relative_change": derivative_change,
    }
    for name in (
        "sigma_two_point",
        "sigma_joint",
        "sigma_ratio_two_point_over_joint",
    ):
        if name in result:
            report[name] = result[name].copy()
    return report


def plot_ia_errors(result):
    """Compare joint and xi-only marginalized errors in the same IA model.

    Arguments:
      result = saved Fisher arrays with resolved two_point and joint errors.
    Returns:
      Shared figure showing sigma_joint/sigma_xi for varied IA parameters.
    Raises:
      ValueError if either forecast has an unresolved parameter direction.
    """
    for name in ("sigma_two_point", "sigma_joint"):
        if name not in result:
            raise ValueError("Resolve the Fisher rank before plotting errors.")
    names = result["parameter_names"].tolist()
    indices = []
    labels = []
    selected_names = []
    display = {
        "a1": r"$A_1$",
        "eta1": r"$\eta_1$",
        "a2": r"$A_2$",
        "eta2": r"$\eta_2$",
        "bta": r"$b_{\rm TA}$",
    }
    for index, name in enumerate(names):
        if name in display:
            indices.append(index)
            labels.append(display[name])
            selected_names.append(name)
    plots = importlib.import_module(name="cosmolike_notebook_utils.3pcf")
    return plots.plot_parameter_errors(
        names=selected_names, labels=labels,
        errors=[result["sigma_joint"][indices]],
        errors_ref=result["sigma_two_point"][indices],
        legend=[r"$\xi_\pm+M_{\rm ap}^3$ relative to $\xi_\pm$"],
        figsize=(9, 4.5), show=None,
    )


def plot_common_contours(results, metadata, labels, parameters):
    """Marginalize each full model before comparing common Gaussian contours.

    Arguments:
      results, metadata, labels = matching model records and figure labels.
      parameters = common parameter names, e.g. omega_m, sigma8, w0, wa.
    Returns:
      Shared Fisher triangle plotter, with each model's other parameters
      marginalized. No flat-prior truncation or pseudoinverse is applied.
    Raises:
      ValueError if a model is unresolved, a parameter is absent, or the
      selected models have different fiducial values for a shown parameter.
    """
    if len(results) != len(metadata) or len(results) != len(labels):
        raise ValueError("Supply one declaration and label per forecast.")
    fisher = importlib.import_module(name="cosmolike_notebook_utils.fisher")
    matrices = []
    center = None
    for result, declaration in zip(results, metadata):
        if "parameter_covariance_joint" not in result:
            raise ValueError("Joint parameter directions are unresolved.")
        names = result["parameter_names"].tolist()
        indices = []
        values = []
        point = dict(declaration["configuration"]["parameters"])
        point.update(declaration["configuration"]["ia"])
        for name in parameters:
            if name not in names:
                raise ValueError(f"Forecast does not vary {name}.")
            indices.append(names.index(name))
            values.append(point[name])
        if center is None:
            center = values
        elif values != center:
            raise ValueError("Shown fiducial parameters must agree.")
        covariance = result["parameter_covariance_joint"]
        marginal = covariance[np.ix_(indices, indices)]
        matrices.append(np.linalg.solve(marginal, np.eye(len(indices))))
    mathematical_labels = {
        "omega_m": r"\Omega_m",
        "sigma8": r"\sigma_8",
        "w0": r"w_0",
        "wa": r"w_a",
    }
    shown = []
    for name in parameters:
        shown.append(mathematical_labels.get(name, name))
    return fisher.plot_Fisher(
        F=matrices, mu=center, names=parameters, labels=shown,
        chain_names=labels, filled=False, flat_priors={},
        select=list(range(len(parameters))),
    )


def derived_amplitude_summary(result, metadata):
    """Propagate full marginalized covariances into Omega_m and S8.

    Arguments:
      result, metadata = saved Fisher arrays and their matching declaration.
    Returns:
      Fiducial S8=sigma8*sqrt(Omega_m/0.3), each experiment's sigma(S8),
      two-dimensional FoM=1/sqrt(det(Cov[Omega_m,S8])), and joint/xi FoM
      ratio. The covariance cross term is retained by the local Jacobian.
    Raises:
      ValueError if either marginalized covariance is unresolved or the
      fiducial point is outside the positive Omega_m/sigma8 domain.
    """
    names = result["parameter_names"].tolist()
    indices = [names.index("omega_m"), names.index("sigma8")]
    point = metadata["configuration"]["parameters"]
    matter = point["omega_m"]
    amplitude = point["sigma8"]
    if matter <= 0.0 or amplitude <= 0.0:
        raise ValueError("S8 requires positive Omega_m and sigma8.")
    scale = np.sqrt(matter / 0.3)
    derived = amplitude * scale
    jacobian = np.array([
        [1.0, 0.0],
        [derived / (2.0 * matter), scale],
    ])
    summary = {"fiducial_s8": float(derived)}
    for probe in ("two_point", "joint"):
        key = f"parameter_covariance_{probe}"
        if key not in result:
            raise ValueError(f"The {probe} parameter covariance is unresolved.")
        original = result[key][np.ix_(indices, indices)]
        covariance = jacobian @ original @ jacobian.T
        factor = np.linalg.cholesky(covariance)
        fom = 1.0 / np.prod(np.diag(factor))
        summary[probe] = {
            "omega_m_s8_covariance": covariance,
            "sigma_s8": float(np.sqrt(covariance[1, 1])),
            "omega_m_s8_fom": float(fom),
        }
    summary["joint_over_two_point_fom"] = (
        summary["joint"]["omega_m_s8_fom"]
        / summary["two_point"]["omega_m_s8_fom"]
    )
    return summary


def fixed_dark_energy_summary(result, metadata):
    """Fix w0=-1 and wa=0 while marginalizing every other varied parameter.

    Partition a posterior Fisher matrix into retained parameters r and
    dark-energy parameters d. Fixing d gives covariance (F_rr)^-1.
    This differs from the rr block of F^-1, which marginalizes over d.
    Select the Fisher submatrix first, then invert its entire retained
    parameter space before selecting Omega_m and sigma8 for S8.

    Arguments:
      result = saved Fisher arrays with parameter_names, fisher_prior,
          fisher_two_point, fisher_joint and their _likelihood matrices.
          Posterior matrices must equal likelihood plus the saved prior.
      metadata = the same archive's metadata, including its configuration
          and fiducial cosmology. Its w0 and wa must be -1 and 0: fixing
          them at a different point would require new physical derivatives.
    Returns:
      A summary only: fixed and marginalized parameter names, sigma(S8),
      the Omega_m/S8 FoM for each probe, the joint/two-point FoM ratio,
      and marginalized IA errors for parameters present in the archive.
      No replacement Fisher archive or parameter covariance is returned.
    Raises:
      ValueError for incompatible fiducials, absent parameters, inconsistent
      saved matrix pieces or an unresolved retained parameter direction.
    Assumptions:
      The saved likelihood Fisher matrices use the same fixed covariance
      and physical derivatives as their original demonstration. All other
      varied parameters remain marginalized and the saved prior is counted
      once. This local Gaussian calculation does not change scale cuts,
      source reconstruction, estimator weights or likelihood assumptions.
      It does not reproduce the published posterior improvements.
    """
    names_array = np.asarray(a=result["parameter_names"])
    if names_array.ndim != 1 or names_array.dtype.kind not in "US":
        raise ValueError("Saved parameter_names must be a string vector.")
    names = names_array.tolist()
    if len(set(names)) != len(names):
        raise ValueError("Saved parameter_names must be unique.")
    for name in (
        "omega_m",
        "sigma8",
        "w0",
        "wa",
    ):
        if name not in names:
            raise ValueError(f"Fixed-dark-energy comparison needs {name}.")
    point = metadata["configuration"]["parameters"]
    fixed = {
        "w0": -1.0,
        "wa": 0.0,
    }
    for name, expected in fixed.items():
        value = point.get(name)
        if (not np.isscalar(value) or not np.isfinite(value)
                or value != expected):
            raise ValueError(
                f"The saved fiducial must have {name}={expected}; "
                "changing the fixed point requires new derivatives."
            )
    for name in ("omega_m", "sigma8"):
        value = point.get(name)
        if (not np.isscalar(value) or not np.isfinite(value)
                or value <= 0.0):
            raise ValueError(f"The saved fiducial requires finite {name}>0.")

    retained = []
    retained_names = []
    for index, name in enumerate(names):
        if name not in fixed:
            retained.append(index)
            retained_names.append(name)
    prior = symmetric_matrix(
        values=result["fisher_prior"],
        name="Saved Fisher prior",
    )
    expected_shape = (len(names), len(names))
    if prior.shape != expected_shape:
        raise ValueError("Saved prior dimensions do not match parameter names.")

    # This temporary dictionary only adapts the existing S8 helper's
    # arguments. It is neither written nor returned as a new run result.
    # Keeping every retained covariance entry here lets the helper select
    # Omega_m/sigma8 only after all remaining nuisance marginalization.
    amplitude_input = {
        "parameter_names": np.asarray(a=retained_names),
    }
    ia_names = (
        "a1",
        "eta1",
        "a2",
        "eta2",
        "bta",
    )
    ia_errors = {}
    for probe in ("two_point", "joint"):
        likelihood = symmetric_matrix(
            values=result[f"fisher_{probe}_likelihood"],
            name=f"Saved {probe} likelihood Fisher matrix",
        )
        posterior = symmetric_matrix(
            values=result[f"fisher_{probe}"],
            name=f"Saved {probe} posterior Fisher matrix",
        )
        if (likelihood.shape != expected_shape
                or posterior.shape != expected_shape):
            raise ValueError(
                f"Saved {probe} matrix dimensions are inconsistent."
            )
        if not np.array_equal(posterior, likelihood + prior):
            raise ValueError(
                f"Saved {probe} posterior does not equal likelihood plus "
                "the saved prior; use an internally consistent archive."
            )
        conditional_fisher = posterior[np.ix_(retained, retained)]
        # The shared helper checks rank in dimensionless parameter units
        # and refuses unresolved directions rather than using a pseudoinverse.
        covariance = parameter_covariance(
            fisher=conditional_fisher,
            names=retained_names,
        )
        amplitude_input[f"parameter_covariance_{probe}"] = covariance
        errors = {}
        for index, name in enumerate(retained_names):
            if name in ia_names:
                errors[name] = float(np.sqrt(covariance[index, index]))
        ia_errors[probe] = errors

    amplitude = derived_amplitude_summary(
        result=amplitude_input,
        metadata=metadata,
    )
    summary = {
        "fixed_parameters": dict(fixed),
        "marginalized_parameters": list(retained_names),
        "fiducial_s8": amplitude["fiducial_s8"],
        "joint_over_two_point_fom": amplitude["joint_over_two_point_fom"],
        "interpretation": (
            "Dark energy fixed at the saved fiducial; every other varied "
            "parameter marginalized; saved Gaussian prior counted once."
        ),
        "scope": (
            "Local demonstration with unchanged inputs and estimator; "
            "not a published posterior reproduction."
        ),
    }
    for probe in ("two_point", "joint"):
        summary[probe] = {
            "sigma_s8": amplitude[probe]["sigma_s8"],
            "omega_m_s8_fom": amplitude[probe]["omega_m_s8_fom"],
            "sigma_ia": ia_errors[probe],
        }
    return summary


def finite_sign_pair_separation(first, second, covariance):
    """Measure the fixed-covariance distance between two saved predictions.

    For delta=mean_first-mean_second, define
    S=C33-C32 C22^-1 C23 and r=delta3-C32 C22^-1 delta2.
    The Gaussian distance obeys chi2_joint=chi2_2+r.T S^-1 r.
    The residual r removes the Map3 change predicted by its covariance
    with xi. It is generally different from the raw Map3 difference.

    Arguments:
      first, second = saved evaluate-result dictionaries with finite mean
          vectors and identical row_labels, row_kind, row_bins, and
          row_angles_radians. The usual example has 480 rows; a selected
          vector is permitted if both xi and Map3 rows remain.
      covariance = positive definite full covariance in exactly that row
          order and in the same observable units. The caller supplies
          the covariance from the verified workflow, including C23.
    Returns:
      Dictionary with dimensionless joint, two-point and conditional
      three-point chi-squared distances; owned delta and conditional
      residual arrays; the Schur covariance and selected row indices.
      The residual carries the units of the Map3 rows. The identity error
      is reported and checked at a stated float64 roundoff tolerance.
    Raises:
      ValueError for mismatched rows, nonfinite means, invalid covariance,
      or a numerically unresolved decomposition.
    Interpretation:
      No parameters are fitted or marginalized, and no prior, mock-count
      correction or detection significance is assigned. This function
      does not verify physical convergence, the covariance sign history,
      or that only A1/A2 differ between the supplied predictions.
    """
    means = []
    required = (
        "mean",
        "row_labels",
        "row_kind",
        "row_bins",
        "row_angles_radians",
    )
    for case in (first, second):
        for key in required:
            if key not in case:
                raise ValueError(f"Saved prediction is missing {key}.")
        original = np.asarray(a=case["mean"])
        if original.dtype.kind not in "fiu":
            raise ValueError("Saved mean must contain real numerical values.")
        mean = np.array(object=original, dtype=np.float64, copy=True)
        if mean.ndim != 1 or mean.size == 0:
            raise ValueError(
                "Saved mean must be a nonempty one-dimensional vector."
            )
        if not np.all(np.isfinite(mean)):
            raise ValueError("Saved mean contains nonfinite values.")
        means.append(mean)
    if means[0].shape != means[1].shape:
        raise ValueError("Saved predictions have different mean-vector shapes.")

    nrow = means[0].size
    row_shapes = {
        "row_labels": (nrow,),
        "row_kind": (nrow,),
        "row_bins": (nrow, 3),
        "row_angles_radians": (nrow, 3),
    }
    for key, shape in row_shapes.items():
        first_rows = np.asarray(a=first[key])
        second_rows = np.asarray(a=second[key])
        if first_rows.shape != shape or second_rows.shape != shape:
            raise ValueError(f"Both {key} arrays must have shape {shape}.")
        if not np.array_equal(first_rows, second_rows):
            raise ValueError(f"Saved predictions have different {key}.")
    labels = np.asarray(a=first["row_labels"])
    if labels.dtype.kind not in "US" or np.unique(labels).size != nrow:
        raise ValueError("Saved row_labels must be unique string labels.")
    angles = np.asarray(a=first["row_angles_radians"])
    if angles.dtype.kind not in "fiu" or not np.all(np.isfinite(angles)):
        raise ValueError("Saved row angles must be finite numerical values.")

    # The shared block algebra acts on any columns. Supplying a finite
    # contrast instead of a derivative gives a distance, without fitting
    # an artificial amplitude parameter or adding parameter priors.
    difference = means[0] - means[1]
    comparison = compare_information(
        derivatives=difference[:, None],
        names=["fixed_sign_pair_contrast"],
        kind=np.asarray(a=first["row_kind"]),
        covariance=covariance,
    )
    joint = float(comparison["fisher_joint_likelihood"][0, 0])
    two_point = float(comparison["fisher_two_point_likelihood"][0, 0])
    conditional = float(comparison["fisher_conditional_three_point"][0, 0])
    values = np.array(object=[
        joint,
        two_point,
        conditional,
    ])
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("Gaussian distances must be finite and nonnegative.")

    # Both contractions must agree even when the mean difference is tiny.
    # A unit-sized denominator would hide relative errors for weak signals.
    sum_of_blocks = two_point + conditional
    identity_absolute_error = abs(joint - sum_of_blocks)
    scale = max(joint, sum_of_blocks)
    identity_relative_error = 0.0
    if scale > 0.0:
        identity_relative_error = identity_absolute_error / scale
    identity_tolerance = 256.0 * nrow * np.finfo(np.float64).eps
    if identity_relative_error > identity_tolerance:
        raise ValueError(
            "Joint and conditional distances disagree: relative error "
            f"{identity_relative_error} exceeds {identity_tolerance}. "
            "Inspect covariance conditioning and row conventions."
        )
    conditional_residual = comparison["conditional_derivatives"][:, 0].copy()
    return {
        "chi2_joint": joint,
        "chi2_two_point": two_point,
        "chi2_conditional_three_point": conditional,
        "delta_mean": difference.copy(),
        "conditional_residual": conditional_residual,
        "conditional_covariance": comparison["conditional_covariance"].copy(),
        "two_point_indices": comparison["two_point_indices"].copy(),
        "three_point_indices": comparison["three_point_indices"].copy(),
        "identity_absolute_error": identity_absolute_error,
        "identity_relative_error": identity_relative_error,
        "identity_relative_tolerance": identity_tolerance,
        "interpretation": "Fixed-model Gaussian distance; no marginalization.",
    }


def ia_figure9_workflows(workflow):
    """Prepare one no-IA case and the two sign choices in NLA and TATT.

    This comparison holds the source curves, cosmology, estimator, grids,
    redshift exponents and power prescription of the supplied TATT run
    fixed. It tests the amplitude choices from Figure 9 of Gomes et al.,
    arXiv:2601.09133, but
    does not claim its exact source reconstruction or measured estimator.

    Arguments:
      workflow = current TATT FisherWorkflow with core_tatt two-point
          predictions and ct=fb=0. Its base eta1 and eta2 are retained.
    Returns:
      Dictionary containing five named workflows, two ordered plot groups,
      their labels, and the complete IA settings of each case. Each group
      begins with the same no_ia workflow, followed by negative_a1 and
      positive_a1. Pass that order to plot_mean_responses so every residual
      uses the same no-IA reference. The existing plot's scaled residual
      uses max(abs(reference)); it is not a pointwise percentage.
    Side effects:
      Writes at most five explicit JSON configurations under chains/ia_response.
      The two TATT files use the exact names and contents already produced
      by ia_sign_workflows; their saved predictions can be reused. Only
      no_ia and the two NLA choices require additional mean predictions.
      Those three new cases are mean-only files: the inherited TATT Fisher
      section is removed because its zero-amplitude slopes are inactive.
      Existing mismatched configuration files are refused, never replaced.
    Raises:
      ValueError for an unsupported model or a changed existing file.
    Physical conventions:
      NLA sets a2=bta=ct=0. Its eta2 is retained but has no effect when
      a2=0. No IA sets a1=a2=bta=ct=0, so both redshift exponents have no
      effect. TATT changes only a1/a2. The supplied eta1/eta2 values are
      reported explicitly; this helper does not silently impose Table 1.
    """
    config, path, _ = driver.load_configuration(
        path=workflow.configuration,
        command="evaluate",
    )
    if config["mean"]["two_point"]["mode"] != "core_tatt":
        raise ValueError("The five-case comparison requires core_tatt.")
    if config["ia"]["ct"] != 0.0 or config["ia"]["fb"] != 0.0:
        raise ValueError(
            "The joint comparison requires ct=fb=0; neither has a complete "
            "shared two-point and three-point response here."
        )
    directory = PROJECT / "chains" / "ia_response"
    cases = (
        {
            "tag": "no_ia",
            "model": "none",
            "a1": 0.0,
            "a2": 0.0,
        },
        {
            "tag": "nla_negative_a1",
            "model": "nla",
            "a1": -0.2,
            "a2": 0.0,
        },
        {
            "tag": "nla_positive_a1",
            "model": "nla",
            "a1": 0.2,
            "a2": 0.0,
        },
        {
            "tag": "negative_a1",
            "model": "tatt",
            "a1": -0.2,
            "a2": 0.4,
        },
        {
            "tag": "positive_a1",
            "model": "tatt",
            "a1": 0.2,
            "a2": -0.4,
        },
    )
    prepared = []
    parameters = {}
    for case in cases:
        # JSON round-tripping gives every case its own nested settings.
        # Retained source/power files are shared inputs, not mutable model
        # state; each evaluate operation still owns its C preparation.
        encoded = json.dumps(obj=config)
        derived = json.loads(s=encoded)
        path_fields = (
            ("source", "path"),
            ("physics", "tables"),
            ("rows", "two_point"),
            ("rows", "aperture"),
            ("covariance", "path"),
        )
        for section, key in path_fields:
            resolved = driver.resolve_path(
                configuration_path=path,
                value=derived[section][key],
            )
            derived[section][key] = str(resolved)
        two_point = derived["mean"]["two_point"]
        directory_two_point = driver.resolve_path(
            configuration_path=path,
            value=two_point["directory"],
        )
        two_point["directory"] = str(directory_two_point)
        derived["ia"]["a1"] = case["a1"]
        derived["ia"]["a2"] = case["a2"]
        if case["model"] in ("none", "nla"):
            derived["ia"]["bta"] = 0.0
            derived["ia"]["ct"] = 0.0
        if case["model"] == "tatt":
            # Preserve the existing two TATT JSON files byte for byte.
            # The base's other settings, including any new numerical
            # controls, must also match before an old output is reused.
            derived["description"] = (
                f"Demonstration IA sign response: A1={case['a1']}, "
                f"A2={case['a2']}; all other inputs fixed."
            )
        elif case["model"] == "nla":
            derived["description"] = (
                f"Demonstration NLA sign response: A1={case['a1']}, "
                "A2=bta=ct=0; base source, cosmology and eta1 retained."
            )
        else:
            derived["description"] = (
                "Common no-IA demonstration reference: A1=A2=bta=ct=0; "
                "base source, cosmology and estimator retained."
            )
        tag = case["tag"]
        output_commands = ("evaluate", "fisher")
        if case["model"] != "tatt":
            if "fisher" in derived:
                del derived["fisher"]
            if "fisher" in derived["outputs"]:
                del derived["outputs"]["fisher"]
            output_commands = ("evaluate",)
            derived["notes"] = [
                "Fixed-parameter mean comparison: no derivatives, parameter "
                "priors or marginalized forecast are defined in this file.",
                "The supplied source reconstruction, selected rows, cosmology, "
                "estimator and numerical grids are retained explicitly below.",
                "Estimator agreement and source-model applicability require "
                "their own checks for the supplied configuration.",
                "The supplied covariance retains its declared convention "
                "and physical assumptions across these amplitude choices.",
                "NLA has A2=bta=ct=0; no IA also has A1=0. Retained redshift "
                "exponents have no effect when their amplitude is zero.",
                "Numerical refinement and matching the published source, "
                "power and estimator choices remain separate checks.",
            ]
        for command in output_commands:
            derived["outputs"][command] = str(
                directory / f"{tag}_{command}.npz"
            )
        target = directory / f"{tag}.yaml"
        text = yaml.safe_dump(derived, sort_keys=False)
        if target.exists() and target.read_text(encoding="utf-8") != text:
            raise ValueError(f"Existing IA comparison differs: {target}")
        prepared.append((tag, target, text))
        parameters[tag] = dict(derived["ia"])

    # Check every destination above before writing any one of the cases.
    # This leaves an inconsistent old run visible for deliberate review.
    directory.mkdir(parents=True, exist_ok=True)
    workflows = {}
    for tag, target, text in prepared:
        if not target.exists():
            with target.open(mode="x", encoding="utf-8") as stream:
                stream.write(text)
        workflows[tag] = FisherWorkflow(configuration=target)
    groups = {
        "nla": {
            "keys": [
                "no_ia",
                "nla_negative_a1",
                "nla_positive_a1",
            ],
            "labels": [
                "No IA",
                r"NLA $A_1=-0.2$",
                r"NLA $A_1=0.2$",
            ],
        },
        "tatt": {
            "keys": [
                "no_ia",
                "negative_a1",
                "positive_a1",
            ],
            "labels": [
                "No IA",
                r"TATT $A_1=-0.2,\ A_2=0.4$",
                r"TATT $A_1=0.2,\ A_2=-0.4$",
            ],
        },
    }
    return {
        "workflows": workflows,
        "groups": groups,
        "ia_parameters": parameters,
        "residual_reference": "no_ia",
        "interpretation": "Declared-input Figure 9 mechanism comparison.",
    }
