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
import os
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))
sys.path.insert(0, str(PROJECT / "scripts"))

import run_3pcf as driver
from des3pcf.mean import physics_model
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
        """Load an existing result, refusing to associate it with changed YAML.

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
          covariance. This is a configuration check, not a code-version check.
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
    manifest_path = PROJECT / "data" / "y3_reference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
    model = physics_model(compiled=compiled, tables=controlled_tables())
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
      Redshift, source density[bin,z], midpoint integrals and JSON sidecar.
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
    sidecar = source.with_suffix(".json")
    record = json.loads(s=sidecar.read_text(encoding="utf-8"))
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
    directory = PROJECT / "runs" / "ia_response"
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
        target = directory / f"{tag}.json"
        text = json.dumps(obj=derived, indent=2, sort_keys=True) + "\n"
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
      and the relative norm of Fjoint-F2-F3|2 before adding priors.
      Missing covariances are reported rather than replaced by a pseudoinverse.
    """
    joint = result["fisher_joint_likelihood"]
    difference = joint - result["fisher_two_point_likelihood"]
    difference -= result["fisher_conditional_three_point"]
    scale = np.linalg.norm(joint)
    report = {
        "names": result["parameter_names"].tolist(),
        "rank_status": metadata["comparison_rank_status"],
        "gaussian_prior_sigmas": metadata["configuration"]["fisher"]["priors"],
        "information_identity_relative_error": float(
            np.linalg.norm(difference) / scale
        ),
        "derivative_relative_change": result["derivative_relative_change"],
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
