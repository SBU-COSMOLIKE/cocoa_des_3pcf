"""Read saved accuracy studies and explicitly launch requested follow-up work.

Loading never evaluates a physical model. A missing result remains pending.
The scan uses the existing shared Fisher-noise estimator; full confirmations
use the ordinary project Fisher driver. No estimator is implemented here.
Every figure receives saved numbers, and a completed full check is distinguished
from a selected-column diagnostic before it enters a plotting array.
"""

import hashlib
import importlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import numpy as np
import yaml

PROJECT = Path(__file__).resolve().parents[1]
CORE = PROJECT.parents[1] / "external_modules/code/cosmolike_core"
sys.path.insert(0, str(PROJECT / "python"))
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(CORE))

import run_3pcf as driver
from cosmolike_des_3pcf_fisher_wrappers import FisherWorkflow
from des3pcf.forecast import figure_of_merit
from des3pcf.provenance import prediction_identity
from des3pcf.provenance import verify_prediction_identity


def summary_file(directory):
    """Prefer YAML records while retaining readable frozen JSON studies."""
    directory = Path(directory)
    current = directory / "summary.yaml"
    archived = directory / "summary.json"
    if current.is_file() or not archived.is_file():
        return current
    return archived


def _digest(path):
    """Hash the exact archived bytes, not just a matching filename."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plotters():
    """Import the shared array-only module whose folder starts with a digit."""
    return importlib.import_module("cosmolike_notebook_utils.3pcf.fisher_accuracy")


def _validate_step_vectors(summary, arrays, baseline, configuration, columns, steps):
    """Check archived coordinate reuse without recomputing any diagnostic.

    One integer coordinate unit is one quarter of the original YAML step.
    For example, unit 2 is the saved +Delta/2 prediction and unit 1 is the
    new +Delta/4 prediction. Integer labels identify shared physical points
    without a tolerance that could confuse two nearby parameter values.

    Arguments:
      summary, arrays = recorded step metadata and owned NPZ arrays.
      baseline = original Fisher arrays; configuration = its recorded inputs.
      columns, steps = validated selected indices and absolute base steps.
    Returns:
      None when every stored stencil and new physical coordinate is consistent.
    Historical reads do not import today's shared noise implementation.
    This verifies saved-node reuse and coordinates, not that a prediction
    was accurate or that a recorded FoM equals a newly recomputed value.
    """
    units = {
        "h4": (-2, -1, 0, 1, 2),
        "h2": (-4, -2, 0, 2, 4),
        "h": (-8, -4, 0, 4, 8),
        "two_h": (-16, -8, 0, 8, 16),
    }
    nrow = len(baseline["row_labels"])
    labels = [case["step_label"] for case in summary["cases"]]
    seen = {}
    for label in labels:
        key = f"stencil_values_{label}"
        derivative_key = f"selected_derivatives_{label}"
        if key not in arrays or derivative_key not in arrays:
            raise ValueError(
                f"Step study lacks retained arrays for {label}: require {key} "
                f"and {derivative_key}; regenerate the complete step archive."
            )
        values = arrays[key]
        derivatives = arrays[derivative_key]
        if (values.dtype != np.float64 or values.shape != (len(columns), 5, nrow)
                or not np.all(np.isfinite(values))
                or derivatives.dtype != np.float64
                or derivatives.shape != (nrow, len(columns))
                or not np.all(np.isfinite(derivatives))):
            raise ValueError(
                f"Step study has invalid retained-array shape/type for {label}: "
                f"stencils {values.shape}/{values.dtype}, derivatives "
                f"{derivatives.shape}/{derivatives.dtype}; require finite float64 "
                f"arrays of shapes {(len(columns), 5, nrow)} and "
                f"{(nrow, len(columns))}. Regenerate the step archive."
            )
        if label in ("h", "h2"):
            expected = baseline[key][columns]
            if not np.array_equal(values.view(np.uint64), expected.view(np.uint64)):
                raise ValueError(f"Step study changes saved baseline nodes for {label}.")
        for index, unit in enumerate(units[label]):
            value = values[:, index]
            if unit in seen and not np.array_equal(
                value.view(np.uint64), seen[unit].view(np.uint64),
            ):
                raise ValueError(f"Step study changes shared quarter-step node {unit}.")
            seen[unit] = value
    reference = dict(configuration["ia"])
    reference.update(configuration["parameters"])
    names = baseline["parameter_names"][columns].tolist()
    new_units = (-1, 1)
    if "two_h" in labels:
        new_units = (-1, 1, -16, 16)
    expected_points = {}
    for index, name in enumerate(names):
        for unit in new_units:
            point = dict(reference)
            point[name] = float(reference[name]) + unit*(steps[index]/4.0)
            expected_points[(name, unit)] = point
    recorded = set()
    for record in summary["new_point_plan"]:
        name = record.get("name")
        unit = record.get("quarter_step_unit")
        if isinstance(unit, bool) or not isinstance(unit, int):
            raise ValueError("Step study quarter-step units must be integers.")
        key = (name, unit)
        if key not in expected_points or key in recorded:
            raise ValueError("Step study repeats or changes a declared new coordinate.")
        point = record.get("point")
        expected = expected_points[key]
        if not isinstance(point, dict) or set(point) != set(expected):
            raise ValueError("Step study changes the physical parameter dictionary.")
        for parameter in expected:
            value = float(point[parameter])
            if (not np.isfinite(value)
                    or value.hex() != float(expected[parameter]).hex()):
                raise ValueError("Step study changes a declared physical coordinate.")
        recorded.add(key)
    if recorded != set(expected_points):
        raise ValueError("Step study is missing a declared new physical coordinate.")


def _case_label(name):
    """Use physical numerical-stage names instead of internal keys on figures."""
    labels = {
        "source_2": "Source sampling ×2",
        "kernel_2": "Source windows ×2",
        "chi_2": "Line of sight ×2",
        "triangle_1p5": "Triangle tables ×1.5",
        "multipole_1p5": "Multipoles ×1.5",
        "angular_2": "Angles ×2",
        "fft_1p5": "FFT grid ×1.5",
        "core_2pt_1p5": "Two-point tables ×1.5",
        "xi_ell_2": "Xi multipole quadrature ×2",
        "xi_theta_2": "Xi angular quadrature ×2",
        "sampling_1p5": "Sampling ×1.5",
        "sampling_2": "Sampling ×2",
        "integration_2": "Integration ×2",
    }
    return labels.get(name, name.replace("_", " "))


def _parameter_label(name):
    """Translate known parameter keys; preserve unknown user-supplied names."""
    labels = {
        "omega_m": r"$\Omega_{\rm m}$",
        "omega_b": r"$\Omega_{\rm b}$",
        "sigma8": r"$\sigma_8$",
        "ns": r"$n_{\rm s}$",
        "w0": r"$w_0$",
        "wa": r"$w_a$",
        "a1": r"$A_1$",
        "a2": r"$A_2$",
        "eta1": r"$\eta_1$",
        "eta2": r"$\eta_2$",
        "bta": r"$b_{\rm TA}$",
    }
    for index in range(1, 5):
        labels[f"m{index}"] = rf"$m_{index}$"
        labels[f"dz{index}"] = rf"$\Delta z_{index}$"
    return labels.get(name, name)


class AccuracyStudy:
    """Read one saved scan against its exact default-accuracy Fisher archive.

    Arguments:
      configuration = baseline Fisher YAML/JSON path.
      output = directory containing summary.yaml (or archived summary.json) and candidate files.
      plan = selection/candidate YAML, used to report unfinished cases.
    Construction and loading write nothing and import no compiled extension.
    Paths are resolved when constructed; file contents are reread each time,
    so an unfinished scan can become available in the same notebook kernel.
    """

    def __init__(self, configuration, output, plan="configs/des_y3_fisher_accuracy.yaml"):
        self.configuration = Path(configuration).resolve()
        self.output = Path(output).resolve()
        self.plan = Path(plan).resolve()

    def _scan_arguments(self, only=None):
        """Construct an explicit child command without starting it."""
        arguments = [
            sys.executable,
            "-B",
            str(PROJECT / "scripts/run_fisher_accuracy.py"),
            str(self.configuration),
            "--plan",
            str(self.plan),
            "--output",
            str(self.output),
        ]
        if only is not None:
            if not only or any(not isinstance(name, str) for name in only):
                raise ValueError("only must be a nonempty list of candidate names.")
            arguments.append("--only")
            arguments.extend(only)
        return arguments

    def scan_command(self, only=None):
        """Return a shell-quoted command; this method never executes it."""
        return shlex.join(self._scan_arguments(only=only))

    def status(self):
        """List file readiness and requested cases without loading a mean model.

        These are availability checks, not validation certificates. load()
        verifies archive identity before any saved values enter a figure.
        """
        config, path, _ = driver.load_configuration(
            path=self.configuration, command="fisher",
        )
        baseline = driver.resolve_path(path, config["outputs"]["fisher"])
        plan = yaml.safe_load(self.plan.read_text(encoding="utf-8"))
        candidates = []
        for case in plan["candidates"]:
            candidates.append(case["name"])
        return {
            "configuration": str(self.configuration),
            "baseline_result": str(baseline),
            "baseline_available": baseline.is_file(),
            "summary_available": summary_file(self.output).is_file(),
            "requested_cases": candidates,
            "scan_command": self.scan_command(),
        }

    def load(self):
        """Validate and return a saved summary, baseline arrays and metadata.

        Each candidate's noise archive binds it to the same baseline bytes,
        parameter steps and selected columns. The full driver records the
        code/input digest map in that archive. This loader preserves that
        record; it does not claim the current checkout matches an old run.
        """
        summary_path = summary_file(self.output)
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Accuracy scan pending: {summary_path}\nRun: {self.scan_command()}"
            )
        report = yaml.safe_load(summary_path.read_text(encoding="utf-8"))
        if report["format"] != "des3pcf-fisher-accuracy-result-v1":
            raise ValueError("Unsupported accuracy summary format.")
        if report["configuration_sha256"] != _digest(path=self.configuration):
            raise ValueError("Accuracy summary belongs to different baseline settings.")
        baseline = Path(report["baseline_result"])
        if report["baseline_result_sha256"] != _digest(path=baseline):
            raise ValueError("The baseline Fisher archive changed after the scan.")
        arrays, metadata = FisherWorkflow(
            configuration=self.configuration,
        ).load_result(command="fisher")
        verify_prediction_identity(
            recorded=metadata.get("prediction_identity"),
            current=report.get("prediction_identity"),
            description="Archived baseline versus accuracy scan",
        )
        config, path, _ = driver.load_configuration(
            path=self.configuration, command="fisher",
        )
        expected = driver.resolve_path(path, config["outputs"]["fisher"])
        if expected.resolve() != baseline.resolve():
            raise ValueError("The summary and baseline YAML point to different archives.")
        if arrays["parameter_names"].tolist() != report["parameter_names"]:
            raise ValueError("The summary changes the baseline parameter order.")
        for case in report["cases"]:
            case_path = Path(case["configuration"])
            noise_path = self.output / f"{case['name']}_noise.npz"
            with np.load(file=noise_path, allow_pickle=False) as archive:
                identity = json.loads(str(archive["identity_json"].item()))
                if identity["configuration_sha256"] != _digest(path=case_path):
                    raise ValueError(f"Candidate settings changed: {case_path}")
                if identity["baseline_result_sha256"] != report["baseline_result_sha256"]:
                    raise ValueError("A candidate uses a different baseline archive.")
                if identity["selected_columns"] != report["selected_columns"]:
                    raise ValueError("A candidate uses a different parameter selection.")
                if identity["absolute_steps"] != report["absolute_steps"]:
                    raise ValueError("A candidate uses different parameter steps.")
                verify_prediction_identity(
                    recorded=identity.get("prediction_identity"),
                    current=report.get("prediction_identity"),
                    description="Archived selected-column case versus baseline",
                )
                if identity.get("dependency_sha256") != report.get("dependency_sha256"):
                    raise ValueError(
                        "A candidate uses a different diagnostic implementation."
                    )
                if not np.array_equal(archive["a"], np.asarray(case["a"])):
                    raise ValueError("Summary noise additions differ from saved arrays.")
        return report, arrays, metadata

    def _available(self):
        """Report pending work without fabricating an empty scientific result."""
        if not summary_file(self.output).is_file():
            print(f"Scan pending for {self.configuration.name}.")
            print(f"Explicit command: {self.scan_command()}")
            return None
        return self.load()

    def summary_markdown(self):
        """Describe coverage and pending checks with no numerical model calls."""
        loaded = self._available()
        if loaded is None:
            return "**Pending:** no saved accuracy scan is available."
        report, _, _ = loaded
        completed = []
        for case in report["cases"]:
            completed.append(case["name"])
        pending = []
        for name in self.status()["requested_cases"]:
            if name not in completed:
                pending.append(name)
        fraction = report["baseline_contribution_fraction"]
        coverage = "undefined because all measured baseline contributions vanish"
        if fraction is not None:
            coverage = f"{100.0 * fraction:.3g}% of the measured baseline contribution"
        selected = ", ".join(report["selected_parameters"])
        omitted = ", ".join(report.get("omitted_parameters", [])) or "none"
        missing = ", ".join(pending) or "none"
        lines = [
            f"**{len(completed)} completed candidate scans.** "
            f"Selected parameters: {selected}.",
            f"They cover {coverage}. Baseline-only parameters: {omitted}.",
            f"Each candidate requests {report['evaluations_per_candidate']} "
            "distinct means; "
            "the default Fisher supplies the fixed marginalization weights.",
            f"Pending planned scans: {missing}.",
            "The omitted share describes the measured baseline only. It does not bound "
            "unknown noise at another setting. Small scores cannot exclude coherent "
            "derivative errors or common physical-model errors.",
        ]
        return "\n\n".join(lines)

    def load_full_checks(self):
        """Load completed full h/h2 recalculations and label pending ones.

        Returns a dictionary with labels, ratios [2,ncase], pending names and
        rank failures. Both rows use the scan's fixed baseline FoM denominator.
        Scientific algebra and rank refusal belong to forecast.figure_of_merit.
        Every candidate must retain the baseline rows, covariance, physical
        inputs and implementation identity. These are comparisons between
        archived records; the current checkout may differ for historical reads.
        """
        report, baseline, _ = self.load()
        records = [("Default settings", baseline)]
        pending = []
        for case in report["cases"]:
            config, path, _ = driver.load_configuration(
                path=case["configuration"], command="fisher",
            )
            filename = driver.resolve_path(path, config["outputs"]["fisher"])
            if not filename.is_file():
                pending.append(case["name"])
                continue
            arrays, metadata = FisherWorkflow(
                configuration=case["configuration"],
            ).load_result(command="fisher")
            verify_prediction_identity(
                recorded=metadata.get("prediction_identity"),
                current=report.get("prediction_identity"),
                description="Archived full candidate versus baseline",
            )
            for key in ("parameter_names", "row_labels", "row_kind", "row_bins",
                        "row_angles_radians", "fiducial_covariance"):
                if not np.array_equal(arrays[key], baseline[key]):
                    raise ValueError(f"Full candidate changes {key}: {filename}")
            records.append((_case_label(name=case["name"]), arrays))
        ratios = np.full(shape=(2, len(records)), fill_value=np.nan)
        labels = []
        failures = []
        for column, (name, arrays) in enumerate(records):
            labels.append(name)
            for row, step in enumerate(("h", "h2")):
                try:
                    merit = figure_of_merit(
                        fisher=arrays[f"fisher_{step}"], names=report["parameter_names"],
                        pair=report["pair"],
                    )
                    ratios[row, column] = merit / report["baseline_fom"]
                except ValueError as error:
                    failures.append(f"{name}, {step}: {error}")
        return {
            "labels": labels,
            "ratios": ratios,
            "pending": pending,
            "rank_failures": failures,
        }

    def plot_noise_contributions(self, show=1):
        """Draw all measured baseline columns, preserving the selected subset."""
        loaded = self._available()
        if loaded is None:
            return None
        report, _, _ = loaded
        selected = np.zeros(shape=len(report["parameter_names"]), dtype=bool)
        selected[report["selected_columns"]] = True
        labels = []
        for name in report["parameter_names"]:
            labels.append(_parameter_label(name=name))
        return _plotters().plot_noise_contributions(
            contributions=report["baseline_contributions"], names=labels,
            selected=selected, show=show,
        )

    def plot_accuracy_screen(self, show=1):
        """Draw completed cheap scans; missing columns remain explicit omissions."""
        loaded = self._available()
        if loaded is None:
            return None
        report, _, _ = loaded
        ratios = np.full(shape=(2, len(report["cases"])), fill_value=np.nan)
        labels = []
        for column, case in enumerate(report["cases"]):
            labels.append(_case_label(name=case["name"]))
            ratios[0, column] = case["predicted_ratio_to_default"]
            partial = case["selected_derivative_replacement_fom_ratio"]
            if partial is not None:
                ratios[1, column] = partial
            else:
                print(f"{case['name']}: {case['selected_derivative_replacement_status']}")
        return _plotters().plot_accuracy_screen(ratios=ratios, labels=labels, show=show)


    def load_step_study(self, directory):
        """Read a historical step study associated with this exact baseline.

        Arguments:
          directory = study folder containing summary.yaml (or archived summary.json) and stencils.npz.
        Returns:
          Labels, two plot-ratio rows and the complete recorded step summary.
          The noise row is normalized to this same subset's h2 noise estimate;
          the derivative row already uses the fixed full h2 Fisher denominator.
        Raises:
          ValueError for inconsistent summaries, identities, rows or step values.
          FileNotFoundError for incomplete saved work. No model is evaluated.
        """
        baseline_report, baseline, baseline_metadata = self.load()
        # The dedicated producer fixes its FoM pair. Do not relabel those
        # saved numbers through a scan configured for another ellipse.
        if baseline_report.get("pair") != ["w0", "wa"]:
            raise ValueError("The cosmology step producer supports only the w0-wa pair.")
        directory = Path(directory).resolve()
        summary = yaml.safe_load(summary_file(directory).read_text(encoding="utf-8"))
        if (summary.get("format") != "des3pcf-cosmology-step-study-v1"
                or summary.get("executed") is not True):
            raise ValueError("Step study is not a completed supported calculation.")
        with np.load(file=directory/"stencils.npz", allow_pickle=False) as archive:
            recorded = json.loads(str(archive["metadata_json"].item()))
            if recorded != summary:
                raise ValueError("Step summary differs from its stencil metadata.")
            if not np.array_equal(
                a1=archive["parameter_names"], a2=baseline["parameter_names"],
            ):
                raise ValueError("Step study changes the full parameter order.")
            if not np.array_equal(archive["row_labels"], baseline["row_labels"]):
                raise ValueError("Step study changes the data-row labels.")
            columns = archive["selected_columns"]
            if (columns.ndim != 1 or columns.size != 2
                    or not np.issubdtype(columns.dtype, np.integer)
                    or np.any(columns < 0)
                    or np.any(columns >= len(baseline["parameter_names"]))
                    or len(np.unique(columns)) != len(columns)):
                raise ValueError("Step study has invalid selected-column indices.")
            selected = baseline["parameter_names"][columns].tolist()
            if selected != summary["selected_parameters"]:
                raise ValueError("Step summary changes its selected parameters.")
            base_steps = archive["absolute_base_steps"].copy()
            retained = {}
            for name in archive.files:
                if (name.startswith("stencil_values_")
                        or name.startswith("selected_derivatives_")):
                    retained[name] = archive[name].copy()
        expected_steps = []
        configuration = baseline_metadata["configuration"]
        for parameter in selected:
            expected_steps.append(configuration["fisher"]["steps"][parameter])
        if not np.array_equal(base_steps, np.asarray(expected_steps)):
            raise ValueError("Step study changes the original absolute steps.")
        identity = summary["identity"]
        if (identity["baseline_sha256"] != baseline_report["baseline_result_sha256"]
                or Path(summary["baseline"]).resolve()
                != Path(baseline_report["baseline_result"]).resolve()
                or identity["configuration_sha256"]
                != baseline_report["configuration_sha256"]
                or Path(summary["configuration"]).resolve() != self.configuration):
            raise ValueError(
                "Step study belongs to a different baseline or configuration."
            )
        verify_prediction_identity(
            recorded=identity.get("prediction_identity"),
            current=baseline_report.get("prediction_identity"),
            description="Archived step study versus baseline",
        )
        by_label = {}
        for case in summary["cases"]:
            label = case["step_label"]
            if label in by_label:
                raise ValueError("Step study repeats a stencil label.")
            by_label[label] = case
        order = ["h4", "h2", "h"]
        if "two_h" in by_label:
            order.append("two_h")
        if set(by_label) != set(order):
            raise ValueError("Step study needs h4, h2, h and optional two_h.")
        expected_count = len(selected)*(4 if "two_h" in by_label else 2)
        if (summary["new_mean_count"] != expected_count
                or len(summary["new_point_plan"]) != expected_count):
            raise ValueError("Step-study new-mean count disagrees with its stencil plan.")
        factors = {
            "h4": 0.25,
            "h2": 0.5,
            "h": 1.0,
            "two_h": 2.0,
        }
        names = {
            "h4": r"$\Delta/4$",
            "h2": r"$\Delta/2$",
            "h": r"$\Delta$",
            "two_h": r"$2\Delta$",
        }
        denominator = float(by_label["h2"]["diagonal_noise_inflation_ratio"])
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("The same-column h2 noise denominator must be positive.")
        ratios = np.full(shape=(2, len(order)), fill_value=np.nan)
        labels = []
        ordered_cases = []
        for index, label in enumerate(order):
            case = by_label[label]
            if not np.array_equal(case["absolute_steps"], base_steps*factors[label]):
                raise ValueError(
                    f"Step study has inconsistent absolute steps for {label}."
                )
            noise = float(case["diagonal_noise_inflation_ratio"])
            if not np.isfinite(noise) or noise <= 0:
                raise ValueError("Step noise ratios must be finite and positive.")
            ratios[0, index] = noise/denominator
            partial = case["selected_derivative_replacement_fom_ratio"]
            if partial is not None:
                if not np.isfinite(partial) or partial <= 0:
                    raise ValueError("Step replacement FoM ratios must be positive.")
                ratios[1, index] = partial
            labels.append(names[label])
            ordered_cases.append(case)
        _validate_step_vectors(
            summary=summary, arrays=retained, baseline=baseline,
            configuration=configuration, columns=columns, steps=base_steps,
        )
        return {
            "summary": summary,
            "labels": labels,
            "ratios": ratios,
            "ordered_cases": ordered_cases,
        }

    def plot_step_study(self, directory, show=1):
        """Plot saved step diagnostics with the existing shared array-only plotter.

        Arguments:
          directory = completed step-study folder; show = shared display control.
        Returns:
          Shared plot result, or None with a pending message if files are absent.
        Side effects:
          Prints selected columns, actual steps and the number of new means.
          Historical identities are checked; no physical calculation is launched.
        """
        directory = Path(directory).resolve()
        files = (summary_file(directory), directory/"stencils.npz")
        if not all(path.is_file() for path in files):
            print(f"Step study pending: {directory}")
            return None
        loaded = self.load_step_study(directory=directory)
        summary = loaded["summary"]
        selected = summary["selected_parameters"]
        print("Selected parameter columns: " + ", ".join(selected))
        print(f"New mean evaluations: {summary['new_mean_count']}")
        print("Both curves concern this selected subset; neither is a full new Fisher.")
        for case in loaded["ordered_cases"]:
            entries = []
            for name, step in zip(selected, case["absolute_steps"]):
                entries.append(f"{name}={step:.8g}")
            print(case["step_label"] + " absolute steps: " + ", ".join(entries))
            if case["selected_derivative_replacement_fom_ratio"] is None:
                print(case["selected_derivative_replacement_status"])
        return _plotters().plot_accuracy_screen(
            ratios=loaded["ratios"], labels=loaded["labels"], show=show,
        )

    def plot_full_checks(self, show=1):
        """Draw only completed full checks and print which scans lack them."""
        if self._available() is None:
            return None
        checks = self.load_full_checks()
        if checks["pending"]:
            print("Full recalculations pending: " + ", ".join(checks["pending"]))
        for failure in checks["rank_failures"]:
            print("Unresolved Fisher direction: " + failure)
        return _plotters().plot_full_checks(
            ratios=checks["ratios"], labels=checks["labels"], show=show,
        )

    def run_scan(self, only=None):
        """Explicitly run selected noise stencils in a separate BLAS=1 process.

        This can be expensive. The caller decides when the machine is free.
        Completed matching archives are reused by the driver; stale ones fail.
        """
        environment = dict(os.environ)
        for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                     "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS",
                     "NUMEXPR_NUM_THREADS"):
            environment[name] = "1"
        environment["OMP_NUM_THREADS"] = "4"
        environment["OMP_THREAD_LIMIT"] = "4"
        subprocess.run(args=self._scan_arguments(only=only), env=environment, check=True)
        return self.load()

    def run_full_check(self, case):
        """Run a full candidate only if current files match the archived baseline.

        Historical plotting compares saved identities with one another. This
        executable method additionally checks current input and code bytes
        before launching work. The child Fisher calculation repeats its own
        beginning/end check; a later historical read must still match the base.

        Arguments:
          case = exact name of one completed selected-column scan.
        Returns:
          path to the new full Fisher archive after the child exits successfully.
        Raises:
          ValueError for missing cases or different input/implementation records.
        Side effects:
          Runs the declared physical means and writes that candidate's full archive.
        """
        report, baseline, _ = self.load()
        for record in report["cases"]:
            if record["name"] == case:
                config, path, _ = driver.load_configuration(
                    path=record["configuration"], command="fisher",
                )
                second_extension = None
                core = config["mean"]["two_point"]
                if (core["mode"] == "core_tatt"
                        and np.any(baseline["row_kind"] != "map3")):
                    directory = driver.resolve_path(path, core["directory"])
                    second_extension = directory / (core["module"] + ".so")
                current = prediction_identity(
                    config=config,
                    configuration_path=path,
                    project=PROJECT,
                    three_point_path=(
                        PROJECT / "interface/cosmolike_des_3pcf_interface.so"
                    ),
                    two_point_path=second_extension,
                )
                verify_prediction_identity(
                    recorded=report.get("prediction_identity"),
                    current=current,
                    description="Current full-check inputs versus archived baseline",
                )
                return FisherWorkflow(
                    configuration=record["configuration"],
                ).run(command="fisher")
        raise ValueError(f"No completed noise scan named {case} is in this study.")


def plot_xi_preflight(directory="runs/xi_refinement", show=1):
    """Plot the saved fiducial xi quadrature check, with no mean evaluations.

    The summary's finest vector is the comparison reference. N_ell, N_theta
    and the upper integration bound all belong in the labels: an ell-count
    plateau alone need not imply convergence in the other two controls.
    """
    path = summary_file(Path(directory).resolve())
    if not path.is_file():
        print(f"Xi quadrature preflight pending: {path}")
        return None
    report = yaml.safe_load(path.read_text(encoding="utf-8"))
    distances = []
    labels = []
    for case in report["cases"]:
        distances.append(case["quadratic_distance_to_finest"])
        upper = f"{case['ell_upper']:g}"
        labels.append(f"{case['ell_count']} / {case['theta_count']} / {upper}")
    print(report["interpretation"])
    return _plotters().plot_xi_preflight(distances=distances, labels=labels, show=show)
