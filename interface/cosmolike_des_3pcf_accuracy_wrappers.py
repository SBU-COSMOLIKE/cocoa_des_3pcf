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


def _digest(path):
    """Hash the exact archived bytes, not just a matching filename."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _plotters():
    """Import the shared array-only module whose folder starts with a digit."""
    return importlib.import_module("cosmolike_notebook_utils.3pcf.fisher_accuracy")


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
      output = directory containing summary.json and candidate files.
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
            "summary_available": (self.output / "summary.json").is_file(),
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
        summary_path = self.output / "summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(
                f"Accuracy scan pending: {summary_path}\nRun: {self.scan_command()}"
            )
        report = json.loads(summary_path.read_text(encoding="utf-8"))
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
                if not np.array_equal(archive["a"], np.asarray(case["a"])):
                    raise ValueError("Summary noise additions differ from saved arrays.")
        return report, arrays, metadata

    def _available(self):
        """Report pending work without fabricating an empty scientific result."""
        if not (self.output / "summary.json").is_file():
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
        Every candidate must retain the baseline rows and covariance exactly.
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
            arrays, _ = FisherWorkflow(
                configuration=case["configuration"],
            ).load_result(command="fisher")
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
        """Explicitly run every derivative for one saved candidate configuration."""
        report, _, _ = self.load()
        for record in report["cases"]:
            if record["name"] == case:
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
    path = Path(directory).resolve() / "summary.json"
    if not path.is_file():
        print(f"Xi quadrature preflight pending: {path}")
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    distances = []
    labels = []
    for case in report["cases"]:
        distances.append(case["quadratic_distance_to_finest"])
        upper = f"{case['ell_upper']:g}"
        labels.append(f"{case['ell_count']} / {case['theta_count']} / {upper}")
    print(report["interpretation"])
    return _plotters().plot_xi_preflight(distances=distances, labels=labels, show=show)
