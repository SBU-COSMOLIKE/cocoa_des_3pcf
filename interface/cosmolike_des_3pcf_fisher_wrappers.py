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
        missing = driver.required_files(config=config, path=path, command=command)
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
                raise ValueError("Saved result belongs to a different configuration.")
            for name in archive.files:
                if name != "metadata_json":
                    arrays[name] = archive[name].copy()
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
        "interpretation": "Controlled inputs; no DES inference or accuracy claim.",
    }
