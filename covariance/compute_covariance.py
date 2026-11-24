"""Compute the des_3pcf analytic covariance from the command line and save it.

The program evaluates the joint xi_+/xi_-/Map3 covariance at one fiducial
point. It reads one evaluate YAML with Cobaya's YAML reader, builds the
Cobaya model (CAMB, the BiHalofit theory block and the likelihood
des_3pcf.joint_shear), evaluates it once at sampler.evaluate.override, and
hands the evaluated likelihood to the adapter des_3pcf_covariance.py,
which calls the C covariance kernels with the likelihood's own rows and
physics model. No sampler runs. The likelihoods never read the matrix
written here; covariance/README.md lists its physical limits.

From the Cocoa/ folder, after `source start_cocoa.sh`, with covariance
generation compiled in (unset IGNORE_COSMOLIKE_DES_3PCF_COVARIANCE, then
recompile with scripts/compile_des_3pcf.sh):

    OMP_NUM_THREADS=8 python projects/des_3pcf/covariance/compute_covariance.py \\
        projects/des_3pcf/EXAMPLE_EVALUATE_COVARIANCE.yaml

The result is the .npz archive named by the YAML's output key (--output
overrides it); an existing archive is replaced only with --overwrite, and
--save-terms also stores the six terms.
"""

import argparse
import os
from pathlib import Path
import sys
import time

# BLAS libraries read these variables when they load, so one BLAS thread
# per process is set before numpy is imported. CosmoLike's OpenMP threads
# follow OMP_NUM_THREADS. COBAYA_NOMPI = 1: one process, no MPI workers.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["COBAYA_NOMPI"] = "1"

# project = projects/des_3pcf, one level above this file's covariance/
# folder. sys.path is the list of folders Python searches on import: the
# compiled interface and this folder, which holds the adapter
# des_3pcf_covariance, go first.
project = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project/"covariance"))
sys.path.insert(0, str(project/"interface"))

import numpy as np
from cobaya.model import get_model
from cobaya.parameterization import Parameterization
from cobaya.yaml import yaml_load_file
from getdist import IniFile

import cosmolike_des_3pcf_interface as ci
import des_3pcf_covariance as survey


def main(argv=None):
    """Evaluate the fiducial likelihood, compute the covariance and save it.

    Arguments:
        argv = explicit command arguments; None reads the command line.
    Returns:
        Path of the written archive.
    Raises:
        SystemExit (status 2, through argparse) for a covariance-off
        build, an unset OMP_NUM_THREADS, a bad or existing output, or a
        malformed YAML block; ValueError from the adapter checks for a
        fiducial point or theory block the covariance cannot describe.
    Side effects:
        runs CAMB and the likelihood once; writes the archive.
    """
    # --- 1. ARGUMENTS, BUILD AND YAML, BEFORE ANY NUMERICAL WORK ---
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="covariance evaluate YAML")
    parser.add_argument("--output", type=Path, help="override the YAML output")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing output archive")
    parser.add_argument("--save-terms", action="store_true",
                        help="also store the six covariance terms")
    parser.add_argument("--xi-only", action="store_true",
                        help="compute only the xi rows (required for now)")
    args = parser.parse_args(args=argv)

    # The Map3 rows are refused before any numerical work: the likelihood's
    # Map3 mean uses the finite-SAS estimator, the C covariance codes only
    # the full-filter one (des_3pcf_covariance.compute explains).
    if not args.xi_only:
        parser.error("the Map3 covariance is refused for now (finite-SAS mean, "
                     "full-filter covariance); pass --xi-only")
    if not hasattr(ci, "shape_noise"):
        parser.error(f"{ci.__file__} is the covariance-off build; unset "
                     f"IGNORE_COSMOLIKE_DES_3PCF_COVARIANCE and recompile")
    threads = os.environ.get("OMP_NUM_THREADS", "")
    if not threads.isdigit() or int(threads) < 1:
        parser.error(f"OMP_NUM_THREADS = {threads!r}; set a positive integer")

    # The covariance and output blocks are this program's; Cobaya never
    # sees them. The evaluate override is the fiducial point.
    info = yaml_load_file(file_name=str(args.input))
    block = info.pop("covariance", None) or {}
    output = info.pop("output", None)
    if set(block) - {"neutrino_model"}:
        parser.error(f"covariance block keys {sorted(block)}; only "
                     f"neutrino_model is accepted")
    if args.output is not None:
        output = args.output
    if output is None or Path(output).suffix != ".npz":
        parser.error(f"output = {output!r}; name a .npz archive")
    output = Path(output)
    if output.exists() and not args.overwrite:
        parser.error(f"{output} exists; use a new path or --overwrite")
    if not output.parent.is_dir():
        parser.error(f"output directory {output.parent} does not exist")
    sampler = info.pop("sampler", None) or {}
    override = (sampler.get("evaluate") or {}).get("override")
    if not override:
        parser.error("sampler.evaluate.override must give the fiducial point")

    # to_input resolves fixed values and value expressions at the override.
    settings = survey.configuration(neutrino_model=block.get(
        "neutrino_model", "massless"))
    parameters = Parameterization(info_params=info["params"]).to_input(
        sampled_params_values=override)
    nbin = len(settings["source_density_arcmin2"])
    survey.check_point(settings=settings, parameters=parameters, nbin=nbin)

    # --- 2. THE FIDUCIAL LIKELIHOOD: CAMB, BIHALOFIT AND ONE EVALUATION ---
    started = time.perf_counter()
    model = get_model(info_or_yaml_or_file=info)
    survey.check_theory(settings=settings, theory=model.theory)
    point = model.logposterior(override)
    if not np.isfinite(point.logpost):
        parser.error(f"log-posterior = {point.logpost} at the fiducial point")
    likelihood = model.likelihood["des_3pcf.joint_shear"]
    evaluated = time.perf_counter()

    # --- 3. THE COVARIANCE ---
    # The likelihood sets its own thread count at every evaluation; the
    # covariance takes the OpenMP team of the environment.
    ci.set_omp_threads(n=int(threads))
    # --xi-only: the xi_+ and xi_- rows, every row whose kind is not map3.
    xi_rows = np.flatnonzero(likelihood.rows.kind != "map3")
    result = survey.compute(interface=ci, likelihood=likelihood,
                            settings=settings, indices=xi_rows)
    finished = time.perf_counter()

    # --- 4. THE ARCHIVE AND ITS INPUT FILES ---
    # The .dataset file names the measurement and n(z) files; the record
    # names each by its path relative to the project folder.
    folder = project/"data"
    if likelihood.path is not None:
        folder = Path(likelihood.path)
    dataset = IniFile(str(folder/likelihood.data_file))
    files = {"evaluate_yaml": args.input.resolve()}
    for key in ("data_file", "cov_file", "nz_source_file"):
        files[key] = Path(dataset.relativeFileName(key)).resolve()
    files["dataset"] = (folder/likelihood.data_file).resolve()
    named = {}
    for key in files:
        named[os.path.relpath(files[key], project)] = files[key]
    survey.save(path=output, result=result, settings=settings,
                input_files=named, save_terms=args.save_terms,
                overwrite=args.overwrite)

    # --- 5. SUMMARY ---
    eigenvalues = np.linalg.eigvalsh(result["covariance"])
    size = result["covariance"].shape[0]
    print(f"Saved the {size} x {size} covariance to {output}")
    print(f"Minimum eigenvalue: {eigenvalues[0]:.6e}")
    print(f"Likelihood evaluation: {evaluated - started:.2f} s; covariance: "
          f"{finished - evaluated:.2f} s ({threads} threads)")
    return output


# __name__ is "__main__" only when this file runs as a script.
if __name__ == "__main__":
    main()
