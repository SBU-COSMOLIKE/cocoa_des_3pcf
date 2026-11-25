"""Compare the candidate matter emulator with CAMB at random des_3pcf cosmologies.

Run from activated Cocoa (one project at a time):
    python projects/des_3pcf/tests/compare_emulator_camb.py
    OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
      mpirun -n 2 --bind-to none python -m mpi4py \
      projects/des_3pcf/tests/compare_emulator_camb.py --mpi

MPI assigns independent parameter columns or cosmologies to each rank;
OpenMP accelerates each rank's model evaluations. Keep BLAS at one thread.

The shared Cocoa test uses the numbered example's cosmological priors,
restricted to 0.2 < Omega_m < 0.5 and fixed mnu=0.06 eV. Nuisance parameters
stay at its declared point. Both calculations use plain mead2020 and CAMB
background distances. BiHalofit consumes the selected provider power.
The native mask and MOPED compression apply before covariance weighting;
the Fisher overlay is the Gaussian fixed-covariance approximation. This does not change the production nonlinear
prescription or its sampled neutrino-mass prior.

Use EMULATOR_CODE_DIR and EMULATOR_DATA_DIR to select candidate checkouts.
COCOA_TEST_INTERFACE_DIR optionally selects an independently built native
interface. Results, including failed points, go under chains/.

The optional --points argument reads explicit named-column parameter points.
The default Fisher calculation varies the sampled cosmological and nuisance
parameters for this project's full retained observable. The shared test
then selects 25 points from its bounded Fisher distribution and evaluates
both theories again. --local-points replaces that selection with a named
parameter table. The E5 test uses --advisory and retains all outcomes.
"""

import argparse
import os
from pathlib import Path
import sys

# Set thread ownership before NumPy, TensorFlow or the native library loads.
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = os.environ["OMP_NUM_THREADS"]
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["COBAYA_NOMPI"] = "1"

project = Path(__file__).resolve().parents[1]
root = project.parents[1]
core = root / "external_modules/code/cosmolike_core"
interface = Path(os.environ.get("COCOA_TEST_INTERFACE_DIR", project / "interface"))
sys.path.insert(0, str(core))
sys.path.insert(0, str(interface))
sys.path.insert(0, str(root / "cobaya"))

import numpy as np
import cosmolike_des_3pcf_interface as ci
from cocoa_testing import (
    run_matter_emulator_prior_check,
    plot_matter_emulator_fisher,
    sample_matter_emulator_fisher,
)


def get_precision():
    """Invert the native covariance after the dataset mask and compression.

    The shared test applies ci.select_joint_data_vector to every full
    prediction, so this metric scores exactly the same measured modes.
    This is a fixed-covariance Gaussian Fisher diagnostic; the numbered
    example's Percival likelihood is still used to verify each saved mean.
    """
    from cosmolike_notebook_utils.fisher import symmetric_matrix

    covariance = symmetric_matrix(ci.get_joint_covariance(), name="Joint covariance")
    # Xi and Map3 carry very different units. Remove those units before
    # solving; otherwise raw inversion can amplify floating-point asymmetry.
    diagonal = np.diag(covariance)
    if np.any(diagonal <= 0.0):
        raise ValueError("Joint covariance must have positive diagonal entries.")
    scale = np.sqrt(diagonal)
    correlation = covariance / np.outer(scale, scale)
    factor = np.linalg.cholesky(correlation)
    inverse_factor = np.linalg.solve(factor, np.eye(len(factor)))
    # If R=L L^T, then R^{-1}=L^{-T} L^{-1}. This Gram product computes
    # both triangles from the same factors and preserves their symmetry.
    return (inverse_factor.T @ inverse_factor) / np.outer(scale, scale)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configuration", default="EXAMPLE_EVALUATE1.yaml")
    parser.add_argument("--models", type=int, default=50)
    parser.add_argument("--seed", type=int, default=137)
    parser.add_argument("--limit", type=float, default=1.0)
    parser.add_argument("--points", type=Path, help="Named-column parameter table instead of random draws")
    parser.add_argument("--no-fisher", action="store_true",
                        help="Skip the derivative and contour stage")
    parser.add_argument("--output", type=Path,
                        help="New result directory below project chains/")
    parser.add_argument("--plot-only", type=Path,
                        help="Redraw a completed comparison directory")
    parser.add_argument("--fisher-sample-power", type=int, default=18,
                        help="log2 bounded-Fisher proposals per repetition")
    parser.add_argument("--resample-fisher", action="store_true",
                        help="With --plot-only, repeat bounded sampling")
    parser.add_argument("--advisory", action="store_true",
                        help="Report error sizes; fail only incomplete calculations")
    parser.add_argument("--local-points", type=Path,
                        default=None,
                        help="Replace the default bounded-Fisher local points")
    parser.add_argument("--mpi", action="store_true",
                        help="Distribute points and Fisher columns across MPI ranks")
    args = parser.parse_args()
    # Each rank owns its Cobaya model and native state. The shared test
    # distributes different jobs and lets rank zero write combined results.
    # Cobaya's own MPI remains disabled; this driver owns the work split.
    mpi_comm = None
    if args.mpi:
        from mpi4py import MPI
        mpi_comm = MPI.COMM_WORLD
    elif int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1")) > 1:
        parser.error("Use --mpi with mpirun to distribute work without duplicate writes.")
    if args.models < 1 or args.limit <= 0:
        parser.error("models and limit must be positive")
    points_file = args.points.resolve() if args.points is not None else None
    if args.plot_only is not None:
        if mpi_comm is not None and mpi_comm.Get_rank() != 0:
            sys.exit(0)
        output = args.plot_only.resolve()
        if args.resample_fisher:
            sample_matter_emulator_fisher(
                output=output / "fisher",
                sample_power=args.fisher_sample_power,
            )
        local = output / "local"
        plot_matter_emulator_fisher(
            output=output / "fisher",
            points_file=output / "points.txt",
            distances_file=output / "distances.txt",
            project_name=project.name,
            local_points_file=local / "points.txt" if local.is_dir() else None,
            local_distances_file=local / "distances.txt" if local.is_dir() else None,
        )
        sys.exit(0)
    output_dir = args.output.resolve() if args.output is not None else None
    local_points = args.local_points.resolve() if args.local_points is not None else None
    os.chdir(root)
    failed = run_matter_emulator_prior_check(
        project=project,
        mpi_comm=mpi_comm,
        configuration=args.configuration,
        get_precision=get_precision,
        select_vector=ci.select_joint_data_vector,
        use_emulator_switch=False,
        candidate_extra_args={"cold_power": "eisenstein_hu", "cold_neutrino_species": 3},
        n_models=args.models,
        seed=args.seed,
        limit=args.limit,
        fisher=not args.no_fisher,
        output_dir=output_dir,
        fisher_sample_power=args.fisher_sample_power,
        points_file=points_file,
        advisory=args.advisory,
        local_points_file=local_points,
    )
    sys.exit(1 if failed else 0)
