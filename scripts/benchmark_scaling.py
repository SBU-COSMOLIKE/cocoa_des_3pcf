"""Run the declared ten-point CLI study after the machine is quiet.

Preflight reads inputs and draws points only. It performs no CAMB calculation,
compiled prediction or timing. Run prepares physical tables, then starts one
fresh child for each workload/thread block. Plot consumes measured records.
"""

import argparse
import os
from pathlib import Path
import sys


# This must precede numerical imports, including those in project helpers.
for variable in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "BLIS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ["OMP_THREAD_LIMIT"] = "8"
os.environ["OMP_DYNAMIC"] = "FALSE"
os.environ["MKL_DYNAMIC"] = "FALSE"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from des3pcf.scaling import preflight, run_study, plot_study, prepare_job, run_job


def main():
    """Select preparation, execution or plotting without implicit numerical work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "plot",
                                            "_prepare", "_worker"))
    parser.add_argument("--config", type=Path,
                        default=PROJECT / "configs" / "openmp_scaling.yaml")
    parser.add_argument("--job", type=Path)
    parser.add_argument("--reuse-from", type=Path,
                        help="Reuse verified LSST physical tables from an earlier run.")
    args = parser.parse_args()
    if args.command == "preflight":
        import json

        print(json.dumps(preflight(args.config), indent=2, sort_keys=True))
    elif args.command == "run":
        print(run_study(args.config, reuse_from=args.reuse_from))
    elif args.command == "plot":
        print(plot_study(args.config))
    elif args.job is None:
        parser.error("Internal child commands require --job.")
    elif args.command == "_prepare":
        prepare_job(args.job)
    else:
        run_job(args.job)


if __name__ == "__main__":
    main()
