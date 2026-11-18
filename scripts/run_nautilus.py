"""Sample the declared joint xi±/Map3 likelihood with Nautilus.

Run from the activated Cocoa directory, with one BLAS thread:
  python projects/des_3pcf/EXAMPLE_NAUTILUS1.py --validate-config
  python projects/des_3pcf/EXAMPLE_NAUTILUS1.py --check
  python projects/des_3pcf/EXAMPLE_NAUTILUS1.py --nlive 1000 --neff 10000

--check calculates a fiducial, a displaced IA point and a restored fiducial;
it starts no sampler. --validate-config only reads settings and checks paths.
Ordinary execution uses the explicit YAML sampling budget and retains a
Nautilus checkpoint. Budget exhaustion is reported as nonconvergence.

An optional MPI launch creates one physical-model owner per process, as in
the DES Y3 examples. The master handles Nautilus's proposal calculations;
worker processes evaluate likelihoods. No compiled model is pickled or
shared between processes. The product of ranks and C workers may not
exceed four in this demonstration. Larger allocations need an explicit
resource policy rather than an unannounced change to these examples.
"""

import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import sys


# These assignments precede NumPy/SciPy and the compiled-interface imports.
# C workers are set explicitly from the YAML by the common prediction driver.
for variable in (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "OMP_NUM_THREADS",
):
    os.environ[variable] = "1"

import numpy as np
from threadpoolctl import threadpool_limits


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

import run_3pcf as driver
from des3pcf.inference import inspect_inference, load_inference
from des3pcf.inference import prepare_likelihood, check_likelihood
from des3pcf.inference import file_digest


# MPI workers install their own owner before pool.wait(). Only parameter
# dictionaries travel through the pool; neither FFT plans nor CAMB objects do.
_LIKELIHOOD = None
_PARAMETER_NAMES = None


def nested_likelihood(values):
    """Return a likelihood without including the independent prior again.

    Arguments:
      values = vector in the printed YAML prior order.
    Returns:
      normalized Gaussian log likelihood. An out-of-domain cosmology returns
      minus infinity before CAMB; unexpected numerical/model failures raise.
    """
    sampled = np.asarray(values, dtype=np.float64)
    if sampled.shape != (len(_PARAMETER_NAMES),) or not np.all(np.isfinite(sampled)):
        raise ValueError("Nautilus supplied a malformed sampled-parameter vector.")
    point = dict(_LIKELIHOOD.parameters)
    for index, name in enumerate(_PARAMETER_NAMES):
        point[name] = float(sampled[index])
    if not 0.0 < point["omega_b"] < point["omega_m"] < 1.0:
        return -np.inf
    if point["h"] <= 0.0 or point["sigma8"] <= 0.0:
        return -np.inf
    return _LIKELIHOOD(parameters=point)


def output_prefix(specification, location, override, resume=False):
    """Resolve the destination and refuse replacement before physical setup."""
    value = specification["output"]
    if override is not None:
        value = override
    prefix = (location.parent / value).resolve()
    if resume:
        for suffix in ("_run.json", "_checkpoint.hdf5", "_data.npz"):
            filename = Path(str(prefix) + suffix)
            if not filename.is_file():
                raise FileNotFoundError(
                    f"Resume requires {filename}; choose its original output prefix."
                )
        return prefix
    for suffix in (
        ".json",
        ".1.txt",
        ".paramnames",
        "_checkpoint.hdf5",
        "_data.npz",
        "_run.json",
    ):
        filename = Path(str(prefix) + suffix)
        if filename.exists():
            raise FileExistsError(f"Output {filename} exists; choose another --outroot.")
    return prefix


def resume_identity(record, settings):
    """Separate the statistical target from explicitly extendable stop criteria.

    The input YAML hashes, effective prediction choices, prior order, data
    hashes and compiled model hash must stay identical. A caller may change
    only stopping budgets through CLI overrides; live points, networks and
    seed retain their original values. Changing YAML bytes requires a fresh
    output prefix even if the edit only changes a comment.
    """
    identity = dict(record)
    identity.pop("check", None)
    identity["sampler_definition"] = {
        "n_live": settings["n_live"],
        "n_networks": settings["n_networks"],
        "seed": settings["seed"],
    }
    return identity


def write_json(path, value, replace=False):
    """Publish JSON only after its full contents have been written.

    A resumed run may replace its own previously verified record. The
    temporary sibling is created exclusively and renamed after writing;
    a partial write cannot erase the preceding record.
    """
    if not replace:
        with path.open(mode="x") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
        return
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open(mode="x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(src=temporary, dst=path)


def verify_resume(prefix, identity):
    """Read the saved target and verify its data before opening a checkpoint.

    Returns:
      the saved run record, when its identity and selected-data hash agree.
    Raises:
      ValueError without changing any file when the target differs.
    """
    filename = Path(str(prefix) + "_run.json")
    saved = json.loads(filename.read_text(encoding="utf-8"))
    if saved.get("identity") != identity:
        raise ValueError(
            "Resume target differs in configuration, priors, rows, inputs or "
            "compiled model; restore the exact target or choose another --outroot."
        )
    data_file = Path(str(prefix) + "_data.npz")
    if saved.get("selected_data_sha256") != file_digest(path=data_file):
        raise ValueError("Saved likelihood data changed; restore them before --resume.")
    return saved


def sample(likelihood, distributions, specification, record, pool, prefix, resume):
    """Run the installed Nautilus API and save weighted posterior rows.

    The prior object maps unit-cube draws to normalized physical densities.
    The likelihood callback therefore contains only the likelihood. Saved
    chain columns follow GetDist's weight, negative-log-likelihood, parameters
    convention; parameter order is also recorded in JSON and .paramnames.
    """
    from nautilus import Prior, Sampler

    prior = Prior()
    for name, distribution in distributions.items():
        prior.add_parameter(key=name, dist=distribution)
    settings = specification["sampler"]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    # Saving selected data, labels and C makes the statistical target
    # inspectable without repeating any physical prediction or sampling.
    identity = resume_identity(record=record, settings=settings)
    data_file = Path(str(prefix) + "_data.npz")
    if resume:
        saved = verify_resume(prefix=prefix, identity=identity)
    else:
        with data_file.open(mode="xb") as stream:
            np.savez_compressed(
                stream,
                data=likelihood.data,
                covariance=likelihood.covariance,
                row_labels=likelihood.rows.labels,
                selected_indices=likelihood.indices,
            )
        saved = {
            "identity": identity,
            "selected_data_sha256": file_digest(path=data_file),
            "attempts": [],
        }
    saved["attempts"].append(dict(settings))
    # This record is written before sampling, so an interrupted first run
    # still has a target identity against which --resume can be checked.
    write_json(
        path=Path(str(prefix) + "_run.json"), value=saved, replace=resume,
    )
    sampler = Sampler(
        prior=prior, likelihood=nested_likelihood, pass_dict=False,
        pool=(pool, None), n_live=settings["n_live"],
        n_networks=settings["n_networks"], seed=settings["seed"],
        filepath=str(prefix) + "_checkpoint.hdf5", resume=resume,
    )
    converged = sampler.run(
        f_live=settings["f_live"], n_eff=settings["n_eff"],
        n_like_max=settings["n_like_max"], verbose=True,
        discard_exploration=True,
    )
    record["sampler"] = dict(settings)
    record["converged"] = bool(converged)
    record["likelihood_evaluations"] = int(sampler.n_like)
    record["effective_samples"] = float(sampler.n_eff)
    record["posterior_samples"] = 0
    record["log_evidence"] = None
    if sampler.n_eff > 0:
        points, log_weight, log_like = sampler.posterior()
        columns = np.column_stack((np.exp(log_weight), -log_like, points))
        names = list(distributions)
        chain_file = Path(str(prefix) + ".1.txt")
        chain_temporary = chain_file.with_name(chain_file.name + ".tmp")
        with chain_temporary.open(mode="x") as stream:
            np.savetxt(
                fname=stream, X=columns,
                header="weight neg_log_likelihood " + " ".join(names),
            )
        os.replace(src=chain_temporary, dst=chain_file)
        names_file = Path(str(prefix) + ".paramnames")
        names_temporary = names_file.with_name(names_file.name + ".tmp")
        with names_temporary.open(mode="x") as stream:
            for name in names:
                label = specification["priors"][name].get("latex", name)
                stream.write(f"{name} {label}\n")
        os.replace(src=names_temporary, dst=names_file)
        record["posterior_samples"] = int(len(points))
        record["log_evidence"] = float(sampler.log_z)
    else:
        print("Sampling budget exhausted without retained posterior samples.")
    record["resume_attempts"] = saved["attempts"]
    write_json(path=Path(str(prefix) + ".json"), value=record, replace=resume)
    return record


def main(default_configuration="EXAMPLE_NAUTILUS1.yaml"):
    """Inspect, check or sample one explicitly named inference configuration.

    Arguments:
      default_configuration = project-relative YAML used by a thin example.
    Side effects:
      A sampling run writes its named outputs; inspection and --check do not.
    """
    global _LIKELIHOOD, _PARAMETER_NAMES
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "configuration", nargs="?", type=Path,
        default=PROJECT / default_configuration,
    )
    parser.add_argument("--validate-config", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--nlive", type=int)
    parser.add_argument("--neff", type=int)
    parser.add_argument("--maxfeval", type=int)
    parser.add_argument("--outroot")
    args = parser.parse_args()
    if args.validate_config and args.check:
        parser.error("Choose either --validate-config or --check.")
    if args.resume and (args.validate_config or args.check):
        parser.error("--resume starts sampling; do not combine it with an inspection.")
    for name in ("nlive", "neff", "maxfeval"):
        value = getattr(args, name)
        if value is not None and value < 1:
            parser.error(f"--{name} must be positive.")
    specification, location = load_inference(path=args.configuration)
    if args.validate_config:
        record = inspect_inference(path=location, driver=driver)
        print(json.dumps(record, indent=2, allow_nan=False))
        return
    # mpi4py is only imported by an MPI launch. Serial execution does not
    # initialize an MPI runtime just to discover that it has one process.
    ranks = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
    if ranks * specification["threads"] > 4:
        parser.error("This example requires MPI ranks times C workers <= 4.")
    if args.check and ranks != 1:
        parser.error("Run --check with one process before an MPI sampling launch.")
    prefix = None
    if not args.check:
        prefix = output_prefix(
            specification=specification, location=location, override=args.outroot,
            resume=args.resume,
        )
    # threadpool_limits also constrains a BLAS library already loaded by
    # another importer. Environment assignments alone cannot change that pool.
    with threadpool_limits(limits=1, user_api="blas"):
        likelihood, distributions, specification, record = prepare_likelihood(
            path=location, driver=driver,
        )
        _LIKELIHOOD = likelihood
        _PARAMETER_NAMES = list(distributions)
        for argument, key in (
            (args.nlive, "n_live"),
            (args.neff, "n_eff"),
            (args.maxfeval, "n_like_max"),
        ):
            if argument is not None:
                specification["sampler"][key] = argument
        context = nullcontext(None)
        if ranks > 1:
            from schwimmbad import MPIPool

            context = MPIPool()
        with context as pool:
            if pool is not None and not pool.is_master():
                pool.wait()
                return
            record["check"] = check_likelihood(
                likelihood=likelihood, distributions=distributions,
                specification=specification,
            )
            record["mpi_ranks"] = ranks
            record["blas_threads"] = 1
            if args.check:
                print(json.dumps(record, indent=2, allow_nan=False))
                return
            result = sample(
                likelihood=likelihood, distributions=distributions,
                specification=specification, record=record,
                pool=pool, prefix=prefix, resume=args.resume,
            )
            print(json.dumps({
                "output": str(prefix),
                "converged": result["converged"],
                "posterior_samples": result["posterior_samples"],
            }, indent=2))


if __name__ == "__main__":
    main()
