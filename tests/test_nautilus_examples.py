"""Exercise the shared LSST/DES Nautilus driver with an analytic posterior.

The target has a Gaussian likelihood, a Gaussian one-parameter prior and
an external correlated prior. Chain scores can therefore be checked
independently: Nautilus must include the correlated factor once and must
not include the Gaussian prior twice. No CAMB solve or DES forecast runs.
"""

import hashlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import yaml

from cocoa_hybrid_sampling import load_model, parser_for
from cobaya.input import update_info


PROJECT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("example", [
    "EXAMPLE_EVALUATE1.yaml",
    "EXAMPLE_EVALUATE3.yaml",
    "EXAMPLE_EVALUATE4.yaml",
    "EXAMPLE_EVALUATE2.yaml",
    "EXAMPLE_EVALUATE5.yaml",
    "EXAMPLE_MCMC1.yaml",
    "EXAMPLE_MCMC3.yaml",
    "EXAMPLE_MCMC4.yaml",
    "EXAMPLE_MCMC2.yaml",
    "EXAMPLE_MCMC5.yaml",
    "EXAMPLE_NAUTILUS1.yaml",
    "EXAMPLE_NAUTILUS3.yaml",
    "EXAMPLE_NAUTILUS4.yaml",
    "EXAMPLE_NAUTILUS2.yaml",
    "EXAMPLE_NAUTILUS5.yaml",
])
def test_cobaya_loads_project_likelihood_defaults(example):
    """Resolve the installed likelihood and its defaults without a prediction.

    Cobaya rejects duplicate Python/YAML defaults and unknown likelihood
    names before a model is constructed. Exercise that actual loader for
    every shipped example, including the plain evaluate entry point.

    Arguments:
      example = project YAML filename.
    Returns:
      nothing; asserts that the resolved component retains its dataset.
    """
    settings = yaml.safe_load((PROJECT / example).read_text(encoding="utf-8"))
    updated = update_info(info=settings)
    component = "des_3pcf.joint_shear"
    data_file = updated["likelihood"][component]["data_file"]
    assert (PROJECT / "data" / data_file).is_file()


# The dataset of each numbered case (real data plain, simulated or synthetic
# vectors with the _sim suffix).
DATASETS = {1: "des_y3_joint1.dataset", 2: "des_y3_joint2.dataset",
            3: "des_y3_joint3_sim.dataset", 4: "des_y3_joint4_sim.dataset",
            5: "des_y3_joint5_sim.dataset"}


@pytest.mark.parametrize("number", [1, 2, 3, 4])
def test_joint_likelihood_initializes_without_generating_covariance(number):
    """Prepare real measurement rows through the installed Cobaya class.

    Loading YAML alone does not exercise the native mean bindings. This
    initialization needs quadrature and xi-bin weights even when the
    covariance generator is excluded from the build. It reads the supplied
    covariance; it never computes a cosmology or a new covariance matrix.
    Case 5 needs an independently trained compression file and is separate.
    Case 1 keeps 307 rows after the DES Y3 xi scale cuts and compresses them
    to 96 (16 published MOPED modes plus the 80 Map3 rows); the others keep 480.
    """
    from cobaya.likelihoods.des_3pcf.joint_shear import joint_shear
    import cosmolike_des_3pcf_interface as ci

    likelihood = joint_shear(info={"data_file": DATASETS[number]})
    assert likelihood.xi_theta.shape == (400, 32)
    np.testing.assert_allclose(
        actual=np.sum(likelihood.xi_weight, axis=1),
        desired=1.0,
        rtol=2.0e-15,
    )
    assert callable(ci.shear_power_rows)
    assert callable(ci.production.xi_from_power)
    assert callable(ci.notebook.xi_from_power)
    assert ci.get_joint_data().size == (96 if number == 1 else 480)
    likelihood.close()


@pytest.mark.parametrize("number", [1, 2, 3, 4, 5])
def test_evaluate_mcmc_and_nautilus_use_the_same_science(number):
    """Changing samplers must preserve data, theory, parameters and priors."""
    reference = yaml.safe_load(
        (PROJECT / f"EXAMPLE_NAUTILUS{number}.yaml").read_text(encoding="utf-8"),
    )
    for sampler in ("EVALUATE", "MCMC"):
        settings = yaml.safe_load(
            (PROJECT / f"EXAMPLE_{sampler}{number}.yaml").read_text(encoding="utf-8"),
        )
        for key in ("likelihood", "theory", "params", "prior"):
            assert settings.get(key) == reference.get(key)
        assert settings["output"] == (
            f"./projects/des_3pcf/chains/EXAMPLE_{sampler}{number}"
        )
        if sampler == "EVALUATE":
            assert settings["sampler"] == reference["sampler"]
        else:
            assert set(settings["sampler"]) == {"mcmc"}


def gaussian_loglike(x, y):
    """Return a two-dimensional unit-width Gaussian log likelihood."""
    return -0.5 * (x*x + y*y) - np.log(2.0*np.pi)


def correlated_prior(x, y):
    """Apply a known external prior along the sum of the two coordinates."""
    return -0.5 * ((x+y)/0.7)**2


@pytest.fixture
def analytic_configuration(tmp_path):
    """Write an actual Cobaya model with finite, independently checked scores."""
    settings = {
        "likelihood": {
            "gaussian": {
                "external": 'import_module("test_nautilus_examples").gaussian_loglike',
            },
        },
        "prior": {
            "correlated": 'import_module("test_nautilus_examples").correlated_prior',
        },
        "params": {
            "x": {
                "prior": {"dist": "norm", "loc": 0.0, "scale": 0.8},
                "latex": "x",
            },
            "y": {
                "prior": {"min": -3.0, "max": 3.0},
                "latex": "y",
            },
        },
        "sampler": {
            "evaluate": {"override": {"x": 0.2, "y": -0.1}},
        },
    }
    path = tmp_path / "analytic.yaml"
    path.write_text(yaml.safe_dump(settings, sort_keys=False), encoding="utf-8")
    return path


def run_example(number, configuration, root, options):
    """Invoke the real project launcher, with test functions on the import path."""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join([
        str(PROJECT / "tests"),
        str(PROJECT / "scripts"),
        environment.get("PYTHONPATH", ""),
    ])
    environment["OMP_NUM_THREADS"] = "1"
    environment["OMP_THREAD_LIMIT"] = "6"
    for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        environment[name] = "1"
    arguments = [
        sys.executable,
        "-B",
        str(PROJECT / f"EXAMPLE_NAUTILUS{number}.py"),
        "--input",
        str(configuration),
        "--root",
        str(root),
    ]
    arguments.extend(options)
    return subprocess.run(
        args=arguments, env=environment, capture_output=True,
        text=True, timeout=120, check=False,
    )


def test_hybrid_default_keeps_emulator_restrictions(analytic_configuration):
    """Physical opt-in must not relax the LSST hybrid default."""
    args = parser_for(mode="nautilus", project=PROJECT, example=2).parse_args([])
    assert args.input.name == "EXAMPLE_EMUL2_EVALUATE2.yaml"
    assert args.outroot == "EXAMPLE_EMUL2_NAUTILUS2"
    with pytest.raises(ValueError, match="use_emulator: 2"):
        load_model(filename=analytic_configuration)


@pytest.mark.parametrize("number", [1, 2, 3, 4, 5])
def test_each_launcher_uses_shared_check(number, analytic_configuration, tmp_path):
    """All five launchers accept the LSST options and leave no chain on --check."""
    result = run_example(
        number=number, configuration=analytic_configuration,
        root=tmp_path, options=["--check"],
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Sampled parameter order:" in result.stdout
    assert "log likelihood=" in result.stdout
    assert not (tmp_path / "chains").exists()


def test_sample_scores_and_existing_output_refusal(analytic_configuration, tmp_path):
    """Check real Nautilus output against the analytic score and refuse overwrite."""
    options = [
        "--nlive", "40",
        "--nnetworks", "1",
        "--neff", "50",
        "--maxfeval", "4000",
        "--flive", "0.05",
        "--seed", "42",
        "--outroot", "analytic",
    ]
    result = run_example(
        number=1, configuration=analytic_configuration,
        root=tmp_path, options=options,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    prefix = tmp_path / "chains" / "analytic"
    record = yaml.safe_load(prefix.with_suffix(".yaml").read_text())
    assert record["posterior_samples"] > 0
    chain = np.loadtxt(fname=str(prefix) + ".1.txt")
    x = chain[:, 2]
    y = chain[:, 3]
    expected = gaussian_loglike(x=x, y=y) + correlated_prior(x=x, y=y)
    np.testing.assert_allclose(actual=-chain[:, 1], desired=expected, atol=2.e-14)
    before = {}
    for path in prefix.parent.iterdir():
        before[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    repeated = run_example(
        number=1, configuration=analytic_configuration,
        root=tmp_path, options=options,
    )
    assert repeated.returncode != 0
    assert "select a new --outroot" in repeated.stdout + repeated.stderr
    after = {}
    for path in prefix.parent.iterdir():
        after[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    assert after == before
