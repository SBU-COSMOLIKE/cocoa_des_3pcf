"""Check the Cocoa covariance driver and its DES Y3 adapter on the fiducial.

The module builds the Cobaya model of EXAMPLE_EVALUATE_COVARIANCE.yaml
(CAMB, the BiHalofit theory block and the likelihood des_3pcf.joint_shear)
and evaluates it once at the yaml's evaluate override, as
covariance/compute_covariance.py does. The adapter
covariance/des_3pcf_covariance.py then computes the covariance of the
likelihood's 480 rows. The checks are structural (a symmetric, finite,
positive definite matrix that is the same byte for byte on a second call),
one physical check against a hand calculation of the white shape noise of
xi, and the refusals of the driver and the adapter. None of them is an
accuracy or convergence study of the covariance.

The whole module is skipped on the covariance-off build of the project
interface (IGNORE_COSMOLIKE_DES_3PCF_COVARIANCE set at compile time), which
has no shape_noise binding.
"""

from pathlib import Path
import runpy
import sys

import numpy as np
import pytest
from cobaya.model import get_model
from cobaya.yaml import yaml_load_file

import cosmolike_des_3pcf_interface as ci


# PROJECT = projects/des_3pcf; COCOA = the Cocoa/ folder, against which the
# yaml's relative CAMB path is written. The adapter lives in covariance/.
PROJECT = Path(__file__).resolve().parents[2]
COCOA = PROJECT.parents[1]
YAML = PROJECT / "EXAMPLE_EVALUATE_COVARIANCE.yaml"
DRIVER = PROJECT / "covariance" / "compute_covariance.py"
sys.path.insert(0, str(PROJECT / "covariance"))

import des_3pcf_covariance as survey


# Four OpenMP threads: the determinism check must run multi-threaded, or a
# race between threads could never show up.
THREADS = 4

pytestmark = pytest.mark.skipif(
    not hasattr(ci, "shape_noise"),
    reason="the project interface is the covariance-off build "
           "(IGNORE_COSMOLIKE_DES_3PCF_COVARIANCE); no shape_noise binding",
)


@pytest.fixture(scope="module")
def evaluated():
    """Return the fiducial likelihood after one Cobaya evaluation.

    The yaml's covariance and output blocks belong to the driver and are
    removed before Cobaya reads the mapping; the CAMB path is resolved
    against the Cocoa/ folder, because the tests run from the project
    folder.

    Returns:
        dict with "model" (the Cobaya model, kept alive), "likelihood"
        (des_3pcf.joint_shear after the evaluation) and "settings" (the
        adapter configuration with the yaml's neutrino_model).
    """
    info = yaml_load_file(file_name=str(YAML))
    block = info.pop("covariance")
    info.pop("output")
    sampler = info.pop("sampler")
    info["theory"]["camb"]["path"] = str(COCOA / "external_modules/code/CAMB")
    model = get_model(info_or_yaml_or_file=info)
    point = model.logposterior(sampler["evaluate"]["override"])
    assert np.isfinite(point.logpost)
    settings = survey.configuration(neutrino_model=block["neutrino_model"])
    survey.check_theory(settings=settings, theory=model.theory)
    return {
        "model": model,
        "likelihood": model.likelihood["des_3pcf.joint_shear"],
        "settings": settings,
    }


def xi_indices(likelihood):
    """Return the indices of the xi_+ and xi_- rows (kind other than map3)."""
    return np.flatnonzero(likelihood.rows.kind != "map3")


@pytest.fixture(scope="module")
def result(evaluated):
    """Return the adapter's covariance of the 400 xi rows.

    The Map3 rows are refused for now: the likelihood's Map3 mean uses the
    finite-SAS estimator, the C covariance codes only the full-filter one.
    """
    ci.set_omp_threads(n=THREADS)
    return survey.compute(
        interface=ci,
        likelihood=evaluated["likelihood"],
        settings=evaluated["settings"],
        indices=xi_indices(likelihood=evaluated["likelihood"]),
    )


def test_map3_rows_are_refused(evaluated):
    """A request that includes Map3 rows fails with the estimator reason."""
    with pytest.raises(ValueError, match="finite-SAS"):
        survey.compute(
            interface=ci,
            likelihood=evaluated["likelihood"],
            settings=evaluated["settings"],
        )


def test_matrix_is_symmetric_finite_and_positive(result):
    """The summed matrix is exactly symmetric, finite and positive definite.

    The Cholesky factorization C = L L^T exists only for a positive
    definite matrix, so numpy raising LinAlgError would fail the test.
    The smallest eigenvalue is checked too, so a failure names its value.
    """
    covariance = result["covariance"]
    assert covariance.shape == (400, 400)
    assert np.all(np.isfinite(covariance))
    assert np.array_equal(covariance, covariance.T)
    np.linalg.cholesky(covariance)
    smallest = np.linalg.eigvalsh(covariance)[0]
    assert smallest > 0.0, f"minimum eigenvalue {smallest}"
    assert result["terms"].shape == (6, 400, 400)
    assert np.array_equal(np.sum(result["terms"], axis=0), covariance)


def test_two_calls_are_byte_identical(evaluated, result):
    """A second call on the same likelihood returns the same bytes."""
    ci.set_omp_threads(n=THREADS)
    again = survey.compute(
        interface=ci,
        likelihood=evaluated["likelihood"],
        settings=evaluated["settings"],
        indices=xi_indices(likelihood=evaluated["likelihood"]),
    )
    for name in ("covariance", "terms", "mean"):
        assert again[name].tobytes() == result[name].tobytes(), name


def test_xi_shape_noise_matches_the_white_noise_formula(evaluated):
    """The xi noise-noise diagonal equals 2 (N_ik N_jl + N_il N_jk)/(A Omega).

    covariance/README.md gives the pure-noise covariance of two xi rows of
    the same kind; on the diagonal (same bins, same annulus) it is
    4 N_i^2 / (A Omega) for an auto pair and 2 N_i N_j / (A Omega) for a
    cross pair, with N_i = sigma_e,i^2 / n_i (n_i per steradian), A the
    survey area in steradians and Omega = pi (theta_max^2 - theta_min^2)
    the annulus area.

    The Gaussian term also holds signal-signal (SS) and signal-noise (SN)
    parts. With every N_i scaled by s, G(s) = SS + s SN + s^2 NN, so three
    runs with s = 1, 2, 4 isolate NN = (G(4) - 3 G(2) + 2 G(1)) / 6. The
    scaling enters through sigma_e (times sqrt(s)), an explicit setting.
    """
    likelihood = evaluated["likelihood"]
    rows = likelihood.rows

    # Three rows: the smallest xi_+ annulus of the auto pair (1,1), the
    # same annulus of the cross pair (1,2), and a xi_- annulus of (4,4).
    # np.flatnonzero returns the indices where the condition holds.
    chosen = []
    for kind, first, second in (("xip", 0, 0), ("xip", 0, 1), ("xim", 3, 3)):
        match = np.flatnonzero(
            (rows.kind == kind)
            & (rows.bins[:, 0] == first)
            & (rows.bins[:, 1] == second)
        )
        chosen.append(int(match[0]))
    indices = np.asarray(chosen, dtype=np.int64)

    # Gaussian diagonal at the three noise scalings.
    diagonal = {}
    ci.set_omp_threads(n=THREADS)
    for scale in (1.0, 2.0, 4.0):
        settings = dict(evaluated["settings"])
        sigma = np.asarray(settings["sigma_e_component"]) * np.sqrt(scale)
        settings["sigma_e_component"] = sigma.tolist()
        scaled = survey.compute(interface=ci, likelihood=likelihood,
                                settings=settings, indices=indices)
        diagonal[scale] = np.diag(scaled["terms"][0])
    noise_noise = (diagonal[4.0] - 3.0 * diagonal[2.0] + 2.0 * diagonal[1.0]) / 6.0

    # Hand calculation, independent of the C code: 10800/pi arcminutes per
    # radian converts a density per arcmin^2 into one per steradian.
    settings = evaluated["settings"]
    arcmin_per_radian = 10800.0 / np.pi
    density = np.asarray(settings["source_density_arcmin2"]) * arcmin_per_radian**2
    white = np.asarray(settings["sigma_e_component"]) ** 2 / density
    area = 4143.0 * (np.pi / 180.0) ** 2
    expected = np.empty(indices.size)
    for position in range(indices.size):
        row = indices[position]
        first = rows.bins[row, 0]
        second = rows.bins[row, 1]
        lower = rows.angles[row, 0]
        upper = rows.angles[row, 1]
        annulus = np.pi * (upper**2 - lower**2)
        pairs = 2.0 * white[first] * white[second]
        if first == second:
            pairs = 4.0 * white[first] ** 2
        expected[position] = pairs / (area * annulus)
    np.testing.assert_allclose(actual=noise_noise, desired=expected, rtol=1.0e-8)


def test_driver_refuses_an_existing_output(tmp_path, capsys):
    """The driver and save() refuse to replace a file without --overwrite.

    runpy.run_path executes the driver file without running main() (its
    __name__ is not "__main__") and returns its namespace; the call below
    then runs main() in this process. The refusal comes before any model
    is built, so the existing file keeps its bytes.
    """
    existing = tmp_path / "existing.npz"
    existing.write_bytes(b"keep")
    namespace = runpy.run_path(str(DRIVER))
    with pytest.raises(SystemExit) as stopped:
        namespace["main"](argv=[str(YAML), "--xi-only",
                                "--output", str(existing)])
    assert stopped.value.code == 2
    assert "exists" in capsys.readouterr().err
    with pytest.raises(FileExistsError):
        survey.save(path=existing, result={}, settings={}, input_files={},
                    save_terms=False, overwrite=False)
    assert existing.read_bytes() == b"keep"


class SharedVarianceBlock:
    """Stands in for a bihalofit theory block set to the shared route."""

    variance_source = "shared"


def test_adapter_refusals_name_the_value():
    """Nonzero m_i, an undeclared positive mnu and a shared variance fail."""
    massless = survey.configuration(neutrino_model="massless")
    single = survey.configuration(neutrino_model="single_fluid_effective_growth")
    with pytest.raises(ValueError, match="DES_M2 = 0.01"):
        survey.check_point(settings=single, parameters={"DES_M2": 0.01, "mnu": 0.0},
                           nbin=4)
    with pytest.raises(ValueError, match="mnu = 0.06 eV"):
        survey.check_point(settings=massless, parameters={"mnu": 0.06}, nbin=4)
    with pytest.raises(ValueError, match="mnu is missing"):
        survey.check_point(settings=single, parameters={}, nbin=4)
    survey.check_point(settings=single, parameters={"mnu": 0.06}, nbin=4)
    with pytest.raises(ValueError, match="'shared'"):
        survey.check_theory(settings=single,
                            theory={"bihalofit": SharedVarianceBlock()})
    with pytest.raises(ValueError, match="neutrino_model"):
        survey.configuration(neutrino_model="cold_baryon")
