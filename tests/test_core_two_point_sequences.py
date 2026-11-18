"""Restore the real DES Y3 core after unannounced notebook state changes.

These checks use the existing ordinary-core extension, not a mock or the
new three-point extension. Three multipoles and two/three smooth source
curves suffice to expose source-count, IA, interpolation and selector
contamination. Reference arrays come from separate clean interpreters.
The reused interpreter keeps old adapter instances while replacing raw
core state, including deliberately incomplete configurations.

The numerical controls are small, explicit test settings. Their equality
checks establish restoration, not a converged survey prediction. A tiny
constant baryon suppression file is generated only to verify that a
previously loaded contamination is removed; it is not a feedback model.
"""

import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


PROJECT = Path(__file__).resolve().parents[1]
CORE_DIRECTORY = PROJECT.parent / "des_y3" / "interface"
CORE_MODULE = CORE_DIRECTORY / "cosmolike_des_y3_interface.so"


def physical_inputs(nbin):
    """Build matching h-unit tables without CAMB or a survey data file."""
    wave = np.geomspace(start=1.0e-5, stop=1.0e3, num=96)
    redshift = np.linspace(start=0.0, stop=2.0, num=41)
    linear = 2.0e4 * wave / (1.0 + (wave / 0.15) ** 2) ** 1.8
    growth = 1.0 / (1.0 + redshift)
    nonlinear = linear * (1.0 + (wave / 0.4) ** 2) ** 0.2
    sources = np.empty((nbin, redshift.size))
    for index in range(nbin):
        scale = 0.55 + 0.25 * index
        sources[index] = redshift**2 * np.exp(-(redshift / scale) ** 1.5)
    tables = {
        "k": wave,
        "p_linear0": linear,
        "p_linear": growth[:, None] ** 2 * linear[None, :],
        "p_nonlinear": growth[:, None] ** 2 * nonlinear[None, :],
        "z": redshift,
        "chi": 3000.0 * redshift / (1.0 + 0.35 * redshift),
        "dz_dchi": (1.0 + 0.35 * redshift) ** 2 / 3000.0,
        "growth": growth,
        "sources": sources,
        "los_z": np.array([0.1, 0.4, 0.8, 1.5]),
        "omega_m": 0.3,
        "omega_de": 0.7,
        "w0": -1.0,
        "wa": 0.0,
        "ns": 0.965,
        "sigma8": 0.8,
    }
    parameters = {"h": 0.7, "omega_b": 0.05, "mnu": 0.0}
    ia = {
        "a1": 0.5,
        "a2": 0.3,
        "eta1": -0.7,
        "eta2": 0.4,
        "bta": 0.8,
        "zpivot": 0.62,
        "c1rho": 0.0134,
    }
    bins = np.array([[0, 0], [0, nbin - 1], [nbin - 1, nbin - 1]],
                    dtype=np.int32)
    return {
        "parameters": parameters,
        "tables": tables,
        "metadata": {},
        "ia": ia,
        "ell": np.array([50.0, 170.0, 530.0]),
        "bins": bins,
    }


def adapter_for_case(core, nbin):
    """Use different declared source/numerical models for the A/B/A test."""
    from des3pcf.core_two_point import CoreTwoPoint

    if nbin == 2:
        return CoreTwoPoint(
            core=core, nbin=2, threads=1, interpolation_type=0,
            accuracy_boost=0.5, integration_accuracy=0,
            fpt_internal_boost=0.5, ell_internal=32,
        )
    return CoreTwoPoint(
        core=core, nbin=3, threads=2, interpolation_type=1,
        accuracy_boost=0.75, integration_accuracy=0,
        fpt_internal_boost=0.4, ell_internal=48,
    )


def record(result, prefix, spectra):
    """Retain owned float arrays before another ordinary-core call occurs."""
    result[prefix + "_e"] = np.array(spectra[0], copy=True)
    result[prefix + "_b"] = np.array(spectra[1], copy=True)


def assert_bits(actual, expected, context):
    """Include signed zeros in the fresh-versus-reused comparison."""
    first = np.ascontiguousarray(actual, dtype=np.float64).view(np.uint64)
    second = np.ascontiguousarray(expected, dtype=np.float64).view(np.uint64)
    np.testing.assert_array_equal(first, second, err_msg=context)


def baryon_test_file(path):
    """Write the existing core's HDF5 schema with constant P_hydro/P_DMO."""
    import h5py

    with h5py.File(path, mode="w") as output:
        group = output.create_group(name="controlled")
        # The core converts these to increasing scale factor a=1/(1+z).
        redshift = group.create_dataset(
            name="zBins", data=np.array([3.0, 1.0, 0.3, 0.0], dtype=np.float32),
        )
        redshift.attrs["Na_bins"] = np.int32(4)
        wave = group.create_dataset(
            name="logkBins", data=np.array([-6.0, -2.0, 1.0, 4.0],
                                          dtype=np.float32),
        )
        wave.attrs["Nk_bins"] = np.int32(4)
        ratios = group.create_group(name="logPkR")
        ratios.create_dataset(name="sim1",
                              data=np.full((4, 4), np.log10(0.65),
                                           dtype=np.float32))


def child_calculation(case, destination):
    """Run one clean reference or a complete reused-process setting sequence."""
    sys.path.insert(0, str(PROJECT / "python"))
    sys.path.insert(0, str(CORE_DIRECTORY))
    import cosmolike_des_y3_interface as core
    from des3pcf.mean import core_power_rows

    first = adapter_for_case(core=core, nbin=2)
    arguments = physical_inputs(nbin=2)
    result = {}
    if case == "fresh3":
        second = adapter_for_case(core=core, nbin=3)
        record(result=result, prefix="reference",
               spectra=second.spectra(**physical_inputs(nbin=3)))
    elif case == "fresh2":
        record(result=result, prefix="reference", spectra=first.spectra(**arguments))
    else:
        original = first.spectra(**arguments)
        record(result=result, prefix="before", spectra=original)
        if case == "adapters":
            second = adapter_for_case(core=core, nbin=3)
            record(result=result, prefix="other",
                   spectra=second.spectra(**physical_inputs(nbin=3)))
            # Source-count growth must not change previously returned arrays.
            assert_bits(actual=original[0], expected=result["before_e"],
                        context="old E output")
            assert_bits(actual=original[1], expected=result["before_b"],
                        context="old B output")
            record(result=result, prefix="after", spectra=first.spectra(**arguments))
            record(result=result, prefix="other_again",
                   spectra=second.spectra(**physical_inputs(nbin=3)))
        elif case == "selectors":
            core.init_cosmo_runmode(is_linear=True)
            core.init_IA(ia_model=0, ia_redshift_evolution=0, ia_code=0)
            raw = core_power_rows(core=core, ell=arguments["ell"],
                                  bins=arguments["bins"])
            record(result=result, prefix="changed", spectra=raw)
            # These are complete public mutations. The adapter must restore
            # its own choices without relying on callers to announce them.
            core.init_accuracy_boost(accuracy_boost=0.8, integration_accuracy=1)
            core.init_fpt_internal_boost(internal_boost=0.9)
            core.init_ntable_ell_internal(nell_internal=40)
            core.init_ntable_lmax(lmax=3000)
            core.init_photoz_conventions(interpolation_type=1, zmid_convention=0)
            core.init_photoz_histogram_akima(enabled=1, preparation_factor=5)
            core.init_include_halo_IA(include_halo_IA=1)
            core.init_probes(possible_probes="wtheta")
            core.set_nuisance_shear_photoz(bias=np.array([0.04, -0.03]))
            core.set_nuisance_shear_calib(M=np.array([0.2, -0.1]))
            core.set_omp_threads(n=2)
            record(result=result, prefix="after", spectra=first.spectra(**arguments))
        elif case == "half_state":
            # Even a count change without a corresponding n(z) install must
            # not make a subsequent adapter read old arrays with new sizes.
            core.init_source_sample_size(Ntomo=4)
            core.init_lens_sample_size(Ntomo=2)
            record(result=result, prefix="after", spectra=first.spectra(**arguments))
            # A hidden accuracy baseline can also be captured from a raw
            # partial setup. Resetting before the adapter's own boost avoids
            # silently inheriting this unrelated coarse-node count.
            core.initial_setup()
            core.init_ntable_ell_internal(nell_internal=40)
            core.init_accuracy_boost(accuracy_boost=0.7, integration_accuracy=1)
            core.init_source_sample_size(Ntomo=3)
            record(result=result, prefix="after_reset",
                   spectra=first.spectra(**arguments))
        elif case == "baryons":
            table = destination.with_suffix(".h5")
            baryon_test_file(path=table)
            core.init_baryons_contamination(sim="controlled-1", allsims=str(table))
            raw = core_power_rows(core=core, ell=arguments["ell"],
                                  bins=arguments["bins"])
            record(result=result, prefix="changed", spectra=raw)
            record(result=result, prefix="after", spectra=first.spectra(**arguments))
        else:
            raise ValueError("Unknown core sequence: " + case)
    np.savez(file=destination, **result)


def run_child(case, directory):
    """Keep fatal core errors isolated and return every saved numerical row."""
    if not CORE_MODULE.is_file():
        pytest.skip("Existing DES Y3 extension is required for this integration test.")
    destination = directory / (case + ".npz")
    environment = dict(os.environ)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS"):
        environment[name] = "1"
    process = subprocess.run(
        args=[sys.executable, "-B", str(Path(__file__)), case, str(destination)],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    assert "runtime error:" not in process.stderr
    with np.load(destination, allow_pickle=False) as arrays:
        return {name: arrays[name].copy() for name in arrays.files}


@pytest.fixture(scope="module")
def clean_references(tmp_path_factory):
    """Produce separate clean-process references for both owned models."""
    directory = tmp_path_factory.mktemp("ordinary_core_references")
    first = run_child(case="fresh2", directory=directory)
    second = run_child(case="fresh3", directory=directory)
    assert np.any(first["reference_b"] != 0.0)
    assert np.any(second["reference_b"] != 0.0)
    return first, second


@pytest.mark.parametrize("case", ["adapters", "selectors", "half_state", "baryons"])
def test_ordinary_core_notebook_restoration(case, clean_references, tmp_path):
    """Old adapters recover their full E/B spectra without a Python restart."""
    first, second = clean_references
    reused = run_child(case=case, directory=tmp_path)
    for prefix in ("before", "after", "after_reset"):
        if prefix + "_e" in reused:
            for component in ("e", "b"):
                assert_bits(actual=reused[prefix + "_" + component],
                            expected=first["reference_" + component],
                            context=f"{case}:{prefix}:{component}")
    if case == "adapters":
        for prefix in ("other", "other_again"):
            for component in ("e", "b"):
                assert_bits(actual=reused[prefix + "_" + component],
                            expected=second["reference_" + component],
                            context=f"{prefix}:{component}")
    if "changed_e" in reused:
        # Establish that the raw selector/baryon edit affected the actual
        # spectrum before testing whether the adapter undid that change.
        assert not np.array_equal(reused["changed_e"], first["reference_e"])
    if case == "selectors":
        np.testing.assert_array_equal(reused["changed_b"], 0.0)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Use CASE OUTPUT.npz.")
    child_calculation(case=sys.argv[1], destination=Path(sys.argv[2]))
