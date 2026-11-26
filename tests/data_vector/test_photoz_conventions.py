"""Check the native n(z) reading of the cosmic-shear part of the joint vector.

The DES Y3 source file gives cell midpoints. The likelihood reconstructs
the shifted source curves with PCHIP before it sends the resulting dense
grid to CosmoLike. This test changes the native reader of that prepared
grid. Map3 consumes the reconstructed curves directly, so changing the
native reader must leave its rows unchanged.
"""

from contextlib import chdir
from ctypes import CDLL
import os
from pathlib import Path

from cobaya.model import get_model
from cobaya.yaml import yaml_load_file
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from threadpoolctl import threadpool_info, threadpool_limits

import cosmolike_des_3pcf_interface as ci


PROJECT = Path(__file__).resolve().parents[2]
COCOA = PROJECT.parents[1]


def test_photoz_conventions():
    """Compare ξ and joint prediction distances under three native alternatives.

    At the fixed point in Evaluate3, change only the reader used by the
    native ξ projection. Restore the linear, sample-point default after
    each alternative while keeping the same likelihood alive.

    Side effects:
      Prints the prediction differences on the ξ covariance and on
      the complete ξ and Map3 covariance, including their cross terms.
    """
    info = yaml_load_file(str(PROJECT / "EXAMPLE_EVALUATE3.yaml"))
    point = info.pop("sampler")["evaluate"]["override"]
    info.pop("output")
    info["timing"] = False
    like_name = "des_3pcf.joint_shear"
    options = info["likelihood"][like_name]
    options["omp_threads"] = int(os.environ.get("COCOA_TEST_OMP_THREADS", "1"))
    alternatives = (
        ("cubic spline", 0, 1),
        ("Steffen", 2, 1),
        ("left edges", 1, 0),
    )
    with (
        chdir(COCOA),
        threadpool_limits(limits=1, user_api="blas"),
        get_model(info_or_yaml_or_file=info) as model,
    ):
        likelihood = model.likelihood[like_name]
        assert likelihood.interpolation_type == 1
        assert likelihood.source_interpolation == "pchip_midpoint"
        assert likelihood.source_z.size == 9601
        xi = likelihood.xi_indices
        map3 = likelihood.map_indices
        assert xi.size == 400 and map3.size == 80
        covariance = ci.get_joint_covariance()
        assert covariance.shape == (480, 480)
        joint_factor = cho_factor(covariance)
        xi_factor = cho_factor(covariance[np.ix_(xi, xi)])

        # Keep the spectra cache at the fixed cosmology. Only the
        # likelihood must rerun when its native reader changes.
        cache_size = likelihood._states.maxlen
        likelihood.set_cache_size(0)

        def prediction(interpolation, zmid):
            """Read the joint vector after applying one native convention.

            Arguments:
              interpolation = native spline (0), linear (1), or Steffen (2).
              zmid = read the prepared nodes as left edges (0) or samples (1).

            Returns:
              A copy of the 480-row prediction in the measurement order.
            """
            ci.init_photoz_conventions(
                interpolation_type=interpolation,
                zmid_convention=zmid,
            )
            result = model.logposterior(params_values=point)
            assert np.isfinite(result.logpost)
            return np.array(likelihood.last_datavector, dtype=np.float64, copy=True)

        try:
            reference = prediction(interpolation=1, zmid=1)
            np.testing.assert_array_equal(
                ci.select_joint_data_vector(prediction=reference), reference,
            )
            sources = likelihood.sources.copy()
            native_threads = CDLL(ci.__file__).omp_get_max_threads()
            blas = [
                pool["num_threads"] for pool in threadpool_info()
                if pool["user_api"] == "blas"
            ]
            assert native_threads == options["omp_threads"]
            assert blas and set(blas) == {1}
            print(f"Native OpenMP={native_threads}, BLAS={blas}.", flush=True)
            for label, interpolation, zmid in alternatives:
                vector = prediction(interpolation=interpolation, zmid=zmid)
                np.testing.assert_array_equal(likelihood.sources, sources)
                np.testing.assert_array_equal(
                    vector[map3].view(np.uint64), reference[map3].view(np.uint64),
                )
                delta = vector - reference
                xi_delta = delta[xi]
                distance = delta @ cho_solve(joint_factor, delta)
                xi_distance = xi_delta @ cho_solve(xi_factor, xi_delta)
                assert np.isfinite(distance) and distance > 0.0, label
                assert np.isfinite(xi_distance) and xi_distance > 0.0, label
                print(f"{label} Δχ² for ξ={xi_distance:.2g}, joint={distance:.2g}.")
                restored = prediction(interpolation=1, zmid=1)
                np.testing.assert_array_equal(
                    restored.view(np.uint64), reference.view(np.uint64),
                )
        finally:
            ci.init_photoz_conventions(interpolation_type=1, zmid_convention=1)
            likelihood.set_cache_size(cache_size)
