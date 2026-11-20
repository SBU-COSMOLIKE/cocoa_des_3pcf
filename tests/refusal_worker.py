"""Run one deliberately invalid call in a separate Python process.

The C++ adapters follow the core's stop-on-invalid-input convention. A
separate process allows a test to inspect that diagnostic without ending
the other checks. No method is replaced and no production state is shared
between the child process and its caller.
"""

from theory.bihalofit.variance import reference_physics_model

from pathlib import Path
import sys

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "interface"))


def run_case(case, adapter_name):
    """Make one invalid call, which must terminate before returning data.

    Arguments:
        case = named malformed geometry or covariance input.
        adapter_name = production or notebook adapter.

    Returns:
        Nothing on the expected failure path. Returning indicates a defect.
    """
    import cosmolike_des_3pcf_interface as compiled

    compiled.set_omp_threads(nthreads=1)
    if adapter_name == "production":
        adapter = compiled.production
    else:
        adapter = compiled.notebook

    if case in ("collinear", "underflow", "radius_overflow", "negative_width"):
        phi = np.array([0.8])
        filters = np.array([[1.0, 1.0, 1.0]])
        width = 0.1
        if case == "collinear":
            phi[0] = 0.0
        elif case == "underflow":
            phi[0] = 1.0e-180
        elif case == "radius_overflow":
            filters[0, 1] = 1.0e200
        else:
            width = -0.1
        adapter.aperture_weights(
            d2=np.array([2.0]), d3=np.array([1.0]), phi=phi,
            dlnd=width, dphi=0.1, filters=filters
        )
    elif case == "negative_gram_weight":
        adapter.connected_gram(
            responses=np.ones((2, 3)), weights=np.array([1.0, -0.1, 1.0])
        )
    elif case == "nonfinite":
        adapter.connected_gram(
            responses=np.array([[1.0, np.nan]]), weights=np.ones(2)
        )
    elif case == "result_overflow":
        adapter.connected_gram(responses=np.full((2, 3), 1.0e200), weights=np.ones(3))
    elif case == "field_index":
        adapter.gaussian_aperture_covariance(
            triplets=np.array([[0, 0, 1]], dtype=np.int32),
            pair=np.ones((1, 1, 2)), pair_zero=np.ones((1, 1)),
            radial_weights=np.ones(2)
        )
    elif case == "triangle_shape":
        adapter.aperture_apply(
            wr=np.ones((1, 4, 3)), wi=np.zeros((1, 4, 3)),
            gr=np.ones((1, 4, 2)), gi=np.zeros((1, 4, 2))
        )
    elif case == "asymmetric_pair":
        pair = np.ones((2, 2, 3))
        pair[0, 1] = 0.7
        adapter.gaussian_aperture_covariance(
            triplets=np.array([[0, 1, 0]], dtype=np.int32),
            pair=pair, pair_zero=np.ones((2, 2)), radial_weights=np.ones(3)
        )
    elif case == "unaligned":
        storage = bytearray(25)
        weights = np.ndarray(shape=(3,), dtype=np.float64, buffer=storage, offset=1)
        weights[:] = 1.0
        adapter.connected_gram(responses=np.ones((2, 3)), weights=weights)
    elif case == "pixel_covariance":
        sys.path.insert(0, str(PROJECT / "tests" / "data_vector"))
        sys.path.insert(0, str(PROJECT / "tests" / "covariance"))
        from test_physics_model import physical_inputs
        from test_physical_covariance import covariance_arguments

        model = reference_physics_model(factory=adapter.PhysicsModel, **physical_inputs(), pixel_window=np.ones(10))
        adapter.physical_covariance(model=model, inputs=covariance_arguments())
    else:
        raise ValueError(f"Unknown refusal case: {case}")
    raise RuntimeError("Invalid input returned a numerical result.")


if __name__ == "__main__":
    run_case(case=sys.argv[1], adapter_name=sys.argv[2])
