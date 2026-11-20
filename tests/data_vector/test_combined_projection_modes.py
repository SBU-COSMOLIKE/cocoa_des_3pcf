"""Check simultaneous signed-baseline and scaled-residual preparation.

Different source triplets have different positive GGG normalization tables.
Preparing them together must retain each table's association with its source
labels. The same physical triplet is also moved between SIMD lanes and a
scalar tail; adding an unrequested output must not change its bits.
"""

from theory.bihalofit.variance import reference_physics_model

import numpy as np
import pytest

from test_grid_transform_reuse import assert_bits
from test_physics_model import compact_grid_settings, physical_inputs
from test_residual_scaling import alignment_parameters, physical_puv_knots


def combined_settings():
    """Keep both optional representations active, including their shared GGG work."""
    settings = compact_grid_settings(reference_mode=0)
    settings.update(
        signed_baseline=1,
        residual_scaling=1,
        shape_stretch=10.106,
        np=7,
        nshape=17,
        lmax=3,
        prepare_multipoles=1,
    )
    return settings


def make_combined_model(compiled, adapter):
    """Use short explicit radial integration, with independent source shapes."""
    inputs = physical_inputs()
    inputs["los_z"] = np.asarray([0.15, 0.5, 1.0, 1.8])
    inputs["source_readers"] = False
    if adapter == "notebook":
        inputs["sources"] = np.asfortranarray(inputs["sources"])
        inputs["p_nonlinear"] = np.asfortranarray(inputs["p_nonlinear"])
        return reference_physics_model(factory=compiled.notebook.PhysicsModel, **inputs)
    return reference_physics_model(factory=compiled.production.PhysicsModel, **inputs)


def prepare_single(model, adapter, bins, parameters, settings):
    """Prepare a one-triplet reference without the multi-triplet entry point."""
    if adapter == "notebook":
        return model.projection_grid_notebook(
            bins=bins, parameters=parameters, settings=settings,
        )
    return model.projection_grid(bins=bins, parameters=parameters, settings=settings)


def prepare_batch(model, adapter, bins, parameters, settings):
    """The notebook path receives a column-major source-label matrix."""
    if adapter == "notebook":
        return model.projection_grids_notebook(
            bins=np.asfortranarray(bins), parameters=parameters, settings=settings,
        )
    return model.projection_grids(bins=bins, parameters=parameters, settings=settings)


@pytest.mark.parametrize("adapter", ["production", "notebook"])
def test_combined_modes_keep_source_scales_and_triplet_lanes(compiled, adapter):
    """Batched 000 and 011 match separate owners; a third leg tests the odd tail.

    The direct physical residual at original PUV knots checks more than
    agreement between two prepared routes. A fourth unused triplet moves the
    third output from a scalar tail into a SIMD pair without changing its
    source labels, physical parameters, or integration nodes.
    """
    compiled.set_omp_threads(nthreads=1)
    model = make_combined_model(compiled=compiled, adapter=adapter)
    settings = combined_settings()
    parameters = alignment_parameters()
    bins = np.asarray(
        [
            [0, 0, 0],
            [0, 1, 1],
            [1, 0, 1],
            [0, 0, 0],
        ],
        dtype=np.int32,
    )
    singles = []
    for row in bins[:3]:
        singles.append(prepare_single(
            model=model,
            adapter=adapter,
            bins=row.copy(),
            parameters=parameters,
            settings=settings,
        ))
    sides = physical_puv_knots(grid=singles[0], settings=settings)
    gravity_parameters = dict(parameters)
    gravity_parameters.update(a1=0.0, a2=0.0, bta=0.0, ct=0.0)
    gravity = []
    for row, grid in zip(bins[:3], singles):
        scale = model.projected(
            sides=sides, bins=row.copy(), parameters=gravity_parameters, component=1,
        )
        gravity.append(scale)
        assert np.all(scale > 0.0)
        direct = model.projected(
            sides=sides, bins=row.copy(), parameters=parameters, component=2,
        )
        prepared = grid.projected(sides=sides, component=2)
        assert np.any(direct != 0.0)
        np.testing.assert_allclose(
            actual=prepared / scale,
            desired=direct / scale,
            rtol=5.0e-10,
            atol=5.0e-10,
        )
    assert not np.array_equal(gravity[0], gravity[1])
    queries = np.asarray(
        [
            [190.0, 280.0, 360.0],
            [430.0, 170.0, 310.0],
            [1600.0, 1900.0, 2700.0],
        ]
    )
    ell = np.asarray([130.0, 280.0, 610.0])
    for count in (2, 3, 4):
        batched = prepare_batch(
            model=model,
            adapter=adapter,
            bins=bins[:count].copy(),
            parameters=parameters,
            settings=settings,
        )
        for row_index in range(min(count, 3)):
            for component in (0, 1, 2):
                assert_bits(
                    actual=batched[row_index].projected(
                        sides=queries, component=component,
                    ),
                    expected=singles[row_index].projected(
                        sides=queries, component=component,
                    ),
                    context=f"{count} triplets changed row {row_index}, B{component}",
                )
                assert_bits(
                    actual=batched[row_index].multipoles(ell=ell, component=component),
                    expected=singles[row_index].multipoles(ell=ell, component=component),
                    context=f"{count} triplets changed row {row_index}, bL{component}",
                )
