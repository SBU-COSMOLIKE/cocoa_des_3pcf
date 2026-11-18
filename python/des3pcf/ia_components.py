"""Separate intrinsic shapes by physical cause and by galaxy placement.

G denotes gravitational shear and I denotes an intrinsic galaxy shape.
A GGI term puts the intrinsic shape on the third ordered source leg.
Each placement can contain linear alignment, density weighting, tidal
torque and velocity shear. These are different questions: summing over
placements answers which operator matters; summing over operators answers
which intrinsic-galaxy combinations matter.

The C projector selects vertices and radial weights before integration.
It does not subtract large complete predictions to extract a small term.
The same linear shear transform and finite aperture weights then act on
every term. This diagnostic costs multiple projections and is optional;
ordinary likelihood evaluations retain their single total calculation.
"""

import numpy as np


OPERATORS = (
    "gravity",
    "linear_alignment",
    "density_weighting",
    "tidal_torque",
    "velocity_shear",
)
PLACEMENTS = ("GGG", "GGI", "GIG", "IGG", "GII", "IIG", "IGI", "III")
LEG_MASKS = (0, 4, 2, 1, 6, 3, 5, 7)


def map3_components(engine, model, ia, calibration=None):
    """Return the signed operator/placement decomposition of an aperture vector.

    Arguments:
      engine = Map3Engine with its fixed finite triangle/filter definition.
        Prepared projection requires signed_baseline=1: all signed components
        divide by the same positive GGG before interpolation.
      model = immutable C PhysicsModel with the cosmology and source tables.
      ia = complete physical IA dictionary; diagnostic selectors must be absent
        or set to their defaults (operator_component=0, intrinsic_legs=-1).
      calibration = optional multiplicative shear offsets [source bin].
    Returns:
      dictionary with operators, placements, values[5,8,nrow], total[nrow],
      and closure[nrow]=sum(values)-total, all dimensionless. Zero entries
      are physical absences: gravity has only GGG; IA has no GGG.
    Raises:
      ValueError for an already selected component or unsigned prepared grids.

    The returned total comes from the ordinary evaluator. Its agreement with
    the sum tests the entire projection, interpolation, transform and aperture
    calculation. It does not establish agreement with observed survey data.
    """
    if ia.get("operator_component", 0) != 0 or ia.get("intrinsic_legs", -1) != -1:
        raise ValueError("Pass full IA parameters without a component selection.")
    if (engine.projection_mode == "prepared"
            and engine.projection_settings.get("signed_baseline", 0) != 1):
        raise ValueError(
            "Component grids need signed_baseline=1; construct an engine "
            "with that projection setting so signed terms share one scale."
        )
    values = np.zeros(shape=(5, 8, engine.rows.values.size), dtype=np.float64)
    total = engine.evaluate(model=model, ia=ia, calibration=calibration)
    for operator in range(5):
        for placement in range(8):
            if (operator == 0 and placement != 0
                    or operator != 0 and placement == 0):
                continue
            selected = dict(ia)
            # The native NLA operator contains both gravity and all linear
            # alignment powers. The field selector separates those here.
            selected["operator_component"] = max(1, operator)
            selected["intrinsic_legs"] = LEG_MASKS[placement]
            values[operator, placement] = engine.evaluate(
                model=model, ia=selected, calibration=calibration,
            )
    return {
        "operators": OPERATORS,
        "placements": PLACEMENTS,
        "values": values,
        "total": total,
        "closure": np.sum(values, axis=(0, 1)) - total,
    }
