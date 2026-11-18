"""Physical-model checks against analytic limits and pinned fastnc values.

The smooth spectra below are controlled test inputs, not a DES cosmology.
They exercise the actual C preparation, power interpolation, matter fit,
all 19 intrinsic-shape terms, and radial projection. The saved reference
comes from the original Python files; it checks implementation agreement.
Separate resolution studies are still required for survey predictions.
"""

from decimal import Decimal, localcontext
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import simpson
from scipy.special import eval_legendre


REFERENCE = Path(__file__).with_name("reference_physics_fastnc.json")


def physical_inputs():
    """Return deterministic h/Mpc, Mpc/h and growth-normalized tables.

    Each source density vanishes at redshift zero. The nonlinear table
    deliberately evolves differently from D squared: this lets a test
    distinguish the source Pnl(k,0) D^2 rule from the optional full table.
    """
    k = np.geomspace(start=1.0e-5, stop=1.0e3, num=96)
    redshift = np.linspace(start=0.0, stop=2.0, num=41)
    linear = 2.0e4 * k / (1.0 + (k / 0.15) ** 2) ** 1.8
    nonlinear0 = linear * (1.0 + (k / 0.35) ** 2) ** 0.22
    nonlinear = nonlinear0[None, :] / (1.0 + redshift[:, None]) ** 2.4
    distance = 3000.0 * redshift / (1.0 + 0.35 * redshift)
    jacobian = (1.0 + 0.35 * redshift) ** 2 / 3000.0
    sources = np.empty(shape=(2, redshift.size), dtype=np.float64)
    sources[0] = redshift**2 * np.exp(-(redshift / 0.6) ** 1.5)
    sources[1] = redshift**2 * np.exp(-(redshift / 0.95) ** 1.5)
    return {
        "k": k,
        "p_linear0": linear,
        "z": redshift,
        "p_nonlinear": nonlinear,
        "chi": distance,
        "dz_dchi": jacobian,
        "growth": 1.0 / (1.0 + redshift),
        "sources": sources,
        "omega_m": 0.3,
        "omega_de": 0.7,
        "w0": -1.0,
        "wa": 0.0,
        "ns": 0.965,
        "sigma8": 0.8,
        "los_z": np.linspace(start=0.005, stop=2.0, num=25),
        "kernel_nz": 101,
        "source_readers": True,
        "table_boost": 8,
    }


@pytest.fixture(scope="module")
def physical_model(compiled):
    """Build one immutable toy model; parameter queries do not mutate it."""
    return compiled.PhysicsModel(**physical_inputs())


def test_pinned_matter_and_all_alignment_components(physical_model):
    """Compare nonsymmetric triangles and nonzero velocity shear to fastnc.

    The 2e-4 matter tolerance includes the upstream nonlinear-radius
    bisection's relative 1e-5 stopping error. IA does not use that root
    and should agree near double precision, including signed B entries.
    """
    record = json.loads(REFERENCE.read_text())
    sides = np.asarray(record["sides"], dtype=np.float64)
    redshift = np.asarray(record["redshifts"], dtype=np.float64)
    for component in range(4):
        actual = physical_model.matter(
            sides=sides, redshifts=redshift, component=component
        )
        expected = np.asarray(record["matter"][component])
        tolerance = 2.0e-10 if component == 1 else 2.0e-4
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=0.0)
    for mode in (0, 1):
        parameters = dict(record["parameters"])
        parameters["power_mode"] = mode
        actual = physical_model.ia(
            sides=sides,
            redshifts=redshift,
            parameters=parameters,
            remove_alignment=False,
        )
        expected = np.asarray(record["alignment"][str(mode)])
        # The first triangle has the intended side ratio 3:4:5. Its ddB
        # shape factor is area*(b^2-a^2)*(c^2-a^2-b^2)/(a^2*b^2*c^2),
        # which vanishes for a right triangle. Binary rounding of the
        # input lengths gives a tiny residual, and either double-precision
        # implementation can change that residual's sign. Check this one
        # entry against its vertex's roundoff scale, not against a relative
        # tolerance on a value whose physical limit is zero. Every other
        # pinned entry retains the original comparison and tolerances.
        np.testing.assert_array_equal(sides[0], [0.03, 0.04, 0.05])
        selected = np.ones(shape=expected.shape, dtype=bool)
        selected[0, 7] = False
        np.testing.assert_allclose(
            actual[selected], expected[selected], rtol=5.0e-11, atol=1.0e-12
        )
        power = physical_model.power(
            k=sides[0].copy(), z=np.zeros(shape=3), mode=mode
        )
        ratio = 1.0 / (1.0 + parameters["zpivot"])
        c1 = -parameters["a1"] * ratio ** parameters["eta1"] * 0.0134 * 0.3
        c2 = parameters["a2"] * ratio ** parameters["eta2"] * 5 * 0.0134 * 0.3
        vertex = power[0] * power[1] * (c2 / 3 + parameters["bta"] * c1)
        vertex_scale = power[0] * power[1]
        vertex_scale *= abs(c2) / 3 + abs(parameters["bta"] * c1)
        # These sides have order-one ratios, so 64 machine epsilons of
        # the unsubtracted vertex cover its short sequence of products,
        # divisions and angular subtractions. This is an absolute bound
        # for this known zero, not an accuracy allowance for other terms.
        roundoff = 64 * np.finfo(np.float64).eps * vertex_scale
        with localcontext() as context:
            context.prec = 80
            first = Decimal.from_float(float(sides[0, 0]))
            second = Decimal.from_float(float(sides[0, 1]))
            closing = Decimal.from_float(float(sides[0, 2]))
            semiperimeter = (first + second + closing) / 2
            area_squared = semiperimeter * (semiperimeter - first)
            area_squared *= (semiperimeter - second) * (semiperimeter - closing)
            angular = area_squared.sqrt() * (second**2 - first**2)
            angular *= closing**2 - first**2 - second**2
            angular /= first**2 * second**2 * closing**2
            zero_reference = float(Decimal.from_float(float(vertex)) * angular)
        assert abs(zero_reference) < roundoff
        np.testing.assert_allclose(
            [actual[0, 7], expected[0, 7]],
            zero_reference,
            rtol=0.0,
            atol=roundoff,
        )


def test_tree_limit_and_sigma8_normalization(physical_model):
    """For equilateral closure F2=2/7, hence B_tree=12 P_L^2/7."""
    wave = np.asarray([0.01, 0.1, 0.8, 7.0], dtype=np.float64)
    redshift = np.asarray([0.0, 0.3, 0.8, 1.7], dtype=np.float64)
    sides = np.column_stack((wave, wave, wave))
    power = physical_model.power(k=wave, z=redshift, mode=0)
    tree = physical_model.matter(sides=sides, redshifts=redshift, component=1)
    np.testing.assert_allclose(tree, 12.0 * power**2 / 7.0, rtol=3.0e-14)

    # This is the source's documented normalization interval and window.
    wave = np.geomspace(start=1.0e-3, stop=1.0e2, num=1000)
    power = physical_model.power(k=wave, z=np.zeros_like(wave), mode=0)
    argument = 8.0 * wave
    window = 3.0 * (np.sin(argument) - argument * np.cos(argument)) / argument**3
    variance = simpson(
        y=wave**3 * power * window**2 / (2.0 * np.pi**2), x=np.log(wave)
    )
    np.testing.assert_allclose(variance, 0.8**2, rtol=3.0e-12)


def test_power_evolution_and_model_ownership(compiled, physical_model):
    """Queries distinguish the two nonlinear prescriptions and own tables."""
    inputs = physical_inputs()
    wave = inputs["k"][[12, 35, 60, 80]].copy()
    redshift = inputs["z"][[0, 8, 20, 35]].copy()
    source = physical_model.power(k=wave, z=redshift, mode=1)
    nonlinear0 = inputs["p_nonlinear"][0, [12, 35, 60, 80]]
    expected_source = nonlinear0 / (1.0 + redshift) ** 2
    evolved = physical_model.power(k=wave, z=redshift, mode=2)
    expected_evolved = nonlinear0 / (1.0 + redshift) ** 2.4
    np.testing.assert_allclose(source, expected_source, rtol=1.0e-12)
    np.testing.assert_allclose(evolved, expected_evolved, rtol=1.0e-12)
    owned = compiled.PhysicsModel(**inputs)
    inputs["p_nonlinear"][:] *= 2.0
    np.testing.assert_array_equal(owned.power(k=wave, z=redshift, mode=2), evolved)


def test_alignment_limits_and_independent_parameters(physical_model):
    """Pure NLA has zero residual; eta2 becomes absent when A2 is zero."""
    sides = np.asarray([[0.12, 0.19, 0.26], [0.7, 0.9, 1.4]])
    redshift = np.asarray([0.3, 0.9])
    nla = {"a1": 0.8, "eta1": 0.4, "power_mode": 0}
    residual = physical_model.ia(
        sides=sides,
        redshifts=redshift,
        parameters=nla,
        remove_alignment=True,
    )
    np.testing.assert_array_equal(residual, np.zeros_like(residual))
    first = dict(nla)
    first["eta2"] = -0.7
    second = dict(nla)
    second["eta2"] = 2.1
    np.testing.assert_array_equal(
        physical_model.ia(sides=sides, redshifts=redshift, parameters=first),
        physical_model.ia(sides=sides, redshifts=redshift, parameters=second),
    )
    first["a2"] = 0.6
    second["a2"] = 0.6
    actual_first = physical_model.ia(
        sides=sides, redshifts=redshift, parameters=first
    )
    actual_second = physical_model.ia(
        sides=sides, redshifts=redshift, parameters=second
    )
    assert not np.allclose(actual_first, actual_second, rtol=1.0e-3)


def test_radial_projection_and_multipole_definition(compiled, physical_model):
    """An explicit radial sum and external Legendre quadrature check C paths."""
    sides = np.asarray([[170.0, 240.0, 330.0], [600.0, 800.0, 1050.0]])
    bins = np.asarray([0, 1, 0], dtype=np.int32)
    radial = physical_model.radial()
    expected = []
    for triangle in sides:
        physical_sides = triangle[None, :] / radial["chi"][:, None]
        matter = physical_model.matter(
            sides=np.ascontiguousarray(physical_sides), redshifts=radial["z"]
        )
        weight = radial["dchi"] * (1.0 + radial["z"]) ** 3 / radial["chi"]
        for bin_index in bins:
            weight *= radial["lens"][bin_index]
        expected.append(np.sum(weight * matter))
    actual = physical_model.projected(sides=sides, bins=bins, parameters={})
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=0.0)
    ell = np.asarray([140.0, 260.0], dtype=np.float64)
    nmu = 8
    lmax = 3
    nodes, weights = np.polynomial.legendre.leggauss(deg=nmu)
    legendre = np.polynomial.legendre.legvander(x=nodes, deg=lmax)
    expected = np.zeros(shape=(lmax + 1, ell.size, ell.size))
    for first in range(ell.size):
        for second in range(ell.size):
            third = np.sqrt(
                (ell[first] - ell[second]) ** 2
                + 2.0 * ell[first] * ell[second] * (1.0 - nodes)
            )
            triangles = np.column_stack(
                (np.full(nmu, ell[first]), np.full(nmu, ell[second]), third)
            )
            signal = physical_model.projected(sides=triangles, bins=bins)
            for order in range(lmax + 1):
                expected[order, first, second] = (
                    (order + 0.5) * np.sum(weights * legendre[:, order] * signal)
                )
    actual = physical_model.multipoles(
        ell=ell, bins=bins, parameters={}, lmax=lmax, nmu=nmu
    )
    np.testing.assert_allclose(actual, expected, rtol=2.0e-12, atol=1.0e-30)
    compiled.set_omp_threads(nthreads=2)
    parallel = physical_model.multipoles(
        ell=ell, bins=bins, parameters={}, lmax=lmax, nmu=nmu
    )
    compiled.set_omp_threads(nthreads=1)
    np.testing.assert_array_equal(actual, parallel)


def test_notebook_layouts_and_baryon_factor(compiled, physical_model):
    """Notebook copies preserve columns, and TNG strength is affine in fb."""
    inputs = physical_inputs()
    inputs["p_nonlinear"] = np.asfortranarray(inputs["p_nonlinear"])
    inputs["sources"] = np.asfortranarray(inputs["sources"])
    notebook = compiled.physics_model_notebook(**inputs)
    sides = np.asarray([[0.1, 0.17, 0.22], [1.0, 1.4, 2.2]])
    redshift = np.asarray([0.2, 0.9])
    direct = physical_model.matter(sides=sides, redshifts=redshift)
    copied = notebook.matter_notebook(
        sides=np.asfortranarray(sides), redshifts=redshift
    )
    np.testing.assert_array_equal(direct, copied)
    full = physical_model.matter(sides=sides, redshifts=redshift, fb=1.0)
    half = physical_model.matter(sides=sides, redshifts=redshift, fb=0.5)
    np.testing.assert_allclose(half, 0.5 * (direct + full), rtol=3.0e-14)


def test_analytic_source_kernel_and_nla_power(compiled):
    """A linear source density gives an exactly known lensing efficiency.

    With chi=L*z and n(z)=2z/zmax^2, the lensing integral is
    g(z)=3 Om/(2 c100^2) (1-z/zmax)^2. Every lens node lies on the
    source grid, where the trapezoidal integral of its linear integrand
    is exact. This checks the physical prefactor and the source integral,
    independently of the projected-bispectrum assembly test above.
    """
    inputs = physical_inputs()
    redshift = inputs["z"]
    maximum = redshift[-1]
    inputs["chi"] = 3000.0 * redshift
    inputs["dz_dchi"] = np.full_like(redshift, 1.0 / 3000.0)
    inputs["sources"] = np.asarray([2.0 * redshift / maximum**2])
    inputs["los_z"] = redshift[1:].copy()
    inputs["kernel_nz"] = redshift.size - 1
    model = compiled.PhysicsModel(**inputs)
    radial = model.radial()
    expected = (
        1.5 * inputs["omega_m"] * (100.0 / 299792.0) ** 2
        * (1.0 - radial["z"] / maximum) ** 2
    )
    np.testing.assert_allclose(radial["lens"][0], expected, rtol=2.0e-12,
                               atol=1.0e-25)
    ell = np.asarray([150.0, 510.0])
    parameters = {"a1": 0.6, "eta1": 0.0}
    c1 = -0.6 * 0.0134 * inputs["omega_m"] * (1.0 + radial["z"])
    # The historical NLA grid starts at the positive LOS lower boundary.
    # Its sampled source normalization therefore omits [0,zmin].
    normalization = 1.0 - (radial["z"][0] / maximum) ** 2
    source_per_distance = 2.0 * radial["z"] / maximum**2 / 3000.0
    kernel = (
        expected * radial["chi"] * (1.0 + radial["z"])
        + c1 * source_per_distance / normalization
    )
    reference = []
    for multipole in ell:
        power = model.power(
            k=multipole / radial["chi"], z=radial["z"], mode=2
        )
        reference.append(np.sum(radial["dchi"] * kernel**2 * power
                                / radial["chi"]**2))
    actual = model.nla_power(
        ell=ell, bins=np.asarray([0, 0], dtype=np.int32), parameters=parameters
    )
    np.testing.assert_allclose(actual, reference, rtol=2.0e-13, atol=0.0)


def test_full_aperture_measure_ordering_and_layout(compiled, physical_model):
    """An external Fourier sum checks the three filters and (2pi)^-3.

    Unequal radii and different source bins prevent a permutation error
    from being hidden by symmetry. This checks the explicitly finite
    Fourier rule; it does not establish its support for a DES forecast.
    """
    ell = np.asarray([90.0, 180.0, 360.0, 720.0])
    dell = np.asarray([70.0, 130.0, 260.0, 480.0])
    phi = (np.arange(8) + 0.5) * (2.0 * np.pi / 8.0)
    dphi = np.full(8, 2.0 * np.pi / 8.0)
    filters = np.asarray(
        [[0.001, 0.0016, 0.0025], [0.002, 0.001, 0.0016],
         [0.0018, 0.0018, 0.0018]]
    )
    bins = np.asarray([0, 1, 0], dtype=np.int32)
    parameters = {"a1": 0.5, "a2": -0.4, "bta": 0.7}
    first, second, angle = np.meshgrid(ell, ell, phi, indexing="ij")
    closure = np.hypot(first + second * np.cos(angle), second * np.sin(angle))
    sides = np.column_stack((first.ravel(), second.ravel(), closure.ravel()))
    projected = physical_model.projected(
        sides=sides, bins=bins, parameters=parameters
    ).reshape(first.shape)
    measure = (
        first * dell[:, None, None] * second * dell[None, :, None]
        * dphi[None, None, :] / (2.0 * np.pi) ** 3
    )
    expected = []
    for radii in filters:
        arguments = (first * radii[0], second * radii[1], closure * radii[2])
        kernel = np.ones_like(first)
        for argument in arguments:
            kernel *= 0.5 * argument**2 * np.exp(-0.5 * argument**2)
        expected.append(np.sum(measure * projected * kernel))
    arguments = {
        "ell": ell,
        "dell": dell,
        "phi": phi,
        "dphi": dphi,
        "filters": filters,
        "bins": bins,
        "parameters": parameters,
    }
    actual = physical_model.aperture_full(**arguments)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-13, atol=0.0)
    arguments["filters"] = np.asfortranarray(filters)
    copied = physical_model.aperture_full_notebook(**arguments)
    np.testing.assert_array_equal(actual, copied)
    arguments["filters"] = filters
    compiled.set_omp_threads(nthreads=2)
    parallel = physical_model.aperture_full(**arguments)
    compiled.set_omp_threads(nthreads=1)
    np.testing.assert_array_equal(actual, parallel)


def test_uniform_production_reader_refinement(compiled, physical_model):
    """Dense linear readers converge to the retained source cubic reader.

    The reference fixture tolerances above remain unchanged. This separate
    test varies the production table resolution and measures its own error
    on wavenumbers and redshifts between the supplied physical nodes.
    """
    inputs = physical_inputs()
    inputs["source_readers"] = False
    inputs["table_boost"] = 2
    coarse = compiled.PhysicsModel(**inputs)
    inputs["table_boost"] = 8
    fine = compiled.PhysicsModel(**inputs)
    wave = np.geomspace(start=0.0023, stop=18.7, num=31)
    redshift = np.linspace(start=0.023, stop=1.91, num=wave.size)
    reference = physical_model.power(k=wave, z=redshift, mode=0)
    coarse_value = coarse.power(k=wave, z=redshift, mode=0)
    fine_value = fine.power(k=wave, z=redshift, mode=0)
    coarse_error = np.max(np.abs(coarse_value / reference - 1.0))
    fine_error = np.max(np.abs(fine_value / reference - 1.0))
    assert fine_error < coarse_error / 5.0
    assert fine_error < 3.0e-4


def compact_grid_settings(reference_mode):
    """Small explicit grids exercise the machinery without a survey run."""
    return {
        "ell_min": 10.0,
        "ell_max": 1.0e5,
        "nr": 8,
        "nu": 9,
        "nv": 8,
        "np": 8,
        "npu_log": 3,
        "npu_linear": 8,
        "nell": 7,
        "npsi": 9,
        "nshape": 18,
        "npsi_log": 3,
        "npsi_linear": 7,
        "nmu_log": 4,
        "nmu_linear": 8,
        "lmax": 4,
        "reference_mode": reference_mode,
        "table_boost": 2,
        "multipole_boost": 2,
    }


def test_reference_grid_nodes_and_source_angular_integral(physical_model):
    """Check original coordinates and piecewise-linear Legendre integration.

    Independent Gauss integration of each linear segment is exact for
    the low Legendre orders used here. Unequal source bins require two
    angular charts, which are tested separately rather than transposed.
    """
    bins = np.asarray([0, 1, 0], dtype=np.int32)
    parameters = {"a1": 0.5, "a2": 0.8, "bta": 0.7}
    settings = compact_grid_settings(reference_mode=1)
    grid = physical_model.projection_grid(
        bins=bins, parameters=parameters, settings=settings
    )
    axes = grid.axes()
    radius = np.exp(axes["log_r"][3])
    ratio = np.exp(axes["log_u"][5])
    difference = axes["v"][3]
    sides = np.asarray([[radius * (1.0 + ratio * difference),
                         radius, radius * ratio]])
    expected = physical_model.projected(
        sides=sides, bins=bins, parameters=parameters, component=1
    )
    np.testing.assert_allclose(
        grid.projected(sides=sides, component=1), expected, rtol=2.0e-13
    )
    magnitude = np.exp(axes["log_ell"][3])
    psi = np.exp(axes["log_psi"][-3])
    ell = np.asarray([magnitude * np.cos(psi), magnitude * np.sin(psi)])
    actual = grid.multipoles(ell=ell)
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(deg=5)
    mu = axes["mu"]
    for first, second in ((0, 1), (1, 0)):
        third = np.sqrt((ell[first] - ell[second]) ** 2
                        + 2.0 * ell[first] * ell[second] * (1.0 - mu))
        triangles = np.column_stack(
            (np.full(mu.size, ell[first]), np.full(mu.size, ell[second]), third)
        )
        sampled = grid.projected(sides=triangles)
        expected = np.zeros(settings["lmax"] + 1)
        for interval in range(mu.size - 1):
            lower = mu[interval]
            upper = mu[interval + 1]
            coordinate = 0.5 * (lower + upper + (upper - lower) * gauss_nodes)
            value = sampled[interval] + (coordinate - lower) / (upper - lower) * (
                sampled[interval + 1] - sampled[interval]
            )
            for order in range(settings["lmax"] + 1):
                expected[order] += (order + 0.5) * 0.5 * (upper - lower) * np.sum(
                    gauss_weights * value * eval_legendre(order, coordinate)
                )
        np.testing.assert_allclose(actual[:, first, second], expected,
                                   rtol=1.0e-10, atol=1.0e-28)
    copied = grid.multipoles_notebook(ell=ell)
    np.testing.assert_array_equal(copied, actual)


def test_uniform_grid_owned_axes_and_thread_determinism(compiled, physical_model):
    """Production axes are uniform in their declared interpolation charts."""
    settings = compact_grid_settings(reference_mode=0)
    bins = np.asarray([0, 1, 0], dtype=np.int32)
    grid = physical_model.projection_grid(
        bins=bins, parameters={"a1": 0.4, "a2": 0.7}, settings=settings
    )
    axes = grid.axes()
    for name in ("log_r", "log_u", "v", "log_p", "log_ell", "log_psi"):
        spacing = np.diff(axes[name])
        np.testing.assert_allclose(spacing, spacing[0], rtol=2.0e-12,
                                   atol=1.0e-14)
    ell = np.asarray([130.0, 280.0, 610.0])
    expected = grid.multipoles(ell=ell)
    axes["log_r"][:] = 0.0
    compiled.set_omp_threads(nthreads=2)
    actual = grid.multipoles(ell=ell)
    compiled.set_omp_threads(nthreads=1)
    np.testing.assert_array_equal(actual, expected)


def test_batched_ordered_grids_and_aperture_sum(compiled, physical_model):
    """Shared physical vertices retain every ordered source association.

    Independent single-triplet preparation must give exactly the same
    arrays. The aperture check below integrates the interpolated B with a
    separate NumPy expression; it checks Fourier weights and leg order,
    while the next test measures interpolation error against direct B.
    """
    bins = np.asarray([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.int32)
    settings = compact_grid_settings(reference_mode=0)
    settings["prepare_multipoles"] = 0
    parameters = {"a1": 0.5, "a2": 0.8, "bta": 0.6, "ct": 0.2}
    grids = physical_model.projection_grids(
        bins=bins, parameters=parameters, settings=settings
    )
    triangles = np.asarray([[190.0, 280.0, 360.0], [430.0, 170.0, 310.0]])
    for triplet, grid in zip(bins, grids):
        separate = physical_model.projection_grid(
            bins=triplet, parameters=parameters, settings=settings
        )
        np.testing.assert_array_equal(
            grid.projected(sides=triangles), separate.projected(sides=triangles)
        )
    copied = physical_model.projection_grids_notebook(
        bins=np.asfortranarray(bins), parameters=parameters, settings=settings
    )
    for original, notebook in zip(grids, copied):
        np.testing.assert_array_equal(
            original.projected(sides=triangles),
            notebook.projected(sides=triangles),
        )
    ell = np.asarray([130.0, 260.0, 510.0])
    dell = np.asarray([90.0, 180.0, 280.0])
    phi = (np.arange(8) + 0.5) * (2.0 * np.pi / 8.0)
    dphi = np.full(shape=phi.size, fill_value=2.0 * np.pi / phi.size)
    filters = np.asarray([[0.003, 0.005, 0.004], [0.004, 0.002, 0.006]])
    first, second, angle = np.meshgrid(ell, ell, phi, indexing="ij")
    closure = np.hypot(first + second * np.cos(angle), second * np.sin(angle))
    sides = np.column_stack((first.ravel(), second.ravel(), closure.ravel()))
    values = grids[0].projected(sides=sides).reshape(first.shape)
    measure = (first * dell[:, None, None] * second * dell[None, :, None]
               * dphi[None, None, :] / (2.0 * np.pi) ** 3)
    expected = []
    for radii in filters:
        integrand = values * measure
        for side, radius in zip((first, second, closure), radii):
            argument = side * radius
            integrand = integrand * (0.5 * argument**2
                                     * np.exp(-0.5 * argument**2))
        expected.append(np.sum(integrand))
    arguments = {"ell": ell, "dell": dell, "phi": phi, "dphi": dphi,
                 "filters": filters}
    actual = grids[0].aperture_full(**arguments)
    np.testing.assert_allclose(actual, expected, rtol=3.0e-13)
    arguments["filters"] = np.asfortranarray(filters)
    np.testing.assert_array_equal(
        actual, grids[0].aperture_full_notebook(**arguments)
    )
    compiled.set_omp_threads(nthreads=2)
    parallel = physical_model.projection_grids(
        bins=bins, parameters=parameters, settings=settings
    )
    compiled.set_omp_threads(nthreads=1)
    for serial, threaded in zip(grids, parallel):
        np.testing.assert_array_equal(
            serial.projected(sides=triangles), threaded.projected(sides=triangles)
        )
    nla = physical_model.projection_grid(
        bins=bins[0], parameters={"a1": 0.5}, settings=settings
    )
    np.testing.assert_array_equal(nla.projected(sides=triangles, component=2), 0.0)


def test_bihalofit_redshift_preparation_refines(compiled):
    """Refine the expensive uniform-z fit anchors separately from P readers.

    Identical dense input-power settings isolate the R_sigma, effective
    slope and fit-coefficient preparation. A denser fixed anchor grid is
    the resolution reference; the pinned-source tests remain independent.
    """
    inputs = physical_inputs()
    inputs.update(source_readers=False, table_boost=8)
    sides = np.asarray([[0.18, 0.27, 0.33], [0.9, 1.3, 1.7],
                        [2.1, 3.2, 4.1], [0.04, 0.07, 0.09]])
    redshift = np.asarray([0.073, 0.417, 0.923, 1.713])
    predictions = []
    for anchors in (9, 33, 129):
        inputs["fit_nz"] = anchors
        model = compiled.PhysicsModel(**inputs)
        predictions.append(model.matter(sides=sides, redshifts=redshift))
    coarse_error = np.max(np.abs(predictions[0] / predictions[2] - 1.0))
    fine_error = np.max(np.abs(predictions[1] / predictions[2] - 1.0))
    assert fine_error < coarse_error / 3.0
    assert fine_error < 3.0e-4


def test_projected_grid_refines_against_direct_ordered_physics(physical_model):
    """Check off-node triangles, including squeezed and flattened cases.

    Signed residuals can pass through zero, where relative errors alone
    have no meaning. Their absolute discrepancy is scaled by the direct
    total/baseline amplitude at the same triangle. Refinement varies the
    physical coarse sampling, not merely the number of interpolated nodes.
    """
    triangles = np.asarray([
        [237.0, 410.0, 530.0], [1000.0, 1003.0, 4.0],
        [160.0, 303.0, 462.9], [97.0, 110.0, 180.0],
        [800.0, 130.0, 700.0], [50.0, 62.0, 88.0],
    ])
    triangles = np.concatenate((triangles, triangles[:, [1, 0, 2]]), axis=0)
    bins = np.asarray([0, 1, 0], dtype=np.int32)
    parameters = {"a1": 0.5, "a2": 0.8, "bta": 0.7, "ct": 0.3}
    reference = np.asarray([
        physical_model.projected(sides=triangles, bins=bins,
                                 parameters=parameters, component=component)
        for component in (1, 2)
    ])
    scale = np.maximum(np.abs(reference[0]), np.abs(reference.sum(axis=0)))
    errors = []
    for count, nshape in ((17, 33), (33, 65)):
        settings = compact_grid_settings(reference_mode=0)
        settings.update(ell_min=1.0, ell_max=3000.0, nr=count, nu=count,
                        nv=count, np=count, nshape=nshape, table_boost=4,
                        prepare_multipoles=0)
        grid = physical_model.projection_grid(
            bins=bins, parameters=parameters, settings=settings
        )
        actual = np.asarray([grid.projected(sides=triangles, component=value)
                             for value in (1, 2)])
        errors.append(np.max(np.abs(actual - reference) / scale, axis=1))
    assert np.all(errors[1] < errors[0] / 2.0), errors
    assert np.all(errors[1] < 0.01), errors


def test_nla_power_simd_tiles_and_extrapolation(compiled):
    """Compare the prepared ell readers to a separate radial sum.

    There are 259 outputs: one full 256-element preparation tile and an
    odd final tile. The broad ell range also reaches both extrapolation
    regions of the supplied P(k) table. Thus SIMD pairs, the scalar last
    output, and endpoint power laws all enter the comparison.
    """
    inputs = physical_inputs()
    inputs["source_readers"] = False
    model = compiled.PhysicsModel(**inputs)
    radial = model.radial()
    ell = np.geomspace(start=1.0e-5, stop=2.0e7, num=259)
    bins = np.asarray([0, 1], dtype=np.int32)
    # With zero IA, W_i=chi*(1+z)*g_i. The supplied radial g arrays
    # therefore give an independent expression for each integration node.
    weight = (radial["dchi"] * (1.0 + radial["z"])**2
              * radial["lens"][0] * radial["lens"][1])
    reference = np.empty_like(ell)
    for index, multipole in enumerate(ell):
        power = model.power(k=multipole / radial["chi"], z=radial["z"], mode=2)
        reference[index] = np.sum(weight * power)
    actual = model.nla_power(ell=ell, bins=bins)
    np.testing.assert_allclose(actual, reference, rtol=3.0e-13, atol=0.0)
    sliced = np.empty(shape=2 * ell.size, dtype=np.float64)
    sliced[::2] = ell
    copied = model.nla_power_notebook(ell=sliced[::2], bins=bins)
    np.testing.assert_array_equal(actual, copied)
    compiled.set_omp_threads(nthreads=2)
    parallel = model.nla_power(ell=ell, bins=bins)
    compiled.set_omp_threads(nthreads=1)
    np.testing.assert_array_equal(actual, parallel)


def test_shared_triangle_geometry_all_ia_components(compiled, physical_model):
    """Common side scaling preserves orientation, but changes all powers.

    Direct point queries reconstruct the triangle independently, whereas
    ia_same_shape prepares its geometry once. Compare every E/B component
    with nonzero velocity shear over six decades in scale and varying z;
    repeat for all power prescriptions and the alignment subtraction.
    """
    scale = np.geomspace(start=1.0e-3, stop=1.0e3, num=13)
    sides = scale[:, None] * np.asarray([0.17, 0.31, 0.39])[None, :]
    redshift = np.linspace(start=0.07, stop=1.93, num=scale.size)
    parameters = {
        "a1": 0.7,
        "a2": 1.2,
        "eta1": 0.3,
        "eta2": -0.4,
        "bta": 0.8,
        "ct": 0.4,
    }
    for mode in (0, 1, 2):
        parameters["power_mode"] = mode
        for remove_alignment in (False, True):
            arguments = {
                "sides": sides,
                "redshifts": redshift,
                "parameters": parameters,
                "remove_alignment": remove_alignment,
            }
            reference = physical_model.ia(**arguments)
            actual = physical_model.ia_same_shape(**arguments)
            scale_by_point = np.max(np.abs(reference), axis=1, keepdims=True)
            error = np.max(np.abs(actual - reference) / scale_by_point)
            assert error < 2.0e-13
    arguments["sides"] = np.asfortranarray(sides)
    copied = physical_model.ia_same_shape_notebook(**arguments)
    np.testing.assert_array_equal(actual, copied)
    arguments["sides"] = sides
    compiled.set_omp_threads(nthreads=2)
    parallel = physical_model.ia_same_shape(**arguments)
    compiled.set_omp_threads(nthreads=1)
    np.testing.assert_array_equal(actual, parallel)


def test_signed_residual_zero_crossing_has_absolute_error_control(physical_model):
    """Check a cancellation without dividing by its vanishing residual.

    At tree level only one quadratic shape vertex enters each term, so
    fixing A1 makes the residual linear in A2. Two direct evaluations
    determine an A2 where the chosen triangle has zero residual. Nearby
    triangles then test both signs using the nonzero NLA baseline as scale.
    """
    center = np.asarray([[237.0, 410.0, 530.0]])
    bins = np.asarray([0, 1, 0], dtype=np.int32)
    parameters = {"a1": 0.5, "a2": 0.0, "bta": 0.7, "ct": 0.3}
    at_zero = physical_model.projected(
        sides=center, bins=bins, parameters=parameters, component=2
    )[0]
    parameters["a2"] = 1.0
    at_one = physical_model.projected(
        sides=center, bins=bins, parameters=parameters, component=2
    )[0]
    assert at_one != at_zero
    parameters["a2"] = -at_zero / (at_one - at_zero)
    sides = np.repeat(center, repeats=3, axis=0)
    sides[:, 0] *= np.asarray([0.99, 1.0, 1.01])
    baseline = physical_model.projected(
        sides=sides, bins=bins, parameters=parameters, component=1
    )
    reference = physical_model.projected(
        sides=sides, bins=bins, parameters=parameters, component=2
    )
    assert reference[0] * reference[2] < 0.0
    assert abs(reference[1] / baseline[1]) < 1.0e-12
    settings = compact_grid_settings(reference_mode=0)
    settings.update(
        ell_min=1.0,
        ell_max=3000.0,
        nr=33,
        nu=33,
        nv=33,
        np=33,
        nshape=65,
        table_boost=4,
        prepare_multipoles=0,
    )
    grid = physical_model.projection_grid(
        bins=bins, parameters=parameters, settings=settings
    )
    actual = grid.projected(sides=sides, component=2)
    assert np.max(np.abs((actual - reference) / baseline)) < 5.0e-5


def test_production_multipole_interpolation_refines_off_nodes(physical_model):
    """Refine only the b_L axes while holding physical projected B fixed.

    Five-point Gauss integration on each mu interval exactly integrates
    a linear B times every Legendre polynomial used here. This independent
    calculation isolates the dense b_L interpolation and component/chart
    packing, including both orders of unequal source-associated sides.
    """
    bins = np.asarray([0, 1, 0], dtype=np.int32)
    parameters = {"a1": 0.4, "a2": 0.7, "bta": 0.6, "ct": 0.2}
    ell = np.asarray([137.0, 291.0, 619.0])
    pairs = ((0, 1), (1, 0), (1, 2), (2, 1))
    gauss_nodes, gauss_weights = np.polynomial.legendre.leggauss(deg=5)
    discrepancies = []
    first_reference = None
    for scale_nodes, angle_nodes in ((17, 33), (129, 257)):
        settings = compact_grid_settings(reference_mode=0)
        settings.update(nr=17, nu=17, nv=17, np=17, nshape=33)
        settings["nell"] = scale_nodes
        settings["npsi"] = angle_nodes
        grid = physical_model.projection_grid(
            bins=bins, parameters=parameters, settings=settings
        )
        mu = grid.axes()["mu"]
        norder = settings["lmax"] + 1
        reference = np.zeros(shape=(2, norder, len(pairs)))
        actual = np.zeros_like(reference)
        for component_index, component in enumerate((1, 2)):
            interpolated = grid.multipoles(ell=ell, component=component)
            for pair_index, (first, second) in enumerate(pairs):
                actual[component_index, :, pair_index] = (
                    interpolated[:, first, second]
                )
                closure = np.sqrt(
                    (ell[first] - ell[second])**2
                    + 2.0 * ell[first] * ell[second] * (1.0 - mu)
                )
                triangles = np.column_stack((
                    np.full(shape=mu.size, fill_value=ell[first]),
                    np.full(shape=mu.size, fill_value=ell[second]),
                    closure,
                ))
                sampled = grid.projected(sides=triangles, component=component)
                for interval in range(mu.size - 1):
                    width = mu[interval + 1] - mu[interval]
                    coordinate = (mu[interval] + 0.5 * width
                                  * (1.0 + gauss_nodes))
                    value = (sampled[interval] + 0.5 * (1.0 + gauss_nodes)
                             * (sampled[interval + 1] - sampled[interval]))
                    for order in range(norder):
                        reference[component_index, order, pair_index] += (
                            (order + 0.5) * 0.5 * width
                            * np.sum(gauss_weights * value
                                     * eval_legendre(order, coordinate))
                        )
        if first_reference is None:
            first_reference = reference.copy()
        else:
            np.testing.assert_array_equal(reference, first_reference)
        # Each component uses its own largest multipole as amplitude scale;
        # a small signed residual cannot hide behind the larger baseline.
        amplitude = np.max(np.abs(reference), axis=1, keepdims=True)
        error = np.max(np.abs(actual - reference) / amplitude, axis=(1, 2))
        discrepancies.append(error)
    assert np.all(discrepancies[1] < discrepancies[0] / 3.0), discrepancies
    assert np.all(discrepancies[1] < 0.005), discrepancies
