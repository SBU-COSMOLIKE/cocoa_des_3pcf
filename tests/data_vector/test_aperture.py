"""Check the finite-grid aperture map before connecting a cosmology model.

A natural component is a complex three-shear correlation with a specified
triangle-dependent rotation. Aperture skewness is a real linear sum of four
such components. These checks compare the numerical weights with a pinned
external kernel, then test complex multiplication, layout, linearity and
reuse. They do not establish that the supplied natural components model a
particular galaxy catalogue.
"""

import json
from pathlib import Path

import numpy as np
import pytest


REFERENCE = Path(__file__).with_name("reference_aperture_weights.json")
# Different implementations round complex products in different orders.
WEIGHT_RTOL = 2.0e-12
WEIGHT_ATOL = 2.0e-15


def reference_inputs():
    """Read the saved external calculation and its complete input geometry.

    Returns:
        arguments = named arrays, in radians, and dimensionless bin widths.
        expected = two arrays [filter, component, triangle], real then imaginary.
    """
    record = json.loads(REFERENCE.read_text())
    arguments = {}
    for name in ("d2", "d3", "phi", "filters"):
        arguments[name] = np.asarray(record[name], dtype=np.float64)
    arguments["dlnd"] = record["dlnd"]
    arguments["dphi"] = record["dphi"]
    expected = (
        np.asarray(record["weights_real"], dtype=np.float64),
        np.asarray(record["weights_imag"], dtype=np.float64),
    )
    return arguments, expected


def test_weights_against_external_reference(compiled):
    """Compare equal and unequal filters with the pinned Python equations."""
    arguments, expected = reference_inputs()
    for adapter in (compiled.production, compiled.notebook):
        actual = adapter.aperture_weights(**arguments)
        for part in range(2):
            np.testing.assert_allclose(
                actual[part], expected[part], rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL
            )


@pytest.mark.parametrize("nfilter", [1, 2, 3, 4])
def test_complex_contraction_and_owned_outputs(compiled, nfilter):
    """Check all four components and both complex parts with five samples.

    Real(w gamma) = Re(w) Re(gamma) - Im(w) Im(gamma). A complex
    conjugation would reverse the second sign and fail this comparison.
    Five samples and three filters also exercise vector-loop remainders.
    """
    arguments, expected = reference_inputs()
    weights_real, weights_imag = expected
    # Repeating one valid operator creates a fourth output without changing
    # its physics. Prefixes of this bank exercise both vector-loop tails.
    weights_real = np.concatenate((weights_real, weights_real[:1]), axis=0)
    weights_imag = np.concatenate((weights_imag, weights_imag[:1]), axis=0)
    weights_real = weights_real[:nfilter]
    weights_imag = weights_imag[:nfilter]
    ntriangle = weights_real.shape[2]
    rng = np.random.default_rng(seed=4703)
    gamma_real = rng.normal(size=(5, 4, ntriangle))
    gamma_imag = rng.normal(size=(5, 4, ntriangle))
    complex_weights = weights_real + 1j * weights_imag
    complex_gamma = gamma_real + 1j * gamma_imag
    reference = np.real(
        complex_gamma.reshape(5, -1) @ complex_weights.reshape(nfilter, -1).T
    )

    for adapter in (compiled.production, compiled.notebook):
        actual = adapter.aperture_apply(
            wr=weights_real, wi=weights_imag, gr=gamma_real, gi=gamma_imag
        )
        np.testing.assert_allclose(actual, reference, rtol=2.0e-12, atol=1.0e-14)
        saved = actual.copy()
        adapter.aperture_weights(**arguments)
        np.testing.assert_array_equal(actual, saved)
        assert not np.shares_memory(actual, gamma_real)


def test_geometry_changes_with_same_dimensions(compiled):
    """Changing coordinate values must produce a new measurement operator."""
    arguments, _ = reference_inputs()
    original = compiled.production.aperture_weights(**arguments)
    changed = dict(arguments)
    changed["phi"] = arguments["phi"] + 0.03
    shifted = compiled.production.aperture_weights(**changed)
    assert not np.array_equal(original[0], shifted[0])
    restored = compiled.production.aperture_weights(**arguments)
    np.testing.assert_array_equal(original[0], restored[0])
    np.testing.assert_array_equal(original[1], restored[1])


def test_common_angular_rescaling(compiled):
    """The finite operator depends on side-to-radius ratios, not angle units.

    The public interface specifies radians. Scaling every side and every
    aperture by the same factor leaves these dimensionless weights fixed.
    The opening angle and logarithmic widths are unchanged.
    """
    arguments, _ = reference_inputs()
    original = compiled.production.aperture_weights(**arguments)
    scaled = dict(arguments)
    for name in ("d2", "d3", "filters"):
        scaled[name] = arguments[name] * 2.0
    actual = compiled.production.aperture_weights(**scaled)
    np.testing.assert_allclose(actual, original, rtol=2.0e-12, atol=2.0e-15)


def test_production_rejects_implicit_copy(compiled):
    """The production adapter must not silently convert float32 inputs."""
    arguments, _ = reference_inputs()
    arguments["d2"] = arguments["d2"].astype(np.float32)
    with pytest.raises(TypeError):
        compiled.production.aperture_weights(**arguments)


def test_notebook_accepts_noncontiguous_inputs(compiled):
    """The notebook adapter copies strided inputs into its owned arrays."""
    arguments, expected = reference_inputs()
    padded = np.zeros(2 * arguments["d2"].size)
    padded[::2] = arguments["d2"]
    arguments["d2"] = padded[::2]
    assert not arguments["d2"].flags.c_contiguous
    actual = compiled.notebook.aperture_weights(**arguments)
    np.testing.assert_allclose(actual, expected, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)


def test_notebook_accepts_unaligned_readonly_inputs(compiled):
    """The copying adapter can accept a byte-offset view without mutating it."""
    arguments, expected = reference_inputs()
    original = arguments["d2"]
    storage = bytearray(original.nbytes + 1)
    unaligned = np.ndarray(
        shape=original.shape, dtype=np.float64, buffer=storage, offset=1
    )
    unaligned[:] = original
    unaligned.setflags(write=False)
    assert not unaligned.flags.aligned
    arguments["d2"] = unaligned
    actual = compiled.notebook.aperture_weights(**arguments)
    np.testing.assert_allclose(actual, expected, rtol=WEIGHT_RTOL, atol=WEIGHT_ATOL)
    np.testing.assert_array_equal(unaligned, original)


def test_thread_count_and_repeated_call(compiled):
    """Workers own independent outputs, so changing their count changes no sum."""
    arguments, _ = reference_inputs()
    baseline = compiled.production.aperture_weights(**arguments)
    ntriangle = baseline[0].shape[2]
    rng = np.random.default_rng(seed=4704)
    gamma_real = rng.normal(size=(5, 4, ntriangle))
    gamma_imag = rng.normal(size=(5, 4, ntriangle))
    reference = compiled.production.aperture_apply(
        wr=baseline[0], wi=baseline[1], gr=gamma_real, gi=gamma_imag
    )
    for nthreads in (2, 3, 4, 1):
        compiled.set_omp_threads(nthreads=nthreads)
        for repeat in range(2):
            weights = compiled.production.aperture_weights(**arguments)
            actual = compiled.production.aperture_apply(
                wr=weights[0], wi=weights[1], gr=gamma_real, gi=gamma_imag
            )
            np.testing.assert_array_equal(weights, baseline)
            np.testing.assert_array_equal(actual, reference)
