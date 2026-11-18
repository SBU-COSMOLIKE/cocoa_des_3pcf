"""Check benchmark control and result validation without measuring a workload."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from des3pcf.scaling import load_configuration, validate_vectors, _runtime_check


PROJECT = Path(__file__).resolve().parents[1]


def test_configuration_keeps_the_requested_worker_counts():
    """The benchmark accepts eight workers without changing routine test settings."""
    result = load_configuration(PROJECT / "configs" / "openmp_scaling.yaml")
    assert result["threads"] == [1, 4, 8]
    assert result["points"] == 10
    assert Path(result["threepcf_configuration"]).name == "des_y3_tatt_scaling.yaml"


@pytest.mark.parametrize("field,value", [
    ("threads", [True, 4, 8]),
    ("thread_order", [1, 4, 4]),
    ("points", 9),
    ("child_timeout_seconds", 0),
])
def test_configuration_refuses_an_ambiguous_study(tmp_path, field, value):
    """Malformed counts cannot silently change the intended sixty evaluations."""
    source = PROJECT / "configs" / "openmp_scaling.yaml"
    config = yaml.safe_load(source.read_text())
    config[field] = value
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError):
        load_configuration(path)


def _saved_vectors(directory):
    """Write small controlled vectors with distinct physical-point identifiers."""
    for workload in ("threepcf", "lsst"):
        for threads in (1, 4, 8):
            folder = directory / f"{workload}_{threads}"
            folder.mkdir()
            for index in range(11):
                values = np.array([index + 1.0, -0.0, 0.25 * index])
                np.save(folder / f"point_{index:02d}.npy", values)


def test_thread_check_requires_bits_and_changed_point_outputs(tmp_path):
    """A sign-of-zero change fails even though a tolerance comparison would pass."""
    _saved_vectors(tmp_path)
    validate_vectors(tmp_path)
    path = tmp_path / "threepcf_8" / "point_03.npy"
    values = np.load(path)
    values[1] = 0.0
    np.save(path, values)
    with pytest.raises(RuntimeError, match="output changed at 8 threads"):
        validate_vectors(tmp_path)


def test_thread_check_rejects_a_stale_point(tmp_path):
    """An identical vector at a changed point cannot be labeled a recomputation."""
    _saved_vectors(tmp_path)
    for threads in (1, 4, 8):
        folder = tmp_path / f"lsst_{threads}"
        np.save(folder / "point_02.npy", np.load(folder / "point_01.npy"))
    with pytest.raises(RuntimeError, match="identical vector"):
        validate_vectors(tmp_path)


def test_runtime_report_reads_the_extension_runtime(compiled):
    """Read one-worker settings only; perform no physical model or timed call."""
    compiled.set_omp_threads(nthreads=1)
    report = _runtime_check(compiled.__file__, threads=1)
    assert report["omp_max_threads"] == 1
    assert report["omp_thread_limit"] >= 1
