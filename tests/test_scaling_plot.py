"""Check a timing display using controlled numbers, never measured runtimes.

These arrays deliberately assign different values to ten parameter-point
indices, so a ratio of medians can be distinguished from a median of
ratios. The retained ``repetition`` field names those point indices; it
does not imply repeated evaluation at fixed parameters. Neither source
code nor a compiled prediction is run to construct these controlled values.
"""

import copy
import os

for name in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
             "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["OMP_THREAD_LIMIT"] = "4"

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np
import pytest

from cosmolike_notebook_utils.plot_thread_scaling import plot_thread_scaling


def controlled_records(workload_count=2):
    """Return controlled times for one, two or four declared model/workload labels."""
    records = []
    names = ["Controlled LSST Y1 NLA", "Controlled LSST Y1 TATT",
             "Controlled 3pcf NLA", "Controlled 3pcf TATT"]
    baselines = [40.0, 48.0, 28.0, 56.0]
    variations = np.array([0.8, 0.85, 0.9, 0.95, 0.98,
                           1.03, 1.05, 1.12, 1.2, 1.3])
    for workload, baseline in zip(names[:workload_count], baselines[:workload_count]):
        for repetition in range(10):
            for threads, speedup in ((1, 1.0), (4, 2.7), (8, 3.8)):
                records.append({
                    "workload": workload, "threads": threads,
                    "repetition": repetition,
                    "seconds": baseline * variations[repetition] / speedup,
                })
    return records


@pytest.mark.parametrize("workload_count", [1, 2, 4])
def test_complete_points_median_baseline_and_ownership(workload_count):
    """Every supplied ordinate survives, with its own workload's median T1."""
    records = controlled_records(workload_count=workload_count)
    original = copy.deepcopy(records)
    figure, axes = plot_thread_scaling(records=records, title="Controlled test arrays")
    names = list(dict.fromkeys(record["workload"] for record in records))
    for workload, name in enumerate(names):
        data = np.array([
            [next(row["seconds"] for row in records if row["workload"] == name
                  and row["threads"] == threads and row["repetition"] == repetition)
             for repetition in range(10)] for threads in (1, 4, 8)
        ])
        baseline = np.median(data[0])
        # One scatter and one errorbar collection per workload. The scatter
        # stores all thirty times, in thread-major then parameter-point order.
        elapsed = axes[0].collections[2 * workload].get_offsets()[:, 1]
        speedup = axes[1].collections[2 * workload].get_offsets()[:, 1]
        np.testing.assert_array_equal(elapsed, data.ravel())
        np.testing.assert_allclose(speedup, (baseline / data).ravel(), rtol=2.e-16)
        summary = axes[1].containers[workload].lines[0].get_ydata()
        np.testing.assert_allclose(summary, baseline / np.median(data, axis=1))
        assert summary[0] == 1.0
    ideal = axes[1].lines[-1]
    np.testing.assert_array_equal(ideal.get_xdata(), [1, 4, 8])
    np.testing.assert_array_equal(ideal.get_ydata(), [1, 4, 8])
    explanation = figure.legends[0].get_title().get_text()
    assert "10 parameter points" in explanation
    assert "machine variation" in explanation
    assert "not CI or repeatability" in explanation
    labels = [text.get_text() for text in figure.legends[0].get_texts()]
    assert labels == names + ["Ideal linear speedup"]
    assert len(axes[0].containers) == workload_count
    assert records == original
    plt.close(figure)


def test_speedup_uses_ratio_of_medians_and_transformed_extrema():
    """The speedup summary differs from the median of the individual ratios."""
    records = controlled_records()
    changed_times = [1., 2., 3., 4., 5., 9., 10., 11., 12., 13.]
    for record in records:
        if record["workload"] == "Controlled LSST Y1 NLA" and record["threads"] == 4:
            record["seconds"] = changed_times[record["repetition"]]
    figure, axes = plot_thread_scaling(records=records)
    baseline = np.median([
        row["seconds"] for row in records
        if row["workload"] == "Controlled LSST Y1 NLA" and row["threads"] == 1
    ])
    result = axes[1].containers[0]
    summary = result.lines[0].get_ydata()[1]
    assert summary == baseline / 7.0
    assert not np.isclose(summary, np.median(baseline / np.array(changed_times)))
    bar = result.lines[2][0].get_segments()[1]
    np.testing.assert_allclose(bar[:, 1], [baseline / 13.0, baseline])
    plt.close(figure)


@pytest.mark.parametrize("change", ["missing", "duplicate", "negative", "nonfinite",
                                    "thread", "repetition", "workload"])
def test_invalid_records_are_refused_before_drawing(change):
    """Never replace a failed or absent run with a fabricated observation."""
    records = controlled_records()
    if change == "missing":
        records.pop()
    elif change == "duplicate":
        records[-1] = dict(records[0])
    elif change == "negative":
        records[0]["seconds"] = -1.0
    elif change == "nonfinite":
        records[0]["seconds"] = np.nan
    elif change == "thread":
        records[0]["threads"] = True
    elif change == "repetition":
        records[0]["repetition"] = 10
    else:
        records[0]["workload"] = "Third workload"
    before = plt.get_fignums()
    with pytest.raises(ValueError):
        plot_thread_scaling(records=records)
    assert plt.get_fignums() == before
