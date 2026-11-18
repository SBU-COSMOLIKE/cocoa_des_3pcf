"""Draw supplied mean-vector and covariance timings without running a model.

The JSON schema is des3pcf-optimization-timings-v1. Its required fields are
format, about, mean and covariance. about names the measurement producer and
timing scope. mean contains exactly four series: case nla or tatt crossed
with variant original or optimized. covariance contains two series with one
shared case name and the same two variants. Each series contains:

  label: a distinct, nonempty legend label;
  case: the model name specified above;
  variant: original or optimized;
  threads: the integer list [1, 4, 8];
  seconds: three rows of ten positive finite elapsed times, in thread order;
  warmup_seconds: optional three positive times, retained only in the JSON.

The ten columns identify the same saved parameter points across thread
counts and variants. The producer must preserve that correspondence; this
plot cannot reconstruct cosmological parameters from elapsed times. Warmups
are never included in medians, ranges or speedups. Ranges combine parameter
dependence and machine variation and are not confidence intervals.

Call plot_optimization(input_path, mean_output, covariance_output), where
outputs are filename stems or PNG/PDF paths. Both formats are written for
each figure, and the function returns captions for use outside the figures.
To run this file alone, place data.json beside it and include outputs.mean
and outputs.covariance stems relative to that JSON. Copying these two files
to another directory requires only NumPy and Matplotlib, not Cocoa.
"""

from pathlib import Path
import json

import numpy as np
import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


# Figure style of the whole notebook: the shared plotting functions never change
# rcParams, so these lines set every font, tick and grid default.
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['xtick.bottom'] = True
matplotlib.rcParams['xtick.top'] = False
matplotlib.rcParams['ytick.right'] = False
matplotlib.rcParams['axes.edgecolor'] = 'black'
matplotlib.rcParams['axes.linewidth'] = '1.0'
matplotlib.rcParams['axes.labelsize'] = 'medium'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.linewidth'] = '0.0'
matplotlib.rcParams['grid.alpha'] = '0.18'
matplotlib.rcParams['grid.color'] = 'lightgray'
matplotlib.rcParams['legend.labelspacing'] = 0.77
matplotlib.rcParams['savefig.bbox'] = 'tight'
matplotlib.rcParams['savefig.format'] = 'pdf'


def read_series(records, kind):
    """Validate each measured series before allocating a figure.

    Arguments:
      records = JSON list for mean or covariance, following the module schema.
      kind = mean or covariance; determines the required model/variant pairs.
    Returns:
      List of owned dictionaries with seconds[3,10] float64 arrays.
    Raises:
      ValueError for missing, duplicated, nonnumeric or nonpositive data.
    """
    expected_count = 4 if kind == "mean" else 2
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError(f"{kind} requires exactly {expected_count} series.")
    result = []
    pairs = set()
    labels = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Every timing series must be a JSON object.")
        for name in (
            "label",
            "case",
            "variant",
        ):
            value = record.get(name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Series {name} must be a nonempty string.")
        if record["variant"] not in ("original", "optimized"):
            raise ValueError("variant must be original or optimized.")
        thread_values = record.get("threads")
        expected_threads = [
            1,
            4,
            8,
        ]
        if thread_values != expected_threads:
            raise ValueError(
                "Each series needs threads=[1,4,8], in that order."
            )
        for count in thread_values:
            if type(count) is not int:
                raise ValueError("Thread counts must be JSON integers.")
        key = (record["case"], record["variant"])
        if key in pairs or record["label"] in labels:
            raise ValueError(
                "Model/variant pairs and legend labels must differ."
            )
        pairs.add(key)
        labels.add(record["label"])
        rows = record.get("seconds")
        if not isinstance(rows, list) or len(rows) != 3:
            raise ValueError("seconds requires one row per thread count.")
        for row in rows:
            if not isinstance(row, list) or len(row) != 10:
                raise ValueError(
                    "Each thread count requires ten measured times."
                )
            for value in row:
                if (isinstance(value, bool)
                        or not isinstance(value, (int, float))):
                    raise ValueError("Elapsed times must be JSON numbers.")
        seconds = np.array(rows, dtype=np.float64, copy=True)
        if not np.all(np.isfinite(seconds)) or np.any(seconds <= 0.0):
            raise ValueError("Elapsed times must be finite positive seconds.")
        if "warmup_seconds" in record:
            warmup = record["warmup_seconds"]
            if not isinstance(warmup, list) or len(warmup) != 3:
                raise ValueError("warmup_seconds needs three excluded times.")
            for value in warmup:
                if (isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not np.isfinite(value) or value <= 0.0):
                    raise ValueError("Warmup times must be positive seconds.")
        result.append({
            "label": record["label"],
            "case": record["case"],
            "variant": record["variant"],
            "seconds": seconds,
        })
    if kind == "mean":
        expected = {
            ("nla", "original"),
            ("nla", "optimized"),
            ("tatt", "original"),
            ("tatt", "optimized"),
        }
        if pairs != expected:
            raise ValueError(
                "Mean series must contain NLA/TATT and both variants."
            )
    else:
        if result[0]["case"] != result[1]["case"]:
            raise ValueError("Covariance variants must describe the same case.")
    return result


def series_style(series, kind):
    """Assign the declared model and implementation encodings.

    Arguments:
      series = validated model/variant dictionary; kind = mean or covariance.
    Returns:
      Color, marker, line width, line style and marker-face color.
      Original curves are dashed/open; optimized curves are solid/filled.
    """
    if kind == "mean":
        blue = series["case"] == "nla"
    else:
        blue = series["variant"] == "original"
    if blue:
        color = "#0072B2"
        marker = "o"
        width = 2.4
    else:
        color = "#E69F00"
        marker = "s"
        width = 1.8
    original = series["variant"] == "original"
    linestyle = "--" if original else "-"
    face = "white" if original else color
    return color, marker, width, linestyle, face


def line_points(line, axis):
    """Sample a drawn line in display pixels, retaining only visible points.

    Arguments:
      line = Matplotlib Line2D, including point-only artists and range bars.
      axis = owning axis, whose rectangle clips the displayed data.
    Returns:
      points[n,2] in pixels. Sixty-four subdivisions per segment resolve
      crossings with label boxes; point-only artists retain their centers.
    """
    coordinates = np.asarray(line.get_xydata(), dtype=np.float64)
    if coordinates.size == 0:
        return np.empty(shape=(0, 2))
    if line.get_linestyle() not in ("None", "none", "", " "):
        pieces = []
        fractions = np.linspace(start=0.0, stop=1.0, num=65)
        for index in range(coordinates.shape[0] - 1):
            start = coordinates[index]
            difference = coordinates[index + 1] - start
            pieces.append(start[None, :] + fractions[:, None] * difference)
        if pieces:
            coordinates = np.concatenate(pieces, axis=0)
    pixels = line.get_transform().transform(coordinates)
    boundary = axis.get_window_extent()
    visible = np.all(np.isfinite(pixels), axis=1)
    visible &= pixels[:, 0] >= boundary.x0
    visible &= pixels[:, 0] <= boundary.x1
    visible &= pixels[:, 1] >= boundary.y0
    visible &= pixels[:, 1] <= boundary.y1
    return pixels[visible]


def check_layout(figure, axes, legend):
    """Keep the outside legend and labels clear of plotted data after drawing.

    Arguments:
      figure = rendered figure; axes = its two panels; legend = shared legend.
    Returns:
      None. The whole-axis legend check is conservative: its rectangle must
      also clear ticks and axis labels. Sampled lines check text against
      reference lines, range bars and point markers in display coordinates.
    Raises:
      RuntimeError if a legend leaves the canvas or text touches plotted data.
    """
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    legend_box = legend.get_window_extent(renderer=renderer).padded(5.0)
    canvas = figure.bbox
    if (legend_box.x0 < canvas.x0 or legend_box.x1 > canvas.x1
            or legend_box.y1 > canvas.y1):
        raise RuntimeError("Legend exceeds the canvas; shorten its labels.")
    for axis in axes:
        if legend_box.overlaps(axis.get_tightbbox(renderer=renderer)):
            raise RuntimeError(
                "Legend touches an axis; enlarge the top margin."
            )
        texts = [axis.xaxis.label, axis.yaxis.label]
        texts.extend(axis.get_xticklabels())
        texts.extend(axis.get_yticklabels())
        texts.extend(axis.texts)
        texts.extend(figure.texts)
        texts.append(axis.xaxis.get_offset_text())
        texts.append(axis.yaxis.get_offset_text())
        for line in axis.get_lines():
            points = line_points(line=line, axis=axis)
            marker_radius = 0.5 * line.get_markersize() * figure.dpi / 72.0
            stroke_radius = 0.5 * line.get_linewidth() * figure.dpi / 72.0
            margin = max(marker_radius, stroke_radius) + 2.0
            for text in texts:
                if not text.get_visible() or not text.get_text():
                    continue
                box = text.get_window_extent(renderer=renderer).padded(margin)
                inside = points[:, 0] >= box.x0
                inside &= points[:, 0] <= box.x1
                inside &= points[:, 1] >= box.y0
                inside &= points[:, 1] <= box.y1
                if np.any(inside):
                    raise RuntimeError(
                        "An axis label touches plotted data; enlarge margins."
                    )


def draw_figure(series, kind):
    """Draw elapsed seconds and speedup with every measured point visible.

    Arguments:
      series = validated timing dictionaries; kind = mean or covariance.
    Returns:
      Figure with two panels and one outside legend. No title or footer is
      drawn. Its lines and labels have passed the display-coordinate checks.

    Each series uses median(T1)/Tn for its speedup points. The summary uses
    median(T1)/median(Tn), which differs from median(T1/Tn) for ten points.
    Horizontal offsets only separate markers; tick labels remain 1,4,8.
    The ideal grey line S=n is clipped to the data-fitted panel range.
    """
    figure, axes = plt.subplots(nrows=1, ncols=2, figsize=(14.5, 5.7))
    figure.subplots_adjust(left=0.09, right=0.99, bottom=0.19, wspace=0.34)
    threads = np.asarray([
        1.0,
        4.0,
        8.0,
    ])
    offsets = np.linspace(start=-0.24, stop=0.24, num=len(series))
    jitter = np.linspace(start=-0.045, stop=0.045, num=10)
    maxima = np.zeros(2)
    handles = []
    labels = []
    for index, record in enumerate(series):
        seconds = record["seconds"]
        ordered = np.sort(seconds, axis=1)
        medians = ordered[:, 4] + 0.5 * (ordered[:, 5] - ordered[:, 4])
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            speedups = medians[0] / seconds
            summaries = medians[0] / medians
        if (not np.all(np.isfinite(speedups)) or np.any(speedups <= 0.0)
                or not np.all(np.isfinite(summaries))):
            raise ValueError(
                "Speedup overflows; check the supplied time units."
            )
        color, marker, width, linestyle, face = series_style(
            series=record, kind=kind,
        )
        centers = threads + offsets[index]
        positions = centers[:, None] + jitter[None, :]
        panels = (
            (seconds, medians),
            (speedups, summaries),
        )
        for panel, (values, summary) in enumerate(panels):
            axis = axes[panel]
            maxima[panel] = max(maxima[panel], float(np.max(values)))
            lower = np.min(values, axis=1)
            upper = np.max(values, axis=1)
            for thread in range(3):
                # Whiskers retain the observed extrema, not fitted errors.
                axis.plot(
                    [centers[thread], centers[thread]],
                    [lower[thread], upper[thread]],
                    color=color, linewidth=1.0, zorder=1,
                )
                for endpoint in (lower[thread], upper[thread]):
                    axis.plot(
                        [centers[thread] - 0.035, centers[thread] + 0.035],
                        [endpoint, endpoint], color=color, linewidth=1.0,
                        zorder=1,
                    )
            axis.plot(
                positions.ravel(), values.ravel(), linestyle="None",
                marker=marker, markersize=4.5, markerfacecolor=face,
                markeredgecolor=color, markeredgewidth=0.9, zorder=2,
            )
            median_artists = axis.plot(
                centers, summary, color=color, linewidth=width,
                linestyle=linestyle, marker=marker, markersize=9.0,
                markerfacecolor=face, markeredgecolor=color,
                markeredgewidth=1.4, zorder=3, label=record["label"],
            )
            median_line = median_artists[0]
            if panel == 0:
                handles.append(median_line)
                labels.append(record["label"])
    axes[1].plot(
        threads, threads, color="0.55", linewidth=1.2, linestyle="--",
        label="_nolegend_", zorder=0,
    )
    axis_labels = ["Elapsed time [s]", "Speedup"]
    for index, axis in enumerate(axes):
        upper = 1.1 * maxima[index]
        if not np.isfinite(upper):
            raise ValueError("Plotting range overflows; check the time units.")
        axis.set_xlim(left=0.55, right=8.45)
        axis.set_ylim(bottom=0.0, top=upper)
        axis.set_xticks(ticks=threads, labels=[
            "1",
            "4",
            "8",
        ])
        axis.set_ylabel(ylabel=axis_labels[index], fontsize=24, labelpad=8)
        axis.tick_params(axis="x", labelsize=22)
        axis.tick_params(axis="y", labelsize=19)
    figure.supxlabel("OpenMP threads", fontsize=24, y=0.035)
    legend = figure.legend(
        handles=handles, labels=labels, loc="upper center",
        bbox_to_anchor=(0.5, 0.99), ncol=len(series), frameon=False,
        fontsize=22, handlelength=2.2, columnspacing=1.0, borderaxespad=0.0,
    )
    # Measure the legend before placing the panel tops. A fixed legend area
    # would leave unused space when the covariance has only two entries.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    box = legend.get_window_extent(renderer=renderer)
    gap_pixels = 24.0 * figure.dpi / 72.0
    top = (box.y0 - gap_pixels - figure.bbox.y0) / figure.bbox.height
    if top <= 0.45:
        raise RuntimeError(
            "Legend leaves too little plot area; shorten labels."
        )
    figure.subplots_adjust(top=top)
    check_layout(figure=figure, axes=axes, legend=legend)
    return figure


def output_paths(stem):
    """Resolve the two file formats requested by one caller-selected path.

    Arguments:
      stem = filename stem, PNG filename or PDF filename.
    Returns:
      Absolute PNG and PDF Paths sharing the supplied stem.
    Raises:
      ValueError if another suffix would make the destination ambiguous.
    """
    supplied = Path(stem)
    if supplied.suffix.lower() in (".png", ".pdf"):
        supplied = supplied.with_suffix("")
    elif supplied.suffix:
        raise ValueError(
            "Output paths must be stems, .png files or .pdf files."
        )
    resolved = supplied.resolve()
    return resolved.with_suffix(".png"), resolved.with_suffix(".pdf")


def plot_optimization(input_path, mean_output, covariance_output):
    """Read portable JSON and save separate mean/covariance timing figures.

    Arguments:
      input_path = JSON file following the module's timing schema.
      mean_output, covariance_output = caller-selected distinct output stems.
    Returns:
      Dictionary with each figure's PNG/PDF filenames and external caption.
    Raises:
      ValueError for invalid records, ambiguous destinations or missing about.
      RuntimeError if labels or the outside legend fail the collision check.
    Side effects:
      Creates destination directories and writes PNG/PDF files. Existing plot
      files at those explicit paths are replaced. No numerical model is run.
    """
    filename = Path(input_path)
    contents = filename.read_text()
    data = json.loads(s=contents)
    if not isinstance(data, dict):
        raise ValueError("The timing JSON must contain one named object.")
    if data.get("format") != "des3pcf-optimization-timings-v1":
        raise ValueError("Input must use des3pcf-optimization-timings-v1.")
    if not isinstance(data.get("about"), str) or not data["about"].strip():
        raise ValueError("about must identify the timing producer and scope.")
    means = read_series(records=data.get("mean"), kind="mean")
    covariance = read_series(records=data.get("covariance"), kind="covariance")
    mean_paths = output_paths(stem=mean_output)
    covariance_paths = output_paths(stem=covariance_output)
    if mean_paths[0] == covariance_paths[0]:
        raise ValueError("Mean and covariance output stems must differ.")
    for destination in mean_paths + covariance_paths:
        if destination == filename.resolve():
            raise ValueError(
                "A plot destination must not replace its input JSON."
            )
    caption = (
        "Points show ten saved parameter points at each thread count; "
        "bars span the observed minimum and maximum, and curves join "
        "median summaries. Each speedup uses its own series' one-thread "
        "median. The grey line is ideal linear scaling, S=n, clipped to "
        "the displayed range. Warmups are excluded. Ranges include "
        "parameter dependence and machine variation, not confidence intervals."
    )
    result = {}
    requests = (
        (
            "mean",
            means,
            mean_paths,
        ),
        (
            "covariance",
            covariance,
            covariance_paths,
        ),
    )
    for kind, series, paths in requests:
        figure = draw_figure(series=series, kind=kind)
        png, pdf = paths
        png.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(fname=png, dpi=180)
        figure.savefig(fname=pdf)
        plt.close(fig=figure)
        result[kind] = {
            "png": str(png),
            "pdf": str(pdf),
            "caption": caption,
        }
    return result


def main():
    """Draw adjacent data.json using output paths declared in that JSON.

    Returns:
      None; prints the saved paths and captions as JSON.
    Raises:
      ValueError if either explicit output stem is missing.
    """
    script = Path(__file__).resolve()
    input_path = script.with_name("data.json")
    contents = input_path.read_text()
    data = json.loads(s=contents)
    if not isinstance(data, dict):
        raise ValueError("Standalone data.json must contain one named object.")
    outputs = data.get("outputs")
    if not isinstance(outputs, dict):
        raise ValueError("Standalone data.json requires an outputs dictionary.")
    for kind in ("mean", "covariance"):
        if not isinstance(outputs.get(kind), str) or not outputs[kind].strip():
            raise ValueError(
                f"outputs.{kind} must be an explicit filename stem."
            )
    result = plot_optimization(
        input_path=input_path,
        mean_output=input_path.parent / outputs["mean"],
        covariance_output=input_path.parent / outputs["covariance"],
    )
    print(json.dumps(obj=result, indent=2))


if __name__ == "__main__":
    main()
