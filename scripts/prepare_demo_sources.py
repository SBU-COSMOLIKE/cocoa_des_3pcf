r"""Write explicitly modified source curves for a demonstration Fisher forecast.

Negative calibrated n(z) samples are not a source probability distribution.
For this demonstration, each negative midpoint value is replaced by zero.
The clipped curve is divided by its trapezoidal integral over the supplied
Z_MID samples. This changes the calibration and does not propagate its
uncertainty. A forecast using these curves is not a survey science result.

Run with all input choices explicit:
  python scripts/prepare_demo_sources.py --input original.fits \
    --output demonstration_sources.fits --extension nz_source --nbin 4

The original file is read without modification. The output contains its
copied primary HDU and selected source table, with the selected BIN columns
stored as float64. Other source-table columns, including all redshift
columns, retain their values and column definitions. A YAML sidecar uses
the output stem, for example demonstration_sources.yaml. Both destinations
must be absent. No overwrite option, CAMB call or native model is used.
"""

import argparse
import hashlib
from io import BytesIO
import yaml
import os
from pathlib import Path
import re

import numpy as np
from astropy.io import fits


POLICY = (
    "Demonstration Fisher forecast only: replace negative BIN midpoint "
    "samples by zero, then divide each bin by the trapezoidal integral "
    "of its clipped curve over the supplied Z_MID samples."
)
LIMITATIONS = (
    "Clipping changes the supplied calibration; these are not recalibrated "
    "survey source distributions.",
    "Uncertainty in the removed negative samples is not propagated.",
    "The integration treats supplied midpoint values as a piecewise-linear "
    "curve, not a reconstructed histogram.",
    "No redshift nodes, bin boundaries, extrapolated tails or source shifts "
    "are introduced.",
    "Fisher results using these curves are demonstrations, not validated "
    "survey constraints or a reproduction of a published likelihood.",
)


def normalize_midpoints(redshift, densities):
    """Clip and normalize midpoint curves while recording the removed area.

    The negative-area magnitude integrates max(-n,0); the positive area
    integrates max(n,0). Their difference is the original signed integral.
    All three use the same supplied midpoint grid and trapezoidal rule.
    Clipping acts on samples first; zero crossings between nodes are not
    inserted into the grid.

    Arguments:
      redshift = finite increasing nonnegative z[nz], with nz at least two.
      densities = finite samples[nbin,nz], with one row per source bin.
    Returns:
      normalized[nbin,nz] float64 values and one statistics dictionary per bin.
      Inputs are unchanged. Redshift and n(z) have dimensionless units.
    Raises:
      ValueError for incompatible shapes, nonfinite values, bad redshift
      order, or a nonpositive signed or clipped integral in any bin.
    """
    z = np.array(redshift, dtype=np.float64, copy=True)
    values = np.array(densities, dtype=np.float64, copy=True)
    if z.ndim != 1 or z.size < 2:
        raise ValueError("Z_MID must be a one-dimensional array with >=2 rows.")
    if not np.all(np.isfinite(z)) or np.any(z < 0.0):
        raise ValueError("Z_MID must contain finite nonnegative redshifts.")
    if np.any(np.diff(z) <= 0.0):
        raise ValueError("Z_MID must increase strictly without repeated nodes.")
    if (values.ndim != 2 or values.shape[0] == 0
            or values.shape[1] != z.size):
        raise ValueError("Source samples must have shape [nbin,len(Z_MID)].")
    if not np.all(np.isfinite(values)):
        raise ValueError("Every BIN source sample must be finite.")
    clipped = np.maximum(values, 0.0)
    negative = np.maximum(-values, 0.0)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        signed_area = np.trapz(values, x=z, axis=1)
        positive_area = np.trapz(clipped, x=z, axis=1)
        negative_area = np.trapz(negative, x=z, axis=1)
        factors = 1.0 / positive_area
        normalized = clipped / positive_area[:, None]
    for index in range(values.shape[0]):
        integrals = np.asarray([
            signed_area[index],
            positive_area[index],
            negative_area[index],
        ])
        if not np.all(np.isfinite(integrals)):
            raise ValueError(f"BIN{index + 1} has a nonfinite integrated area.")
        if signed_area[index] <= 0.0 or positive_area[index] <= 0.0:
            raise ValueError(
                f"BIN{index + 1} needs positive signed and clipped integrals."
            )
        if not np.isfinite(factors[index]):
            raise ValueError(
                f"BIN{index + 1} has a nonfinite normalization scale."
            )
    if not np.all(np.isfinite(normalized)):
        raise ValueError("Normalized source samples must remain finite.")
    final_area = np.trapz(normalized, x=z, axis=1)
    records = []
    for index in range(values.shape[0]):
        records.append({
            "column": f"BIN{index + 1}",
            "negative_sample_count": int(np.count_nonzero(values[index] < 0)),
            "negative_area_magnitude": float(negative_area[index]),
            "original_signed_integral": float(signed_area[index]),
            "original_positive_integral": float(positive_area[index]),
            "negative_area_fraction_of_positive": float(
                negative_area[index] / positive_area[index],
            ),
            "normalization_factor": float(factors[index]),
            "output_midpoint_integral": float(final_area[index]),
        })
    return normalized, records


def write_pair(output, fits_content, sidecar, yaml_content):
    """Create two absent files after all scientific validation has finished.

    Exclusive creation refuses a destination created by another process
    after the earlier checks. If writing fails, remove only partial files
    whose filesystem identities still match the files opened by this call.

    Arguments:
      output, sidecar = distinct destination Paths in existing directories.
      fits_content = already serialized FITS bytes.
      yaml_content = already serialized UTF-8 audit text.
    Returns:
      None; creates the requested files without replacing an existing file.
    Raises:
      OSError for destination collisions or failed writes.
    """
    created = []
    try:
        with output.open(mode="xb") as fits_stream:
            state = os.fstat(fits_stream.fileno())
            created.append((output, state.st_dev, state.st_ino))
            with sidecar.open(mode="x", encoding="utf-8") as audit_stream:
                state = os.fstat(audit_stream.fileno())
                created.append((sidecar, state.st_dev, state.st_ino))
                fits_stream.write(fits_content)
                audit_stream.write(yaml_content)
    except OSError:
        for path, device, inode in created:
            try:
                current = path.stat()
            except FileNotFoundError:
                continue
            if current.st_dev == device and current.st_ino == inode:
                path.unlink()
        raise


def prepare_demo_sources(input_path, output_path, extension, nbin):
    """Write a source-only FITS copy and its demonstration-policy record.

    BIN1 through BINnbin must be exactly the source-bin columns present.
    Selected density columns use FITS D storage, including when their input
    uses integer or float32 storage. That prevents output conversion from
    undoing the float64 midpoint normalization. Other columns are copied.

    Arguments:
      input_path = original FITS file, read without modification.
      output_path = absent destination with a .fits suffix.
      extension = explicit source binary-table HDU name.
      nbin = positive integer number of source columns.
    Returns:
      Plain metadata record also saved beside output_path with suffix .yaml.
    Raises:
      ValueError for column/count mismatches, unsuitable numerical inputs,
      ambiguous extension names, colliding paths or missing output directory.
      FileExistsError if either output file already exists.
      OSError for FITS reads or exclusive output creation failures.
    """
    if isinstance(nbin, bool) or not isinstance(nbin, int) or nbin < 1:
        raise ValueError("nbin must be an explicit positive integer.")
    if not isinstance(extension, str) or not extension.strip():
        raise ValueError("extension must be an explicit source HDU name.")
    supplied_input = Path(input_path)
    original = supplied_input.resolve(strict=True)
    requested_output = Path(output_path)
    if requested_output.suffix.lower() != ".fits":
        raise ValueError("The demonstration output must have a .fits suffix.")
    requested_sidecar = requested_output.with_suffix(".yaml")
    output = requested_output.resolve()
    sidecar = requested_sidecar.resolve()
    paths = {
        original,
        output,
        sidecar,
    }
    if len(paths) != 3:
        raise ValueError(
            "Input FITS, output FITS and audit paths must be distinct."
        )
    for path in (requested_output, requested_sidecar):
        if path.exists() or path.is_symlink():
            raise FileExistsError(
                f"Destination exists; choose another path: {path}"
            )
    if not output.parent.is_dir() or not sidecar.parent.is_dir():
        raise ValueError("Create the output directory before preprocessing.")

    # Parse the same bytes that are hashed, so replacing the source file
    # concurrently cannot detach the recorded identity from these samples.
    raw_content = original.read_bytes()
    raw_identifier = hashlib.sha256(raw_content)
    raw_hash = raw_identifier.hexdigest()
    raw_stream = BytesIO(raw_content)
    with fits.open(name=raw_stream, memmap=False) as hdus:
        matching = []
        for index, hdu in enumerate(hdus):
            if hdu.name.casefold() == extension.casefold():
                matching.append(index)
        if len(matching) != 1:
            raise ValueError(
                "The source extension name must match exactly one HDU."
            )
        source = hdus[matching[0]]
        if not isinstance(source, fits.BinTableHDU) or source.data is None:
            raise ValueError(
                "The source extension must be a populated binary table."
            )
        names = list(source.columns.names)
        if len(names) != len(set(names)):
            raise ValueError("Source-table column names must be unique.")
        if "Z_MID" not in names:
            raise ValueError(
                "The source table lacks the required Z_MID column."
            )
        selected = {}
        for index in range(nbin):
            selected[f"BIN{index + 1}"] = index
        actual_bins = set()
        for name in names:
            if re.fullmatch(pattern=r"BIN[0-9]+", string=name) is not None:
                actual_bins.add(name)
        if actual_bins != set(selected):
            raise ValueError(
                f"Source BIN columns must match requested nbin={nbin} exactly."
            )
        z = np.array(source.data["Z_MID"], dtype=np.float64, copy=True)
        raw = []
        for name in selected:
            samples = np.array(source.data[name], dtype=np.float64, copy=True)
            raw.append(samples)
        densities = np.asarray(raw, dtype=np.float64)
        normalized, bin_records = normalize_midpoints(
            redshift=z, densities=densities,
        )
        columns = []
        unchanged_columns = []
        for column in source.columns:
            name = column.name
            if name in selected:
                index = selected[name]
                replacement = fits.Column(
                    name=name, format="D", unit=column.unit,
                    array=normalized[index],
                )
                columns.append(replacement)
                bin_records[index]["original_fits_format"] = str(column.format)
                bin_records[index]["output_fits_format"] = "D"
            else:
                columns.append(column.copy())
                unchanged_columns.append(name)
        output_source = fits.BinTableHDU.from_columns(
            columns=columns, header=source.header.copy(), name=source.name,
        )
        primary = hdus[0].copy()
    output_source.header["DEMONZ"] = (True, "Demonstration source curves only")
    output_source.header["NZCLIP"] = (
        True, "Negative midpoint values set to zero",
    )
    output_source.header["NZNORM"] = (
        "TRAPZMID", "Integral over supplied Z_MID",
    )
    output_source.header["ORIGSHA"] = raw_hash
    output_source.header.add_history(POLICY)
    output_hdus = fits.HDUList([primary, output_source])
    serialized = BytesIO()
    output_hdus.writeto(fileobj=serialized, checksum=True)
    fits_content = serialized.getvalue()
    output_identifier = hashlib.sha256(fits_content)
    output_hash = output_identifier.hexdigest()
    record = {
        "format": "des3pcf-demonstration-sources-v1",
        "demonstration_only": True,
        "policy": POLICY,
        "limitations": list(LIMITATIONS),
        "input": {
            "path": str(original),
            "sha256": raw_hash,
            "bytes": len(raw_content),
        },
        "output": {
            "path": str(output),
            "sha256": output_hash,
            "bytes": len(fits_content),
        },
        "audit_path": str(sidecar),
        "extension": output_source.name,
        "nbin": nbin,
        "redshift_count": int(z.size),
        "redshift_minimum": float(z[0]),
        "redshift_maximum": float(z[-1]),
        "unchanged_source_columns": unchanged_columns,
        "integration_rule": "trapezoid over supplied midpoint samples",
        "signed_integral_requirement": "strictly positive before clipping",
        "bins": bin_records,
    }
    yaml_content = yaml.safe_dump(record, sort_keys=False)
    write_pair(
        output=output, fits_content=fits_content,
        sidecar=sidecar, yaml_content=yaml_content,
    )
    return record


def main():
    """Read explicit CLI choices and write the demonstration source pair.

    Returns:
      None; prints the output and audit filenames after successful writing.
    Raises:
      File and scientific-input errors propagate without replacing inputs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extension", required=True)
    parser.add_argument("--nbin", type=int, required=True)
    args = parser.parse_args()
    record = prepare_demo_sources(
        input_path=args.input, output_path=args.output,
        extension=args.extension, nbin=args.nbin,
    )
    print(record["output"]["path"])
    print(record["audit_path"])


if __name__ == "__main__":
    main()
