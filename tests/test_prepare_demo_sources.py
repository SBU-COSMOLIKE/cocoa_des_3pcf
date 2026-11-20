"""Check demonstration source preprocessing with generated FITS files.

The irregular midpoint grid has independently calculable positive and
negative trapezoidal areas. Checks cover the saved float64 values, audit
hashes, untouched redshift columns and original file, and refusal before
writing for invalid inputs. No CAMB calculation or native model is needed.
"""

import hashlib
import yaml
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from astropy.io import fits


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
sys.path.insert(0, str(PROJECT.parents[1] / "external_modules/code/cosmolike_core"))
from prepare_demo_sources import prepare_demo_sources, write_pair


def example_samples():
    """Return four source curves on an irregular four-node redshift grid.

    Returns:
      redshift[4], densities[4,4], both independent float64 arrays.
      BIN1 has signed area 1.325, positive area 1.375 and negative area 0.05.
    """
    redshift = np.asarray([
        0.0,
        0.5,
        1.5,
        3.0,
    ])
    densities = np.asarray([
        [-0.2, 1.0, 0.5, 0.0],
        [0.0, 0.5, 1.0, -0.1],
        [0.2, 0.8, 0.3, 0.1],
        [0.0, 1.0, 1.0, 0.0],
    ])
    return redshift, densities


def write_source(path, redshift=None, densities=None, extension="nz_source",
                 bin_format="D", include_midpoint=True):
    """Create a test FITS with source metadata and an unrelated image HDU.

    Arguments:
      path = absent temporary FITS destination.
      redshift, densities = optional replacement arrays for the four-node case.
      extension = explicit source-table HDU name.
      bin_format = FITS scalar format; E checks float32-to-float64 promotion.
      include_midpoint = omit Z_MID when testing missing-column refusal.
    Returns:
      None; writes the generated FITS file without replacing another file.
    """
    default_redshift, default_densities = example_samples()
    if redshift is None:
        redshift = default_redshift
    if densities is None:
        densities = default_densities
    columns = []
    columns.append(fits.Column(
        name="Z_LOW", format="D", array=redshift - 0.05,
    ))
    if include_midpoint:
        columns.append(fits.Column(name="Z_MID", format="D", array=redshift))
    columns.append(fits.Column(
        name="Z_HIGH", format="D", array=redshift + 0.05,
    ))
    for index in range(densities.shape[0]):
        columns.append(fits.Column(
            name=f"BIN{index + 1}", format=bin_format,
            unit="1", array=densities[index],
        ))
    columns.append(fits.Column(
        name="AUXILIARY", format="J", array=np.arange(redshift.size),
    ))
    source = fits.BinTableHDU.from_columns(columns=columns, name=extension)
    source.header["TESTMETA"] = "preserve source metadata"
    primary = fits.PrimaryHDU()
    primary.header["RAWID"] = "preserve primary metadata"
    unrelated = fits.ImageHDU(data=np.eye(2), name="unrelated")
    hdus = fits.HDUList([
        primary,
        source,
        unrelated,
    ])
    hdus.writeto(fileobj=path)


def test_clipping_accounting_and_original_preservation(tmp_path):
    """Check analytic areas, normalized output and the untouched original.

    Arguments:
      tmp_path = pytest-owned directory for generated input and output files.
    Returns:
      None; assertions verify numerical values and the saved file identities.
    """
    original = tmp_path / "original.fits"
    output = tmp_path / "demonstration.fits"
    write_source(path=original)
    raw_bytes = original.read_bytes()
    raw_identifier = hashlib.sha256(raw_bytes)
    raw_digest = raw_identifier.hexdigest()
    record = prepare_demo_sources(
        input_path=original, output_path=output, extension="nz_source", nbin=4,
    )
    assert original.read_bytes() == raw_bytes
    sidecar = output.with_suffix(".yaml")
    sidecar_contents = sidecar.read_text()
    stored = yaml.safe_load(sidecar_contents)
    assert stored == record
    assert record["demonstration_only"] is True
    assert record["input"]["sha256"] == raw_digest
    output_bytes = output.read_bytes()
    output_identifier = hashlib.sha256(output_bytes)
    output_digest = output_identifier.hexdigest()
    assert record["output"]["sha256"] == output_digest
    first = record["bins"][0]
    assert first["negative_sample_count"] == 1
    np.testing.assert_allclose(
        actual=first["negative_area_magnitude"], desired=0.05,
        rtol=1.0e-15, atol=0.0,
    )
    np.testing.assert_allclose(
        actual=first["original_signed_integral"], desired=1.325,
        rtol=1.0e-15, atol=0.0,
    )
    assert first["original_positive_integral"] == 1.375
    assert record["bins"][2]["negative_sample_count"] == 0
    with fits.open(name=original, memmap=False) as raw_hdus:
        with fits.open(name=output, memmap=False) as changed:
            assert len(changed) == 2
            assert changed[0].header["RAWID"] == raw_hdus[0].header["RAWID"]
            source = changed["nz_source"]
            assert source.header["DEMONZ"]
            assert source.header["TESTMETA"] == "preserve source metadata"
            assert source.header["ORIGSHA"] == raw_digest
            for name in (
                "Z_LOW",
                "Z_MID",
                "Z_HIGH",
                "AUXILIARY",
            ):
                np.testing.assert_array_equal(
                    x=source.data[name],
                    y=raw_hdus["nz_source"].data[name],
                )
            z = source.data["Z_MID"]
            for index in range(4):
                name = f"BIN{index + 1}"
                density = source.data[name]
                assert np.all(density >= 0.0)
                integral = np.trapz(density, x=z)
                np.testing.assert_allclose(
                    actual=integral, desired=1.0, rtol=3.0e-15, atol=0.0,
                )
            expected_first = np.asarray([0.0, 1.0, 0.5, 0.0]) / 1.375
            np.testing.assert_array_equal(
                x=source.data["BIN1"], y=expected_first,
            )


def test_float32_columns_are_written_as_normalized_float64(tmp_path):
    """Prevent float32 FITS storage from rounding the normalized output.

    Arguments:
      tmp_path = temporary directory for two generated FITS files.
    Returns:
      None; checks storage format and normalization after reading the output.
    """
    original = tmp_path / "float32.fits"
    output = tmp_path / "demonstration.fits"
    write_source(path=original, bin_format="E")
    record = prepare_demo_sources(
        input_path=original, output_path=output, extension="nz_source", nbin=4,
    )
    with fits.open(name=output, memmap=False) as hdus:
        source = hdus["nz_source"]
        for index in range(4):
            name = f"BIN{index + 1}"
            assert str(source.columns[name].format) == "D"
            assert record["bins"][index]["original_fits_format"] == "E"
            area = np.trapz(source.data[name], x=source.data["Z_MID"])
            np.testing.assert_allclose(
                actual=area, desired=1.0, rtol=3.0e-15, atol=0.0,
            )


@pytest.mark.parametrize(("problem", "message"), [
    ("nonfinite_redshift", "finite nonnegative"),
    ("negative_redshift", "finite nonnegative"),
    ("repeated_redshift", "increase strictly"),
    ("reversed_redshift", "increase strictly"),
    ("nonfinite_density", "source sample must be finite"),
    ("zero_signed_area", "positive signed and clipped"),
    ("no_positive_area", "positive signed and clipped"),
])
def test_invalid_samples_refused_before_writing(tmp_path, problem, message):
    """Reject each invalid numerical input without creating either output.

    Arguments:
      tmp_path = temporary directory; problem = selected invalid input class.
      message = expected numerical refusal text, not an unrelated I/O failure.
    Returns:
      None; checks refusal, absent outputs and the unchanged original bytes.
    """
    redshift, densities = example_samples()
    if problem == "nonfinite_redshift":
        redshift[1] = np.inf
    elif problem == "negative_redshift":
        redshift[0] = -0.1
    elif problem == "repeated_redshift":
        redshift[1] = redshift[0]
    elif problem == "reversed_redshift":
        redshift = redshift[::-1].copy()
    elif problem == "nonfinite_density":
        densities[2, 1] = np.nan
    elif problem == "zero_signed_area":
        densities[0] = [-3.0, 1.0, 0.0, 0.0]
    else:
        densities[0] = 0.0
    original = tmp_path / "original.fits"
    output = tmp_path / "demonstration.fits"
    write_source(path=original, redshift=redshift, densities=densities)
    raw_bytes = original.read_bytes()
    with pytest.raises(ValueError, match=message):
        prepare_demo_sources(
            input_path=original, output_path=output,
            extension="nz_source", nbin=4,
        )
    assert not output.exists()
    assert not output.with_suffix(".yaml").exists()
    assert original.read_bytes() == raw_bytes


@pytest.mark.parametrize(("problem", "message"), [
    ("missing_bin", "nbin=4 exactly"),
    ("extra_bin", "nbin=4 exactly"),
    ("missing_midpoint", "required Z_MID"),
    ("wrong_extension", "exactly one HDU"),
])
def test_column_and_extension_mismatches(tmp_path, problem, message):
    """Require the requested extension and exact BIN1..BINnbin column set.

    Arguments:
      tmp_path = temporary directory; problem = chosen FITS mismatch.
      message = expected column or extension refusal text.
    Returns:
      None; verifies mismatches cannot write a partial output pair.
    """
    redshift, densities = example_samples()
    if problem == "missing_bin":
        densities = densities[:3]
    elif problem == "extra_bin":
        densities = np.concatenate((densities, densities[:1]), axis=0)
    original = tmp_path / "original.fits"
    output = tmp_path / "demonstration.fits"
    write_source(
        path=original, redshift=redshift, densities=densities,
        include_midpoint=problem != "missing_midpoint",
    )
    extension = "absent" if problem == "wrong_extension" else "nz_source"
    with pytest.raises(ValueError, match=message):
        prepare_demo_sources(
            input_path=original, output_path=output,
            extension=extension, nbin=4,
        )
    assert not output.exists()
    assert not output.with_suffix(".yaml").exists()


@pytest.mark.parametrize("existing", ["fits", "json"])
def test_existing_outputs_are_not_overwritten(tmp_path, existing):
    """Refuse either occupied destination and retain its original bytes.

    Arguments:
      tmp_path = temporary directory; existing = destination to pre-create.
    Returns:
      None; checks that neither existing data nor the input is overwritten.
    """
    original = tmp_path / "original.fits"
    output = tmp_path / "demonstration.fits"
    write_source(path=original)
    destination = output if existing == "fits" else output.with_suffix(".yaml")
    destination.write_bytes(b"retain these original bytes")
    with pytest.raises(FileExistsError):
        prepare_demo_sources(
            input_path=original, output_path=output,
            extension="nz_source", nbin=4,
        )
    assert destination.read_bytes() == b"retain these original bytes"
    other = output.with_suffix(".yaml") if existing == "fits" else output
    assert not other.exists()


def test_input_cannot_be_an_output_or_sidecar(tmp_path):
    """Refuse both ways an output path could name the source file.

    Arguments:
      tmp_path = temporary directory containing generated inputs.
    Returns:
      None; checks direct FITS collision and sidecar collision.
    """
    original = tmp_path / "original.fits"
    write_source(path=original)
    raw_bytes = original.read_bytes()
    with pytest.raises(ValueError, match="distinct"):
        prepare_demo_sources(
            input_path=original, output_path=original,
            extension="nz_source", nbin=4,
        )
    disguised = tmp_path / "demonstration.yaml"
    disguised.write_bytes(raw_bytes)
    with pytest.raises(ValueError, match="distinct"):
        prepare_demo_sources(
            input_path=disguised, output_path=tmp_path / "demonstration.fits",
            extension="nz_source", nbin=4,
        )
    assert original.read_bytes() == raw_bytes
    assert disguised.read_bytes() == raw_bytes


def test_exclusive_creation_removes_only_its_partial_file(tmp_path):
    """Exercise a sidecar collision at the exclusive-write boundary.

    Arguments:
      tmp_path = temporary directory with a sidecar owned by another writer.
    Returns:
      None; the sidecar is preserved and the partial FITS file is removed.
    """
    output = tmp_path / "demonstration.fits"
    sidecar = tmp_path / "demonstration.yaml"
    sidecar.write_text(data="retain the existing sidecar")
    with pytest.raises(FileExistsError):
        write_pair(
            output=output, fits_content=b"prepared FITS bytes",
            sidecar=sidecar, yaml_content="{}\n",
        )
    assert not output.exists()
    assert sidecar.read_text() == "retain the existing sidecar"


def test_cli_uses_the_explicit_extension(tmp_path):
    """Run the declared command on a custom source-HDU name.

    Arguments:
      tmp_path = temporary directory for input and CLI outputs.
    Returns:
      None; verifies that CLI options reach the saved demonstration record.
    """
    original = tmp_path / "original.fits"
    output = tmp_path / "demonstration.fits"
    write_source(path=original, extension="custom_sources")
    command = [
        sys.executable,
        "-B",
        str(PROJECT / "scripts" / "prepare_demo_sources.py"),
        "--input",
        str(original),
        "--output",
        str(output),
        "--extension",
        "custom_sources",
        "--nbin",
        "4",
    ]
    result = subprocess.run(
        args=command, capture_output=True, text=True, check=True,
    )
    audit = output.with_suffix(".yaml")
    audit_contents = audit.read_text()
    record = yaml.safe_load(audit_contents)
    assert record["extension"].casefold() == "custom_sources"
    assert record["demonstration_only"] is True
    assert str(output) in result.stdout
    assert str(audit) in result.stdout
