"""Fetch the pinned public Y3 reference inputs used to validate this port.

Rafael Gomes's 3pcf_integrator contains several generations of simulated
and observed data. The checked-in manifest selects the files named by
one 796-simulation joint NLA configuration, including its exact settings.
Downloading them does not establish that every setting matches the final
paper. The comparison record must check the scale cuts, sign convention,
compression and likelihood separately.

Run from any directory with the existing Cocoa Python:
  python projects/des_3pcf/scripts/fetch_y3_reference.py

Only the listed public files are fetched, about 8.5 MB in total. Git blob
hashes are checked before writing. Downloaded scientific data are kept in
the project's ignored data/upstream_y3 directory; their original repository
and revision remain recorded in data/y3_reference_manifest.json.
"""

import hashlib
import json
from pathlib import Path
import urllib.request


PROJECT = Path(__file__).resolve().parents[1]  # Owns the manifest and local data.


def blob_hash(content):
    """Calculate Git's blob identifier for downloaded bytes.

    Git hashes 'blob <length>\0' followed by the exact contents. Checking
    this identifier ties a file to the pinned source tree rather than
    merely trusting its remote filename.

    Arguments:
      content = raw file bytes.
    Returns:
      forty-character SHA-1 Git blob identifier.
    """
    header = f"blob {len(content)}\0".encode(encoding="ascii")
    return hashlib.sha1(header + content).hexdigest()


def main():
    """Download or verify each exact manifest file without executing it.

    Returns:
      nothing; prints the count and destination after all checks pass.
    Raises:
      ValueError for an unexpected repository, unsafe path or content hash.
      OSError for a network or filesystem failure.
    Side effects:
      Creates only the manifest's paths beneath data/upstream_y3.
      Existing files are read and verified, never silently overwritten.
    """
    manifest_path = PROJECT / "data" / "y3_reference_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["repository"] != "https://github.com/rchgomes/3pcf_integrator":
        raise ValueError("The reference manifest must name Rafael's 3pcf_integrator.")
    commit = manifest["commit"]
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("The reference revision must be a full hexadecimal Git commit.")
    destination = PROJECT / "data" / "upstream_y3"
    for item in manifest["files"]:
        relative = Path(item["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"Reference path must stay below the data folder: {relative}"
            )
        output = destination / relative
        if output.exists():
            content = output.read_bytes()
        else:
            url = (
                "https://raw.githubusercontent.com/rchgomes/3pcf_integrator/"
                + commit + "/" + relative.as_posix()
            )
            # Reading one byte beyond the recorded length detects an
            # unexpected replacement without accepting an unbounded download.
            with urllib.request.urlopen(url=url, timeout=30) as response:
                content = response.read(item["bytes"] + 1)
        if (len(content) != item["bytes"]
                or blob_hash(content=content) != item["git_blob_sha1"]):
            raise ValueError(
                f"Reference hash/length mismatch for {relative}; no file written."
            )
        if not output.exists():
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open(mode="xb") as stream:
                stream.write(content)
    print(f"Verified {len(manifest['files'])} pinned files in {destination}")


if __name__ == "__main__":
    main()
