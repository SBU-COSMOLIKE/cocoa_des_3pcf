"""Fetch the pinned observed DES Y3 vectors paired by Rafael Gomes's analysis.

The selected joint NLA configuration names an unblinded two-point vector,
an unblinded Map3 vector, and their simulation-estimated joint covariance.
The companion data paper is https://arxiv.org/abs/2508.14018v2. These files
contain measurements; they are separate from the simulated reference in
fetch_y3_reference.py. Their availability does not establish that this
port's angular cuts, bin averages or likelihood reproduce the paper.

Run from the activated Cocoa directory:
  python projects/des_3pcf/scripts/fetch_y3_observed.py

Git blob hashes and byte lengths are checked before writing. Downloaded
files stay under the ignored data/upstream_y3_observed directory. The
manifest data/y3_observed_manifest.json records their repository, commit,
filenames and measurement provenance. Existing files are verified and
never silently replaced. No numerical model or sampler is run.
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
      Creates only the manifest's paths beneath data/upstream_y3_observed.
      Existing files are read and verified, never silently overwritten.
    """
    manifest_path = PROJECT / "data" / "y3_observed_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["repository"] != "https://github.com/rchgomes/3pcf_integrator":
        raise ValueError("The reference manifest must name Rafael's 3pcf_integrator.")
    commit = manifest["commit"]
    if len(commit) != 40 or any(
        character not in "0123456789abcdef" for character in commit
    ):
        raise ValueError("The reference revision must be a full hexadecimal Git commit.")
    destination = PROJECT / "data" / "upstream_y3_observed"
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
