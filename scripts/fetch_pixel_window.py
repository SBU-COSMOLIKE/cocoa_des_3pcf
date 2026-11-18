"""Fetch or verify the pinned 72-kB scalar HEALPix window for the Y3 comparison.

Run explicitly from Cocoa with:
  python projects/des_3pcf/scripts/fetch_pixel_window.py

The manifest pins the upstream commit, byte count and SHA256. Existing
files must match it; this script refuses to overwrite different bytes.
Physical evaluation only reads that local file and never downloads it.
The HEALPix authors and healpy maintainers provide the original table.
"""

import hashlib
import json
from pathlib import Path
from urllib.request import urlopen


def main():
    """Verify bytes before writing the declared local comparison input.

    Side effects:
      Downloads only the manifest's pixel-window file when absent, then
      creates its local directory and writes the verified bytes exclusively.
    """
    project = Path(__file__).resolve().parents[1]
    manifest_path = project / "data" / "pixel_window_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    destination = project / "data" / manifest["path"]
    existing = destination.is_file()
    if existing:
        content = destination.read_bytes()
    else:
        with urlopen(url=manifest["url"], timeout=30) as response:
            content = response.read(manifest["bytes"] + 1)
    digest = hashlib.sha256(content).hexdigest()
    if len(content) != manifest["bytes"] or digest != manifest["sha256"]:
        raise ValueError("Pixel-window size or SHA256 differs from the pinned manifest.")
    if not existing:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open(mode="xb") as stream:
            stream.write(content)
    print(f"Verified {destination}: {len(content)} bytes, SHA256 {digest}")


if __name__ == "__main__":
    main()
