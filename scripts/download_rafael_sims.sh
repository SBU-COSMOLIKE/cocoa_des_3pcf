#!/bin/bash
# ------------------------------------------------------------------------------
# download_rafael_sims.sh: download Rafael Gomes' CosmoGrid DES Y3
# simulation products from the group's server and check every file against
# the SHA-256 manifests committed in data/.
#
# Usage (from projects/des_3pcf):
#
#   bash scripts/download_rafael_sims.sh datavector_2pt3pt   # 800 files, 21 MB
#   bash scripts/download_rafael_sims.sh maps_cov_512_v2     # 15,999 files, 378 GB
#   bash scripts/download_rafael_sims.sh all
#
# Files go to chains/rafael/<folder>/ (ignored by Git), the place the
# notebooks and scripts read them from; a second argument chooses another
# root folder. The two folders are:
#
#   datavector_2pt3pt  400 joint xi_+/xi_-/Map3 vectors (.npy, in the
#                      released row order) and their FITS files (all bins
#                      and scales), measured on NSIDE 1024 maps: 4
#                      independent DES cutouts of each of 100 CosmoGrid
#                      full-sky simulations (the paper used 199 x 4 = 796).
#   maps_cov_512_v2    the NSIDE 512 shear maps of the same skies with extra
#                      noise realizations (run = parent lightcone, ncut =
#                      cutout, nreal = noise realization; RING ordering),
#                      plus indices_512.npy (footprint pixels) and
#                      missing.txt. See
#                      notes/covariance_from_simulations/MAPS_README.md.
#
# Each manifest line is "<sha256>  <folder>/<file>  <bytes>". A file is kept
# only after its SHA-256 matches; an existing file with the right hash is
# skipped, so an interrupted download can simply be restarted. An existing
# file with a different hash stops the script; it is never overwritten.
# ------------------------------------------------------------------------------

SERVER="http://138-128-240-126.cloud-xip.com/Rafael"
WHICH="${1:-}"
ROOT="${2:-chains/rafael}"

fail () {
  echo "download_rafael_sims.sh: ${1}" >&2
  exit 1
}

case "${WHICH}" in
  datavector_2pt3pt|maps_cov_512_v2) FOLDERS=("${WHICH}") ;;
  all) FOLDERS=("datavector_2pt3pt" "maps_cov_512_v2") ;;
  *) fail "choose datavector_2pt3pt, maps_cov_512_v2 or all." ;;
esac

command -v curl >/dev/null 2>&1 || fail "curl is not installed."

# sha256sum on Linux, shasum -a 256 on macOS; both print the hash first.
if command -v sha256sum >/dev/null 2>&1; then
  HASH=(sha256sum)
elif command -v shasum >/dev/null 2>&1; then
  HASH=(shasum -a 256)
else
  fail "neither sha256sum nor shasum is installed."
fi

for FOLDER in "${FOLDERS[@]}"; do
  MANIFEST="data/rafael_${FOLDER}.sha256"
  [ -f "${MANIFEST}" ] || fail "missing ${MANIFEST}; run from projects/des_3pcf."
  mkdir -p "${ROOT:?}/${FOLDER:?}" || fail "cannot create ${ROOT}/${FOLDER}."
  COUNT=0

  while read -r EXPECTED NAME SIZE; do
    TARGET="${ROOT:?}/${NAME:?}"

    # An existing file is kept if it is the recorded one.
    if [ -f "${TARGET}" ]; then
      if [ "$("${HASH[@]}" "${TARGET}" | cut -d' ' -f1)" == "${EXPECTED}" ]; then
        COUNT=$((COUNT + 1))
        continue
      fi
      fail "${TARGET} exists with a different SHA-256; move it away first."
    fi

    # Download next to the target, check the hash, then rename it in place.
    PARTIAL="${TARGET}.partial"
    curl -fsS --retry 3 "${SERVER}/${NAME}" -o "${PARTIAL}" \
      || fail "download failed: ${SERVER}/${NAME}"
    if [ "$("${HASH[@]}" "${PARTIAL}" | cut -d' ' -f1)" != "${EXPECTED}" ]; then
      rm -f "${PARTIAL:?}"
      fail "${NAME} does not match its SHA-256 in ${MANIFEST}."
    fi
    mv "${PARTIAL}" "${TARGET}" || fail "cannot write ${TARGET}."
    COUNT=$((COUNT + 1))
  done < "${MANIFEST}"

  echo "${FOLDER}: ${COUNT} files match ${MANIFEST}."
done
