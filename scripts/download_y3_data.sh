#!/bin/bash
# ------------------------------------------------------------------------------
# download_y3_data.sh: download the DES Y3 files this project reads, from
# Rafael Gomes' public repository rchgomes/3pcf_integrator at one pinned
# commit, and check each one against the git hash recorded in the project's
# manifests (data/y3_reference_manifest.yaml, data/y3_observed_manifest.yaml).
#
# Usage (from projects/des_3pcf):
#
#   bash scripts/download_y3_data.sh            # into data/
#   bash scripts/download_y3_data.sh /some/dir  # into /some/dir instead
#
# The files are already committed to this project; the script documents
# where they come from and lets anyone recreate or verify them. Two sets:
#
#   upstream_y3          simulated DES Y3 xi and Map3 vectors at the
#                        CosmoGrid cosmology, the joint xi/Map3 covariance
#                        from 796 simulations, and the CosmoSIS configs that
#                        used them (arXiv:2503.03964). The numbered examples
#                        read the row labels, angles and covariance here.
#   upstream_y3_observed the observed, unblinded DES Y3 xi and Map3
#                        vectors, the same covariance and their configs
#                        (arXiv:2508.14018).
#
# A file is written only after its git blob hash matches the manifest. An
# existing file with the right hash is left alone; one with a different hash
# stops the script (it is never overwritten silently).
# ------------------------------------------------------------------------------

REPOSITORY="https://raw.githubusercontent.com/rchgomes/3pcf_integrator"
COMMIT="d6098d326dcb03bc750adcc9627119f52d928944"
DESTINATION="${1:-data}"

# Every file: local folder, path inside the upstream repository, and the git
# blob hash the manifest records for it. --no-filters below hashes the raw
# bytes: this project stores *.fits with Git LFS, and without it git would
# hash the small LFS pointer instead of the file.
FILES=(
  "upstream_y3 data/dv/sim_2pt-NLA-cosmoCosmogrid-04Nov24.fits 574553f15e301a6665e1d658ac326c8a5224e9cc"
  "upstream_y3 data/dv/sim_map3_emu-NLA-cosmoCosmogrid-24FEBRUARY2025_WITH_COV.fits 5e26a7bafd42fdcbe837b2eb0639f18c3845ae5a"
  "upstream_y3 data/covariance/xip_xim_map3_covariance_8ARCMINCUT_25Feb25.fits 963bb0fcb42c00e9b7b47d4a87c99efc00edd688"
  "upstream_y3 config/3pt.ini f0d69f9b8c9488f39b75b5dd0fc026bab2690ed6"
  "upstream_y3 config/joint.ini 621fb3cdb0ae63d1d99e681602b7c851418a4916"
  "upstream_y3 config/cosmic-shear.ini a9b2b0016c61fb294291387a01e448fc1afcdeb2"
  "upstream_y3 config/moped.ini 56dc9e7bcc97f0d6fe045b98a050256f7e1dc5a4"
  "upstream_y3 config/astrophysical.ini 103e239bbd2e2d59bfa068ba7a6eecf2d83bd81d"
  "upstream_y3 config/cosmo.ini 047df7629876b40a5d5bccc3a2d8b818793318ac"
  "upstream_y3 config/linear-power.ini e9a1aae22da192803a9ad1f1ae01c929c158a1eb"
  "upstream_y3 config/test-cosmogridc-simdv-joint-2pt-moped-run-796sims-february-Percival/des-y3-shear-values.ini 0c6ff8feb7364066d78ab6d8f51ac69cd95c6aa6"
  "upstream_y3 config/test-cosmogridc-simdv-joint-2pt-moped-run-796sims-february-Percival/des-y3-shear-2pt-map3-NLA-all.ini f10b87a7fbc60f72e4ebd64e896ef904ecfc4f78"
  "upstream_y3_observed data/dv/real_2pt_desy3_unblind_g2flip_matchElisa_2025Mar10.fits f782d0eabf7265a35bc1dd1d6725f7de16a3f7a8"
  "upstream_y3_observed data/dv/real_map3_desy3_unblind_scalecut3pcf_796simcov_2025Mar11.fits bc296c07b48b7d3f7be37d8fa29289fb25df1dd2"
  "upstream_y3_observed data/covariance/xip_xim_map3_covariance_8ARCMINCUT_25Feb25.fits 963bb0fcb42c00e9b7b47d4a87c99efc00edd688"
  "upstream_y3_observed config/3pt.ini f0d69f9b8c9488f39b75b5dd0fc026bab2690ed6"
  "upstream_y3_observed config/joint.ini 621fb3cdb0ae63d1d99e681602b7c851418a4916"
  "upstream_y3_observed config/cosmic-shear.ini a9b2b0016c61fb294291387a01e448fc1afcdeb2"
  "upstream_y3_observed config/moped.ini 56dc9e7bcc97f0d6fe045b98a050256f7e1dc5a4"
  "upstream_y3_observed config/astrophysical.ini 103e239bbd2e2d59bfa068ba7a6eecf2d83bd81d"
  "upstream_y3_observed config/cosmo.ini 047df7629876b40a5d5bccc3a2d8b818793318ac"
  "upstream_y3_observed config/linear-power.ini e9a1aae22da192803a9ad1f1ae01c929c158a1eb"
  "upstream_y3_observed config/unblind-metacal-796sims-february-Percival-2ptfix-3pcfsc/des-y3-shear-values.ini d6a1d9e58d4220cca0f14241a795741601e325a2"
  "upstream_y3_observed config/unblind-metacal-796sims-february-Percival-2ptfix-3pcfsc/unblind-metacal-shear-2pt-map3-NLA-all.ini cbf1d7b0f0538443e5c9f89dcb1ecee0bc0f7ded"
)

fail () {
  echo "download_y3_data.sh: ${1}" >&2
  exit 1
}

command -v curl >/dev/null 2>&1 || fail "curl is not installed."
command -v git >/dev/null 2>&1 || fail "git is needed for git hash-object."

TEMPORARY=$(mktemp -d) || fail "cannot create a temporary folder."
trap 'rm -rf "${TEMPORARY:?}"' EXIT

for ENTRY in "${FILES[@]}"; do
  read -r FOLDER UPSTREAM HASH <<< "${ENTRY}"
  TARGET="${DESTINATION:?}/${FOLDER:?}/${UPSTREAM:?}"

  # An existing file is kept if it is the recorded one, and stops the run
  # if it is not.
  if [ -f "${TARGET}" ]; then
    if [ "$(git hash-object --no-filters "${TARGET}")" == "${HASH}" ]; then
      echo "present  ${FOLDER}/${UPSTREAM}"
      continue
    fi
    fail "${TARGET} exists with a different git hash; move it away first."
  fi

  # Download into the temporary folder, check the hash, then move it in.
  DOWNLOAD="${TEMPORARY:?}/file"
  curl -fsSL "${REPOSITORY}/${COMMIT}/${UPSTREAM}" -o "${DOWNLOAD}" \
    || fail "download failed: ${REPOSITORY}/${COMMIT}/${UPSTREAM}"
  if [ "$(git hash-object --no-filters "${DOWNLOAD}")" != "${HASH}" ]; then
    fail "${UPSTREAM} does not match its recorded git hash ${HASH}."
  fi
  mkdir -p "$(dirname "${TARGET}")" || fail "cannot create $(dirname "${TARGET}")."
  mv "${DOWNLOAD}" "${TARGET}" || fail "cannot write ${TARGET}."
  echo "fetched  ${FOLDER}/${UPSTREAM}"
done

echo "All ${#FILES[@]} files match the manifests (commit ${COMMIT})."
