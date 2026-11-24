# Input data and generated calculations

1. [Input files and their sources](#inputs)
2. [Acknowledgements](#acknowledgements)
3. [Generated calculations](#generated)
4. [Demonstration source processing](#sources)

## Input files and their sources <a name="inputs"></a>

The project includes these inputs in version control (the FITS and NPZ
files through Git LFS), so nothing needs to be downloaded before running
the examples. Every input names its origin:

| Input | Origin and pinned revision | Paper |
|---|---|---|
| `upstream_y3/`: simulated DES Y3 $`\xi_\pm`$ and Map3 vectors at the CosmoGrid cosmology, the joint covariance from 796 mocks (four cutouts of each of 199 CosmoGrid skies), and nine CosmoSIS configurations | [3pcf_integrator](https://github.com/rchgomes/3pcf_integrator) at commit `d6098d326dcb03bc750adcc9627119f52d928944`; manifest [y3_reference_manifest.yaml](y3_reference_manifest.yaml) | [Methods paper](https://arxiv.org/abs/2503.03964); simulations from [CosmoGrid](https://arxiv.org/abs/2209.04662) |
| `upstream_y3_observed/`: observed unblinded DES Y3 $`\xi_\pm`$ and Map3 vectors, the same covariance and nine configurations | Same repository and commit; manifest [y3_observed_manifest.yaml](y3_observed_manifest.yaml) | [DES Y3 third-order shear results](https://arxiv.org/abs/2508.14018) |
| `upstream_healpix/pixel_window_n1024.fits`: scalar HEALPix pixel window at NSIDE 1024 | [healpy-data](https://github.com/healpy/healpy-data) at the commit and URL in [pixel_window_manifest.yaml](pixel_window_manifest.yaml) | [HEALPix, Górski et al. 2005](https://arxiv.org/abs/astro-ph/0409513) |
| `demonstration/`: the processed source distribution, two fixed synthetic data vectors and the xi-only row mask | Produced in this project from the `upstream_y3/` inputs ([demonstration source processing](#sources)) | |
| `chains/rafael/datavector_2pt3pt/` (local, never committed): 400 measured joint $`\xi_\pm`$/Map3 vectors (`.npy`, in the released row order) and their FITS files (all bin and scale information), the four independent DES cutouts of each of 100 CosmoGrid skies at NSIDE 1024 | Supplied by Rafael Gomes on the group's server; manifest [rafael_datavector_2pt3pt.sha256](rafael_datavector_2pt3pt.sha256); downloaded by [scripts/download_rafael_sims.sh](../scripts/download_rafael_sims.sh) | [Methods paper](https://arxiv.org/abs/2503.03964); [CosmoGrid](https://arxiv.org/abs/2209.04662) |
| `chains/rafael/maps_cov_512_v2/` (local, never committed): CosmoGrid DES Y3 shear maps at NSIDE 512 with repeated noise realizations | Supplied by Rafael Gomes on the group's server; manifest [rafael_maps_cov_512_v2.sha256](rafael_maps_cov_512_v2.sha256); downloaded by the same script and described in the [map download guide](../notes/covariance_from_simulations/MAPS_README.md) | [Methods paper](https://arxiv.org/abs/2503.03964); [CosmoGrid](https://arxiv.org/abs/2209.04662) |

The manifests record the source commit, the Git blob hash and the byte
length of every copied file, and the upstream INI files document the
source analysis; they are not scripts that this project executes.
[scripts/download_y3_data.sh](../scripts/download_y3_data.sh) names all
24 DES Y3 files: it downloads each from the pinned commit and checks its
hash against the manifests. A present file with the right hash is left
alone; a mismatching file stops the script and is never overwritten.

**Step :one:**: to recreate or verify the DES Y3 inputs (11.7 MB in
total), run from `projects/des_3pcf`:

```bash
bash scripts/download_y3_data.sh
```

An optional directory argument
(`bash scripts/download_y3_data.sh /some/dir`) downloads the set
somewhere else for inspection.

**Step :two:**: Rafael Gomes' simulation products download separately,
because of their size, into the ignored `chains/rafael/`. From
`projects/des_3pcf`, choose `datavector_2pt3pt` (800 files, 21 MB),
`maps_cov_512_v2` (15,999 files, 378 GB) or `all`:

```bash
bash scripts/download_rafael_sims.sh datavector_2pt3pt
```

The script checks every file against the committed manifests named in the
table and keeps it only when its SHA-256 matches, so an interrupted
download can simply be restarted. The released joint covariance in
`upstream_y3/` came from these simulations. Every covariance there used
796 mocks, the four cutouts of each of 199 CosmoGrid skies: the joint
xi/Map3 matrix and the Map3-only matrix, which is a block of the joint
one (the measurement's author, private communication, 2026-11-23). The
COVMAT headers disagree and are typos: `NSIM=500` in
`xip_xim_map3_covariance_8ARCMINCUT_25Feb25.fits` and in the observed
`real_map3_desy3_unblind_scalecut3pcf_796simcov_2025Mar11.fits`, and
`NSIM=400` in `sim_map3_emu-NLA-cosmoCosmogrid-24FEBRUARY2025_WITH_COV.fits`.
The files are kept as released; 796 is the count to use.

The released simulated vectors and the observed measurements are different
scientific inputs. They belong in `data/`; predictions saved to detect a
change in our calculation belong in `tests/`. Neither input establishes
that this port matches the published estimator, cuts or likelihood. See
[the case table](../README.md#running_cosmolike_projects) for the examples' choices and
[the paper comparison](../README.md#estimator) for the remaining differences.
The full simulation realizations are not included, and the released
covariance does not replace the simulations needed to calibrate a
covariance correction.

## Acknowledgements <a name="acknowledgements"></a>

**We thank Rafael C. H. Gomes for making the analysis inputs available
through [3pcf_integrator](https://github.com/rchgomes/3pcf_integrator).**
The released files retain their original names and contents.

The observed vectors are the Dark Energy Survey collaboration's Y3
measurement, built on its Y3 shape catalogue; credit the collaboration
and cite the [third-order shear results](https://arxiv.org/abs/2508.14018)
when using them. The simulated vectors, the 796-mock covariance, the measured
simulation vectors and the local shear maps come from CosmoGrid-based
DES Y3 simulations
([Kacprzak et al.](https://arxiv.org/abs/2209.04662)). The pixel window
is the official HEALPix table distributed with healpy
([Górski et al. 2005](https://arxiv.org/abs/astro-ph/0409513)).

## Generated calculations <a name="generated"></a>

No preparation command exists: the Cobaya model computes CAMB and the
matter bispectrum at each evaluation, so there is no saved theory table
to create first. Generated products are written under the project's
ignored `chains/` directory by the notebooks that explain them.

| Generated file | How to create it |
|---|---|
| `chains/independent_pilot_compression.npz` | Run [EXAMPLE_COMPRESSION.ipynb](../EXAMPLE_COMPRESSION.ipynb) including its direct TATT derivative calculation; the case-4 datasets read this matrix |
| `chains/conditional_xi_compression.npz` | Run the conditional-xi section of [EXAMPLE_COMPRESSION.ipynb](../EXAMPLE_COMPRESSION.ipynb) |
| `chains/joint_feedback/` | Run [EXAMPLE_JOINT_FEEDBACK.ipynb](../EXAMPLE_JOINT_FEEDBACK.ipynb) |
| `chains/EXAMPLE_EVALUATE1_fisher.npz` | Run the Fisher section of [EXAMPLE_EVALUATE1.ipynb](../EXAMPLE_EVALUATE1.ipynb) |

The compression matrix is trained from the chosen derivatives using an
independent teaching covariance; it is not the published compression.
Generated calculations stay in `chains/`; frozen numerical comparisons,
including timing vectors, stay in `tests/`. Sampler chains use their
configured chain directory.

## Demonstration source processing <a name="sources"></a>

The included `demonstration/y3_demo_sources.fits` is produced from the
`NZ_SOURCE` extension of
`upstream_y3/data/dv/sim_2pt-NLA-cosmoCosmogrid-04Nov24.fits`: it clips
negative samples and renormalizes each source bin. Its YAML record
contains the input and output hashes and the amount removed. This is the
explicitly approved teaching assumption, not a recalibration of the DES
source model. Recorded absolute paths describe where the processing
occurred; consumers resolve the included file from their YAML
configuration and check its hash.

Reproduce the processing in the source-distribution section of
[EXAMPLE_EVALUATE1.ipynb](../EXAMPLE_EVALUATE1.ipynb). After its setup cell,
you can jump directly to that section without running a cosmology or Fisher
calculation. It shows clipping, the removed area and normalization, and
checks that the resulting samples exactly match the included file;
`tests/test_prepare_demo_sources.py` executes those same cells.
An optional cell writes a FITS copy under `chains/`; it never replaces the
input in `data/`. Changing the source model of an inference requires changing
`nz_source_file` in its `.dataset` file explicitly.

The other demonstration files are this project's own:
`y3_nla_synthetic.txt` and `y3_tatt_synthetic.txt` are fixed synthetic
fiducial mean vectors whose headers state their fiducial point and IA
convention, and `y3_xi_rows_mask.txt` is the xi-only row mask that
`des_y3_xi1.dataset` reads to cut the Map3 rows.
