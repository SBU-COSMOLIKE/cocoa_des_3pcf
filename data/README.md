# Input data and generated calculations

1. [Input files](#inputs)
2. [Generated theory tables and forecasts](#generated)
3. [Demonstration source processing](#sources)

## Input files <a name="inputs"></a>

The project includes these inputs in version control. No Python download
script is needed.

| Directory | Contents and purpose |
|---|---|
| `upstream_y3/` | Released simulated vectors, source distributions, joint covariance and the upstream configurations that identify their use |
| `upstream_y3_observed/` | Observed unblinded DES Y3 two-point and Map3 vectors, their joint covariance and identifying configurations |
| `upstream_healpix/` | Scalar HEALPix pixel window at NSIDE 1024 |
| `demonstration/` | The processed source distribution and its YAML record |

**We thank Rafael Gomes for making the analysis inputs available through
[3pcf_integrator](https://github.com/rchgomes/3pcf_integrator).** The released
files retain their original names and contents. The reference and observed
YAML manifests record the source commit, Git blob hashes and byte lengths.
The pixel-window manifest identifies the HEALPix/healpy source and hash.
The upstream INI files document the source analysis; they are not scripts
that this project executes.

The released simulated vectors and the observed measurements are different
scientific inputs. They belong in `data/`; predictions saved to detect a
change in our calculation belong in `tests/`. Neither input establishes
that this port matches the published estimator, cuts or likelihood. See
[the inference guide](../README.md#sampling) for the examples' choices and
[the paper comparison](../README.md#estimator) for the remaining differences.

The full simulation realizations are not included. These released vectors
and covariance matrices do not replace the simulations needed to calibrate
a covariance correction.

## Generated theory tables and forecasts <a name="generated"></a>

We assume Cocoa is activated and the shell is in `Cocoa/`. Generated files
are written under the project's ignored `chains/` directory. Each configuration
sets its destination in `physics.tables` and `outputs`.

**Step :one:**: prepare the fiducial tables shared by the NLA and TATT examples.

```bash
python projects/des_3pcf/scripts/run_3pcf.py prepare \
  projects/des_3pcf/configs/des_y3_nla_demo_fisher.yaml
```

This writes `chains/y3_demo_pchip9601_physics.npz`. A likelihood varying
cosmology recomputes its prediction; this saved fiducial table is not a
substitute for that calculation. Existing files require explicit
`--overwrite`. Recompute dependent results after changing the inputs or
numerical settings; do not relabel an old forecast as a current result.

**Step :two:**: prepare the separate neutrino tables for a BACCO calculation.

```bash
python projects/des_3pcf/scripts/run_3pcf.py prepare \
  projects/des_3pcf/configs/des_y3_bacco_fisher.yaml
```

This writes `chains/y3_bacco_physics.npz`. The [BACCO guide](../../../external_modules/code/cosmolike_core/theory/bacco_bispectrum/README.md#bacco)
describes its emulator inputs and physical assumptions.

| Generated file | How to create it |
|---|---|
| `chains/paired_feedback_demo.npz` | Run the response-construction cell in [EXAMPLE_JOINT_FEEDBACK.ipynb](../EXAMPLE_JOINT_FEEDBACK.ipynb) |
| `chains/y3_tatt_measured_edge_demo_fisher.npz` | Run `run_3pcf.py fisher` with `configs/des_y3_tatt_demo_fisher.yaml` after preparation |
| `chains/independent_pilot_compression.npz` | Run [EXAMPLE_COMPRESSION.ipynb](../EXAMPLE_COMPRESSION.ipynb) after generating its TATT Fisher input |

The artificial feedback response illustrates the calculation; it is not a
hydrodynamic calibration. The compression matrix is trained from the chosen
derivatives using an independent teaching covariance; it is not the
published compression. Both are generated calculations and stay in `chains/`.
Frozen numerical comparisons, including timing vectors, stay in `tests/`.
Figures use `chains/`; sampler chains use their configured chain directory.

## Demonstration source processing <a name="sources"></a>

The included `demonstration/y3_demo_sources.fits` clips negative samples
and renormalizes each source bin. Its YAML record contains the input and
output hashes and the amount removed. This is the explicitly approved
teaching assumption, not a recalibration of the DES source model.
Recorded absolute paths describe where the processing occurred; consumers
resolve the included file from their YAML configuration and check its hash.

To reproduce the processing without modifying the input data:

```bash
python projects/des_3pcf/scripts/prepare_demo_sources.py \
  --input projects/des_3pcf/data/upstream_y3/data/dv/sim_2pt-NLA-cosmoCosmogrid-04Nov24.fits \
  --output projects/des_3pcf/chains/y3_demo_sources.fits \
  --extension nz_source --nbin 4
```

The command writes a FITS file and YAML record under `chains/`. The included
processed distribution remains the examples' declared input. To adopt a
different source model, change `source.path` explicitly and prepare new
theory tables.
