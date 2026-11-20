"""Run the DES Y3 joint two-/three-point Nautilus example 3 (NLA with observed data).

This file follows lsst_y1/EXAMPLE_EMUL2_NAUTILUS1.py: it binds the shared
run() in cosmolike_core/cocoa_hybrid_sampling.py to this project and example.
The shared driver explains the sampler options, prior accounting and MPI.

EXAMPLE_NAUTILUS3.yaml defines the Cobaya model and its priors.
Its sampler.evaluate.override block is the fiducial evaluated by --check.
The likelihood dataset names the fixed inputs in data/. CAMB and the
matter-bispectrum theory block run through the YAML's Cobaya provider.

From activated Cocoa, check without sampling:

    python projects/des_3pcf/EXAMPLE_NAUTILUS3.py --check

Results go to projects/des_3pcf/chains/EXAMPLE_NAUTILUS3.*.
--outroot changes the name; existing results are never overwritten.
"""

from pathlib import Path
import sys

# Match the LSST launcher; Cocoa registers the likelihood separately.
project = Path(__file__).resolve().parent
core = project.parents[1]/"external_modules/code/cosmolike_core"
sys.path.insert(0, str(core))

from cocoa_hybrid_sampling import run


# Physical theory uses the same sampler, with no hybrid-emulator restrictions.
if __name__ == "__main__":
    run(
        mode="nautilus",
        project=project,
        example=3,
        hybrid=False,
    )
