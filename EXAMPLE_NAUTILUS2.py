"""Run the DES Y3 joint two-/three-point Nautilus example 2 (TATT with synthetic data).

This file follows lsst_y1/EXAMPLE_EMUL2_NAUTILUS1.py: it binds the shared
run() in cosmolike_core/cocoa_hybrid_sampling.py to this project and example.
The shared driver explains the sampler options, prior accounting and MPI.

EXAMPLE_NAUTILUS2.yaml defines the Cobaya model and its priors.
Its sampler.evaluate.override block is the fiducial evaluated by --check.
The likelihood dataset names the inputs in data/ and the prediction YAML;
prepare that YAML's theory tables in runs/ before checking or sampling.

From activated Cocoa, check without sampling:

    python projects/des_3pcf/EXAMPLE_NAUTILUS2.py --check

Results go to projects/des_3pcf/chains/EXAMPLE_NAUTILUS2.*.
--outroot changes the name; existing results are never overwritten.
"""

from pathlib import Path
import sys

# Match the LSST launcher. The additional Python and interface directories
# locate this project's physical likelihood and compiled 3PCF module.
project = Path(__file__).resolve().parent
core = project.parents[1]/"external_modules/code/cosmolike_core"
sys.path.insert(0, str(project / "python"))
sys.path.insert(0, str(project / "interface"))
sys.path.insert(0, str(core))

from cocoa_hybrid_sampling import run


# Physical theory uses the same sampler, with no hybrid-emulator restrictions.
if __name__ == "__main__":
    run(mode="nautilus", project=project, example=2, hybrid=False)
