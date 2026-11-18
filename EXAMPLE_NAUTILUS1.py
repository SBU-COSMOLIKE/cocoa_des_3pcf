"""Run the joint shear inference declared in EXAMPLE_NAUTILUS1.yaml.

The YAML identifies the mean, data, priors, covariance and sampling budget.
--validate-config checks paths only; --check evaluates and restores an IA
parameter displacement; ordinary execution starts Nautilus. The shared
script saves weighted GetDist rows and records budget exhaustion explicitly.
See INFERENCE.md for the assumptions and activated-Cocoa commands.
"""

from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT / "scripts"))

from run_nautilus import main


if __name__ == "__main__":
    main(default_configuration="EXAMPLE_NAUTILUS1.yaml")
