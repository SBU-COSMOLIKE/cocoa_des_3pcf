"""Cobaya likelihood for the joint DES two-point and Map3 data vector.

The dataset names local measurements, covariance, numerical settings and a
fixed synthetic fiducial when requested. The example YAML defines the
sampled parameters and priors. Cobaya passes one parameter point to logp;
the returned number contains the likelihood alone.

This adapter retains the existing physical provider, which calls CAMB and
the Python BiHalofit producer. It does not yet obtain those arrays from a
separate Cobaya CAMB/BiHalofit dependency graph. Changing the sampler must
not silently change that scientific prediction.
"""

from pathlib import Path
import os
import sys

import numpy as np
from cobaya.likelihood import Likelihood

from .inference import read_dataset, cobaya_specification, prepare_likelihood
from .inference import DISCRETE_PARAMETERS


class JointShear(Likelihood):
    """Retain one joint prediction and covariance for each Cobaya process."""

    path = None
    data_file = None

    def initialize(self):
        """Read the dataset and declare supported parameters before setup."""
        if self.data_file is None:
            raise ValueError("JointShear requires data_file in the likelihood YAML.")
        self.dataset, self.location = read_dataset(
            data_file=self.data_file, path=self.path,
        )
        # Every MPI process owns a compiled model. Count all ranks, including
        # the coordinator, when enforcing this project's six-core allocation.
        ranks = int(os.environ.get("OMPI_COMM_WORLD_SIZE", "1"))
        if ranks * self.dataset["threads"] > 6:
            raise ValueError("MPI ranks times dataset threads must not exceed six.")
        self.joint = None
        self.record = None

    def get_can_support_params(self):
        """Name the physical and nuisance parameters accepted by the model."""
        return list(self.dataset["initial"])

    def initialize_with_provider(self, provider):
        """Construct the physical owner after Cobaya has resolved its priors."""
        super().initialize_with_provider(provider=provider)
        project = Path(__file__).resolve().parents[2]
        # The command-line prediction builder is also used by Fisher. Import
        # its constructors here; no sampler or CLI main function is called.
        sys.path.insert(0, str(project / "scripts"))
        import run_3pcf as driver

        specification = cobaya_specification(
            parameters=provider.model.info()["params"], dataset=self.dataset,
        )
        self.joint, _, _, self.record = prepare_likelihood(
            path=self.location, driver=driver, specification=specification,
        )

    def logp(self, _derived=None, **params_values):
        """Return only log likelihood; independent priors belong to Cobaya."""
        point = dict(self.joint.parameters)
        point.update(params_values)
        # Cobaya may represent fixed numeric values as floats. Native IA
        # selectors are integers; accept only exactly integral values here.
        # These fields cannot be sampled as continuous parameters.
        for name in DISCRETE_PARAMETERS:
            if name in point:
                value = point[name]
                if not np.isfinite(value) or value != int(value):
                    raise ValueError(f"Fixed selector {name}={value} must be an integer.")
                point[name] = int(value)
        if not 0.0 < point["omega_b"] < point["omega_m"] < 1.0:
            return -np.inf
        if point["h"] <= 0.0 or point["sigma8"] <= 0.0:
            return -np.inf
        return self.joint(parameters=point)

    def close(self):
        """Release this process's immutable model and covariance owners."""
        self.joint = None
