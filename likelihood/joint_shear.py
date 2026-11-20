"""Joint DES shear two-point and third-order aperture likelihood."""

from ._cosmolike_prototype_base import _cosmolike_prototype_base


class joint_shear(_cosmolike_prototype_base):
    """Use the joint data rows and covariance selected by the dataset."""

    def initialize(self):
        """Select the joint xi+/xi-/Map3 measurement."""
        super().initialize(probe="xi_map3")
