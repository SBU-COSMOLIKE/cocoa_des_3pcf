"""Defines the Cobaya likelihood des_3pcf.joint_shear (joint DES shear).

The entry `des_3pcf.joint_shear` in the likelihood block of a yaml file
makes Cobaya import this module and build the class of the same name,
with the options and defaults of joint_shear.yaml (next to this file).
The base class _cosmolike_prototype_base does all the work (its module
docstring explains the data vector, the probes and the order of the
calls); this subclass selects the probe "xi_map3": the joint xi_+, xi_-
and third-order aperture-mass (Map3) rows.
"""

from ._cosmolike_prototype_base import _cosmolike_prototype_base


class joint_shear(_cosmolike_prototype_base):
    """Evaluates the joint shear likelihood (probe "xi_map3")."""

    def initialize(self):
        """Configure the base likelihood for probe "xi_map3".

        Cobaya calls initialize once, when it builds the likelihood;
        super() reaches the method of the base class, which reads the
        .dataset file and sets up cosmolike.
        """
        super().initialize(probe="xi_map3")
