"""DES two- and three-point calculations using the isolated CosmoLike port.

The compiled extension owns the numerical integrals. This Python package
joins its arrays to survey labels, prepares cosmological input tables, and
calculates parameter information from derivatives of the mean. Importing
the package does not run CAMB, change an existing CosmoLike cosmology, or
load a survey file. The same functions serve DES Y3 and Y6; their data,
selection cuts and priors must be supplied separately.
"""

from .accuracy import Accuracy
