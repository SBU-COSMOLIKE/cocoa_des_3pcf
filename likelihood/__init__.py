# Marks this directory as a Python package. Cocoa links
# projects/des_3pcf/likelihood into Cobaya's likelihoods directory under
# the name des_3pcf, so Python imports it as cobaya.likelihoods.des_3pcf.
# The package holds the likelihood module joint_shear (with its yaml file
# of option defaults, joint_shear.yaml), the shared base class
# _cosmolike_prototype_base.py, and compression.py, which trains and
# stores the optional fixed linear compression of the data vector.
