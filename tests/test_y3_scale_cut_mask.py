"""Check the DES Y3 xi scale-cut mask against its source.

data/y3_xi_scale_cut_mask.txt holds one 0/1 flag per row of the joint data
vector (200 xi+, 200 xi-, then 80 Map3), in the likelihood's row order. Its
xi flags come from the DES Y3 cosmic-shear scale cuts of the CosmoSIS
standard library (data/upstream_cosmosis/des-y3-scale-cuts.ini): a row is
kept when its angle ANG lies inside the [minimum, maximum] range given for
its bin pair, as CosmoSIS's 2pt likelihood does. Every Map3 row is kept,
because the 8-arcmin Map3 cut is already applied to the released rows.
"""

import configparser
from pathlib import Path

import numpy as np
from astropy.io import fits

DATA = Path(__file__).resolve().parents[1] / "data"


def test_mask_follows_the_y3_cosmic_shear_cuts():
    """Rebuild the mask from the cut file and the FITS rows; 227 xi rows kept."""
    # Step 1: the angle ranges, one per (correlation, bin pair).
    cuts = configparser.ConfigParser()
    cuts.read(DATA / "upstream_cosmosis/des-y3-scale-cuts.ini")
    ranges = cuts["2pt_like"]

    # Step 2: flag each xi row of the released file, in file order.
    flags = []
    measurement = DATA / "upstream_y3/data/dv/sim_2pt-NLA-cosmoCosmogrid-04Nov24.fits"
    for extension in ("xip", "xim"):
        for row in fits.getdata(measurement, extname=extension):
            key = f"angle_range_{extension}_{row['BIN1']}_{row['BIN2']}"
            minimum, maximum = map(float, ranges[key].split())
            flags.append(int(minimum <= row["ANG"] <= maximum))

    # Step 3: all Map3 rows follow, then compare with the shipped mask.
    flags += [1] * 80
    shipped = np.loadtxt(DATA / "y3_xi_scale_cut_mask.txt", dtype=int)
    np.testing.assert_array_equal(shipped, flags)
    assert 227 == int(np.sum(shipped[:400]))
