"""Check shared-pixel Map3 polarization on tiny spherical catalogues.

No production map is opened. Direct triangle enumeration supplies an
independent oracle, including the source-permutation normalization. The
multipole check tests cubic linearity, not angular-truncation convergence.
"""
import os
for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS'):
    os.environ[name] = '1'
import ast
from itertools import combinations, combinations_with_replacement
from pathlib import Path
import numpy as np
import treecorr

source = Path(__file__).resolve().parents[2] / 'scripts/measure_full_joint_mpi.py'
tree = ast.parse(source.read_text())
function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == 'polarized_map3')
environment = dict(np=np, treecorr=treecorr)
exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'),
     environment)
measure = environment['polarized_map3']
rng = np.random.default_rng(221129)
ra = rng.uniform(-45, 45, 24) / (60 * np.cos(np.deg2rad(-35)))
dec = -35 + rng.uniform(-45, 45, 24) / 60
# TreeCorr stores weighted shears as complex<float>. Dyadic inputs and their
# small integer sums are exactly representable, separating algebra errors
# from the rounding introduced by constructing a combined shear catalogue.
shear = rng.integers(-32, 33, size=(4, 2, 24)) / 1024.
def catalog(values):
    return treecorr.Catalog(ra=ra, dec=dec, ra_units='deg', dec_units='deg',
                           g1=values[0], g2=values[1], flip_g2=True)
catalogs = [catalog(g) for g in shear]
triples = list(combinations_with_replacement(range(4), 3))
radii = np.array([7., 14., 25., 40.])
edges = np.geomspace(.5, 80., 21)
first = np.flatnonzero(edges >= 8.)[0]
configuration = dict(min_sep=edges[first], max_sep=80., nbins=20-first,
                     sep_units='arcmin', metric='Arc', bin_slop=0., angle_slop=0.)
angular = dict(min_phi=.016, max_phi=3.126, nphi_bins=63, phi_units='radians')
unit = np.eye(4, dtype=int)
keys = [tuple(u) for u in unit]
for i, j in combinations(range(4), 2):
    keys.extend((tuple(unit[i]+unit[j]), tuple(unit[i]-unit[j])))
for i, j, k in combinations(range(4), 3):
    keys.append(tuple(unit[i]+unit[j]+unit[k]))
assert len(keys) == 20

# Fill the same small cache with direct-SAS auto measurements, then exercise
# the production polarization coefficients against explicit mixed triples.
dyadic_shear = shear.copy()
random_shear = np.random.default_rng(39211).normal(0., .03, shear.shape)
# Ordinary inputs also test the small rounding difference between a summed
# field and separately stored fields. TreeCorr's complex<float> leaf storage
# prevents machine-precision equality in this case. Test a normwise error;
# nearly zero individual moments do not define useful relative tolerances.
for shear, tolerance in ((dyadic_shear, 2e-12), (random_shear, 3e-7)):
    catalogs = [catalog(g) for g in shear]
    cache = {}
    for key in keys:
        corr = treecorr.GGGCorrelation(bin_type='LogSAS', **configuration, **angular)
        corr.process(catalog(np.einsum('i,ijk->jk', key, shear)),
                     num_threads=1, algo='triangle')
        cache[key] = corr.calculateMap3(R=radii)[0]
    predictions = []
    references = []
    for triple in triples:
        predictions.append(measure(catalogs, triple, radii, {}, cache))
        i, j, k = triple
        corr = treecorr.GGGCorrelation(bin_type='LogSAS', **configuration, **angular)
        if i == k:
            corr.process(catalogs[i], num_threads=1, algo='triangle')
        elif i == j:
            corr.process(catalogs[k], catalogs[i], ordered=False,
                         num_threads=1, algo='triangle')
        elif j == k:
            corr.process(catalogs[i], catalogs[j], ordered=False,
                         num_threads=1, algo='triangle')
        else:
            corr.process(catalogs[i], catalogs[j], catalogs[k], ordered=False,
                         num_threads=1, algo='triangle')
        references.append(corr.calculateMap3(R=radii)[0])
    error = np.linalg.norm(np.array(predictions)-references)/np.linalg.norm(references)
    assert error < tolerance, error
    print('Direct spherical SAS relative difference:', error, 'tolerance:', tolerance)

shear = dyadic_shear
catalogs = [catalog(g) for g in shear]

# Now exercise the actual multipole auto calls and cache. The recovered
# symmetric coefficients must predict a fresh auto measurement of any
# linear combination. This also tests the radian coordinates and shear sign
# used when production creates the combined catalogues.
configuration.update(bin_type='LogMultipole', max_n=100)
cache.clear()
values = [measure(catalogs, t, radii, configuration, cache) for t in triples]
assert len(cache) == 20
assert all(value.shape == (4,) for value in cache.values())
for coefficients in (np.array([1, 2, -1, 1]), np.array([2, -1, 3, -2])):
    predicted = np.zeros(4)
    for (i, j, k), value in zip(triples, values):
        multiplicity = 1 if i == k else (3 if i == j or j == k else 6)
        predicted += multiplicity*coefficients[i]*coefficients[j]*coefficients[k]*value
    corr = treecorr.GGGCorrelation(**configuration)
    corr.process(catalog(np.einsum('i,ijk->jk', coefficients, shear)), num_threads=1)
    actual = corr.toSAS(**angular).calculateMap3(R=radii)[0]
    error = np.linalg.norm(predicted-actual)/np.linalg.norm(actual)
    assert error < 2e-11, error
    print('Multipole cubic reconstruction relative difference:', error)
# A resumed map can request only a subset of triples and can reach cached
# sums in a different order. It must reproduce the uninterrupted calculation
# exactly. Use ordinary random shears for these actual multipole calls.
shear = random_shear
catalogs = [catalog(g) for g in shear]
cache.clear()
full = np.array([measure(catalogs, t, radii, configuration, cache)
                 for t in triples])
assert len(cache) == 20
cache.clear()
shuffled = np.empty_like(full)
for index in np.random.default_rng(923).permutation(len(triples)):
    shuffled[index] = measure(catalogs, triples[index], radii, configuration, cache)
np.testing.assert_array_equal(shuffled, full)
assert len(cache) == 20
for unfinished in ([1], [5, 19], [19, 18, 4, 11]):
    cache.clear()
    partial = np.array([measure(catalogs, triples[index], radii, configuration, cache)
                        for index in unfinished])
    np.testing.assert_array_equal(partial, full[unfinished])

# A worker clears its dictionary when it loads the next map. Changing every
# shear by a factor of two must then change every cubic moment by eight,
# rather than returning any of the previous map's cached aperture values.
cache.clear()
catalogs = [catalog(2*g) for g in shear]
doubled = np.array([measure(catalogs, t, radii, configuration, cache)
                    for t in triples])
np.testing.assert_array_equal(doubled, 8*full)
print('PASS: all triples, float rounding, normalization, shuffled/partial cache and new map')
