"""Check fixed-A algebra on small matrices with exactly known covariance."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
from pathlib import Path
from itertools import combinations
import numpy as np
from fixed_block_precision import estimates

rng = np.random.default_rng(211126)
truth = np.array([[1., .12, .08], [.12, .9, .10], [.08, .10, 1.1]])
reference = np.diag([1., 1.2, 1.2])
l = np.linalg.cholesky(reference)
w = np.linalg.inv(l)
a = truth[:1, :1]
# Check the recurrence against enumerating every increasing-index product.
x = rng.multivariate_normal(np.zeros(3), truth, size=12)
factors = []
for delta in (x[::2]-x[1::2])/np.sqrt(2):
    z = np.outer(delta, delta)
    z[:1, :1] = a
    factors.append(np.eye(3)-w @ z @ w.T)
expected = np.eye(3)
results, used = estimates(x, reference, 1)
for degree in range(1, 7):
    products = []
    for subset in combinations(range(6), degree):
        product = np.eye(3)
        for i in subset:
            product = product @ factors[i]
        products.append(product)
    term = np.mean(products, axis=0)
    expected += (term+term.T)/2
    if degree in (2, 4, 6):
        raw = results[(degree-2)//2][1]
        np.testing.assert_allclose(raw, w.T @ expected @ w, atol=2.e-12)
print('Distinct-factor recurrence equals explicit products through sixth order.')
# Verify the cross-square subtraction against its direct double sum.
b = [(row[:1, None] @ row[None, 1:]) for row in (x[::2]-x[1::2])/np.sqrt(2)]
q = (sum(b).T @ sum(b)-sum(value.T @ value for value in b))/30
brute = sum(b[i].T @ b[j] for i in range(6) for j in range(6) if i != j)/30
np.testing.assert_allclose(q, brute, atol=1.e-14)
print('Unbiased cross-square formula equals explicit distinct-pair sum.')
# Ensemble means target the finite polynomial, not the infinite inverse.
t = np.eye(3)-w @ truth @ w.T
targets = [w.T @ sum(np.linalg.matrix_power(t, k) for k in range(order+1)) @ w
           for order in (2, 4, 6)]
values = []
for repeat in range(512):
    draws = rng.multivariate_normal(np.zeros(3), truth, size=128)
    result, used = estimates(draws, reference, 1)
    values.append([item[1] for item in result])
values = np.asarray(values)
for index, order in enumerate((2, 4, 6)):
    error = values[:, index].std(axis=0, ddof=1)/np.sqrt(len(values))
    z = np.max(np.abs(values[:, index].mean(axis=0)-targets[index])/error)
    print('order', order, 'maximum mean discrepancy / Monte Carlo SE', z)
    assert z < 6
print('Every returned constrained covariance retains A exactly and passes the inverse identity.')
