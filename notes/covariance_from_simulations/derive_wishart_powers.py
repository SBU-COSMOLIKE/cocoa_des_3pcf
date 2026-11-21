"""Derive unbiased matrix powers through degree six by exact Gaussian moments.

Run with Cocoa's Python and SymPy. No simulation or DES input is read.
The accompanying teaching note explains the contraction graph and the
notebook uses the resulting coefficients. This script verifies polynomial
identities, rather than fitting coefficients to Monte Carlo measurements.
"""

from collections import defaultdict
import sympy as sp

# nu = N - 1 for a mean-subtracted covariance of N independent Gaussian
# vectors. Keeping nu symbolic makes the checks valid for every allowed N.
nu = sp.symbols("nu", positive=True)


def pairings(items):
    """List every way to pair Gaussian factors in Wick's moment identity.

    A product of 2k zero-mean Gaussian variables has (2k-1)!! pairings.
    Choosing a partner for the first remaining factor and recursing lists
    each pairing once: 3, 15 and 105 at matrix degrees 2, 3 and 4.
    """
    if not items:
        yield []
        return
    first = items[0]
    for j in range(1, len(items)):
        rest = items[1:j] + items[j + 1:]
        for paired in pairings(rest):
            yield [(first, items[j])] + paired


def components(count, edges):
    """Return the connected-component label of each vertex in a small graph.

    The same operation counts equal Gaussian-realization indices and
    collects covariance factors into matrix paths or trace cycles. These
    graphs contain at most eight vertices; no numerical graph library is
    needed. Joining two roots records that their labels must be equal.
    """
    roots = list(range(count))

    def root(i):
        while roots[i] != i:
            i = roots[i]
        return i

    for i, j in edges:
        roots[root(i)] = root(j)
    return [root(i) for i in range(count)]


def moment(chain, traces):
    """Expand E[S**chain times the specified traces] in powers of R.

    Here S = sum_a z_a z_a^T / nu, with independent Gaussian z_a of
    covariance R. A chain of length r stands for (S^r)_ij; a closed
    cycle of length q stands for tr(S^q). Every S factor contributes two
    Gaussian factors. Wick pairs force their realization labels equal
    and replace their coordinate factors by an entry of R.

    After pairing, each free realization label supplies nu. The remaining
    R factors form one open path and zero or more closed cycles. A path
    of length r is R^r; a cycle of length q is tr(R^q). This enumeration
    returns the exact coefficient of each such matrix monomial.
    """
    edges = [(i, i + 1) for i in range(chain)]
    vertices = chain + 1
    for length in traces:
        edges += [
            (vertices + i, vertices + (i + 1) % length)
            for i in range(length)
        ]
        vertices += length
    coordinates = [vertex for edge in edges for vertex in edge]
    degree = len(edges)
    counts = defaultdict(int)
    for pairs in pairings(list(range(2 * degree))):
        # The two factors from S number a initially share realization a.
        # A Gaussian cross-covariance is zero unless its labels agree.
        replica = components(degree, [(a // 2, b // 2) for a, b in pairs])
        covariance_edges = [(coordinates[a], coordinates[b]) for a, b in pairs]
        coordinate = components(vertices, covariance_edges)
        assert coordinate[0] == coordinate[chain]
        lengths = defaultdict(int)
        for a, b in covariance_edges:
            lengths[coordinate[a]] += 1
        power = lengths.pop(coordinate[0])
        key = (power, tuple(sorted(lengths.values(), reverse=True)))
        counts[key, len(set(replica)) - degree] += 1
    # Count equal contractions as integers first. Symbolic addition inside
    # every pairing is unnecessary; sixth order has 10,395 pairings per term.
    result = defaultdict(lambda: 0)
    for (key, exponent), count in counts.items():
        result[key] += count * nu**exponent
    return dict(result)


# A basis entry (r, (a,b,...)) means S^r tr(S^a) tr(S^b) ... .
# All terms have the same total degree. There is no assumption that the
# matrix dimension is two; the contractions are valid in any dimension.
BASES = {
    2: [(2, ()), (1, (1,))],
    3: [(3, ()), (2, (1,)), (1, (2,)), (1, (1, 1))],
    4: [
        (4, ()), (3, (1,)), (2, (2,)), (2, (1, 1)),
        (1, (3,)), (1, (2, 1)), (1, (1, 1, 1)),
    ],
}


def partitions(total, largest=None):
    """List trace powers in descending order, without repeated partitions."""
    if total == 0:
        yield ()
        return
    if largest is None:
        largest = total
    for first in range(min(total, largest), 0, -1):
        for rest in partitions(total - first, first):
            yield (first,) + rest


for degree in (5, 6):
    BASES[degree] = [
        (power, traces)
        for power in range(degree, 0, -1)
        for traces in partitions(degree - power)
    ]


def verify(max_degree=4):
    """Solve and verify the exact moment equations; print the coefficients."""
    for degree, basis in BASES.items():
        if degree > max_degree:
            continue
        table = [moment(*entry) for entry in basis]
        matrix = sp.Matrix([
            [column.get(entry, 0) for column in table]
            for entry in basis
        ])
        # Column j contains the expectation of observed basis term j.
        # Find c such that matrix*c selects only R^degree, with no trace
        # terms. Polynomial-domain elimination avoids slow expression growth.
        integer_matrix = sp.simplify(matrix * nu**degree)
        coefficients = integer_matrix.inv(method="DM")[:, 0] * nu**degree
        coefficients = [sp.factor(value) for value in coefficients]
        target = sp.eye(len(basis))[:, 0]
        residual = matrix * sp.Matrix(coefficients) - target
        assert all(sp.cancel(value) == 0 for value in residual)
        # In one dimension every basis term is the same scalar power.
        # The coefficients must reduce to the chi-square moment correction.
        scalar = nu**degree
        for j in range(degree):
            scalar /= nu + 2 * j
        assert sp.cancel(sum(coefficients) - scalar) == 0
        print("DEGREE", degree, "exact matrix and scalar identities passed")
        for term, coefficient in zip(basis, coefficients):
            print(" ", term, coefficient)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-degree", type=int, choices=[4, 5, 6], default=4)
    arguments = parser.parse_args()
    verify(max_degree=arguments.max_degree)
