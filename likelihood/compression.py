"""Keep chosen local parameter responses in a fixed linear data compression.

A compression multiplies data, model and derivatives by one matrix T, and
the covariance by T on the left and T transpose on the right. Rebuilding T
at each sampled cosmology would change the measured statistic. Here T is
prepared once from declared training derivatives and a covariance model.

The xi-only mode follows the MOPED response subspace. The conditional-xi
mode additionally accounts for correlations with retained Map3 rows. It
preserves the joint Gaussian Fisher matrix at the training point, even
when xi-only MOPED loses joint information. Neither statement guarantees
preservation of nonlinear posteriors or a finite-simulation likelihood.
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from cosmolike_notebook_utils.fisher import symmetric_matrix


class CompressedRows:
    """Label numerical modes without assigning fictitious angles or bin indices."""

    def __init__(self, labels, values):
        """Own mode labels and their one-dimensional measured values."""
        self.labels = np.array(labels, dtype=str, copy=True)
        self.values = np.array(values, dtype=float, copy=True)


class LinearCompression:
    """Own a fixed transformation and the exact input row identities.

    Arguments:
      matrix = finite [noutput,ninput] matrix with independent rows.
      input_labels = unique labels in original measurement order.
      output_labels = unique names for modes and retained measurements.
      metadata = JSON-compatible provenance of training or an imported matrix.
    Side effects:
      Copies all arrays. Subsequent caller edits cannot change the statistic.
    """

    def __init__(self, matrix, input_labels, output_labels, metadata):
        """Validate dimensions and numerical rank before accepting a statistic."""
        matrix = np.array(matrix, dtype=float, copy=True)
        original = np.asarray(input_labels, dtype=str)
        output = np.asarray(output_labels, dtype=str)
        if (matrix.ndim != 2 or min(matrix.shape) < 1
                or matrix.shape[0] > matrix.shape[1]
                or not np.all(np.isfinite(matrix))):
            raise ValueError("Compression needs a finite, nonempty short-wide matrix.")
        for labels, count in ((original, matrix.shape[1]), (output, matrix.shape[0])):
            if (labels.shape != (count,) or np.unique(labels).size != count
                    or np.any(np.char.str_len(labels) == 0)):
                raise ValueError("Compression labels must be unique and match its axes.")
        # Normalize rows before the rank test: changing a mode's units must
        # not make it appear linearly dependent on another mode.
        scale = np.max(np.abs(matrix), axis=1)
        if np.any(scale == 0.0):
            raise ValueError("Compression has a zero row; remove that mode.")
        if np.linalg.matrix_rank(matrix / scale[:, None]) != matrix.shape[0]:
            raise ValueError("Compression rows are dependent; remove duplicate modes.")
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError("Compression requires explicit training/source metadata.")
        self.metadata_json = json.dumps(metadata, allow_nan=False, sort_keys=True)
        self.matrix = matrix.copy()
        self.input_labels = original.copy()
        self.output_labels = output.copy()
        self.metadata = json.loads(self.metadata_json)

    def check_labels(self, labels):
        """Require exact input ordering before multiplying any data or model."""
        if not np.array_equal(np.asarray(labels, dtype=str), self.input_labels):
            raise ValueError("Compression input labels differ; select the training rows.")

    def apply(self, values):
        """Transform a vector [ninput] or derivative matrix [ninput,nparameter]."""
        values = np.asarray(values, dtype=float)
        if (values.ndim not in (1, 2) or values.shape[0] != self.matrix.shape[1]
                or not np.all(np.isfinite(values))):
            raise ValueError(
                "Compression input must be finite with the trained row count."
            )
        return self.matrix @ values

    def covariance(self, values):
        """Transform both covariance axes, retaining every cross-correlation."""
        matrix = symmetric_matrix(values=values, name="Compression covariance")
        if matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Compression covariance must have the trained row count.")
        return self.matrix @ matrix @ self.matrix.T

    def save(self, path):
        """Write numerical arrays and JSON metadata without pickle.

        Existing files are refused: a saved transform defines an experiment.
        """
        destination = Path(path)
        with destination.open("xb") as stream:
            np.savez_compressed(
                stream,
                matrix=self.matrix,
                input_labels=self.input_labels,
                output_labels=self.output_labels,
                metadata=np.asarray(self.metadata_json),
            )


def load_compression(path):
    """Read a fixed transform, preserving its recorded labels and provenance."""
    with np.load(Path(path), allow_pickle=False) as saved:
        return LinearCompression(
            matrix=saved["matrix"],
            input_labels=saved["input_labels"],
            output_labels=saved["output_labels"],
            metadata=json.loads(str(saved["metadata"].item())),
        )


def fit_compression(covariance, derivatives, labels, kind, mode, metadata, rtol=1.e-10):
    """Train a Gaussian response-subspace compression at one fixed point.

    Arguments:
      covariance = positive [ndata,ndata] training covariance.
      derivatives = [ndata,nparameter], including every varied nuisance column.
      labels, kind = [ndata] row labels and xip/xim/map3 identifiers.
      mode = 'joint', 'xi_moped', or 'conditional_xi'. The latter two
        retain each Map3 row and compress only xi. No implicit default.
      metadata = source and training point, supplied by the caller.
      rtol = relative singular-value cutoff after unit-norm column scaling.
    Returns:
      immutable LinearCompression. Its rank and singular values are recorded.

    If C=L L^T, whitening gives L^-1 D. Its left singular vectors span all
    measured directions excited by the trained parameter changes. Projecting
    onto that span retains D^T C^-1 D without constructing C^-1. For partial
    joint preservation, use S=Cxx-Cxz Czz^-1 Czx and J=Dx-Cxz Czz^-1 Dz.
    The uncompressed Map3 rows already contain the subtracted response.
    """
    if not isinstance(metadata, dict) or not metadata:
        raise ValueError("Training requires explicit source and fiducial metadata.")
    covariance = symmetric_matrix(values=covariance, name="Training covariance",
                                  positive=True)
    derivatives = np.asarray(derivatives, dtype=float)
    labels = np.asarray(labels, dtype=str)
    kind = np.asarray(kind, dtype=str)
    size = covariance.shape[0]
    if (derivatives.ndim != 2 or derivatives.shape[0] != size
            or derivatives.shape[1] < 1 or not np.all(np.isfinite(derivatives))
            or labels.shape != (size,) or kind.shape != (size,)):
        raise ValueError(
            "Training derivatives and row descriptions have inconsistent sizes."
        )
    if mode not in ("joint", "xi_moped", "conditional_xi"):
        raise ValueError("Compression mode must be joint, xi_moped or conditional_xi.")
    if not np.isfinite(rtol) or not 0.0 < rtol < 1.0:
        raise ValueError("Compression rtol must lie strictly between zero and one.")
    retained = np.empty(0, dtype=int)
    compressed = np.arange(size)
    if mode != "joint":
        if not np.all(np.isin(kind, ["xip", "xim", "map3"])):
            raise ValueError("Partial compression requires xip/xim/map3 row kinds.")
        retained = np.flatnonzero(kind == "map3")
        compressed = np.flatnonzero(kind != "map3")
    if compressed.size == 0:
        raise ValueError("There are no rows in the requested compression block.")
    conditional = covariance[np.ix_(compressed, compressed)].copy()
    response = derivatives[compressed].copy()
    if mode == "conditional_xi" and retained.size:
        other = covariance[np.ix_(retained, retained)]
        cross = covariance[np.ix_(compressed, retained)]
        factor = np.linalg.cholesky(other)
        whitened_cross = solve_triangular(a=factor, b=cross.T, lower=True)
        whitened_response = solve_triangular(
            a=factor, b=derivatives[retained], lower=True,
        )
        conditional -= whitened_cross.T @ whitened_cross
        response -= whitened_cross.T @ whitened_response
    factor = np.linalg.cholesky(symmetric_matrix(
        values=conditional, name="Conditional training covariance", positive=True,
    ))
    whitened = solve_triangular(a=factor, b=response, lower=True)
    # Parameter units cannot decide numerical rank. Scale each nonzero
    # response column before SVD, retaining those scales in the record.
    if not np.all(np.isfinite(whitened)):
        raise ValueError("Whitened derivatives overflow; rescale the training units.")
    maxima = np.max(np.abs(whitened), axis=0)
    active = maxima > 0.0
    scales = np.zeros(derivatives.shape[1])
    if np.any(active):
        scaled = whitened[:, active] / maxima[active]
        scales[active] = np.linalg.norm(scaled, axis=0)
        basis, singular, _ = np.linalg.svd(scaled / scales[active], full_matrices=False)
        rank = int(np.count_nonzero(singular > rtol * singular[0]))
    else:
        basis = np.empty((compressed.size, 0))
        singular = np.empty(0)
        rank = 0
    if rank + retained.size == 0:
        raise ValueError("Training derivatives contain no response to retain.")
    weights = solve_triangular(a=factor.T, b=basis[:, :rank], lower=False).T
    matrix = np.zeros((rank + retained.size, size))
    matrix[:rank, compressed] = weights
    matrix[rank + np.arange(retained.size), retained] = 1.0
    output = [f"{mode}_mode_{index + 1}" for index in range(rank)]
    output.extend(labels[retained].tolist())
    record = dict(metadata)
    record.update(mode=mode, rank=rank, rtol=float(rtol),
                  column_maxima=maxima.tolist(), scaled_column_norms=scales.tolist(),
                  singular_values=singular.tolist())
    return LinearCompression(matrix=matrix, input_labels=labels,
                             output_labels=output, metadata=record)
