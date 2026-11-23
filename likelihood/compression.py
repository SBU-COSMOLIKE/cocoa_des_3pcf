"""Train, store and apply a fixed linear compression of the joint data vector.

A compression multiplies data, model and derivatives by one matrix T, and
the covariance by T on the left and T transpose on the right, so the chi2
runs over a few linear modes instead of every measured row. T is prepared
once, from training derivatives and a covariance model at one fixed point;
rebuilding it at each sampled cosmology would change the measured statistic.

Terms used in this file:
  MOPED      = the classic linear compression of Heavens, Jimenez and Lahav
               (2000, arXiv:astro-ph/9911102): for a Gaussian likelihood it
               keeps one mode per parameter, chosen so the Fisher matrix at
               the training point is unchanged, and discards every other
               direction.
  whitening  = the change of coordinates d -> L^-1 d, where C = L L^T is the
               Cholesky factorization of the covariance. In the new
               coordinates the covariance is the identity: every direction
               has variance one, like rescaling each covariance eigenvector
               by one over the square root of its eigenvalue. Ranks, angles
               and singular values read off in these coordinates are then
               statements about information, not about the rows' units.
  response subspace = the span of the whitened derivative columns: the set
               of measured directions that move when the trained parameters
               change. Projecting onto it keeps D^T C^-1 D (the Fisher
               matrix) without ever constructing C^-1.

The xi-only mode applies MOPED to the xi rows and keeps every Map3 row
unchanged. The conditional-xi mode additionally accounts for the
correlations between the xi rows and the retained Map3 rows, so the joint
Gaussian Fisher matrix at the training point is preserved even where
xi-only MOPED loses joint information. Both statements are exact for the
linear Gaussian model at the training point.
"""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import solve_triangular

from cosmolike_notebook_utils.fisher import symmetric_matrix


class CompressedRows:
    """Labels the compressed modes of a data vector.

    A compressed row is an abstract linear mode, not a measurement at an
    angle in a bin pair: the SurveyRows fields (angle, bin indices, row
    kind) have no meaning for it, so only a label and a value exist.
    """

    def __init__(self, labels, values):
        """Copy the mode labels and their one-dimensional values.

        Arguments:
          labels = one text label per mode.
          values = one measured (compressed) value per mode.

        Side effects: copies both arrays, so later caller edits cannot
        change this object.
        """
        self.labels = np.array(labels, dtype=str, copy=True)
        self.values = np.array(values, dtype=float, copy=True)


class LinearCompression:
    """Owns a fixed compression matrix and the exact input row identities.

    The stored input labels pin the measurement order the matrix was
    trained on: check_labels refuses any data whose rows differ, so the
    transform can never be applied to a reordered or re-cut vector.

    Arguments:
      matrix = finite [noutput,ninput] matrix with independent rows.
      input_labels = unique labels in original measurement order.
      output_labels = unique names for modes and retained measurements.
      metadata = JSON-compatible provenance of training or an imported matrix.
    Side effects:
      Copies all arrays. Subsequent caller edits cannot change the statistic.
    """

    def __init__(self, matrix, input_labels, output_labels, metadata):
        """Validate dimensions and numerical rank before accepting a statistic.

        Arguments:
          matrix, input_labels, output_labels, metadata = see the class
            docstring.

        Raises:
          ValueError when the matrix is not a finite short-wide 2D array,
          when either label set is not unique and matched to its matrix
          axis, when a matrix row is zero or linearly dependent on the
          others, or when the metadata is not a nonempty dictionary.

        Side effects: stores copies of every input and the metadata both
        as a dictionary and as its canonical JSON text (used verbatim by
        save).
        """
        # ------------------------------------------------------------------
        # Copy the inputs; the checks below then run on data the caller can
        # no longer change.
        # ------------------------------------------------------------------
        matrix = np.array(matrix, dtype=float, copy=True)
        original = np.asarray(input_labels, dtype=str)
        output = np.asarray(output_labels, dtype=str)

        # ------------------------------------------------------------------
        # Shape: a compression is short and wide (at most as many output
        # modes as input rows), two-dimensional, nonempty and finite.
        # ------------------------------------------------------------------
        if (matrix.ndim != 2 or min(matrix.shape) < 1
                or matrix.shape[0] > matrix.shape[1]
                or not np.all(np.isfinite(matrix))):
            raise ValueError("Compression needs a finite, nonempty short-wide matrix.")

        # ------------------------------------------------------------------
        # Labels: the loop runs the same check twice, once per axis — the
        # input labels against the column count and the output labels
        # against the row count; each set must be unique and nonempty.
        # ------------------------------------------------------------------
        for labels, count in ((original, matrix.shape[1]), (output, matrix.shape[0])):
            if (labels.shape != (count,) or np.unique(labels).size != count
                    or np.any(np.char.str_len(labels) == 0)):
                raise ValueError("Compression labels must be unique and match its axes.")

        # ------------------------------------------------------------------
        # Rank: normalize each row by its largest entry before the rank
        # test, so changing a mode's units cannot make it appear linearly
        # dependent on another mode.
        # ------------------------------------------------------------------
        scale = np.max(np.abs(matrix), axis=1)
        if np.any(scale == 0.0):
            raise ValueError("Compression has a zero row; remove that mode.")
        if np.linalg.matrix_rank(matrix / scale[:, None]) != matrix.shape[0]:
            raise ValueError("Compression rows are dependent; remove duplicate modes.")

        # ------------------------------------------------------------------
        # Provenance: the metadata must say where the matrix came from.
        # Round-tripping it through canonical JSON (sorted keys, no NaN)
        # both validates that it is JSON-compatible and fixes the exact
        # text that save writes.
        # ------------------------------------------------------------------
        if not isinstance(metadata, dict) or not metadata:
            raise ValueError("Compression requires explicit training/source metadata.")
        self.metadata_json = json.dumps(metadata, allow_nan=False, sort_keys=True)
        self.matrix = matrix.copy()
        self.input_labels = original.copy()
        self.output_labels = output.copy()
        self.metadata = json.loads(self.metadata_json)

    def check_labels(self, labels):
        """Refuse any input whose row order differs from the training rows.

        Arguments:
          labels = the row labels of the data about to be transformed.

        Raises:
          ValueError when the labels differ from the stored input labels in
          content or order.
        """
        if not np.array_equal(np.asarray(labels, dtype=str), self.input_labels):
            raise ValueError("Compression input labels differ; select the training rows.")

    def apply(self, values):
        """Transform a data vector or a derivative matrix into mode space.

        Arguments:
          values = vector [ninput] or derivative matrix
            [ninput,nparameter], finite, in the training row order.

        Returns:
          the product T @ values: [noutput] or [noutput,nparameter].

        Raises:
          ValueError when the input is not 1D or 2D with the trained row
          count, or contains nonfinite entries.
        """
        values = np.asarray(values, dtype=float)
        if (values.ndim not in (1, 2) or values.shape[0] != self.matrix.shape[1]
                or not np.all(np.isfinite(values))):
            raise ValueError(
                "Compression input must be finite with the trained row count."
            )
        return self.matrix @ values

    def covariance(self, values):
        """Transform a covariance into mode space, on both axes.

        The two-sided product T C T^T keeps every cross-correlation
        between the output modes, including those between compressed and
        retained rows.

        Arguments:
          values = symmetric covariance [ninput,ninput] in the training
            row order.

        Returns:
          compressed covariance [noutput,noutput].

        Raises:
          ValueError (from symmetric_matrix or the size check) when the
          input is not a symmetric matrix with the trained row count.
        """
        matrix = symmetric_matrix(values=values, name="Compression covariance")
        if matrix.shape[0] != self.matrix.shape[1]:
            raise ValueError("Compression covariance must have the trained row count.")
        return self.matrix @ matrix @ self.matrix.T

    def save(self, path):
        """Write the matrix, labels and JSON metadata to an npz file.

        The file is written without pickle, so it can be loaded with
        allow_pickle=False. An existing file is refused (the "xb" open
        mode fails when the path exists): a saved transform defines an
        experiment and must not be silently replaced.

        Arguments:
          path = destination file name.

        Raises:
          FileExistsError when the destination already exists.

        Side effects: creates the npz file at path.
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
    """Read a saved transform back, with its recorded labels and provenance.

    The npz archive is read with allow_pickle=False, and the returned
    LinearCompression re-runs every constructor check on the loaded
    arrays.

    Arguments:
      path = npz file written by LinearCompression.save.

    Returns:
      LinearCompression with the saved matrix, labels and metadata.

    Raises:
      ValueError (from the constructor) when the saved arrays fail any
      construction check.
    """
    with np.load(Path(path), allow_pickle=False) as saved:
        return LinearCompression(
            matrix=saved["matrix"],
            input_labels=saved["input_labels"],
            output_labels=saved["output_labels"],
            metadata=json.loads(str(saved["metadata"].item())),
        )


def fit_compression(covariance, derivatives, labels, kind, mode, metadata, rtol=1.e-10):
    """Train a Gaussian response-subspace compression at one fixed point.

    The mechanism, in whitened coordinates (module docstring): with
    C = L L^T, the whitened derivatives L^-1 D have unit covariance, and
    their left singular vectors span every measured direction the trained
    parameters excite. Projecting onto that span keeps the Fisher matrix
    D^T C^-1 D without constructing C^-1. In the conditional-xi mode the
    block subscripts are x = the compressed xi rows and z = the retained
    Map3 rows: the xi block is first conditioned on the Map3 rows, using
    the Schur complement S = Cxx - Cxz Czz^-1 Czx as covariance and
    J = Dx - Cxz Czz^-1 Dz as derivatives. This prevents the failure of
    xi-only MOPED, which discards the part of the xi response that is
    correlated with the kept Map3 rows and so loses joint information; the
    uncompressed Map3 rows already carry the subtracted response.

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
    Raises:
      ValueError when the metadata is missing, the covariance is not
      positive, the derivative and row arrays are inconsistent, the mode
      or rtol is invalid, the row kinds do not support a partial
      compression, the compression block is empty, or the training
      derivatives contain no response to retain.
    """
    # ----------------------------------------------------------------------
    # Validation: provenance, a positive training covariance, consistent
    # derivative and row arrays, a named mode and a usable cutoff — all
    # before any linear algebra.
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # Row split: the joint mode compresses every row; the two partial modes
    # retain each Map3 row unchanged and compress only the xi rows.
    # ----------------------------------------------------------------------
    retained = np.empty(0, dtype=int)
    compressed = np.arange(size)
    if mode != "joint":
        if not np.all(np.isin(kind, ["xip", "xim", "map3"])):
            raise ValueError("Partial compression requires xip/xim/map3 row kinds.")
        retained = np.flatnonzero(kind == "map3")
        compressed = np.flatnonzero(kind != "map3")
    if compressed.size == 0:
        raise ValueError("There are no rows in the requested compression block.")

    # ----------------------------------------------------------------------
    # Conditioning (conditional_xi only): subtract from the xi covariance
    # and derivatives the part explained by the retained Map3 rows (the
    # Schur complement S and conditioned derivatives J of the docstring).
    # The triangular solves apply Lz^-1 without forming Czz^-1.
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # Whitening: factor the (conditioned) covariance and map the response
    # into coordinates where that covariance is the identity.
    # ----------------------------------------------------------------------
    factor = np.linalg.cholesky(symmetric_matrix(
        values=conditional, name="Conditional training covariance", positive=True,
    ))
    whitened = solve_triangular(a=factor, b=response, lower=True)

    # ----------------------------------------------------------------------
    # Scaled SVD: parameter units cannot decide numerical rank, so each
    # nonzero response column is scaled to order one before the singular
    # value decomposition; the scales go into the record. The rank counts
    # the singular values above rtol times the largest one. With no
    # nonzero column at all, the basis is empty and the rank is zero.
    # ----------------------------------------------------------------------
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

    # ----------------------------------------------------------------------
    # Matrix assembly: undo the whitening on the kept basis (the transpose
    # solve applies L^-T), place those mode rows on the compressed columns,
    # and give each retained Map3 row an identity row of its own.
    # ----------------------------------------------------------------------
    weights = solve_triangular(a=factor.T, b=basis[:, :rank], lower=False).T
    matrix = np.zeros((rank + retained.size, size))
    matrix[:rank, compressed] = weights
    matrix[rank + np.arange(retained.size), retained] = 1.0
    output = [f"{mode}_mode_{index + 1}" for index in range(rank)]
    output.extend(labels[retained].tolist())

    # ----------------------------------------------------------------------
    # Record: the caller's metadata plus everything needed to audit the
    # training (mode, rank, cutoff, column scalings, singular values).
    # ----------------------------------------------------------------------
    record = dict(metadata)
    record.update(
        mode=mode,
        rank=rank,
        rtol=float(rtol),
        column_maxima=maxima.tolist(),
        scaled_column_norms=scales.tolist(),
        singular_values=singular.tolist(),
    )
    return LinearCompression(
        matrix=matrix,
        input_labels=labels,
        output_labels=output,
        metadata=record,
    )
