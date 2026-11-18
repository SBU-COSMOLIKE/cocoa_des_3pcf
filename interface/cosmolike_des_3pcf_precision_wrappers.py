"""Teach covariance corrections with controlled joint Gaussian measurements.

These arrays are not DES simulations. Two-point-like and three-point-like
rows have different units and correlated fluctuations, but their covariance
is prescribed here so the exact precision and truncation error are known.
The same precision.py functions used by the forecast are called directly.
No compiled model, CAMB calculation, downloads or files are needed to build
the example. Loading external simulation vectors is a separate explicit call.
"""

import importlib
from pathlib import Path
import sys

import numpy as np
from scipy.linalg import solve_triangular


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "python"))

from des3pcf.precision import component_precision_expansion
from des3pcf.precision import precision_expansion, split_precision_expansion
from des3pcf.precision import whiten_covariance


def controlled_joint_covariance(ndata=24):
    """Specify C=A+B and an imperfect independent model M=A+Bm.

    A is a diagonal known random contribution. B has correlations both within
    and between two-point-like and three-point-like rows. Independent draws
    of the A and B random vectors make this covariance sum physical in the
    controlled experiment. It is not a claim that DES noise can be switched
    off to isolate every corresponding connected covariance term.

    Arguments:
      ndata = even integer at least8, split equally into the two row groups.
    Returns:
      Dictionary with true/analytic/component covariances, row labels and a
      three-column illustrative derivative matrix. The row scales 1e-4 and
      1e-8 only expose mixed units; they are not calibrated survey variances.
    Raises:
      ValueError for an invalid dimension.
    """
    if (isinstance(ndata, bool) or int(ndata) != ndata
            or ndata < 8 or ndata % 2):
        raise ValueError("ndata must be an even integer at least8.")
    ndata = int(ndata)
    coordinate = np.arange(ndata, dtype=np.float64)
    separation = np.abs(coordinate[:, None] - coordinate[None, :])
    component = 0.12 * np.exp(-separation / 3.0)
    component += 0.025 * np.ones((ndata, ndata))
    known = np.diag(0.8 + 0.1 * np.cos(coordinate / 4.0))
    component_model = 0.8 * component + 0.01 * np.eye(ndata)
    scales = np.full(ndata, 1.0e-4)
    scales[ndata // 2:] = 1.0e-8
    units = scales[:, None] * scales[None, :]
    component *= units
    known *= units
    component_model *= units
    derivative = np.empty((ndata, 3))
    derivative[:, 0] = scales
    derivative[:, 1] = scales * np.cos(coordinate / 4.0)
    derivative[:, 2] = scales * np.sin(coordinate / 3.0)
    # The last parameter changes the three-point group differently. These
    # are prescribed summary responses, not derivatives of a shear model.
    derivative[ndata // 2:, 2] += scales[ndata // 2:]
    kinds = np.full(ndata, "map3", dtype="U4")
    kinds[:ndata // 2] = "xip"
    labels = []
    for index, kind in enumerate(kinds):
        labels.append(f"controlled_{kind}_{index:03d}")
    return {
        "known": known,
        "component": component,
        "component_model": component_model,
        "truth": known + component,
        "analytic": known + component_model,
        "derivatives": derivative,
        "row_labels": np.asarray(labels),
        "row_kind": kinds,
        "parameter_names": ["amplitude", "alignment", "shape"],
        "scope": "Controlled Gaussian joint vectors; not DES simulations.",
    }


def draw_vectors(covariance, count, generator):
    """Generate independent zero-mean vectors with an explicit covariance.

    Arguments:
      covariance = positive definite [ndata,ndata] matrix.
      count = number of realizations; generator = supplied NumPy Generator.
    Returns:
      Float64 vectors[count,ndata]. Cholesky multiplication preserves the
      intentionally different observable units without an SVD rank threshold.
    """
    factor = np.linalg.cholesky(covariance)
    normal = generator.standard_normal(size=(count, covariance.shape[0]))
    return normal @ factor.T


def covariance_of_vectors(vectors):
    """Subtract the estimated mean and divide by N-1.

    Arguments:
      vectors = finite [N,ndata] independent realizations, N>=2.
    Returns:
      Unbiased sample covariance. Rank deficiency is allowed and no inverse
      is attempted; after mean removal its rank is at most N-1.
    Raises:
      ValueError for invalid arrays or insufficient realizations.
    """
    values = np.asarray(vectors, dtype=np.float64)
    if (values.ndim != 2 or values.shape[0] < 2 or values.shape[1] == 0
            or not np.all(np.isfinite(values))):
        raise ValueError("Vectors must be finite [N,ndata] with N>=2.")
    centered = values - np.mean(values, axis=0)
    return centered.T @ centered / (values.shape[0] - 1)


def precision_diagnostics(precision, truth, derivative, validation):
    """Compare a supplied precision with known truth and held-out vectors.

    Arguments:
      precision = symmetric candidate [ndata,ndata], possibly indefinite.
      truth = known true covariance; derivative = [ndata,nparameter].
      validation = independent zero-mean truth draws, unused in preparation.
    Returns:
      Eigenvalues of Lc.T precision Lc-I and its relative Frobenius error.
      Positive candidates also yield parameter-error and validation statistics.
      Indefinite candidates are retained as failures, without eigenvalue repair.
    """
    factor = np.linalg.cholesky(truth)
    normalized = factor.T @ precision @ factor
    normalized = 0.5 * (normalized + normalized.T)
    eigenvalues = np.linalg.eigvalsh(normalized)
    error = normalized - np.eye(truth.shape[0])
    result = {
        "positive": bool(eigenvalues[0] > 0.0),
        "minimum_precision_eigenvalue_in_true_units": float(eigenvalues[0]),
        "precision_error_eigenvalues": eigenvalues - 1.0,
        "relative_precision_frobenius_error": float(
            np.linalg.norm(error) / np.sqrt(truth.shape[0])
        ),
    }
    if not result["positive"]:
        result["inference_status"] = "refused_nonpositive_precision"
        return result
    information = derivative.T @ precision @ derivative
    parameter_covariance = np.linalg.solve(
        information, np.eye(information.shape[0]),
    )
    estimator = parameter_covariance @ derivative.T @ precision
    fitted = validation @ estimator.T
    empirical = covariance_of_vectors(vectors=fitted)
    # The exact frequentist covariance of the linear estimator depends on C,
    # not on the covariance whose inverse was used to weight the fit.
    sandwich = estimator @ truth @ estimator.T
    result.update({
        "inference_status": "positive_precision",
        "parameter_covariance": parameter_covariance,
        "exact_fit_covariance": sandwich,
        "validation_fit_covariance": empirical,
        "sigma": np.sqrt(np.diag(parameter_covariance)),
        "validation_variance_over_quoted": (
            np.diag(empirical) / np.diag(parameter_covariance)
        ),
        "exact_variance_over_quoted": (
            np.diag(sandwich) / np.diag(parameter_covariance)
        ),
        "validation_mean_quadratic_per_row": float(
            np.mean(np.einsum("ni,ij,nj->n", validation, precision, validation))
            / truth.shape[0]
        ),
    })
    return result


def controlled_precision_study(ndata=24, training_count=18,
                               validation_count=1200, seed=1937):
    """Compare full and component PME with one fixed, independent model.

    Training contains N independent A and B vectors; their sum samples C.
    The B-only sample uses those same N B vectors, so comparisons share the
    available physical realizations. Two disjoint halves supply independent
    B covariance estimates for the split-group polynomial. Validation uses
    a separately seeded stream of fresh joint vectors and never tunes M.

    Arguments:
      ndata = even vector length; training_count = even N>=6 and N<=ndata.
      validation_count = integer number of held-out draws at least100.
      seed = integer entropy for separate training/validation generators.
    Returns:
      Known matrices, samples, component/full/split estimates, positivity,
      true and estimated residual radii, truncation and validation diagnostics.
    Raises:
      ValueError for invalid counts; no repeated draw selects a nicer result.
    """
    if (isinstance(training_count, bool)
            or int(training_count) != training_count
            or training_count < 6 or training_count % 2
            or training_count > ndata):
        raise ValueError("Use even training_count from6 through ndata.")
    if (isinstance(validation_count, bool)
            or int(validation_count) != validation_count
            or validation_count < 100):
        raise ValueError("validation_count must be an integer at least100.")
    system = controlled_joint_covariance(ndata=ndata)
    streams = np.random.SeedSequence(entropy=seed).spawn(2)
    training_rng = np.random.default_rng(seed=streams[0])
    validation_rng = np.random.default_rng(seed=streams[1])
    component_vectors = draw_vectors(
        covariance=system["component"], count=int(training_count),
        generator=training_rng,
    )
    known_vectors = draw_vectors(
        covariance=system["known"], count=int(training_count),
        generator=training_rng,
    )
    joint_vectors = component_vectors + known_vectors
    validation = draw_vectors(
        covariance=system["truth"], count=int(validation_count),
        generator=validation_rng,
    )
    sample_component = covariance_of_vectors(vectors=component_vectors)
    sample_joint = covariance_of_vectors(vectors=joint_vectors)
    half = int(training_count) // 2
    sample_first = covariance_of_vectors(vectors=component_vectors[:half])
    sample_second = covariance_of_vectors(vectors=component_vectors[half:])
    model = system["analytic"]
    factor = np.linalg.cholesky(model)
    normalized_truth = whiten_covariance(
        covariance=system["truth"], factor=factor,
    )
    residual = normalized_truth - np.eye(ndata)
    true_radius = float(np.max(np.abs(np.linalg.eigvalsh(residual))))
    inverse_factor = solve_triangular(factor, np.eye(ndata), lower=True)
    analytic_precision = inverse_factor.T @ inverse_factor
    truth_precision = np.linalg.solve(system["truth"], np.eye(ndata))
    polynomial = np.eye(ndata) - residual + residual @ residual
    truncated = inverse_factor.T @ polynomial @ inverse_factor
    estimates = {
        "truth": truth_precision,
        "analytic": analytic_precision,
        "exact_second_order": truncated,
    }
    estimator_diagnostics = {}
    estimates["component_wishart"], estimator_diagnostics["component"] = (
        component_precision_expansion(
            analytic=model, component_model=system["component_model"],
            component_sample=sample_component,
            independent_count=training_count,
        )
    )
    estimates["full_wishart"], estimator_diagnostics["full"] = (
        precision_expansion(
            analytic=model, sample=sample_joint,
            independent_count=training_count,
        )
    )
    estimates["component_split"] = split_precision_expansion(
        analytic=model, sample_first=system["known"] + sample_first,
        sample_second=system["known"] + sample_second,
    )
    diagnostics = {}
    for name, precision in estimates.items():
        diagnostics[name] = precision_diagnostics(
            precision=precision, truth=system["truth"],
            derivative=system["derivatives"], validation=validation,
        )
    normalized_sample = whiten_covariance(
        covariance=sample_joint, factor=factor,
    )
    return {
        "system": system,
        "sample_joint": sample_joint,
        "sample_component": sample_component,
        "estimates": estimates,
        "diagnostics": diagnostics,
        "estimator_diagnostics": estimator_diagnostics,
        "true_residual_spectral_radius": true_radius,
        "true_second_order_operator_error_bound": true_radius**3
        / (1.0 - true_radius),
        "sample_rank_in_model_units": int(
            np.linalg.matrix_rank(normalized_sample)
        ),
        "training_count": int(training_count),
        "validation_count": int(validation_count),
        "seed": seed,
        "training_stream": streams[0].spawn_key,
        "validation_stream": streams[1].spawn_key,
        "sample_inverse_attempted": False,
    }


def plot_precision_errors(report):
    """Draw dimensionless precision errors, including failed estimates.

    Arguments:
      report = controlled_precision_study result.
    Returns:
      Shared figure/axes. Each line sorts the eigenvalues of Lc.T Psi Lc-I;
      sorted positions are not identifications of shared physical eigenvectors.
    """
    plots = importlib.import_module(name="cosmolike_notebook_utils.3pcf")
    names = (
        "analytic",
        "component_wishart",
        "component_split",
        "full_wishart",
    )
    labels = [
        "Analytic model",
        "Known A + Wishart B",
        "Known A + independent B groups",
        "Full-covariance Wishart",
    ]
    curves = []
    coordinate = np.arange(report["system"]["truth"].shape[0]) + 1
    for name in names:
        errors = report["diagnostics"][name]["precision_error_eigenvalues"]
        curves.append((coordinate, errors[None, :]))
    return plots.plot_ia_components(
        components=curves, component_labels=["Precision error"],
        coordinate_label="Sorted eigenvalue index",
        quantity_label=r"Eigenvalues of $L_C^{\mathsf{T}}\hat{\Psi} L_C-I$",
        legend=labels, xscale="linear", ncols=1, figsize=(10, 5),
        bintextpos=None, colorbar=None, show=None,
    )


def plot_precision_parameter_errors(report):
    """Plot quoted parameter errors only for positive precision estimates.

    Arguments:
      report = controlled_precision_study result with known true precision.
    Returns:
      Shared figure/axis showing sigma/sigma_true. Positive precision alone
      does not establish coverage; validation variance ratios are separate.
    """
    plots = importlib.import_module(name="cosmolike_notebook_utils.3pcf")
    errors = []
    labels = []
    methods = (
        ("analytic", "Analytic model"),
        ("component_wishart", "Known A + Wishart B"),
        ("component_split", "Known A + independent B groups"),
        ("full_wishart", "Full-covariance Wishart"),
    )
    for name, label in methods:
        diagnostics = report["diagnostics"][name]
        if diagnostics["positive"]:
            errors.append(diagnostics["sigma"])
            labels.append(label)
        else:
            print(f"No parameter contour for {label}: nonpositive precision.")
    return plots.plot_parameter_errors(
        names=report["system"]["parameter_names"], errors=errors,
        errors_ref=report["diagnostics"]["truth"]["sigma"],
        legend=labels, figsize=(11, 5), show=None,
    )


def load_joint_simulations(path, expected_labels, expected_kind,
                           expected_estimator, excluded_parent_ids=()):
    """Read explicitly labeled independent joint vectors without reordering.

    The NPZ must contain vectors[N,p], row_labels[p], row_kind[p],
    parent_ids[N], and scalar estimator. One parent ID per independent sky
    must occur exactly once. Multiple patches sharing a parent are refused;
    averaging them here would change the estimator and its covariance.

    Arguments:
      path = an explicit NPZ file, read with allow_pickle=False.
      expected_labels, expected_kind = exact ordered row identities of the
          analytic covariance; each group must contain xi and Map3 rows.
      expected_estimator = declared estimator string, e.g. finite_sas.
          A family name does not verify triangle cuts, weights, masks or
          source calibration; compare their input/measurement manifests
          separately before interpreting a survey correction.
      excluded_parent_ids = parents used for analytic tuning or validation;
          any overlap is refused before a sample covariance is formed.
    Returns:
      Owned vectors, labels, kinds, parent IDs and independent count.
    Raises:
      ValueError for shape, finite-value, row-order, estimator or parent errors.
      Independence is a scientific input declaration; unique IDs alone do not
      prove it. The caller must establish the survey realization construction.
    """
    filename = Path(path)
    required = (
        "vectors",
        "row_labels",
        "row_kind",
        "parent_ids",
        "estimator",
    )
    arrays = {}
    with np.load(filename, allow_pickle=False) as archive:
        for name in required:
            if name not in archive.files:
                raise ValueError(f"Simulation archive lacks {name}.")
            arrays[name] = archive[name].copy()
    vectors = arrays["vectors"]
    if (vectors.dtype.kind not in "fiu" or vectors.ndim != 2
            or vectors.shape[0] < 3
            or not np.all(np.isfinite(vectors))):
        raise ValueError("Need finite independent vectors[N,p] with N>2.")
    labels = arrays["row_labels"]
    kinds = arrays["row_kind"]
    parents = arrays["parent_ids"]
    if (labels.shape != (vectors.shape[1],)
            or kinds.shape != labels.shape
            or parents.shape != (vectors.shape[0],)):
        raise ValueError("Simulation rows or parent IDs have wrong shapes.")
    if (not np.array_equal(labels, expected_labels)
            or not np.array_equal(kinds, expected_kind)):
        raise ValueError("Simulation rows do not match the analytic matrix.")
    if (np.unique(labels).size != labels.size
            or np.any(~np.isin(kinds, ("xip", "xim", "map3")))
            or not np.any(kinds == "map3")
            or not np.any(kinds != "map3")):
        raise ValueError("Require unique labeled joint xi and Map3 rows.")
    if arrays["estimator"].shape != ():
        raise ValueError("Simulation estimator must be one declared string.")
    if str(arrays["estimator"].item()) != expected_estimator:
        raise ValueError("Simulation and analytic estimators differ.")
    if parents.dtype.kind not in "iuUS":
        raise ValueError("Parent IDs must be integer or text identifiers.")
    if parents.dtype.kind in "US" and np.any(parents == ""):
        raise ValueError("Parent IDs must not be empty text identifiers.")
    if np.unique(parents).size != parents.size:
        raise ValueError(
            "Repeated parent IDs: patches are not independent skies."
        )
    if np.intersect1d(parents, np.asarray(excluded_parent_ids)).size:
        raise ValueError("Training parents overlap tuning or validation.")
    return {
        "vectors": vectors.copy(),
        "row_labels": labels.copy(),
        "row_kind": kinds.copy(),
        "parent_ids": parents.copy(),
        "independent_count": int(vectors.shape[0]),
        "estimator": expected_estimator,
    }
