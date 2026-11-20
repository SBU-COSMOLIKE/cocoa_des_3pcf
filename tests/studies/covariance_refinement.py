"""Inspect fixed-point covariance refinement without measuring execution time.

Run this study with explicit --extension, --tables and --output paths.
The project is inferred from this file; --project can select another copy.
The default writes a resolved plan only; --execute performs
its covariance calls sequentially with at most four OpenMP workers and one
BLAS worker. No CAMB calculation or table preparation runs here.

The saved bank's point zero is a random numerical reference point, not a
survey fiducial. The saved source curves include recorded clipping. This
study checks numerical sensitivity of the declared full-filter gravitational
model; it does not establish a calibrated survey source model.
It cannot establish agreement with the finite-triangle TATT mean estimator.

Each coordinate is doubled alone, then all are doubled together. Optional
bound extensions start from the combined rule and retain at least its
number of Gauss nodes per logarithmic interval. Such remeshing is not a
proof that the extended tail is resolved. Every difference is saved; no
threshold or convergence verdict is assigned by this script.
"""

import argparse
import copy
import importlib
import math
import os
from pathlib import Path
import sys


def case_settings(helper, include_bounds):
    """Resolve baseline, individual refinements and a combined refinement."""
    options = argparse.Namespace(
        case="physical",
        nell=64,
        nphi=32,
        nradius=256,
        nmass=32,
        nk=512,
        nwindow=64,
        nlos=33,
    )
    baseline = helper.resolved_settings(args=options)
    cases = {"baseline": baseline}
    for coordinate, count in baseline["counts"].items():
        refined = copy.deepcopy(baseline)
        refined["counts"][coordinate] = 2 * count
        if coordinate == "nlos":
            # LOS uses trapezoids: bisect its 32 intervals, retaining nodes.
            refined["counts"][coordinate] = 2 * (count - 1) + 1
        cases[coordinate] = refined
    combined = copy.deepcopy(baseline)
    for coordinate in baseline["counts"]:
        combined["counts"][coordinate] = cases[coordinate]["counts"][coordinate]
    cases["combined"] = combined
    if include_bounds:
        extensions = (
            (
                "ell_low",
                "ell_bounds",
                "nell",
                0,
                0.5,
            ),
            (
                "ell_high",
                "ell_bounds",
                "nell",
                1,
                60000.0,
            ),
            (
                "radius_low",
                "radius_bounds",
                "nradius",
                0,
                5.0e-8,
            ),
            (
                "radius_high",
                "radius_bounds",
                "nradius",
                1,
                0.5,
            ),
            (
                "mass_low",
                "mass_bounds",
                "nmass",
                0,
                1.0e7,
            ),
            (
                "mass_high",
                "mass_bounds",
                "nmass",
                1,
                1.0e17,
            ),
            (
                "k_low",
                "k_bounds",
                "nk",
                0,
                1.0e-6,
            ),
            (
                "k_high",
                "k_bounds",
                "nk",
                1,
                2000.0,
            ),
            (
                "window_low",
                "window_bounds",
                "nwindow",
                0,
                0.005,
            ),
            (
                "window_high",
                "window_bounds",
                "nwindow",
                1,
                2000.0,
            ),
        )
        for name, bounds_key, count_key, side, endpoint in extensions:
            settings = copy.deepcopy(combined)
            old_low, old_high = settings[bounds_key]
            settings[bounds_key][side] = endpoint
            new_low, new_high = settings[bounds_key]
            width_ratio = math.log(new_high / new_low) / math.log(old_high / old_low)
            settings["counts"][count_key] = math.ceil(
                combined["counts"][count_key] * width_ratio
            )
            cases[name] = settings
        for name, side, endpoint in (
            (
                "los_low",
                0,
                0.0025,
            ),
            (
                "los_high",
                1,
                3.1,
            ),
        ):
            settings = copy.deepcopy(combined)
            old_low, old_high = settings["los_bounds"]
            settings["los_bounds"][side] = endpoint
            new_low, new_high = settings["los_bounds"]
            intervals = combined["counts"]["nlos"] - 1
            settings["counts"]["nlos"] = 1 + math.ceil(
                intervals * (new_high - new_low) / (old_high - old_low)
            )
            cases[name] = settings
    return cases


def array_summary(np, matrix):
    """Describe symmetry and eigenvalues without replacing negative modes."""
    symmetric = np.array_equal(matrix, matrix.T)
    if not symmetric:
        raise RuntimeError("The returned covariance is not exactly symmetric.")
    eigenvalues = np.linalg.eigvalsh(matrix)
    diagonal = np.diag(matrix)
    scale = float(np.max(np.abs(eigenvalues)))
    relative_minimum = None
    if scale > 0.0:
        relative_minimum = float(eigenvalues[0] / scale)
    summary = {
        "negative_eigenvalue_count": int(np.count_nonzero(eigenvalues < 0.0)),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "minimum_eigenvalue_over_max_abs": relative_minimum,
        "nonpositive_diagonal_rows": np.flatnonzero(diagonal <= 0.0).tolist(),
        "maximum_absolute_entry": float(np.max(np.abs(matrix))),
    }
    return summary, eigenvalues


def whitened_difference(np, reference, difference):
    """Express a covariance change in units of its full reference covariance.

    If reference=L L^T is positive definite, two linear solves form
    E=L^-1 difference L^-T without constructing an inverse. The eigenvalues
    of E describe changes along correlated combinations of observable rows.
    Scaling only by diagonal errors cannot measure those combinations.

    The reference matrix is never modified. Failed Cholesky factorization
    returns an explicit reason and undefined output arrays, with no row
    removal, pseudoinverse or eigenvalue adjustment. Roundoff can make the
    two-solve result slightly asymmetric. Its symmetric part supplies the
    eigenvalues; the saved raw result and measured antisymmetric remainder
    record this numerical distinction.
    """
    matrix = np.full(shape=difference.shape, fill_value=np.nan)
    eigenvalues = np.full(shape=difference.shape[0], fill_value=np.nan)
    summary = {
        "available": False,
        "reason": None,
        "definition": "L^-1 (candidate-reference) L^-T; reference=L L^T",
    }
    try:
        lower = np.linalg.cholesky(reference)
    except np.linalg.LinAlgError as error:
        summary["reason"] = (
            "Reference total covariance failed Cholesky factorization: "
            + str(error)
        )
        return matrix, eigenvalues, summary
    left_solved = np.linalg.solve(lower, difference)
    matrix = np.linalg.solve(lower, left_solved.T).T
    if not np.all(np.isfinite(matrix)):
        summary["reason"] = "Whitening solves produced nonfinite entries."
        return matrix, eigenvalues, summary
    antisymmetric = 0.5 * (matrix - matrix.T)
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(symmetric)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    summary["available"] = True
    summary["spectral_norm"] = float(singular_values[0])
    summary["minimum_symmetric_eigenvalue"] = float(eigenvalues[0])
    summary["maximum_symmetric_eigenvalue"] = float(eigenvalues[-1])
    summary["maximum_absolute_antisymmetric_entry"] = float(
        np.max(np.abs(antisymmetric))
    )
    return matrix, eigenvalues, summary


def compare(np, helper, candidate, reference, destination):
    """Save differences using only declared, nonzero reference denominators.

    Term-diagonal fractional differences use that term's own nonzero
    reference diagonal. Diagonal-scaled matrix differences use sqrt(C_ii C_jj)
    from the reference total, excluding nonpositive diagonals explicitly.
    Full covariance whitening is a separate metric and requires Cholesky
    factorization of the entire reference total, without selecting rows.
    Undefined entries are NaN in the NPZ arrays with matching validity masks;
    JSON contains counts and row indices, never a made-up denominator.
    """
    delta = candidate["terms"] - reference["terms"]
    reference_total = np.sum(reference["terms"], axis=0)
    candidate_total = np.sum(candidate["terms"], axis=0)
    total_delta = candidate_total - reference_total
    whitened, whitened_eigenvalues, whitening_summary = whitened_difference(
        np=np, reference=reference_total, difference=total_delta
    )
    reference_diagonal = np.diag(reference_total)
    valid_rows = reference_diagonal > 0.0
    indices = np.flatnonzero(valid_rows)
    normalized = np.full(shape=delta.shape, fill_value=np.nan)
    normalized_total = np.full(shape=total_delta.shape, fill_value=np.nan)
    matrix_summary = None
    if indices.size:
        scale = np.sqrt(reference_diagonal[indices])
        denominator = np.outer(scale, scale)
        selection = np.ix_(indices, indices)
        for term in range(delta.shape[0]):
            normalized[term][selection] = delta[term][selection] / denominator
        normalized_total[selection] = total_delta[selection] / denominator
        values = normalized_total[selection]
        eigenvalues = np.linalg.eigvalsh(values)
        matrix_summary = {
            "maximum_absolute_entry": float(np.max(np.abs(values))),
            "rms_entry": float(np.sqrt(np.mean(values**2))),
            "spectral_norm": float(np.max(np.abs(eigenvalues))),
        }
    fractional = np.full(shape=(6, reference_diagonal.size), fill_value=np.nan)
    valid_diagonals = np.zeros(shape=fractional.shape, dtype=bool)
    families = {}
    for term, name in enumerate(helper.TERM_NAMES):
        prior = np.diag(reference["terms"][term])
        difference = np.diag(delta[term])
        valid = prior != 0.0
        valid_diagonals[term] = valid
        fractional[term, valid] = difference[valid] / prior[valid]
        fractional_max = None
        if np.any(valid):
            fractional_max = float(np.max(np.abs(fractional[term, valid])))
        normalized_max = None
        if indices.size:
            normalized_max = float(np.max(np.abs(normalized[term][selection])))
        families[name] = {
            "maximum_absolute_difference": float(np.max(np.abs(delta[term]))),
            "maximum_absolute_diagonal_fraction": fractional_max,
            "defined_fractional_diagonal_count": int(np.count_nonzero(valid)),
            "zero_reference_diagonal_changed_rows": np.flatnonzero(
                (~valid) & (difference != 0.0)
            ).tolist(),
            "max_difference_in_total_diagonal_units": normalized_max,
        }
    np.savez_compressed(
        destination.with_suffix(".npz"),
        term_difference=delta,
        total_difference=total_delta,
        diagonal_scaled_term_difference=normalized,
        diagonal_scaled_total_difference=normalized_total,
        whitened_total_difference=whitened,
        whitened_symmetric_eigenvalues=whitened_eigenvalues,
        whitening_available=whitening_summary["available"],
        valid_total_diagonal=valid_rows,
        fractional_term_diagonal=fractional,
        valid_term_diagonal=valid_diagonals,
        mean_difference=candidate["mean"] - reference["mean"],
    )
    helper.save_json(path=destination.with_suffix(".json"), value={
        "sign": "candidate minus reference",
        "families": families,
        "diagonal_scaled_total": matrix_summary,
        "covariance_whitened_total": whitening_summary,
        "excluded_nonpositive_reference_diagonal_rows": np.flatnonzero(
            ~valid_rows
        ).tolist(),
        "reference_convergence_established": False,
    })


def main():
    """Write the study plan, and calculate it only with explicit --execute."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project", type=Path, default=Path(__file__).resolve().parents[2]
    )
    parser.add_argument("--extension", type=Path, required=True)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--point", type=int, choices=range(10), default=0)
    parser.add_argument(
        "--workers",
        type=int,
        choices=(
            1,
            2,
            4,
        ),
        default=4,
    )
    parser.add_argument("--include-bounds", action="store_true")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    for variable in (
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "BLIS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "GOTO_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    os.environ["OMP_NUM_THREADS"] = str(args.workers)
    os.environ["OMP_THREAD_LIMIT"] = "4"
    os.environ["OMP_DYNAMIC"] = "FALSE"
    project = args.project.resolve(strict=True)
    sys.path.insert(0, str(project / "tests" / "studies"))
    sys.path.insert(0, str(project / "scripts"))
    sys.path.insert(0, str(project.parents[1] / "external_modules/code/cosmolike_core"))
    import covariance_scaling as helper

    all_cases = case_settings(helper=helper, include_bounds=args.include_bounds)
    names = list(all_cases)
    if args.cases is not None:
        names = args.cases
        if len(set(names)) != len(names) or "baseline" not in names:
            raise ValueError("Select distinct cases, including baseline.")
        for name in names:
            if name not in all_cases:
                raise ValueError(f"Unknown case {name}; available: {list(all_cases)}")
    bank = helper.table_records(directory=args.tables, points=args.point + 1)
    selected = bank[args.point + 1]
    extension = helper.identity(path=args.extension)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("Choose a new output directory; existing results remain.")
    output.mkdir(parents=True)
    plan = {
        "format": "des3pcf-covariance-refinement-v1",
        "reference_point": selected,
        "point_origin": "explicit saved random bank point; not a survey fiducial",
        "extension": extension,
        "driver": helper.identity(path=Path(__file__)),
        "input_helper": helper.identity(path=Path(helper.__file__)),
        "workers": args.workers,
        "blas_workers": 1,
        "cases": {},
        "completed": False,
        "scope": "80 full-filter gravitational Map3 rows; declared area/noise",
        "limits": [
            "Processed sources do not establish a calibrated survey source model.",
            "No nonlinear-IA covariance or finite-SAS triangle selection.",
            "Single-fluid effective-growth massive-neutrino approximation.",
            "Selected halo-abundance response is not all SSC contributions.",
            "Saved P, distance, source samples and dense readers remain fixed.",
            "Extended k or ell bounds can test the saved power-law extrapolation.",
            "One refinement level and one physical point cannot establish convergence.",
            "Bounds remain fixed unless explicit bound cases are selected.",
        ],
    }
    for name in names:
        settings = all_cases[name]
        # B and closure filters alone need 20+16 dense triangle rows.
        geometry = settings["counts"]["nell"] ** 2 * settings["counts"]["nphi"]
        plan["cases"][name] = {
            "settings": settings,
            "triangle_table_bytes_lower_bound": 36 * geometry * 8,
        }
    helper.save_json(path=output / "plan.json", value=plan)
    if not args.execute:
        print(f"Resolved plan only: {output / 'plan.json'}")
        return

    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1, user_api="blas"):
        import numpy as np
        from des3pcf.physical_provider import load_tables
        from theory_inputs import physics_model

        sys.path.insert(0, str(Path(extension["path"]).parent))
        compiled = importlib.import_module(name=helper.EXTENSION_NAME)
        if Path(compiled.__file__).resolve() != Path(extension["path"]):
            raise RuntimeError(
                "Another compiled extension was imported; use a fresh CLI."
            )
        if helper.identity(path=extension["path"])["sha256"] != extension["sha256"]:
            raise RuntimeError("The chosen extension changed after plan preparation.")
        compiled.set_omp_threads(nthreads=args.workers)
        tables, metadata = load_tables(path=selected["path"])
        if metadata.get("scaling_point", {}).get("index") != args.point:
            raise ValueError("The saved physical point does not match --point.")
        if tables["sources"].shape[0] != 4:
            raise ValueError("The controlled row construction requires four source bins.")
        helper.save_json(path=output / "physical_input_metadata.json", value=metadata)
        results = {}
        for name in names:
            settings = all_cases[name]
            lower, upper = settings["los_bounds"]
            if upper > tables["z"][-1]:
                raise ValueError(f"Case {name} exceeds the saved redshift support.")
            current = dict(tables)
            current["los_z"] = np.linspace(
                start=lower, stop=upper, num=settings["counts"]["nlos"]
            )
            from theory.bihalofit.model import bispectrum_axes, prepare_reference_bispectrum

            model_tables, baseline = prepare_reference_bispectrum(
                tables=current, axes=bispectrum_axes(zmax=current["z"][-1]),
                fit_nz=settings["fit_nz"], table_boost=settings["table_boost"],
                source_readers=settings["source_readers"],
            )
            model = physics_model(
                compiled=compiled,
                tables=model_tables, baseline=baseline,
                kernel_nz=settings["kernel_nz"],
                table_boost=settings["table_boost"],
                fit_nz=settings["fit_nz"],
                source_readers=settings["source_readers"],
            )
            rows, inputs = helper.build_inputs(settings=settings)
            mean, terms = compiled.production.physical_covariance(
                model=model, inputs=inputs
            )
            if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(terms)):
                raise RuntimeError(f"Case {name} produced nonfinite entries.")
            total = np.sum(terms, axis=0)
            diagnostics, eigenvalues = array_summary(np=np, matrix=total)
            family_eigenvalues = np.empty(shape=(6, total.shape[0]))
            diagnostics["families"] = {}
            for term, family in enumerate(helper.TERM_NAMES):
                item, values = array_summary(np=np, matrix=terms[term])
                diagnostics["families"][family] = item
                family_eigenvalues[term] = values
            if np.any(terms[3] != 0.0):
                raise RuntimeError("Pure Map3 rows must have zero P*B cross covariance.")
            np.savez_compressed(
                output / f"{name}.npz",
                mean=mean,
                terms=terms,
                total=total,
                total_eigenvalues=eigenvalues,
                family_eigenvalues=family_eigenvalues,
                row_labels=rows.labels,
            )
            helper.save_json(path=output / f"{name}.json", value=diagnostics)
            results[name] = {"mean": mean, "terms": terms}
            print(f"Saved {name}: {rows.values.size} rows", flush=True)
            del model
        for name in names:
            if name == "baseline":
                continue
            reference = "baseline"
            resolution_case = name == "combined"
            resolution_case = resolution_case or name in all_cases["baseline"]["counts"]
            if not resolution_case:
                if "combined" not in results:
                    continue
                reference = "combined"
            compare(
                np=np,
                helper=helper,
                candidate=results[name],
                reference=results[reference],
                destination=output / f"difference_{name}_minus_{reference}",
            )
        plan["completed"] = True
        helper.save_json(path=output / "plan.json", value=plan)


if __name__ == "__main__":
    main()
