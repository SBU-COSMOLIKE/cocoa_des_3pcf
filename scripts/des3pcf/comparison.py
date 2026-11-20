"""Run small, declared comparisons with unmodified original source files.

Live original-code runs use separate Python processes. Each process reads
verified source files and its own configuration, so fastnc's class-level
dictionaries cannot affect a later notebook calculation. No source module
is patched, no missing package is installed, and no survey likelihood is
implied by these finite-operator checks. All physical arrays use radians
and the conventions named in each result.
"""

from theory.bihalofit.variance import reference_physics_model, reference_bihalofit

import argparse
import hashlib
import importlib.util
import json
import yaml
import os
from pathlib import Path
import subprocess
import sys
import tempfile


PROJECT = Path(__file__).resolve().parents[2]
MANIFEST = PROJECT / "data" / "code_comparison_manifest.yaml"
FORMAT = "des3pcf-code-comparison-v1"
BLAS_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "GOTO_NUM_THREADS",
)


def validate_comparison_settings(settings):
    """Check explicit notebook settings without reading a configuration file.

    Arguments:
      settings = source locations, output location and small-grid controls.
        Paths must be absolute; the notebook constructs them from its project
        directory. None marks an unavailable original library.
    Returns:
      an owned dictionary with normalized string paths for child processes.
    Raises:
      ValueError for unknown fields, relative paths or invalid grid/thread counts.
    """
    required = {"format", "sources", "output", "threads", "nfft",
                "lmax", "mmax", "phi_count", "frozen_physics"}
    if not isinstance(settings, dict) or set(settings) != required:
        raise ValueError(
            f"Comparison settings must contain exactly {sorted(required)}."
        )
    settings = dict(settings)
    if settings["format"] != FORMAT:
        raise ValueError(f"Comparison format must be {FORMAT}.")
    limits = {
        "threads": (1, 4),
        "nfft": (8, 64),
        "lmax": (0, 6),
        "mmax": (0, 8),
        "phi_count": (3, 16),
    }
    for name, bounds in limits.items():
        value = settings[name]
        if (isinstance(value, bool) or not isinstance(value, int)
                or not bounds[0] <= value <= bounds[1]):
            raise ValueError(f"{name} must be an integer in {bounds}.")
    if settings["nfft"] % 2:
        raise ValueError("The comparison FFT length must be even.")
    if not isinstance(settings["frozen_physics"], bool):
        raise ValueError("frozen_physics must be true or false.")
    if not isinstance(settings["sources"], dict):
        raise ValueError(
            "sources must name tatt, main and integrator checkouts."
        )
    if set(settings["sources"]) != {"tatt", "main", "integrator"}:
        raise ValueError("sources must name exactly tatt, main and integrator.")
    sources = {}
    for name, value in settings["sources"].items():
        if value is None:
            sources[name] = None
        elif isinstance(value, (str, Path)) and str(value).strip():
            source = Path(value).expanduser()
            if not source.is_absolute():
                raise ValueError(f"Source {name} must be an absolute path.")
            sources[name] = str(source.resolve())
        else:
            raise ValueError(f"Source {name} needs a path or null.")
    if (not isinstance(settings["output"], (str, Path))
            or not str(settings["output"]).strip()):
        raise ValueError("output must name a writable result directory.")
    output = Path(settings["output"]).expanduser()
    if not output.is_absolute():
        raise ValueError("output must be an absolute path.")
    settings["sources"] = sources
    settings["output"] = str(output.resolve())
    return settings


def comparison_status(settings):
    """Check source hashes and dependency availability without scientific work.

    An unavailable or modified checkout is reported explicitly. The runner
    will skip it rather than compare against a different source silently.
    A commit label is supplemented by hashes of every imported TATT Python
    file and of the individual main/integrator files used here.
    """
    settings = validate_comparison_settings(settings=settings)
    manifest = yaml.safe_load(MANIFEST.read_text())
    report = {"format": FORMAT, "sources": {}, "dependencies": {}}
    for package in ("numpy", "scipy", "astropy", "pandas", "mpi4py",
                    "cosmosis", "treecorr"):
        available = importlib.util.find_spec(package) is not None
        report["dependencies"][package] = available
    for name, specification in manifest.items():
        directory = settings["sources"][name]
        record = dict(specification)
        record["path"] = directory
        record["status"] = "verified"
        record["problems"] = []
        if directory is None:
            record["status"] = "unavailable"
            record["problems"].append("No source checkout was configured.")
        else:
            for relative, expected in specification["sha256"].items():
                source_file = Path(directory) / relative
                if not source_file.is_file():
                    record["problems"].append(f"Missing {relative}")
                else:
                    source_bytes = source_file.read_bytes()
                    actual = hashlib.sha256(source_bytes).hexdigest()
                    if actual != expected:
                        record["problems"].append(f"SHA256 differs: {relative}")
            if record["problems"]:
                record["status"] = "unavailable"
        report["sources"][name] = record
    report["scope"] = (
        "Finite transform/aperture checks; no original CosmoSIS chain or "
        "TreeCorr catalogue measurement is executed."
    )
    return report


def _worker_environment(threads):
    """Set thread limits before the child imports NumPy or the extension."""
    environment = os.environ.copy()
    for name in BLAS_VARIABLES:
        environment[name] = "1"
    environment["OMP_NUM_THREADS"] = str(threads)
    environment["OMP_DYNAMIC"] = "FALSE"
    environment["MKL_DYNAMIC"] = "FALSE"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join([
        str(PROJECT / "scripts"),
        str(PROJECT.parents[1] / "external_modules/code/cosmolike_core"),
    ])
    if sys.platform == "darwin":
        environment["OMP_PROC_BIND"] = "disabled"
    else:
        environment["OMP_PROC_BIND"] = "close"
    return environment


def run_code_comparison(settings):
    """Run available comparisons serially and save arrays plus provenance.

    Arguments:
      settings = explicit source paths, output path and numerical controls.
    Returns:
      report dictionary containing source/dependency status and case records.
      A case's NPZ file contains complete port/reference outputs and a small
      plotting slice. An unequal-filter discrepancy is reported as expected,
      never relabeled as agreement.
    Side effects:
      Writes only the configured output directory, starts one child at a
      time, and sets every BLAS vendor to one thread in those children.
    """
    settings = validate_comparison_settings(settings=settings)
    report = comparison_status(settings=settings)
    output = Path(settings["output"])
    output.mkdir(parents=True, exist_ok=True)
    report["configuration"] = settings
    report["cases"] = []
    tasks = ["tatt", "main", "integrator"]
    if settings["frozen_physics"]:
        tasks.append("frozen")
    for task in tasks:
        if task != "frozen" and report["sources"][task]["status"] != "verified":
            report["cases"].append({"name": task, "status": "unavailable"})
            continue
        temporary_directory = tempfile.TemporaryDirectory(
            prefix="comparison-", dir=output
        )
        with temporary_directory as temporary:
            request_file = Path(temporary) / "request.yaml"
            request_file.write_text(yaml.safe_dump(settings, sort_keys=False))
            result_file = output / f"{task}_result.yaml"
            process = subprocess.run(
                args=[sys.executable, "-B", "-m", "des3pcf.comparison",
                      "--worker", task, "--request", str(request_file),
                      "--result", str(result_file)],
                env=_worker_environment(threads=settings["threads"]),
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
            (output / f"{task}_stdout.txt").write_text(process.stdout)
            (output / f"{task}_stderr.txt").write_text(process.stderr)
            if process.returncode != 0:
                report["cases"].append({
                    "name": task,
                    "status": "failed",
                    "returncode": process.returncode,
                    "diagnostic": process.stderr[-4000:],
                })
            else:
                report["cases"].extend(yaml.safe_load(result_file.read_text()))
    report_file = output / "comparison_report.yaml"
    report_file.write_text(yaml.safe_dump(report, sort_keys=False))
    report["report_file"] = str(report_file)
    return report


def load_comparison_case(case):
    """Return owned plotting arrays from a completed case record."""
    import numpy as np

    if "arrays" not in case:
        raise ValueError("This comparison has no numerical arrays.")
    with np.load(case["arrays"], allow_pickle=False) as saved:
        return {
            "x": saved["x"].copy(),
            "reference": saved["plot_reference"].copy(),
            "ported": saved["plot_ported"].copy(),
            "labels": saved["labels"].tolist(),
        }


def comparison_case(report, name):
    """Select one named result, refusing absent or failed calculations."""
    for case in report["cases"]:
        if case["name"] == name:
            if case["status"] != "completed":
                raise ValueError(f"{name} did not complete: {case}")
            return case
    raise ValueError(f"No completed comparison named {name} is present.")


def comparison_summary(report):
    """Return a short display table without hiding unavailable cases."""
    import pandas as pd

    rows = []
    if "cases" in report:
        for case in report["cases"]:
            rows.append({
                "comparison": case["name"],
                "status": case["status"],
                "reference": case.get("kind", "not executed"),
                "max scaled difference": case.get("max_scaled_error"),
                "expected difference": case.get("expected_discrepancy", False),
            })
    else:
        for name, source in report["sources"].items():
            rows.append({
                "source": name,
                "commit": source["commit"],
                "status": source["status"],
                "diagnostic": "; ".join(source["problems"]),
            })
    return pd.DataFrame(rows)


def _load_original_file(path, name):
    """Execute an original standalone file without changing its contents."""
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _save_case(settings, name, reference, ported, x, plot_reference,
               plot_ported, labels, note, expected_discrepancy=False,
               frozen=False):
    """Store arrays and a scaled diagnostic that remains defined at zero."""
    import numpy as np

    if reference.shape != ported.shape:
        raise ValueError(f"{name}: reference and port shapes differ.")
    if not np.all(np.isfinite(reference)) or not np.all(np.isfinite(ported)):
        raise ValueError(f"{name}: comparison contains nonfinite output.")
    scale = float(np.max(np.abs(reference)))
    if scale == 0.0:
        raise ValueError(
            f"{name}: identically zero reference gives no reach check."
        )
    error = float(np.max(np.abs(ported - reference)) / scale)
    filename = Path(settings["output"]) / f"{name}.npz"
    np.savez(
        filename,
        reference=reference,
        ported=ported,
        x=x,
        plot_reference=plot_reference,
        plot_ported=plot_ported,
        labels=np.asarray(labels),
    )
    return {
        "name": name,
        "status": "completed",
        "kind": "frozen_reference" if frozen else "live_original_code",
        "expected_discrepancy": expected_discrepancy,
        "max_scaled_error": error,
        "reference_scale": scale,
        "shape": list(reference.shape),
        "arrays": str(filename),
        "note": note,
    }


def _bessel_comparison(settings, compiled, original, corrected):
    """Compare identical finite FFTLog operators with their declared taper."""
    import numpy as np

    ell = np.geomspace(start=0.05, stop=30.0, num=settings["nfft"])
    integrand = np.exp(-0.2 * ell[:, None] - 0.4 * ell[None, :])
    integrand *= ell[:, None] ** 1.2 * ell[None, :] ** 0.8
    source = original.two_Bessel(
        x1=ell, x2=ell, fx1x2=integrand, nu1=1.13, nu2=0.89,
        xy=1.7, c_window_width=0.25, N_pad=0,
    )
    reference = source.two_Bessel_binave(
        ell1=1, ell2=3, binwidth_dlny1=0.12, binwidth_dlny2=0.12,
    )[2]
    ported = compiled.production.two_bessel(
        ell=ell, integrand=integrand, order1=1, order2=3,
        nu1=1.13, nu2=0.89, xy=1.7, dlnt=0.12, padding=0,
        window=corrected, window_width=0.25,
    )
    name = "bessel_main_taper" if corrected else "bessel_tatt_untapered"
    middle = ell.size // 2
    return _save_case(
        settings=settings, name=name, reference=reference, ported=ported,
        x=1.7 / ell[::-1], plot_reference=reference[middle:middle + 1],
        plot_ported=ported[middle:middle + 1], labels=["J1/J3 transform"],
        note=("Main applies the Fourier-coefficient taper; TATT computes "
              "and discards it. Each port call matches its named source mode."),
    )


class _ControlledMultipoles:
    """Supply a symmetric smooth test bispectrum through fastnc's public API.

    This is a controlled input, not a cosmological fit. Holding these
    multipoles fixed isolates the coupling, Fourier transform, bin average
    and shear-rotation operations from the physical-model calculation.
    Symmetry under ell1/ell2 exchange satisfies TATT's negative-M shortcut.
    """

    multipole_type = "legendre"

    def kappa_bispectrum_multipole(self, L, ELL, PSI, **arguments):
        """Return L-indexed Gaussian test rows on the requested source grid."""
        import numpy as np

        if arguments:
            raise ValueError(
                "The controlled provider has no source-bin parameters."
            )
        rows = []
        for degree in L:
            radial = np.exp(-0.03 * ELL**2)
            shape = np.sin(2 * PSI) ** (2 * int(degree))
            rows.append(radial * shape / (1 + degree)**2)
        return np.asarray(rows, dtype=np.float64)


def _natural_comparison(settings, compiled, package):
    """Call public FastNaturalComponents with cache and grid tuning disabled."""
    import numpy as np

    phi = np.linspace(start=0.18, stop=2.95, num=settings["phi_count"])
    calculator = package.fastnc.FastNaturalComponents(
        verbose=False, Lmax=settings["lmax"], Mmax=settings["mmax"],
        Lmax_diag=settings["lmax"], multipole_type="legendre",
        cache=False, use_GLM_table=False, auto=False,
        ell1min=0.05, ell1max=30.0, nfft=settings["nfft"],
        t1=None, phi=phi, mu=[0, 1, 2, 3], dlnt=0.12,
        nu1=1.01, nu2=1.01, N_pad=0, xy=1.0, projection="cent",
    )
    provider = _ControlledMultipoles()
    calculator.set_bispectrum(bispectrum=provider)
    calculator.compute()
    reference = np.asarray([
        calculator.Gamma0, calculator.Gamma1,
        calculator.Gamma2, calculator.Gamma3,
    ])
    ell = calculator.ell1_fft.copy()
    multipoles = provider.kappa_bispectrum_multipole(
        L=np.arange(settings["lmax"] + 1),
        ELL=calculator.ELL_FFT, PSI=calculator.PSI_FFT,
    )
    middle = ell.size // 2
    opening = phi.size // 2
    reference_slice = reference[:, middle, :, opening]
    source_psi = calculator.GLM.psi.copy()
    source_coupling = calculator.GLM(
        np.arange(settings["lmax"] + 1),
        np.arange(settings["mmax"] + 1), source_psi,
    ).transpose(1, 0, 2)
    analytic_coupling = compiled.production.legendre_coupling(
        psi=source_psi, lmax=settings["lmax"], mmax=settings["mmax"], npsi=0,
    )
    coupling_labels = []
    for mode in range(settings["mmax"] + 1):
        for degree in range(settings["lmax"] + 1):
            coupling_labels.append(f"M={mode}, L={degree}")
    coupling = _save_case(
        settings=settings, name="coupling_tatt_rectangle_bias",
        reference=source_coupling, ported=analytic_coupling, x=source_psi,
        plot_reference=source_coupling.reshape(-1, source_psi.size),
        plot_ported=analytic_coupling.reshape(-1, source_psi.size),
        labels=coupling_labels, expected_discrepancy=True,
        note=("The original left-endpoint rectangle integrator caps at 4096 "
              "samples. At L=M=0, psi=0, its integral is 4*pi^2/4096 "
              "although the continuum result is zero. The coupling class "
              "discards the integrator's convergence flag."),
    )
    coupling["source_left_endpoint_bias"] = float(source_coupling[0, 0, 0])
    coupling["rectangle_bias_prediction"] = float(4 * np.pi**2 / 4096)
    # The integration function itself exposes max_itern even though the
    # public natural-component constructor does not pass that control on.
    # A known zero integral isolates its endpoint rule without changing
    # any method, class or source file.
    def zero_integrand(angle):
        return 4 * np.pi * np.cos(angle)

    ladder = []
    for iteration_limit in (6, 8, 10, 12):
        value, converged = package.integration.aint(
            fnc=zero_integrand, xmin=0.0, xmax=np.pi, Nx=2,
            tol=1.0e-12, max_itern=iteration_limit,
        )
        count = 2 ** (iteration_limit + 2)
        ladder.append({
            "left_rectangle_count": count,
            "source_integral": float(value),
            "predicted_bias": float(4 * np.pi**2 / count),
            "converged": bool(converged),
        })
    coupling["rectangle_bias_ladder"] = ladder
    results = [coupling]
    for exact in (False, True):
        # Read the actual source table length. Its metadata field Npsi
        # says 201 while the default self.psi array contains 200 nodes.
        # Matching that array isolates numerical angular integration from
        # the separate choice to interpolate at all.
        npsi = 0 if exact else calculator.GLM.psi.size
        plan = compiled.production.TransformPlan(
            ell=ell, phi=phi, indices=np.arange(ell.size, dtype=np.int32),
            lmax=settings["lmax"], mmax=settings["mmax"], npsi=npsi,
            nu1=1.01, nu2=1.01, xy=1.0, dlnt=0.12,
            padding=0, window=False, projection="centroid", ordered_legs=False,
        )
        real, imaginary = plan.natural_components(b_l=multipoles)
        ported = real + 1j * imaginary
        ported_slice = ported[:, middle, :, opening]
        if exact:
            name = "natural_exact_coupling"
            note = (
                "The same live TATT output is compared with exact C angular "
                "coupling. This deliberately removes the source's 200-node "
                "interpolation; its difference is not a source-agreement error."
            )
        else:
            name = "natural_tatt_public"
            note = (
                "Live public TATT with centroid projection and matching "
                "200-node linear angular interpolation. Source nodes retain "
                "the demonstrated finite rectangle-integration bias; C "
                "nodes use analytic coupling. This is not a DES error bound."
            )
        results.append(_save_case(
            settings=settings, name=name, reference=reference, ported=ported,
            x=1 / ell[::-1],
            plot_reference=np.concatenate(
                (reference_slice.real, reference_slice.imag)
            ),
            plot_ported=np.concatenate((ported_slice.real, ported_slice.imag)),
            labels=[
                r"$\mathrm{Re}\,\Gamma_0$",
                r"$\mathrm{Re}\,\Gamma_1$",
                r"$\mathrm{Re}\,\Gamma_2$",
                r"$\mathrm{Re}\,\Gamma_3$",
                r"$\mathrm{Im}\,\Gamma_0$",
                r"$\mathrm{Im}\,\Gamma_1$",
                r"$\mathrm{Im}\,\Gamma_2$",
                r"$\mathrm{Im}\,\Gamma_3$",
            ],
            note=note, expected_discrepancy=exact,
        ))
    return results


def _aperture_comparisons(settings, compiled, original):
    """Exercise equal filters and display the original unequal-filter error."""
    import numpy as np

    d2 = np.geomspace(start=0.002, stop=0.02, num=5)
    d3 = np.geomspace(start=0.003, stop=0.03, num=5)
    phi = np.linspace(start=0.2, stop=2.9, num=7)
    dlnd = float(np.log(d2[1] / d2[0]))
    dphi = float(phi[1] - phi[0])
    side2, side3, opening = np.meshgrid(d2, d3, phi, indexing="ij")
    envelope = np.exp(-(side2 / 0.015)**2 - (side3 / 0.019)**2)
    gamma = []
    for component in range(4):
        gamma.append(envelope * np.exp(1j * (component + 1) * opening / 3))
    gamma = np.asarray(gamma)
    radii = np.asarray([0.004, 0.008, 0.012])
    results = []
    for unequal in (False, True):
        ratios = np.asarray([1.0, 1.35, 1.8]) if unequal else np.ones(3)
        filters = np.ascontiguousarray(radii[:, None] * ratios[None, :])
        reference = original.calculateMap3(
            three_pt=gamma, d2_vals=side2, d3_vals=side3, phi_vals=opening,
            logr_bin_size=dlnd, phi_bin_size=dphi, filters=filters.T.copy(),
        )
        wr, wi = compiled.production.aperture_weights(
            d2=d2, d3=d3, phi=phi, dlnd=dlnd, dphi=dphi, filters=filters,
        )
        ported = compiled.production.aperture_apply(
            wr=wr, wi=wi,
            gr=np.ascontiguousarray(gamma.real.reshape(1, 4, -1)),
            gi=np.ascontiguousarray(gamma.imag.reshape(1, 4, -1)),
        )[0]
        name = "aperture_unequal" if unequal else "aperture_equal"
        results.append(_save_case(
            settings=settings, name=name, reference=reference, ported=ported,
            x=radii, plot_reference=reference[None, :],
            plot_ported=ported[None, :], labels=["Map3"],
            expected_discrepancy=unequal,
            note=("Rafael's adapted NumPy function is executed unchanged. "
                  "Its k2.all()==1/k3.all()==1 test treats every nonzero "
                  "ratio as equal to one, selecting the equal-filter kernel "
                  "for these unequal filters. The port uses the general "
                  "kernel."),
        ))
    return results


def _frozen_physics(settings, compiled):
    """Compare the C model with saved outputs; no original physics runs here."""
    import numpy as np

    # These deterministic tables define the existing frozen physics fixture.
    # Keeping the input definition explicit avoids importing pytest in a
    # user notebook. Its identity is checked by the comparison tests.
    k = np.geomspace(start=1.0e-5, stop=1.0e3, num=96)
    z = np.linspace(start=0.0, stop=2.0, num=41)
    linear = 2.0e4 * k / (1.0 + (k / 0.15)**2)**1.8
    nonlinear0 = linear * (1.0 + (k / 0.35)**2)**0.22
    sources = np.empty(shape=(2, z.size), dtype=np.float64)
    sources[0] = z**2 * np.exp(-(z / 0.6)**1.5)
    sources[1] = z**2 * np.exp(-(z / 0.95)**1.5)
    inputs = dict(
        k=k, p_linear0=linear, z=z,
        p_nonlinear=nonlinear0[None, :] / (1 + z[:, None])**2.4,
        chi=3000 * z / (1 + 0.35 * z),
        dz_dchi=(1 + 0.35 * z)**2 / 3000,
        growth=1 / (1 + z), sources=sources,
        omega_m=0.3, omega_de=0.7, w0=-1.0, wa=0.0,
        ns=0.965, sigma8=0.8,
        los_z=np.linspace(start=0.005, stop=2.0, num=25),
        kernel_nz=101, source_readers=True, table_boost=8,
    )
    model = reference_physics_model(factory=compiled.production.PhysicsModel, **inputs)
    theory = reference_bihalofit(inputs=inputs)
    filename = PROJECT / "tests/data_vector/reference_physics_fastnc.json"
    record = json.loads(filename.read_text())
    sides = np.asarray(record["sides"], dtype=np.float64)
    redshift = np.asarray(record["redshifts"], dtype=np.float64)
    matter = []
    for component in range(4):
        matter.append(theory.evaluate(
            k1=sides[:, 0], k2=sides[:, 1], k3=sides[:, 2], z=redshift,
            component=component,
        ))
    matter = np.asarray(matter)
    reference = np.asarray(record["matter"])
    results = [_save_case(
        settings=settings, name="physics_matter_frozen", reference=reference,
        ported=matter, x=np.arange(sides.shape[0]),
        plot_reference=reference, plot_ported=matter,
        labels=["total", "tree", "one halo", "three halo"], frozen=True,
        note=("Python theory versus stored source outputs; no live original-code run. The "
              "source nonlinear-radius root permits the established 2e-4 "
              "matter comparison tolerance; the tree has a tighter test."),
    )]
    for mode in (0, 1):
        parameters = dict(record["parameters"], c1rho=0.0134)
        parameters["power_mode"] = mode
        actual = model.ia(
            sides=sides, redshifts=redshift, parameters=parameters
        )
        expected = np.asarray(record["alignment"][str(mode)])
        labels = []
        for component in range(19):
            labels.append(f"IA {component}")
        results.append(_save_case(
            settings=settings, name=f"physics_ia_mode{mode}_frozen",
            reference=expected, ported=actual, x=np.arange(sides.shape[0]),
            plot_reference=expected.T, plot_ported=actual.T, labels=labels,
            frozen=True,
            note=("Stored 19-component source outputs. A right-triangle ddB "
                  "zero is checked with its vertex's roundoff scale by the "
                  "separate high-precision tests; a relative error at zero "
                  "is not a useful diagnostic."),
        ))
    for result in results:
        fixture_hash = hashlib.sha256(filename.read_bytes()).hexdigest()
        result["fixture_sha256"] = fixture_hash
        result["source_commit"] = record["source_commit"]
    return results


def _worker(task, settings):
    """Dispatch one isolated run after its process thread limits are set."""
    sys.path.insert(0, str(PROJECT / "interface"))
    import cosmolike_des_3pcf_interface as compiled

    compiled.set_omp_threads(nthreads=settings["threads"])
    if task == "frozen":
        return _frozen_physics(settings=settings, compiled=compiled)
    directory = Path(settings["sources"][task])
    if task == "tatt":
        sys.path.insert(0, str(directory))
        import fastnc

        if Path(fastnc.__file__).resolve().parent != directory / "fastnc":
            raise RuntimeError("Python imported a different fastnc checkout.")
        bessel = _bessel_comparison(
            settings=settings, compiled=compiled,
            original=fastnc.twobessel, corrected=False,
        )
        natural = _natural_comparison(
            settings=settings, compiled=compiled, package=fastnc,
        )
        return [bessel] + natural
    if task == "main":
        original = _load_original_file(
            path=directory / "fastnc/hankel/twobessel.py",
            name="original_main_bessel",
        )
        return [_bessel_comparison(
            settings=settings, compiled=compiled, original=original,
            corrected=True,
        )]
    original = _load_original_file(
        path=directory / "python/fast_map3.py", name="original_rafael_map3",
    )
    return _aperture_comparisons(
        settings=settings, compiled=compiled, original=original,
    )


def main():
    """Internal child-process command; notebooks use run_code_comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--worker", choices=("tatt", "main", "integrator", "frozen"),
        required=True,
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    arguments = parser.parse_args()
    settings = yaml.safe_load(arguments.request.read_text())
    result = _worker(task=arguments.worker, settings=settings)
    arguments.result.write_text(yaml.safe_dump(result, sort_keys=False))


if __name__ == "__main__":
    main()
