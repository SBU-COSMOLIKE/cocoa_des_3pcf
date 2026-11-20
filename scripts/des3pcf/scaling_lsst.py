"""Prepare the existing LSST Y1 3x2pt calculation for a scaling study.

This module does not time anything. The controller decides when the
machine is quiet, prepares CAMB for eleven declared points, and measures
LSSTWorkload.evaluate at the ten random points. Each call installs that
point's cosmology and nuisances through the existing setters. Angular
kernels and FFT plans remain allocated between physical evaluations.

The LSST library has process-global state. Use one workload in its own
child process, with a copied extension and one BLAS thread. No LSST file
is changed. The default is the live EXAMPLE_EVALUATE2 cosmology and its
65000 multipole limit, with an all-ones mask to compute every one of the
1560 standard entries: xi+, xi-, tangential shear and lens clustering.
The existing LSST random-model test supplies the ten points. Process the
whole bank for each thread count so successive points change the state.
"""

import copy
import hashlib
import importlib
import importlib.util
import json
import math
from pathlib import Path
import sys


FORMAT = "des3pcf-lsst-scaling-tables-v2"
MODULE = "cosmolike_lsst_y1_interface"
TABLE_NAMES = (
    "log10k_2D",
    "z_2D",
    "lnP_linear",
    "lnP_nonlinear",
    "G",
    "z_G",
    "z_1D",
    "chi",
    "omegan2",
    "lnP_linear_cb",
)


def default_lsst_settings(cocoa_root, ia_model="nla"):
    """Return explicit LSST inputs; cocoa_root is the Cocoa directory.

    Paths are absolute. The returned dictionary owns its nested lists,
    so changing it does not change another workload's configuration.
    The all-ones mask selects the full vector rather than the M1 subset
    of the example likelihood. ia_model selects "nla" (native code 0)
    or "tatt" (code 1), both with CFASTPT code 0 and redshift evolution
    code 3. TATT starts with the existing test's nonzero quadratic IA
    amplitudes; evaluate subsequently installs each sampled point's IA.
    No covariance is computed here.
    """
    if ia_model not in ("nla", "tatt"):
        raise ValueError("LSST ia_model must be 'nla' or 'tatt'.")
    model_code = 0
    quadratic_amplitude = 0.0
    if ia_model == "tatt":
        model_code = 1
        quadratic_amplitude = 0.05
    root = Path(cocoa_root).expanduser().resolve()
    data = root / "projects" / "lsst_y1" / "data"
    extension = root / "projects" / "lsst_y1" / "interface"
    return {
        "cocoa_root": str(root),
        "extension_path": str(extension / (MODULE + ".so")),
        "source_file": str(data / "lsst_y1_source.nz"),
        "lens_file": str(data / "lsst_y1_lens.nz"),
        "covariance_file": str(data / "lsst_y1_cov"),
        "mask_file": str(data / "ones.mask"),
        "data_file": str(data / "lsst_y1_theory.modelvector"),
        "probes": "3x2pt",
        "nsource": 5,
        "nlens": 5,
        "ntheta": 26,
        "theta_min_arcmin": 2.5,
        "theta_max_arcmin": 900.0,
        "lmax": 65000,
        "accuracy_boost": 1.0,
        "integration_accuracy": 0,
        "internal_accuracy_boost": 1.0,
        "nonlimber_accuracy_boost": 1.0,
        "photoz_histogram_akima": 0,
        "photoz_preparation_factor": 3,
        "photoz_interpolation_type": 0,
        "photoz_zmid_convention": 0,
        "adopt_limber_gs": 0,
        "adopt_limber_gg": 0,
        "IA_model": model_code,
        "IA_code": 0,
        "IA_redshift_evolution": 3,
        "bias_model": [
            0,
            0,
            0,
            1,
            0,
            0,
        ],
        "include_HOD_GX": 0,
        "include_halo_IA": 0,
        "nonlimber_lowrank_gg": 0,
        "nonlimber_lowrank_rank": 5,
        "nonlimber_lowrank_lmax": 150,
        "cosmology": {
            "omegam": 0.3,
            "omegab": 0.04,
            "H0": 67.32,
            "ns": 0.96605,
            "As_1e9": 2.1,
            "w": -0.9,
            "w0pwa": -0.9,
            "mnu": 0.06,
        },
        "camb": {
            "AccuracyBoost": 1.0,
            "kmax": 5.0,
            "k_per_logint": 10,
            "CAMBAccuracyBoost": 1.0,
            "CLAccuracyBoost": 1.0,
            "non_linear_emul": 2,
            "lens_potential_accuracy": 1.0,
            "halofit_version": "takahashi",
        },
        "nuisance": {
            "M": [
                0.0191832,
                -0.0431752,
                -0.034961,
                -0.0158096,
                -0.0158096,
            ],
            "source_photoz": [
                0.0414632,
                0.00147332,
                0.0237035,
                -0.0773436,
                -0.0000867127,
            ],
            "lens_photoz": [
                0.00457604,
                0.000309875,
                0.00855907,
                -0.00316269,
                -0.0146753,
            ],
            "A1": [
                0.606102,
                -1.51541,
                0.0,
                0.0,
                0.0,
            ],
            "A2": [
                quadratic_amplitude,
                -1.51541,
                0.0,
                0.0,
                0.0,
            ],
            "B_TA": [
                quadratic_amplitude,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            "B1": [
                1.72716,
                1.65168,
                1.61423,
                1.92886,
                2.11633,
            ],
            "B2": [0.0]*5,
            "B_MAG": [0.0]*5,
            "B3nl": [0.0]*5,
            "BK": [0.0]*5,
            "PMV": [0.0]*5,
        },
    }


def _validated_settings(settings):
    """Check the declared benchmark before touching compiled globals.

    The numerical settings and model families must match the stated
    defaults. A separate sampled-point dictionary supplies cosmology and
    nuisances. Paths may point to copies, but their contents are checked
    again before initialization. This does not parse a second likelihood.
    """
    if not isinstance(settings, dict) or "cocoa_root" not in settings:
        raise ValueError("LSST settings need the explicit cocoa_root field.")
    code = settings.get("IA_model")
    if type(code) is not int or code not in (0, 1):
        raise ValueError("LSST IA_model must be integer 0 (NLA) or 1 (TATT).")
    model_name = "nla"
    if code == 1:
        model_name = "tatt"
    expected = default_lsst_settings(
        settings["cocoa_root"], ia_model=model_name,
    )
    if set(settings) != set(expected):
        raise ValueError("LSST settings must include every declared field.")
    result = copy.deepcopy(settings)
    for name, value in expected.items():
        if name.endswith("_file") or name.endswith("_path"):
            result[name] = str(Path(settings[name]).expanduser().resolve())
        elif name == "cocoa_root":
            result[name] = value
        elif (json.dumps(settings[name], sort_keys=True)
              != json.dumps(value, sort_keys=True)):
            raise ValueError(
                f"LSST scaling field {name} differs from its declared "
                "EXAMPLE_EVALUATE2 workload. Start a separate study for "
                "different scientific settings."
            )
    return result


def _file_record(path):
    """Read a file hash in bounded blocks without loading numerical data."""
    filename = Path(path)
    if not filename.is_file():
        raise FileNotFoundError(f"LSST input file is missing: {filename}")
    digest = hashlib.sha256()
    with filename.open("rb") as stream:
        while True:
            block = stream.read(1024*1024)
            if not block:
                break
            digest.update(block)
    return {
        "path": str(filename.resolve()),
        "bytes": filename.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def preflight_lsst(settings, extension_path=None):
    """Inventory inputs without importing CAMB or the LSST extension.

    The optional extension_path is the copied .so file that the worker
    will import. File hashes describe the actual inputs, including the
    covariance needed by the production vector entry point. This performs
    file I/O only; it neither initializes CosmoLike nor computes a vector.
    """
    resolved = _validated_settings(settings)
    if extension_path is not None:
        resolved["extension_path"] = str(Path(extension_path).resolve())
    extension = Path(resolved["extension_path"])
    if extension.name != MODULE + ".so":
        raise ValueError("The copied LSST extension must keep its .so name.")
    files = {}
    for name in (
        "extension_path",
        "source_file",
        "lens_file",
        "covariance_file",
        "mask_file",
        "data_file",
    ):
        files[name] = _file_record(resolved[name])
    return {
        "settings": resolved,
        "files": files,
        "expected_block_sizes": [
            780,
            650,
            130,
        ],
        "expected_vector_size": 1560,
        "timed_call": "set_cosmology; set_nuisance_*; "
                      "compute_data_vector_masked",
        "excluded": [
            "CAMB preparation and file I/O",
            "extension import and global initialization",
            "covariance loading and inversion",
            "first-call geometry and FFT plans",
            "thread selection and result checks",
        ],
    }


def _point_state(settings, point):
    """Translate the sampled LSST parameter names into native setters.

    The eight cosmology parameters vary through the CAMB inputs. IA
    arrays use amplitude and redshift-exponent slots; the other arrays
    have one value per source or lens bin. Higher-order galaxy biases and
    point masses keep the fixed values of the declared LSST workload.
    """
    if not isinstance(point, dict):
        raise ValueError("An LSST sampled point must be a dictionary.")
    cosmology = {}
    expected = set(settings["cosmology"])
    for name in settings["cosmology"]:
        cosmology[name] = float(point[name])
    nuisance = copy.deepcopy(settings["nuisance"])
    for index in range(5):
        number = index + 1
        names = {
            "M": f"LSST_M{number}",
            "source_photoz": f"LSST_DZ_S{number}",
            "lens_photoz": f"LSST_DZ_L{number}",
            "B1": f"LSST_B1_{number}",
        }
        for target, name in names.items():
            expected.add(name)
            nuisance[target][index] = float(point[name])
    alignment = {
        "A1": ["LSST_A1_1", "LSST_A1_2"],
        "A2": ["LSST_A2_1", "LSST_A2_2"],
        "B_TA": ["LSST_BTA_1"],
    }
    for target, names in alignment.items():
        for index, name in enumerate(names):
            expected.add(name)
            nuisance[target][index] = float(point[name])
    if set(point) != expected:
        raise ValueError("LSST point names differ from the sampled contract.")
    for name, value in point.items():
        if isinstance(value, bool) or not math.isfinite(float(value)):
            raise ValueError(f"LSST parameter {name} must be finite.")
    return cosmology, nuisance


def lsst_point_bank(settings, npoints=10, seed=137):
    """Call the existing LSST test's seeded parameter sampler only.

    This is a small random-number/configuration operation: it imports no
    compiled LSST module and builds no cosmology. Box-prior parameters are
    sampled in sorted-name order from seed+index; Gaussian-prior values
    remain at the frozen fiducial. The project's narrowed boxes, rejected
    high clustering amplitudes and odd-index holds are applied unchanged.
    The result records these rules, the exact points and source hashes.
    NLA and TATT use identical random points. Only TATT's separate warmup
    point receives the unit test's nonzero A2 and B_TA amplitudes, ensuring
    its first evaluation prepares the quadratic IA terms before timing.
    """
    resolved = _validated_settings(settings)
    if type(npoints) is not int or npoints != 10:
        raise ValueError("This scaling study requires exactly ten points.")
    if type(seed) is not int or seed < 0:
        raise ValueError("The LSST point-bank seed must be nonnegative.")
    root = Path(resolved["cocoa_root"])
    tests = root / "projects" / "lsst_y1" / "tests"
    filename = tests / "cocoa_test_utils.py"
    specification = importlib.util.spec_from_file_location(
        "lsst_scaling_test_inputs", filename,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"Cannot load the LSST test inputs from {filename}.")
    inputs = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(inputs)
    points, held = inputs.nmodels_points(
        example="example2",
        box_overrides=inputs.NMODELS_BOX,
        hold=inputs.NMODELS_HOLD,
        amplitude_cut=inputs.NMODELS_AMPLITUDE_CUT,
        n_models=npoints,
        seed=seed,
    )
    warmup_point = inputs.load_frozen_point(example="example2")
    warmup_adjustments = {}
    if resolved["IA_model"] == 1:
        warmup_adjustments = copy.deepcopy(inputs.TATT_POINT)
        warmup_point.update(warmup_adjustments)
    _point_state(resolved, warmup_point)
    for point in points:
        _point_state(resolved, point)
    core = root / "external_modules" / "code" / "cosmolike_core"
    sources = {
        "project_test_inputs": _file_record(filename),
        "sampler": _file_record(core / "cocoa_testing.py"),
        "frozen_manifest": _file_record(tests / "manifest_sha256.json"),
        "frozen_config": _file_record(
            tests / "frozen" / "frozen_config_example2.py",
        ),
    }
    return {
        "format": "des3pcf-lsst-scaling-point-bank-v1",
        "example": "example2",
        "IA_model": resolved["IA_model"],
        "warmup_point": warmup_point,
        "warmup_adjustments": warmup_adjustments,
        "points": points,
        "held": held,
        "draw_settings": {
            "seed": seed,
            "count": npoints,
            "box_overrides": copy.deepcopy(inputs.NMODELS_BOX),
            "odd_index_holds": list(inputs.NMODELS_HOLD),
            "amplitude_cut": inputs.NMODELS_AMPLITUDE_CUT,
            "gaussian_priors": "held at the frozen fiducial",
        },
        "source_hashes": sources,
        "interpretation": "Designed stress bank, not a joint-prior draw.",
    }


def _table_metadata(settings, point):
    """Record the sampled state, public CAMB request and table units."""
    cosmology, nuisance = _point_state(settings, point)
    root = Path(settings["cocoa_root"])
    helper = root / "external_modules" / "code" / "cosmolike_core"
    helper = helper / "cosmolike_notebook_utils" / "camb_cosmology.py"
    return {
        "format": FORMAT,
        "point": copy.deepcopy(point),
        "cosmology": cosmology,
        "nuisance": nuisance,
        "camb": copy.deepcopy(settings["camb"]),
        "helper": _file_record(helper),
        "units": "k h/Mpc; P (Mpc/h)^3; chi Mpc/h; "
                 "G=D(z)*(1+z) up to its declared normalization",
        "power_layout": "(nz,nk) flattened in Fortran order",
    }


def _validate_tables(bundle):
    """Check dimensions and finite values before the fatal C boundary."""
    from copy import deepcopy

    import numpy as np

    if set(bundle) != {"metadata", "arrays"}:
        raise ValueError("LSST table bundle needs metadata and arrays.")
    if bundle["metadata"].get("format") != FORMAT:
        raise ValueError(f"LSST table format must be {FORMAT}.")
    arrays = bundle["arrays"]
    if set(arrays) != set(TABLE_NAMES):
        raise ValueError("LSST table bundle is missing a required array.")
    for name, value in arrays.items():
        if name == "omegan2":
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError("LSST omegan2 must be finite and nonnegative.")
        elif (value.dtype != np.float64 or value.ndim != 1
              or value.size < 2 or not np.isfinite(value).all()):
            raise ValueError(f"LSST table {name} needs finite float64 rows.")
    for name in ("log10k_2D", "z_2D", "z_G", "z_1D"):
        if not np.all(np.diff(arrays[name]) > 0.0):
            raise ValueError(f"LSST coordinate {name} must increase strictly.")
    nk = arrays["log10k_2D"].size
    nz = arrays["z_2D"].size
    for name in ("lnP_linear", "lnP_nonlinear", "lnP_linear_cb"):
        if arrays[name].size != nk*nz:
            raise ValueError(f"LSST table {name} does not have nk*nz entries.")
    if arrays["G"].size != arrays["z_G"].size:
        raise ValueError("LSST G and z_G must have the same length.")
    if arrays["chi"].size != arrays["z_1D"].size:
        raise ValueError("LSST chi and z_1D must have the same length.")
    if np.any(arrays["G"] <= 0.0) or np.any(arrays["chi"] < 0.0):
        raise ValueError("LSST growth must be positive and chi nonnegative.")


def prepare_lsst_tables(settings, output_path, point, reuse_path=None):
    """Run CAMB for one point and save ten public-helper outputs to an NPZ.

    Call only after numerical work is authorized. This is preparation,
    excluded from all data-vector timings. Returns a bundle with arrays,
    the exact sampled point and full request metadata. Prepare the one
    warmup point and ten random points separately. All thread-count blocks
    then load the same archives without repeating CAMB.

    If reuse_path names an earlier NPZ, its cosmology, CAMB settings and
    helper source hash must match exactly. The physical arrays are then
    reused without running CAMB, while the target point and nuisance
    metadata replace the earlier ones. This lets NLA and TATT share all
    matter tables, including their different IA warmup amplitudes. The
    new archive records the hash of the reused file and must be written
    to a different path so that the original run remains intact.
    """
    import numpy as np

    resolved = _validated_settings(settings)
    filename = Path(output_path).expanduser().resolve()
    if filename.suffix != ".npz":
        raise ValueError("LSST table output filename must end in .npz.")
    metadata = _table_metadata(resolved, point)
    if reuse_path is not None:
        previous_file = Path(reuse_path).expanduser().resolve()
        if previous_file == filename:
            raise ValueError("Write reused LSST tables to a new output path.")
        previous = load_lsst_tables(previous_file)
        for name in ("cosmology", "camb"):
            if previous["metadata"].get(name) != metadata[name]:
                raise ValueError(
                    f"Cannot reuse LSST tables: {name} differs from "
                    "the target point. Prepare new tables with CAMB."
                )
        previous_helper = previous["metadata"].get("helper", {})
        if previous_helper.get("sha256") != metadata["helper"]["sha256"]:
            raise ValueError("Cannot reuse LSST tables: the CAMB helper "
                             "source hash changed. Prepare new tables.")
        arrays = previous["arrays"]
        metadata["reused_input"] = _file_record(previous_file)
    else:
        root = Path(resolved["cocoa_root"])
        core = root / "external_modules" / "code" / "cosmolike_core"
        if str(core) not in sys.path:
            sys.path.insert(0, str(core))
        from cosmolike_notebook_utils import get_camb_cosmology

        request = copy.deepcopy(metadata["cosmology"])
        request.update(resolved["camb"])
        result = get_camb_cosmology(**request)
        arrays = {}
        for name, value in zip(TABLE_NAMES, result):
            if name == "omegan2":
                arrays[name] = float(value)
            else:
                arrays[name] = np.array(value, dtype=np.float64, copy=True)
    bundle = {"metadata": metadata, "arrays": arrays}
    _validate_tables(bundle)
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.savez(filename, metadata=json.dumps(metadata, sort_keys=True), **arrays)
    return bundle


def load_lsst_tables(path):
    """Load owned physical arrays with pickle disabled; perform no CAMB run."""
    import numpy as np

    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != set(TABLE_NAMES) | {"metadata"}:
            raise ValueError("LSST archive does not contain declared tables.")
        metadata = json.loads(str(archive["metadata"].item()))
        arrays = {}
        for name in TABLE_NAMES:
            if name == "omegan2":
                arrays[name] = float(archive[name].item())
            else:
                arrays[name] = np.array(archive[name], copy=True)
    bundle = {"metadata": metadata, "arrays": arrays}
    _validate_tables(bundle)
    return bundle


class LSSTWorkload:
    """One global LSST model owned by an isolated timing worker.

    initialize_lsst constructs this object after input checks. It owns
    its settings; each evaluate call borrows a previously loaded table
    bundle. Do not initialize another LSST model in the same child:
    the library globals would replace this one.
    """

    def __init__(self, interface, settings):
        self.interface = interface
        self.settings = copy.deepcopy(settings)

    def set_threads(self, nthreads):
        """Select 1, 4 or 8 OpenMP workers outside the measured interval.

        The native setter also enforces one BLAS thread. The controller
        uses a fresh process for each team size, sets this before native
        initialization, then warms that process before any measurement.
        """
        if type(nthreads) is not int or nthreads not in (1, 4, 8):
            raise ValueError("LSST timing supports exactly 1, 4 or 8 threads.")
        self.interface.set_omp_threads(n=nthreads)

    def evaluate(self, tables):
        """Install one sampled state and return its 1560-entry vector.

        Include this whole call in the timer. The actual changed cosmology
        and nuisance values trigger native cache invalidation. There is
        no artificial invalidation. Evaluate the full ten-point bank for
        each thread count, with a separate fiducial warmup before each
        block; consecutive repeats of one point would be cache hits.
        Validate/load each table bundle outside this measured call.

        The returned Python sequence comes directly from the production
        binding. Convert/check it outside the timer if the controller
        needs a NumPy array or a bitwise comparison between repetitions.
        """
        values = tables["arrays"]
        cosmology = tables["metadata"]["cosmology"]
        nuisance = tables["metadata"]["nuisance"]
        self.interface.set_cosmology(
            omegam=cosmology["omegam"],
            H0=cosmology["H0"],
            omegab=cosmology["omegab"],
            log10k_2D=values["log10k_2D"],
            z_2D=values["z_2D"],
            lnP_linear=values["lnP_linear"],
            lnP_nonlinear=values["lnP_nonlinear"],
            G=values["G"],
            z_G=values["z_G"],
            z_1D=values["z_1D"],
            chi=values["chi"],
            omegan2=values["omegan2"],
            lnP_linear_cb=values["lnP_linear_cb"],
        )
        self.interface.set_nuisance_shear_calib(M=nuisance["M"])
        self.interface.set_nuisance_shear_photoz(bias=nuisance["source_photoz"])
        self.interface.set_nuisance_clustering_photoz(
            bias=nuisance["lens_photoz"],
        )
        self.interface.set_nuisance_ia(
            A1=nuisance["A1"],
            A2=nuisance["A2"],
            B_TA=nuisance["B_TA"],
        )
        self.interface.set_nuisance_bias(
            B1=nuisance["B1"],
            B2=nuisance["B2"],
            B_MAG=nuisance["B_MAG"],
            B3nl=nuisance["B3nl"],
            BK=nuisance["BK"],
        )
        self.interface.set_point_mass(PMV=nuisance["PMV"])
        return self.interface.compute_data_vector_masked()


def initialize_lsst(settings, threads=1, extension_path=None):
    """Initialize full production 3x2pt; exclude this call from timings.

    extension_path names the copied cosmolike_lsst_y1_interface.so FILE.
    The covariance is loaded and inverted only here because the exact
    production data-vector function shares its measurement setup with
    the likelihood. No covariance is recomputed. The controller must
    subsequently warm evaluate before collecting ten measurements at its
    fixed team size. Use a fresh child for each size. All sources have
    five tomographic bins.
    """
    import numpy as np

    resolved = _validated_settings(settings)
    if type(threads) is not int or threads not in (1, 4, 8):
        raise ValueError("Initialize LSST with exactly 1, 4 or 8 workers.")
    if extension_path is not None:
        resolved["extension_path"] = str(Path(extension_path).resolve())
    for name in ("source_file", "lens_file", "covariance_file",
                 "mask_file", "data_file", "extension_path"):
        if not Path(resolved[name]).is_file():
            raise FileNotFoundError(f"LSST input is missing: {resolved[name]}")
    mask = np.loadtxt(resolved["mask_file"])
    expected_indices = np.arange(1560, dtype=np.float64)
    if (mask.shape != (1560, 2)
            or not np.array_equal(mask[:, 0], expected_indices)
            or not np.all(mask[:, 1] == 1.0)):
        raise ValueError("LSST timing requires the full ordered all-ones mask.")
    extension = Path(resolved["extension_path"])
    if extension.name != MODULE + ".so":
        raise ValueError("The copied LSST extension must keep its .so name.")
    if MODULE in sys.modules:
        loaded = Path(sys.modules[MODULE].__file__).resolve()
        if loaded != extension:
            raise RuntimeError("A different LSST extension is already loaded; "
                               "use a fresh timing worker.")
    sys.path.insert(0, str(extension.parent))
    ci = importlib.import_module(MODULE)
    if Path(ci.__file__).resolve() != extension:
        raise RuntimeError("The worker did not import the copied LSST binary.")

    # Configure this child's team before native tables allocate scratch.
    # The data load can invoke BLAS; restore OpenMP afterward.
    workload = LSSTWorkload(ci, resolved)
    workload.set_threads(nthreads=threads)
    ci.initial_setup()
    ci.init_probes(possible_probes=resolved["probes"])
    ci.init_binning(
        ntheta_bins=resolved["ntheta"],
        theta_min_arcmin=resolved["theta_min_arcmin"],
        theta_max_arcmin=resolved["theta_max_arcmin"],
    )
    ci.init_cosmo_runmode(is_linear=False)
    ci.init_IA(
        ia_model=resolved["IA_model"],
        ia_redshift_evolution=resolved["IA_redshift_evolution"],
        ia_code=resolved["IA_code"],
    )
    ci.init_redshift_distributions_from_files(
        lens_multihisto_file=resolved["lens_file"],
        lens_ntomo=resolved["nlens"],
        source_multihisto_file=resolved["source_file"],
        source_ntomo=resolved["nsource"],
    )
    ci.init_bias(bias_model=resolved["bias_model"])
    ci.init_ntable_lmax(lmax=resolved["lmax"])
    ci.init_photoz_histogram_akima(
        enabled=resolved["photoz_histogram_akima"],
        preparation_factor=resolved["photoz_preparation_factor"],
    )
    ci.init_photoz_conventions(
        interpolation_type=resolved["photoz_interpolation_type"],
        zmid_convention=resolved["photoz_zmid_convention"],
    )
    ci.init_fpt_internal_boost(
        internal_boost=resolved["internal_accuracy_boost"],
    )
    ci.init_nonlimber_accuracy_boost(
        nonlimber_boost=resolved["nonlimber_accuracy_boost"],
    )
    ci.init_adopt_limber_gs(adopt_limber_gs=resolved["adopt_limber_gs"])
    ci.init_adopt_limber_gg(adopt_limber_gg=resolved["adopt_limber_gg"])
    ci.init_include_HOD_GX(include_HOD_GX=resolved["include_HOD_GX"])
    ci.init_include_halo_IA(include_halo_IA=resolved["include_halo_IA"])
    ci.init_lowrank_nonlimber_gg(
        flag=resolved["nonlimber_lowrank_gg"],
        nrank=resolved["nonlimber_lowrank_rank"],
        lmax=resolved["nonlimber_lowrank_lmax"],
    )
    ci.init_accuracy_boost(
        accuracy_boost=resolved["accuracy_boost"],
        integration_accuracy=resolved["integration_accuracy"],
    )
    ci.init_data_real(
        COV=resolved["covariance_file"],
        MASK=resolved["mask_file"],
        DATA=resolved["data_file"],
    )
    nuisance = resolved["nuisance"]
    ci.set_nuisance_shear_calib(M=nuisance["M"])
    ci.set_nuisance_shear_photoz(bias=nuisance["source_photoz"])
    ci.set_nuisance_clustering_photoz(bias=nuisance["lens_photoz"])
    ci.set_nuisance_ia(
        A1=nuisance["A1"],
        A2=nuisance["A2"],
        B_TA=nuisance["B_TA"],
    )
    ci.set_nuisance_bias(
        B1=nuisance["B1"],
        B2=nuisance["B2"],
        B_MAG=nuisance["B_MAG"],
        B3nl=nuisance["B3nl"],
        BK=nuisance["BK"],
    )
    ci.set_point_mass(PMV=nuisance["PMV"])
    ci.reset_bary_struct()
    workload.set_threads(nthreads=threads)
    sizes = list(ci.compute_data_vector_3x2pt_real_sizes())
    if sizes != [780, 650, 130]:
        raise RuntimeError(f"LSST block sizes {sizes} differ from 780/650/130.")
    return workload
