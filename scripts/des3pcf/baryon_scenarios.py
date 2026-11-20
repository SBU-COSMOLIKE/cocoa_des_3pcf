"""Evaluate Cocoa's physical feedback models without inventing a bispectrum.

The bfmt Cobaya theory block returns R_P=P_feedback/P_DMO. This module calls
that block, as the LSST Y1 and Roman notebooks do, and keeps its parameter
points, interpolation policy and source-file hash with the results. It
does not infer R_B from R_P. The separate BACCO API can supply R_B, subject
to its own matter definition and triangle calibration.

Wavenumbers entering this module are in h/Mpc. The bfmt public product
uses 1/Mpc, so its request contains h*k. Forgetting this factor moves the
feedback feature along the Fourier axis without producing an exception.
Preparation is serial. The caller sets BLAS to one worker before import;
no sampler, download or calculation starts when this module is imported.
"""

import hashlib
import importlib.util
import logging
from pathlib import Path

import numpy as np


class _TheoryMessages(logging.Handler):
    """Collect warnings so a failed emulator cannot masquerade as no feedback.

    Some bfmt error paths return unity instead of raising an exception.
    This notebook adapter refuses every warning/error from that block.
    Declared high-redshift extension is handled by bfmt without a warning.
    """

    def __init__(self):
        """Create an empty list without changing the process logging policy."""
        super().__init__(level=logging.WARNING)
        self.messages = []

    def emit(self, record):
        """Retain the fully formatted message produced by the theory block."""
        self.messages.append(record.getMessage())


def _checked_grid(z, k):
    """Check a redshift and wavenumber request before calling an emulator.

    Arguments:
      z = increasing nonnegative redshifts, at least four samples.
      k = increasing positive wavenumbers in h/Mpc, at least four samples.
    Returns:
      independent float64 vectors. Uniform spacing is not required here:
      the existing bfmt block prepares its own regular interpolation grid.
    Raises:
      ValueError for invalid dimensions, ordering or physical domains.
    """
    arrays = []
    for name, values in (("z", z), ("k", k)):
        array = np.array(values, dtype=float, copy=True)
        if (array.ndim != 1 or array.size < 4 or not np.all(np.isfinite(array))
                or np.any(np.diff(array) <= 0.0)):
            raise ValueError(f"{name} requires four increasing finite samples.")
        arrays.append(array)
    if arrays[0][0] < 0.0 or arrays[1][0] <= 0.0:
        raise ValueError("Require z>=0 and k>0 in h/Mpc.")
    return arrays


def bfmt_model_input(cocoa_root, parameters, metadata, options, point):
    """Describe the same background cosmology as the saved DES power tables.

    Arguments:
      cocoa_root = Cocoa directory containing external_modules.
      parameters = DES cosmology dictionary with omega_m/omega_b/h/ns/mnu,
        w0/wa, num_massive_neutrinos and nnu.
      metadata = CAMB preparation record containing As and omega_nu_h2.
      options = bfmt model selector and numerical interpolation settings.
      point = that model's fixed feedback parameters.
    Returns:
      Cobaya input for CAMB + bfmt + the constant likelihood named one.
      CAMB supplies the expansion history; no nonlinear-P request is made.
    Raises:
      ValueError for missing preparation evidence or overlapping parameters.
    """
    from des3pcf.pipeline import COSMOLOGY_NAMES

    h = float(parameters["h"])
    required = ("As", "omega_nu_h2")
    for name in required:
        if name not in metadata or not np.isfinite(metadata[name]):
            raise ValueError(f"Saved CAMB metadata must contain finite {name}.")
    if h <= 0.0 or metadata["As"] <= 0.0 or metadata["omega_nu_h2"] < 0.0:
        raise ValueError("Require h>0, As>0 and omega_nu_h2>=0.")
    # A saved primordial amplitude belongs to one cosmology. Reusing it at
    # a different sigma8 would make BACCO and the DMO tables disagree.
    recorded = metadata.get("parameters", metadata.get("configuration", {}).get(
        "parameters", {},
    ))
    for name in COSMOLOGY_NAMES:
        if name in parameters and (name not in recorded
                                   or parameters[name] != recorded[name]):
            raise ValueError(f"Regenerate CAMB metadata after changing {name}.")
    fixed = {
        "As": float(metadata["As"]), "ns": parameters["ns"], "H0": 100.0 * h,
        "mnu": parameters.get("mnu", 0.0), "tau": 0.0543,
        "w": parameters["w0"], "wa": parameters["wa"],
        "ombh2": parameters["omega_b"] * h**2,
        "omch2": (parameters["omega_m"] - parameters["omega_b"]) * h**2
                  - metadata["omega_nu_h2"],
        "omegam": {"derived": True}, "omegab": {"derived": True},
    }
    if fixed["omch2"] <= 0.0 or set(fixed).intersection(point):
        raise ValueError("Require positive CDM density and separate feedback parameters.")
    fixed.update(point)
    theory_path = Path(cocoa_root) / "external_modules/code/baryon_suppression"
    if not (theory_path / "bfmt.py").is_file():
        raise FileNotFoundError("Install bfmt through Cocoa's installation keys.")
    if "python_path" in options:
        raise ValueError("The adapter uses this Cocoa installation's bfmt source.")
    return {
        "likelihood": {"one": None},
        "theory": {
            "camb": {"extra_args": {
                "dark_energy_model": "ppf",
                "num_massive_neutrinos": parameters.get("num_massive_neutrinos", 1),
                "nnu": parameters.get("nnu", 3.046),
            }},
            "bfmt": dict(options, python_path=str(theory_path.resolve())),
        },
        "params": fixed, "stop_at_error": True, "debug": logging.WARNING,
    }


def bfmt_power_response(cocoa_root, parameters, metadata, options, point, z, k):
    """Run a physical R_P scenario through the existing Cobaya theory block.

    Arguments:
      cocoa_root, parameters, metadata, options, point = bfmt_model_input inputs.
      z, k = requested redshifts and wavenumbers in h/Mpc.
    Returns:
      positive finite R_P[nz,nk] and a JSON-compatible preparation record.
      The record deliberately declares that bfmt supplies no bispectrum.
    Raises:
      RuntimeError on a rejected point or any bfmt warning/error, including
      fallback to unity after an emulator failure; ValueError on bad output.
    Side effects:
      Loads installed emulators and runs a serial CAMB background calculation.
      Existing Cocoa installation data must already be available locally.
    """
    from cobaya.model import get_model

    z, k = _checked_grid(z=z, k=k)
    info = bfmt_model_input(
        cocoa_root=cocoa_root, parameters=parameters, metadata=metadata,
        options=options, point=point,
    )
    _require_installed_theory_data(model_number=options["baryon_model"])
    messages = _TheoryMessages()
    bacco_coordinates = None
    # BCEmu's import changes NumPy's global error policy. Restore the
    # caller's policy on success or failure instead of leaking that change.
    with np.errstate(), get_model(info_or_yaml_or_file=info) as model:
        logger = model.theory["bfmt"].log
        previous_level = logger.level
        logger.setLevel(logging.WARNING)
        logger.addHandler(messages)
        try:
            model.add_requirements(requirements={"baryon_suppression": {
                "z": z, "k": parameters["h"] * k,
            }})
            posterior = model.logposterior(params_values={})
            if not np.isfinite(posterior.logpost):
                raise RuntimeError("bfmt rejected the declared feedback/cosmology point.")
            rows = model.provider.get_baryon_suppression()
            response = np.stack([rows[value] for value in z])
            if options["baryon_model"] == 4:
                # Match bfmt's own massive-neutrino density convention and
                # sigma8 emulator. sigma8 for total matter is not sigma8_cb.
                h = parameters["h"]
                omega_cold = (model.provider.get_param("omegam")
                              - parameters.get("mnu", 0.0) / 94.13 / h**2)
                sigma = model.theory["bfmt"].baccoemulator.get_sigma8(
                    cold=True, omega_cold=omega_cold,
                    omega_baryon=parameters["omega_b"], A_s=metadata["As"],
                    hubble=h, ns=parameters["ns"],
                    neutrino_mass=parameters.get("mnu", 0.0),
                    w0=parameters["w0"], wa=parameters["wa"],
                )
                bacco_coordinates = {
                    "omega_cold": float(omega_cold),
                    "omega_baryon": parameters["omega_b"],
                    "sigma8_cold": float(np.asarray(sigma).item()),
                }
                for name in ("M_c", "eta", "beta", "M1_z0_cen", "theta_inn"):
                    bacco_coordinates[name] = point[name + "_baccoemu"]
        finally:
            logger.removeHandler(messages)
            logger.setLevel(previous_level)
    if messages.messages:
        raise RuntimeError("bfmt reported: " + "; ".join(messages.messages))
    if (response.shape != (z.size, k.size) or np.any(response <= 0.0)
            or not np.all(np.isfinite(response))):
        raise ValueError("bfmt must return positive finite R_P[nz,nk].")
    source = Path(info["theory"]["bfmt"]["python_path"]) / "bfmt.py"
    record = {
        "producer": "Cocoa bfmt baryon_suppression", "supplied_spectra": ["P"],
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "options": dict(options), "feedback_parameters": dict(point),
        "cosmology": dict(parameters), "As": metadata["As"],
        "omega_nu_h2": metadata["omega_nu_h2"],
        "input_k_units": "h/Mpc", "bfmt_request_k_units": "1/Mpc",
        "k_conversion": "k_bfmt=h*k_input",
    }
    if bacco_coordinates is not None:
        record["bacco_coordinates"] = bacco_coordinates
        record["bacco_sigma8_source"] = "bfmt Burger2025 sigma8_cold emulator"
        record["matter_field"] = "cold; applying to total P is an approximation"
    return response, record


def _require_installed_theory_data(model_number):
    """Refuse missing weights before the pinned bfmt dependencies can download.

    Arguments:
      model_number = existing bfmt selector, 1 through 5.
    Returns:
      nothing when the dependency's auto-download triggers are absent.
    Raises:
      FileNotFoundError naming a Cocoa setup input that is missing locally.
      Corrupt files are still checked by their ordinary upstream loaders.
    """
    paths = []
    if model_number in (2, 5):
        spec = importlib.util.find_spec(name="BCemu")
        if spec is None:
            raise FileNotFoundError("Install BCemu with Cocoa's installation keys.")
        base = Path(spec.origin).parent / "input_data"
        if model_number == 2:
            for redshift in ("0", "0p5", "1", "1p5", "2"):
                paths.append(base / f"kpls_emulator_z{redshift}_nComp3.pkl")
        else:
            paths.append(base / "BCemu2025_meta.json")
    if model_number == 4:
        installed_bacco_weights()
        spec = importlib.util.find_spec(name="baccoemu")
        base = Path(spec.origin).parent
        # bfmt loads the ordinary power constructor too, even though this
        # example requests only the baryonic ratio. These are its pinned
        # directories; absent directories trigger upstream downloads.
        for name in ("sigma8_emu_2.0.2", "cold_matter_linear_emu_1.0.2",
                     "total_matter_linear_emu_1.0.2", "no_wiggles_emu_1.0.2",
                     "nonlinear_emu_1.0.3"):
            paths.append(base / name)
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"Cocoa setup must supply {path} before evaluation.")
