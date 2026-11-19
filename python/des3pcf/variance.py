"""Prepare owned three-point variance inputs with the shared CosmoLike engine.

The ordinary core already smooths a supplied linear spectrum with Gaussian
and top-hat windows. This adapter installs that spectrum serially, resolves
the shared sampling counts and copies the requested outputs. It leaves the
choice of evolving or factorized linear growth explicit. Neither choice
rescales power to a second sigma8 scalar.
"""

import hashlib
import json

import numpy as np

from .accuracy import accuracy_settings
from .inputs import uniform_segments
from .ownership import immutable_copy


class SharedVariance:
    """Own copied snapshots while reusing the core's retained FFT plans.

    Construction calls core.initial_setup and records its unboosted counts.
    Subsequent physical changes replace inputs without resetting the core,
    allowing plans of the same size to survive. Calls use the core's usual
    serial preparation contract; its builders manage their OpenMP workers.
    Previously returned snapshots remain valid after any later core changes.
    The provider's own saved baseline and Accuracy object select resolution;
    unrelated manual core settings do not redefine that baseline.
    """

    def __init__(self, compiled):
        self.compiled = compiled
        compiled.core.initial_setup()
        self.baseline = immutable_copy(value=compiled.core.variance_sampling())
        self._latest = None

    def prepare(self, tables, parameters, metadata, z, accuracy=None,
                growth_mode="factorized", mass=None, reference_rho=0.0,
                include_fit=True):
        """Return a copied snapshot and its complete numerical input record.

        tables supplies k [h/Mpc], z, positive linear power [(Mpc/h)^3],
        growth D(0)=1 and omega_m. 'factorized' installs P(k,0)D(z)^2,
        retaining the three-point effective-growth approximation. 'evolving'
        requires the full supplied p_linear[z,k]; no fallback is inferred.
        parameters supplies omega_b and h. Massive-neutrino density comes
        from metadata['omega_nu_h2'], as computed by the Boltzmann provider.
        Output z and optional mass [Msun/h] are requested coordinates, not
        the core's internal interpolation nodes. The returned record states
        both. Only fit arrays should be passed to a physical-model factory.
        """
        if growth_mode not in ("factorized", "evolving"):
            raise ValueError("Variance growth_mode must be factorized or evolving.")
        if not isinstance(include_fit, (bool, np.bool_)):
            raise ValueError("Variance include_fit must be Boolean.")
        controls = accuracy_settings(value=accuracy)
        sampling = controls.variance_settings(baseline=self.baseline)
        wave = np.ascontiguousarray(tables["k"], dtype=np.float64)
        redshift = np.ascontiguousarray(tables["z"], dtype=np.float64)
        if np.any(wave <= 0.0):
            raise ValueError("Shared variance requires positive k in h/Mpc.")
        uniform_segments(coordinates=np.log10(wave), name="variance log10(k)", maximum=1)
        uniform_segments(coordinates=redshift, name="variance redshift")
        if redshift[0] != 0.0:
            raise ValueError("Shared variance power must include z=0.")
        if growth_mode == "evolving":
            if "p_linear" not in tables:
                raise ValueError("Evolving variance requires supplied p_linear[z,k].")
            power = np.ascontiguousarray(tables["p_linear"], dtype=np.float64)
        else:
            linear = np.asarray(tables["p_linear0"], dtype=np.float64)
            growth = np.asarray(tables["growth"], dtype=np.float64)
            if linear.shape != wave.shape or growth.shape != redshift.shape:
                raise ValueError("Factorized variance requires P(k,0) and D(z) on their declared axes.")
            if (not np.all(np.isfinite(growth)) or np.any(growth <= 0.0)
                    or abs(growth[0] - 1.0) > 1.0e-12):
                raise ValueError("Factorized variance requires positive growth with D(0)=1.")
            power = growth[:, None]**2 * linear[None, :]
        if (power.shape != (redshift.size, wave.size)
                or not np.all(np.isfinite(power)) or np.any(power <= 0.0)):
            raise ValueError("Shared linear power must be positive finite [nz,nk].")
        query = np.array(z, dtype=np.float64, order="C", copy=True)
        if (query.ndim != 1 or query.size == 0 or not np.all(np.isfinite(query))
                or np.any(np.diff(query) <= 0.0) or query[0] < redshift[0]
                or query[-1] > redshift[-1]):
            raise ValueError("Variance output redshifts must increase within the supplied power range.")
        masses = None
        if mass is not None:
            masses = np.array(mass, dtype=np.float64, order="C", copy=True)
            if (masses.ndim != 1 or masses.size == 0 or not np.all(np.isfinite(masses))
                    or np.any(masses <= 0.0) or np.any(np.diff(masses) <= 0.0)):
                raise ValueError("Variance masses must be a positive finite increasing vector.")
        if not include_fit and masses is None:
            raise ValueError("Variance request must include fit quantities or halo masses.")
        if not np.isfinite(reference_rho) or reference_rho < 0.0:
            raise ValueError("Variance reference_rho must be finite and nonnegative.")
        matter = float(tables["omega_m"])
        baryon = float(parameters["omega_b"])
        hubble = 100.0 * float(parameters["h"])
        if parameters.get("mnu", 0.0) > 0.0 and "omega_nu_h2" not in metadata:
            raise ValueError("Massive-neutrino variance needs provider omega_nu_h2 metadata.")
        neutrino = float(metadata.get("omega_nu_h2", 0.0))
        if (not np.all(np.isfinite([matter, baryon, hubble, neutrino]))
                or hubble <= 0.0 or baryon < 0.0 or matter <= baryon
                or neutrino < 0.0 or neutrino / (hubble / 100.0)**2 >= matter):
            raise ValueError("Variance cosmology requires finite physical matter/baryon/neutrino densities and h>0.")
        arrays = {"k": wave, "z": redshift, "linear_power": power, "output_z": query}
        if masses is not None:
            arrays["mass"] = masses
        hashes = {}
        for name, values in arrays.items():
            hashes[name] = hashlib.sha256(values.tobytes(order="C")).hexdigest()
        record = {
            "engine": "shared CosmoLike FFTLog",
            "growth_mode": growth_mode,
            "normalization": "supplied power amplitude; no scalar sigma8 rescaling",
            "matter_field": "total",
            "sampling": sampling,
            "omega_m": matter, "omega_b": baryon, "hubble": hubble,
            "omega_nu_h2": neutrino, "reference_rho": float(reference_rho),
            "include_fit": bool(include_fit), "input_sha256": hashes,
        }
        key = json.dumps(record, sort_keys=True)
        if self._latest is not None and key == self._latest[0]:
            return self._latest[1], json.loads(self._latest[0])
        # Prepare the layout before installing any global state. Returning
        # an earlier cached snapshot above requires no current-core state:
        # the snapshot owns exactly the values recorded by its input key.
        flattened = np.ascontiguousarray(np.log(power).T).ravel()
        core = self.compiled.core
        core.set_variance_sampling(**sampling)
        core.set_cosmological_parameters(omega_matter=matter, omega_baryon=baryon,
                                         hubble=hubble, omega_nu_h2=neutrino)
        core.set_linear_power_spectrum(log10k=np.log10(wave), z=redshift, lnP=flattened)
        snapshot = core.variance_snapshot(z=query, mass=masses,
                                          reference_rho=reference_rho,
                                          include_fit=bool(include_fit))
        owned = immutable_copy(value=snapshot)
        self._latest = (key, owned)
        return owned, json.loads(key)
