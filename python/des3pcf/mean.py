"""From a physical bispectrum to the measured finite-range aperture vector.

The C model projects the matter and intrinsic-alignment bispectra along
the line of sight. Their angular multipoles pass through a persistent
two-dimensional FFTLog plan, which produces the four complex shear
correlations. A fixed linear integral then gives aperture moments for
the same finite triangle grid used by the measurement. Geometry is
prepared once; changing a cosmology does not rebuild the FFT plans.

The code groups rows by their ordered source triplet. All requested
aperture radii use one shear-transform result for that triplet. Repeated
source bins remain repeated: a (1,1,2) shear-calibration factor is
(1+m1)^2(1+m2), not (1+m1)(1+m2).
"""

import numpy as np

from .inputs import validate_tables, ARRAY_NAMES, SCALAR_NAMES
from .covariance import xi_bin_weights
from .accuracy import accuracy_settings
from .ownership import PreparedGeometry


def aligned_fft_grid(theta_lower, dlnt, ell_min, ell_max, minimum_size):
    """Place every requested radial bin exactly on a logarithmic FFT grid.

    FFTLog pairs theta with xy/ell in reverse order. Here xy=1 and the
    fine logarithmic spacing divides the requested bin spacing by an
    integer. Expanding the Fourier range then preserves every requested
    bin edge; no nearest-node replacement changes a measurement angle.

    Arguments:
      theta_lower = positive, equally log-spaced lower bin edges in radians.
      dlnt = positive natural-log radial-bin width.
      ell_min, ell_max = required positive Fourier range, ell_min<ell_max.
      minimum_size = minimum even-transform length, at least 4.
    Returns:
      ell[nfft] and int32 indices selecting theta_lower from 1/ell[::-1].
    Raises:
      ValueError for invalid geometry or excessively large requested arrays.
    """
    theta = np.asarray(theta_lower, dtype=np.float64)
    if (theta.ndim != 1 or theta.size == 0 or np.any(theta <= 0.0)
            or not np.all(np.isfinite(theta)) or not np.isfinite(dlnt) or dlnt <= 0.0):
        raise ValueError("Theta lower edges and dlnt must be positive and finite.")
    if (theta.size > 1 and not np.allclose(
        np.diff(np.log(theta)), dlnt, rtol=1.0e-11, atol=1.0e-13,
    )):
        raise ValueError(
            "Theta lower edges must be regularly separated by dlnt in log angle."
        )
    if (not 0.0 < ell_min < ell_max or not np.isfinite(ell_max)
            or int(minimum_size) != minimum_size or minimum_size < 4):
        raise ValueError("Require 0<ell_min<ell_max and integer minimum_size>=4.")
    subdivisions = max(1, int(np.ceil(minimum_size * dlnt / np.log(ell_max / ell_min))))
    step = dlnt / subdivisions
    first_index = max(0, int(np.ceil(np.log(theta[0] * ell_max) / step)))
    log_maximum = -np.log(theta[0]) + first_index * step
    length = max(
        int(minimum_size),
        int(np.ceil((log_maximum - np.log(ell_min)) / step)) + 1,
        first_index + (theta.size - 1) * subdivisions + 1,
    )
    # Small prime factors make FFTW's factorization cheap. Rounding always
    # increases the length, including when the requested minimum is odd.
    while True:
        remainder = length
        for prime in (2, 3, 5):
            while remainder % prime == 0:
                remainder //= prime
        if length % 2 == 0 and remainder == 1:
            break
        length += 1
    if length > 65536:
        raise ValueError("FFT length exceeds 65536; reduce the requested grid density.")
    logarithms = log_maximum - step * np.arange(length - 1, -1, -1)
    ell = np.exp(logarithms)
    indices = first_index + subdivisions * np.arange(theta.size, dtype=np.int32)
    if not np.all(np.isfinite(ell)) or ell[0] <= 0.0:
        raise ValueError(
            "Requested FFT range overflows; revise the angular/Fourier ranges."
        )
    if not np.allclose(1.0 / ell[::-1][indices], theta, rtol=2.0e-12, atol=0.0):
        raise ValueError(
            "FFT alignment lost precision; revise the requested angle range."
        )
    return np.ascontiguousarray(ell), np.ascontiguousarray(indices, dtype=np.int32)


def physics_model(
    compiled, tables, kernel_nz=101, source_readers=False, table_boost=8, fit_nz=33,
    accuracy=None, adapter="production", feedback=None, pixel_window=None,
    baseline=None, variance=None,
):
    """Construct the owned C model after validating its physical input arrays.

    Arguments:
      compiled = this project's imported extension module.
      tables = arrays/scalars in the inputs.py physical schema.
      kernel_nz = source-window interpolation node count used by fastnc's baseline.
      source_readers = explicit upstream spline readers for reference comparisons.
      table_boost = dense uniform-table refinement for production linear readers.
      fit_nz = uniform redshift anchors for the prepared matter-fit coefficients.
      accuracy = immutable Accuracy controls or their keyword dictionary.
      adapter = production or notebook; both own the resolved input copies.
      feedback = optional immutable FeedbackPair with paired total-matter
          responses; the IA dictionary selects its common amplitude.
      pixel_window = optional W(ell) at consecutive integers starting at zero.
          Three-point angular observables use one W per leg. Raw projected
          bispectra and ordinary xi retain their declared unsmoothed convention.
      baseline = optional immutable BaselineBispectrum prepared in Python.
          None retains the existing internal BiHalofit calculation. A supplied
          baseline also needs tables['omega_m_z'] from its background provider.
      variance = optional copied core fit arrays z/rnl/neff/sigma8, prepared
          from the same supplied linear power. This route trusts that power's
          amplitude and does not renormalize it with the separate sigma8 scalar.
    Returns:
      production.PhysicsModel, owning copies of its prepared physical state.
    Raises:
      ValueError for invalid physical table shapes or domains.
    """
    checked = validate_tables(tables=tables)
    if checked["k"].size < 4 or checked["z"].size < 4:
        raise ValueError(
            "Physical-model spline preparation requires at least four k/z nodes."
        )
    if checked.get("source_z", checked["z"]).size < 4:
        raise ValueError(
            "Physical-model source preparation requires at least four nodes."
        )
    if int(kernel_nz) != kernel_nz or kernel_nz < 4:
        raise ValueError("kernel_nz must be an integer of at least four.")
    if abs(checked["omega_m"] + checked["omega_de"] - 1.0) > 1.0e-8:
        raise ValueError(
            "The projected physical model requires a spatially flat background."
        )
    controls = accuracy_settings(value=accuracy)
    settings = controls.model_settings(
        kernel_nz=kernel_nz, table_boost=table_boost, fit_nz=fit_nz,
    )
    if adapter not in ("production", "notebook"):
        raise ValueError("Physical adapter must be production or notebook.")
    arguments = {}
    for name in ARRAY_NAMES + SCALAR_NAMES:
        arguments[name] = checked[name]
    if "source_z" in checked:
        arguments["source_z"] = checked["source_z"]
    arguments["los_z"] = controls.los_nodes(baseline=checked["los_z"])
    if variance is not None:
        arguments["variance"] = dict(variance)
    if feedback is not None:
        arguments["feedback"] = dict(feedback.native)
        arguments["feedback"]["boost"] = controls.count(
            baseline=feedback.native["boost"], stage="feedback_table_boost",
        )
    binding = getattr(compiled, adapter)
    if pixel_window is not None:
        from .pixel_window import checked_window

        arguments["pixel_window"] = checked_window(values=pixel_window)
    if baseline is not None:
        from .bihalofit import BaselineBispectrum

        if not isinstance(baseline, BaselineBispectrum):
            raise ValueError(
                "baseline must be a prepared BaselineBispectrum; construct it "
                "from the prescribed bispectrum and its background arrays."
            )
        if "omega_m_z" not in checked:
            raise ValueError(
                "A supplied bispectrum requires provider omega_m_z on the "
                "background grid; regenerate the physical tables with that input."
            )
        if baseline.background_z is not None:
            same_redshifts = np.array_equal(
                baseline.background_z, checked["z"],
            )
            same_matter_fraction = np.array_equal(
                baseline.background_omega_m_z, checked["omega_m_z"],
            )
            if not same_redshifts or not same_matter_fraction:
                raise ValueError(
                    "The baseline bispectrum and physical tables have different "
                    "background z or omega_m_z arrays. Prepare the baseline for "
                    "these tables, or supply its recorded background arrays."
                )
        # The prescribed bispectrum and its background arrive as arrays. C
        # needs no spectral tilt or dark-energy parameters to interpret them.
        for name in ("omega_de", "w0", "wa", "ns"):
            del arguments[name]
        arguments["baseline"] = dict(baseline.native)
        arguments["omega_m_z"] = checked["omega_m_z"]
        return binding.PhysicsModelWithBispectrum(
            **arguments, **settings, source_readers=source_readers,
        )
    return binding.PhysicsModel(
        **arguments, **settings, source_readers=source_readers,
    )


class Map3Engine(PreparedGeometry):
    """Keep the finite-triangle estimator and transform geometry together.

    The stored weights refer to centroid-projected shear correlations.
    The transform's aperture method performs its declared component/sign
    conversion before applying these weights. Calling this engine does
    not change the ordinary CosmoLike data-vector state.

    Arguments:
      compiled = imported project extension, with production adapters.
      rows = SurveyRows containing only map3 rows.
      theta_lower = side-bin lower edges in radians.
      phi = positive opening-angle centers strictly below pi.
      dlnt, dphi = radial log width and uniform opening-angle width.
      ell_min, ell_max = required Fourier integration range.
      minimum_size = minimum FFT length before exact grid alignment.
      lmax, mmax = Legendre and Fourier angular truncation orders.
      nmu = Gauss-Legendre nodes for each bispectrum multipole.
      transform_options = explicit FFT/window/projection convention settings.
      projection_mode = 'prepared' uniform tables or 'direct' reference integrals.
      projection_settings = explicit coarse/dense triangle and multipole controls.
      projection_batch_size = source triplets prepared together, limiting memory.
      accuracy = global/stage refinement from the unchanged baseline counts.
      side_keep = optional boolean mask on the original lower-edge bank.
        The same selection applies to both side axes; FFT bins stay intact.
      representative_rule = 'geometric' or 'area_weighted' side representative.
      representative_offset = additive change to the dimensionless side
        factor, e.g. -0.015226 for the archived Y3 empirical convention.
    Side effects:
      Creates serial FFT plans and stores geometry/weights until released.
    """

    def __init__(
        self, compiled, rows, theta_lower, phi, dlnt, dphi,
        ell_min, ell_max, minimum_size, lmax, mmax, nmu, transform_options=None,
        projection_mode="prepared", projection_settings=None, projection_batch_size=4,
        accuracy=None,
        side_keep=None, representative_rule="geometric", representative_offset=0.0,
    ):
        """Prepare one transform and one aperture-weight table for these rows.

        Arguments:
          All arguments follow the class docstring; arrays are copied.
        Returns:
          nothing; initializes an engine without evaluating a cosmology.
        Raises:
          ValueError for non-aperture rows or invalid angular grids/settings.
        """
        if rows.values.size == 0 or np.any(rows.kind != "map3"):
            raise ValueError("Map3Engine requires at least one row and only map3 rows.")
        lower = np.asarray(theta_lower, dtype=np.float64)
        if representative_rule not in ("geometric", "area_weighted"):
            raise ValueError("Side representative must be geometric or area_weighted.")
        if not np.isfinite(dlnt) or dlnt <= 0.0 or not np.isfinite(representative_offset):
            raise ValueError("Require positive finite dlnt and finite side offset.")
        factor = np.exp(0.5 * dlnt)
        if representative_rule == "area_weighted":
            # The ratio integrates t*(t dt) and divides by the bin area.
            # expm1 avoids cancellation for narrow logarithmic intervals.
            factor = (2.0 / 3.0) * np.expm1(3.0 * dlnt) / np.expm1(2.0 * dlnt)
        factor += representative_offset
        if not np.isfinite(factor) or factor <= 0.0:
            raise ValueError("Representative side factor must be finite and positive.")
        if factor < 1.0 - 1.e-14 or factor > np.exp(dlnt) * (1.0 + 1.e-14):
            raise ValueError(
                "Representative side lies outside its bin; use an offset "
                "calibrated for this radial width."
            )
        self.representative_rule = representative_rule
        self.representative_offset = float(representative_offset)
        self.representative_sides = lower * factor
        if side_keep is None:
            side_keep = np.ones(lower.size, dtype=bool)
        self.side_keep = np.array(side_keep, copy=True)
        if (self.side_keep.dtype != np.dtype(bool)
                or self.side_keep.shape != lower.shape or not np.any(self.side_keep)):
            raise ValueError("side_keep must be a nonempty boolean lower-edge selection.")
        self.accuracy = accuracy_settings(value=accuracy)
        minimum_size = self.accuracy.count(minimum_size, stage="fft_boost")
        nmu = self.accuracy.count(nmu, stage="angular_boost", integration=True)
        self.rows = rows.select(indices=np.arange(rows.values.size, dtype=np.int64))
        self.compiled = compiled
        self.lmax = int(lmax)
        self.nmu = int(nmu)
        if lmax != self.lmax or nmu != self.nmu or self.lmax < 0 or nmu <= lmax:
            raise ValueError("Require integer lmax>=0 and nmu>lmax.")
        self.ell, indices = aligned_fft_grid(
            theta_lower=theta_lower, dlnt=dlnt,
            ell_min=ell_min, ell_max=ell_max, minimum_size=minimum_size,
        )
        if projection_mode not in ("prepared", "direct"):
            raise ValueError("Projection mode must be prepared or direct.")
        if (int(projection_batch_size) != projection_batch_size
                or projection_batch_size < 1):
            raise ValueError("projection_batch_size must be a positive integer.")
        self.projection_mode = projection_mode
        self.projection_batch_size = int(projection_batch_size)
        self.projection_settings = self.accuracy.projection_settings(
            baseline=projection_settings or {},
        )
        self.projection_settings.setdefault("ell_min", float(self.ell[0]))
        self.projection_settings.setdefault("ell_max", float(self.ell[-1]))
        self.projection_settings.setdefault("lmax", self.lmax)
        self.projection_settings.setdefault("prepare_multipoles", 1)
        if (self.projection_settings["lmax"] != self.lmax
                or self.projection_settings["prepare_multipoles"] != 1):
            raise ValueError(
                "Finite-SAS projection requires its declared lmax and multipoles."
            )
        if (self.projection_settings["ell_min"] > self.ell[0]
                or self.projection_settings["ell_max"] < self.ell[-1]):
            raise ValueError("Projection bounds must cover the aligned FFT ell grid.")
        phi = np.array(phi, dtype=np.float64, order="C", copy=True)
        if (phi.ndim != 1 or phi.size < 1 or not np.all(np.isfinite(phi))
                or np.any(phi <= 0.0) or np.any(phi >= np.pi)
                or not np.isfinite(dphi) or dphi <= 0.0
                or np.any(np.diff(phi) <= 0.0)):
            raise ValueError("Require ordered 0<phi<pi and a positive finite dphi.")
        if phi.size > 1 and not np.allclose(np.diff(phi), dphi, rtol=1.0e-12):
            raise ValueError("The supplied phi grid must have uniform width dphi.")
        options = {}
        if transform_options is not None:
            options.update(transform_options)
        if options.get("projection", "centroid") != "centroid":
            raise ValueError("Aperture weights require centroid projection.")
        if "xy" in options and options["xy"] != 1.0:
            raise ValueError("The aligned physical grid fixes xy=1.")
        for name in ("ell", "phi", "indices", "lmax", "mmax", "dlnt"):
            if name in options:
                raise ValueError(
                    f"Set {name} through the engine geometry, not transform_options."
                )
        self.plan = compiled.production.TransformPlan(
            ell=self.ell, phi=phi, indices=indices,
            lmax=self.lmax, mmax=mmax, dlnt=dlnt, **options,
        )
        self.groups = {}
        filter_rows = []
        filter_lookup = {}
        self.filter_indices = np.empty(rows.values.size, dtype=np.int32)
        for index in range(rows.values.size):
            triplet = tuple(int(value) for value in rows.bins[index])
            if triplet not in self.groups:
                self.groups[triplet] = []
            self.groups[triplet].append(index)
            radii = tuple(float(value) for value in rows.angles[index])
            if radii not in filter_lookup:
                filter_lookup[radii] = len(filter_rows)
                filter_rows.append(radii)
            self.filter_indices[index] = filter_lookup[radii]
        filters = np.ascontiguousarray(filter_rows, dtype=np.float64)
        self.weights_real, self.weights_imag = compiled.production.aperture_weights(
            d2=np.ascontiguousarray(self.representative_sides),
            d3=np.ascontiguousarray(self.representative_sides),
            phi=phi, dlnd=dlnt, dphi=dphi, filters=filters,
        )
        # A cut removes quadrature contributions, not bins from the FFT
        # transform. Keep its original spacing and phases. The product of
        # identical side masks also respects the reflected triangle term.
        keep_pairs = self.side_keep[:, None] & self.side_keep[None, :]
        keep_triangles = np.repeat(keep_pairs.ravel(), phi.size)
        self.weights_real[:, :, ~keep_triangles] = 0.0
        self.weights_imag[:, :, ~keep_triangles] = 0.0
        self._seal_geometry()

    def evaluate(self, model, ia, calibration=None, component=0):
        """Evaluate all aperture rows with one multipole transform per triplet.

        Arguments:
          model = owned production.PhysicsModel for this cosmology and n(z).
          ia = explicit IA/baryon parameter dictionary understood by the C model.
          calibration = optional [nbin] multiplicative shear offsets m_i.
          component = projected total=0, baseline=1, or residual IA=2.
        Returns:
          dimensionless [nrow] aperture vector in the original row order.
        Raises:
          ValueError for invalid calibration values or nonfinite predictions.
        """
        if calibration is not None:
            calibration = np.asarray(calibration, dtype=np.float64)
            maximum = int(np.max(self.rows.bins))
            if (calibration.ndim != 1 or calibration.size <= maximum
                    or not np.all(np.isfinite(calibration))):
                raise ValueError("Shear calibration must include every source bin.")
        result = np.empty(self.rows.values.size, dtype=np.float64)
        triplets = list(self.groups)
        for start in range(0, len(triplets), self.projection_batch_size):
            batch = triplets[start:start + self.projection_batch_size]
            grids = None
            if self.projection_mode == "prepared":
                # The batch shares physical matter/IA work before distinct
                # source weights are applied. Limiting its size bounds the
                # number of dense projected tables alive simultaneously.
                grids = model.projection_grids(
                    bins=np.ascontiguousarray(batch, dtype=np.int32),
                    parameters=ia, settings=dict(self.projection_settings),
                )
            for index, triplet in enumerate(batch):
                if grids is not None and component == 0:
                    # The C plan retains only interpolation geometry and a
                    # temporary buffer. Each physical grid refills that buffer;
                    # changing cosmology or IA never reuses a prediction.
                    values = self.plan.aperture_from_grid(
                        grid=grids[index], weights_real=self.weights_real,
                        weights_imag=self.weights_imag,
                    )
                else:
                    if grids is None:
                        bins = np.ascontiguousarray(triplet, dtype=np.int32)
                        multipoles = model.multipoles(
                            ell=self.ell, bins=bins, parameters=ia,
                            lmax=self.lmax, nmu=self.nmu, component=component,
                        )
                    else:
                        multipoles = grids[index].multipoles(
                            ell=self.ell, component=component,
                        )
                    values = self.plan.aperture_from_multipoles(
                        b_l=multipoles, weights_real=self.weights_real,
                        weights_imag=self.weights_imag,
                    )
                factor = 1.0
                if calibration is not None:
                    for source in triplet:
                        factor *= 1.0 + calibration[source]
                for row in self.groups[triplet]:
                    result[row] = factor * values[self.filter_indices[row]]
            # Release this batch before constructing another dense set.
            grids = None
        if not np.all(np.isfinite(result)):
            raise ValueError(
                "Aperture prediction is nonfinite; inspect the physical grids."
            )
        return result


class XiEngine(PreparedGeometry):
    """Average xi± over the survey's actual angular bins without compression.

    NLA spectra can be calculated directly by the isolated physical model.
    Full TATT spectra can instead be supplied from the existing CosmoLike
    CFASTPT calculation through evaluate_spectra. Both routes use the same
    C Bessel integral and bin weights; a nonlinear-TATT request is never
    silently replaced by NLA.

    Arguments:
      compiled = project extension with the C xi_from_power calculation.
      rows = SurveyRows containing only xi+ and xi- entries.
      ell, dell = positive Fourier quadrature nodes and weights in dell.
      theta_count = number of separation nodes within each observed bin.
      pair_weight = explicit normalized weights or 'uniform_area'.
    """

    def __init__(self, compiled, rows, ell, dell, theta_count, pair_weight):
        """Copy the angular definition and retain its physical row labels.

        Arguments:
          compiled, rows, ell, dell, theta_count, pair_weight = as above.
        Returns:
          nothing; prepares arrays without evaluating a cosmology.
        Raises:
          ValueError for non-xi rows or an invalid Fourier quadrature.
        """
        if rows.values.size == 0 or np.any((rows.kind != "xip") & (rows.kind != "xim")):
            raise ValueError("XiEngine requires a nonempty vector of only xip/xim rows.")
        self.compiled = compiled
        self.rows = rows.select(indices=np.arange(rows.values.size, dtype=np.int64))
        self.ell = np.ascontiguousarray(ell, dtype=np.float64)
        self.dell = np.ascontiguousarray(dell, dtype=np.float64)
        if (self.ell.ndim != 1 or self.ell.size < 2 or self.dell.shape != self.ell.shape
                or np.any(self.ell <= 0.0) or np.any(np.diff(self.ell) <= 0.0)
                or np.any(self.dell <= 0.0) or not np.all(np.isfinite(self.ell))
                or not np.all(np.isfinite(self.dell))):
            raise ValueError(
                "Xi integration needs increasing positive ell and positive dell."
            )
        self.theta, self.theta_weight = xi_bin_weights(
            rows=rows, count=theta_count, pair_weight=pair_weight,
        )
        self.sign = np.ones(rows.values.size, dtype=np.int32)
        self.sign[rows.kind == "xim"] = -1
        self._seal_geometry()

    def evaluate(self, model, ia, calibration=None):
        """Calculate NLA spectra from the common model and integrate all xi rows.

        Arguments:
          model = owned PhysicsModel containing the shared P and n(z).
          ia = NLA parameter dictionary; nonzero TATT/VS terms are rejected.
          calibration = optional [nbin] multiplicative offsets m_i.
        Returns:
          [nrow] dimensionless bin-averaged xi in original measurement order.
        Raises:
          ValueError for a request requiring nonlinear-TATT or baryonic power.
        """
        for name in ("a2", "bta", "ct", "fb"):
            if ia.get(name, 0.0) != 0.0:
                raise ValueError(
                    f"Xi's isolated NLA route cannot use {name}; "
                    "supply the corresponding full E/B spectra."
                )
        cache = {}
        spectra = np.empty((self.rows.values.size, self.ell.size), dtype=np.float64)
        for row in range(self.rows.values.size):
            first = int(self.rows.bins[row, 0])
            second = int(self.rows.bins[row, 1])
            pair = (min(first, second), max(first, second))
            if pair not in cache:
                bins = np.ascontiguousarray(pair, dtype=np.int32)
                cache[pair] = model.nla_power(ell=self.ell, bins=bins, parameters=ia)
            spectra[row] = cache[pair]
        return self.evaluate_spectra(
            spectra_e=spectra, spectra_b=np.zeros_like(spectra), calibration=calibration,
        )

    def evaluate_spectra(self, spectra_e, spectra_b, calibration=None):
        """Integrate supplied signal spectra with the stored angular-bin weights.

        Arguments:
          spectra_e, spectra_b = [nrow,nell] signal angular powers; no shape noise.
          calibration = optional [nbin] offsets, applied once after integration.
        Returns:
          [nrow] dimensionless xi±, retaining every selected two-point row.
        Raises:
          ValueError for invalid spectral shape or calibration values.
        """
        even = np.ascontiguousarray(spectra_e, dtype=np.float64)
        odd = np.ascontiguousarray(spectra_b, dtype=np.float64)
        shape = (self.rows.values.size, self.ell.size)
        if (even.shape != shape or odd.shape != shape
                or not np.all(np.isfinite(even)) or not np.all(np.isfinite(odd))):
            raise ValueError("Xi spectra must be finite [nrow,nell] signal arrays.")
        result = self.compiled.production.xi_from_power(
            ell=self.ell, dell=self.dell, theta=self.theta,
            theta_weight=self.theta_weight, spectra_e=even, spectra_b=odd, sign=self.sign,
        )
        if calibration is not None:
            values = np.asarray(calibration, dtype=np.float64)
            if (values.ndim != 1 or values.size <= np.max(self.rows.bins[:, :2])
                    or not np.all(np.isfinite(values))):
                raise ValueError(
                    "Shear calibration must contain every finite source-bin offset."
                )
            for row in range(self.rows.values.size):
                first, second = self.rows.bins[row, :2]
                result[row] *= (1.0 + values[first]) * (1.0 + values[second])
        return result


def core_power_rows(core, ell, bins):
    """Read full E/B power from an already prepared ordinary CosmoLike module.

    The existing CFASTPT calculation supplies the two-point TATT model.
    Its batch wrapper fills only source pairs with i<=j in cubes ordered
    [ell,i,j]. This adapter selects the correct triangle of those cubes
    and never edits the data-vector C source files. The caller must first
    prepare that core with the same physical parameters, source curves,
    IA normalization and redshift pivot as the three-point model.

    Arguments:
      core = configured project extension exposing C_ss_tomo_limber(l=...).
      ell = positive 1D multipoles, identical to the XiEngine integration grid.
      bins = zero-based [nrow,2] source pairs in the requested row order.
    Returns:
      owned C-contiguous E and B signal powers [nrow,nell].
    Raises:
      ValueError for a shape/bin mismatch or nonfinite core output.
    Side effects:
      The existing core may fill its own cosmology-dependent caches.
    """
    ell = np.asarray(ell, dtype=np.float64)
    pairs = np.asarray(bins)
    if (pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype.kind not in "iu"
            or np.any(pairs < 0)):
        raise ValueError(
            "Core shear projection requires nonnegative integer source pairs."
        )
    even_cube, odd_cube = core.C_ss_tomo_limber(l=ell)
    even_cube = np.asarray(even_cube, dtype=np.float64)
    odd_cube = np.asarray(odd_cube, dtype=np.float64)
    if (even_cube.ndim != 3 or even_cube.shape != odd_cube.shape
            or even_cube.shape[0] != ell.size or even_cube.shape[1] != even_cube.shape[2]
            or np.any(pairs >= even_cube.shape[1])):
        raise ValueError(
            "Existing core's [ell,source,source] spectra do not match these rows."
        )
    even = np.empty((pairs.shape[0], ell.size), dtype=np.float64)
    odd = np.empty_like(even)
    for row in range(pairs.shape[0]):
        first = int(min(pairs[row]))
        second = int(max(pairs[row]))
        even[row] = even_cube[:, first, second]
        odd[row] = odd_cube[:, first, second]
    if not np.all(np.isfinite(even)) or not np.all(np.isfinite(odd)):
        raise ValueError("Existing core returned nonfinite E/B power.")
    return even, odd
