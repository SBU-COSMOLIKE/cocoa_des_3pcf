#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cosmolike/generic_interface.hpp"
#include "cosmolike/cosmo2D_wrapper.hpp"
namespace py = pybind11;
#include "cosmolike/3pcf/threepcf_interface.hpp"
#include "cosmolike/3pcf/physics_interface.hpp"
#include "cosmolike/3pcf/transform_interface.hpp"
#ifndef COSMOLIKE_NO_COVARIANCE
#include "cosmolike/3pcf/covariance_interface.hpp"
#endif

// Register the existing two-point initializers and spectra beside the
// three-point kernels. Both probes use this project's native state.
PYBIND11_MODULE(cosmolike_des_3pcf_interface, module)
{
  module.doc() =
      "Physical shear bispectra, transforms and aperture covariance.";
  cosmolike_interface::bind_threepcf(module);
  cosmolike_interface::bind_threepcf_physics(module);
  cosmolike_interface::bind_threepcf_transform(module);
#ifndef COSMOLIKE_NO_COVARIANCE
  cosmolike_interface::bind_threepcf_covariance(module);
  module.attr("has_covariance") = true;
#else
  module.attr("has_covariance") = false;
#endif


  module.def("init_ntable_ell_internal",
      &cosmolike_interface::init_ntable_ell_internal,
      "Coarse exact-quadrature ell nodes of the C_ss/C_gs/C_gk/C_ks "
      "tables and of the scale-cut tables' ell axis, cubic-spline "
      "upsampled to N_ell; 0 = exact per-node quadrature",
      (py::arg("nell_internal") = 192).none(false)
    );

  module.def("init_accuracy_boost",
      &cosmolike_interface::init_accuracy_boost,
      "Init accuracy and sampling Boost (may slow down Cosmolike a lot)",
      (py::arg("accuracy_boost") = 1.0).none(false),
      (py::arg("integration_accuracy") = 0).none(false),
      (py::arg("integration_accuracy_ss") = 1.0).none(false),
      (py::arg("integration_accuracy_gs") = 1.0).none(false),
      (py::arg("integration_accuracy_gg") = 1.0).none(false),
      (py::arg("integration_accuracy_gk") = 1.0).none(false),
      (py::arg("integration_accuracy_ks") = 1.0).none(false),
      (py::arg("integration_accuracy_kk") = 1.0).none(false),
      (py::arg("integration_accuracy_cg") = 1.0).none(false),
      (py::arg("integration_accuracy_cc") = 1.0).none(false),
      (py::arg("integration_accuracy_cs") = 1.0).none(false),
      (py::arg("integration_accuracy_n") = 1.0).none(false)
    );

  module.def("init_photoz_histogram_akima",
      &cosmolike_interface::init_photoz_histogram_akima,
      "Select n(z): 0 original midpoint curve, 1 histogram + Akima",
      (py::arg("enabled") = 0).none(false),
      (py::arg("preparation_factor") = 3).none(false)
    );

  module.def("init_photoz_conventions",
      &cosmolike_interface::init_photoz_conventions,
      "Set the n(z) interpolation type (0: cspline, 1: linear, 2+: steffen) "
      "and the n(z) file z-column convention (0: Z_LOW, 1: Z_MID)",
      (py::arg("interpolation_type") = 0).none(false),
      (py::arg("zmid_convention") = 0).none(false)
    );

  module.def("init_fpt_internal_boost",
      &cosmolike_interface::init_fpt_internal_boost,
      "Set the C-FAST-PT internal (convolution) grid as a fraction of the "
      "output table (1.0 = grids equal, the exact legacy path)",
      (py::arg("internal_boost") = 1.0).none(false)
    );

  module.def("init_include_halo_IA",
      &cosmolike_interface::init_include_halo_IA,
      "Cosmic shear and ggl: 0 = the init_IA model (default), 1 = "
      "halo-model IA (Fortuna et al. 2021; NLA, Limber gs, no HOD)",
      (py::arg("include_halo_IA") = 0).none(false)
    );

  module.def("init_cosmo_runmode",
      &cosmolike_interface::init_cosmo_runmode,
      "Init Run Mode (should we force the matter power spectrum to be linear)",
      py::arg("is_linear").none(false)
    );

  module.def("init_IA",
      &cosmolike_interface::init_IA_fastpt,
      "Init IA related options",
      py::arg("ia_model").none(false).noconvert(),
      py::arg("ia_redshift_evolution").none(false).noconvert(),
      py::arg("ia_code").none(false).noconvert()
    );

  module.def("init_probes",
      &cosmolike_interface::init_probes,
      "Init Probes (cosmic shear or 2x2pt or 3x2pt...)",
      py::arg("possible_probes").none(false)
    );

  module.def("initial_setup",
      []() {
        cosmolike_interface::initial_setup();
        cosmolike_interface::reset_threepcf_joint_data();
      },
      "Initialize Cosmolike Variables to their Default Values"
    );

  module.def("init_lens_sample_size",
      &cosmolike_interface::set_lens_sample_size,
      "Set the lens number of tomo bins",
      py::arg("Ntomo").none(false).noconvert()
    );

  module.def("init_source_sample_size",
      &cosmolike_interface::set_source_sample_size,
      "Set the source number of tomo bins",
      py::arg("Ntomo").none(false).noconvert()
    );

  module.def("init_ntomo_powerspectra",
    &cosmolike_interface::init_ntomo_powerspectra,
    "Set the number of power spectra"
  );

  // Install the feedback-corrected nonlinear spectrum after the owned
  // 3pcf model has copied its DMO inputs. The shared setter performs the
  // same validation and cache invalidation as set_cosmology below.
  module.def("set_non_linear_power_spectrum",
      &cosmolike_interface::set_non_linear_power_spectrum,
      "Set ln P_nonlinear on the current log10(k), z grid in Fortran order",
      py::arg("log10k"),
      py::arg("z"),
      py::arg("lnP")
    );

  module.def("set_cosmology",
      [](const double omega_matter,
         const double hubble,
         arma::Col<double> io_log10k_2D,
         arma::Col<double> io_z_2D,
         arma::Col<double> io_lnP_linear,
         arma::Col<double> io_lnP_nonlinear,
         arma::Col<double> io_G,
         arma::Col<double> io_z_1D,
         arma::Col<double> io_chi,
         const double omega_baryon,
         std::vector<double> io_z_G,
         const double omega_nu_h2,
         std::vector<double> io_lnP_linear_cb)
      {
        spdlog::debug("\x1b[90m{}\x1b[0m: Begins", "set_cosmology");
        using namespace cosmolike_interface;
        set_cosmological_parameters(omega_matter, omega_baryon, hubble,
                                    omega_nu_h2);
        set_linear_power_spectrum(io_log10k_2D,io_z_2D,io_lnP_linear);
        // the linear P_cb (cold dark matter + baryons) on the grid of
        // lnP_linear, after it: sigma_cb^2(M,a) supplies
        // all halo statistics. An empty list removes the table of
        // the previous call, so a stale P_cb is never read.
        if (io_lnP_linear_cb.empty()) {
          clear_linear_power_spectrum_cb();
        }
        else {
          set_linear_power_spectrum_cb(io_log10k_2D, io_z_2D,
                                       arma::Col<double>(io_lnP_linear_cb));
        }
        set_non_linear_power_spectrum(io_log10k_2D,io_z_2D,io_lnP_nonlinear);
        // growfac reads G linearly in z: the likelihood samples G on its
        // dense 1D grid (z_G) instead of the coarse z_2D grid of the power
        // spectra (whose size CAMB's transfer redshifts cap), because the
        // halo model, the IA amplitudes and f_growth inherit the error of
        // that linear read. Without z_G, G is sampled on z_2D.
        if (io_z_G.empty()) {
          set_growth(io_z_2D, io_G);
        }
        else {
          set_growth(arma::Col<double>(io_z_G), io_G);
        }
        set_distances(io_z_1D,io_chi);
        spdlog::debug("\x1b[90m{}\x1b[0m: Ends", "set_cosmology");
      },
      "Set Cosmological Parameters, Distance, Matter Power Spectrum, Growth "
      "Factor, and the massive-neutrino density and linear P_cb of the halo "
      "model",
       py::arg("omegam").none(false),
       py::arg("H0").none(false),
       py::arg("log10k_2D").none(false),
       py::arg("z_2D").none(false),
       py::arg("lnP_linear").none(false),
       py::arg("lnP_nonlinear").none(false),
       py::arg("G").none(false),
       py::arg("z_1D").none(false),
       py::arg("chi").none(false),
       py::arg("omegab") = 0.0,
       py::arg("z_G") = std::vector<double>(),
       // Omega_nu h^2 of the massive neutrinos (CAMB's omnuh2)
       py::arg("omegan2") = 0.0,
       // ln P_cb [(Mpc/h)^3], flattened as lnP_linear; empty = none
       py::arg("lnP_linear_cb") = std::vector<double>(),
       py::return_value_policy::move
    );

  module.def("set_nuisance_ia",
      &cosmolike_interface::set_nuisance_IA,
      "Set nuisance Intrinsic Aligment (IA) amplitudes",
      py::arg("A1").none(false),
      py::arg("A2").none(false),
      py::arg("B_TA").none(false),
      py::return_value_policy::move
    );

  module.def("set_nuisance_shear_calib",
      &cosmolike_interface::set_nuisance_shear_calib,
      "Set nuisance shear calibration amplitudes",
      py::arg("M").none(false),
      py::return_value_policy::move
    );

  module.def("set_nuisance_shear_photoz",
      &cosmolike_interface::set_nuisance_shear_photoz,
      "Set nuisance shear photo-z bias amplitudes",
      py::arg("bias").none(false),
      py::return_value_policy::move
    );

  module.def("set_lens_sample",
      &cosmolike_interface::set_lens_sample,
      "Set the lens n(z) from a numpy n(z) histogram",
      py::arg("nofz").none(false)
    );

  module.def("set_source_sample",
      &cosmolike_interface::set_source_sample,
      "Set the source n(z) from a numpy n(z) histogram",
      py::arg("nofz").none(false)
    );

  module.def("reset_bary_struct",
      &reset_bary_struct,
       "Set the Baryon Functions to not contaminate the MPS w/ Baryon effects"
    );

  module.def("C_ss_tomo_limber",
      py::overload_cast<const double, const int, const int>(
        &cosmolike_interface::C_ss_tomo_limber_cpp
      ),
      "Compute shear-shear (fourier - limber) data vector at a single"
      " tomographic bin and ell value",
      py::arg("l").none(false).noconvert(),
      py::arg("ni").none(false).noconvert(),
      py::arg("nj").none(false).noconvert()
    );

  module.def("C_ss_tomo_limber",
      py::overload_cast<arma::Col<double>>(
        &cosmolike_interface::C_ss_tomo_limber_cpp),
      "Compute shear-shear (fourier - limber) data vector at all tomographic"
      " bins and many ell (vectorized)",
      py::arg("l").none(false),
      py::return_value_policy::move
    );
}
