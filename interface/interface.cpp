#include <pybind11/pybind11.h>
#include "cosmolike/3pcf/threepcf_interface.hpp"
#include "cosmolike/3pcf/physics_interface.hpp"
#include "cosmolike/3pcf/transform_interface.hpp"
#include "cosmolike/3pcf/covariance_interface.hpp"

// The project supplies registration only. Shared C kernels and both
// array-ownership interfaces belong to cosmolike/3pcf.
PYBIND11_MODULE(cosmolike_des_3pcf_interface, module)
{
  module.doc() =
      "Physical shear bispectra, transforms and aperture covariance.";
  cosmolike_interface::bind_threepcf(module);
  cosmolike_interface::bind_threepcf_physics(module);
  cosmolike_interface::bind_threepcf_transform(module);
  cosmolike_interface::bind_threepcf_covariance(module);
}
