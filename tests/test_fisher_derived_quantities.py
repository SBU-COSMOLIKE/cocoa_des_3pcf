"""Check the actual notebook's derived forecasts without a physical mean."""

import ast
import numpy as np
from scipy.linalg import solve_triangular
from cosmolike_notebook_utils import fisher as cnu
from test_fisher_cobaya_notebooks import notebook_cells


def amplitude_reports(forecast):
    """Execute the visible amplitude-report cell on a small supplied forecast."""
    source = next(cell for cell in notebook_cells("notes/intrinsic_alignments/EXAMPLE_FISHER_IA.ipynb")
                  if "for fixed_dark_energy in (False, True)" in cell)
    namespace = dict(np=np, cnu=cnu, forecasts={"toy": forecast})
    exec(source, namespace)
    return namespace["amplitude_forecasts"]["toy"]


def test_s8_error_keeps_cross_covariance_and_fom_ratio():
    """A negative Omega_m/sigma8 correlation reduces the S8 variance.

    The notebook takes dS8/dp from the forecast record (CAMB's sigma8 at the
    stencils). This toy uses the coordinates (Omega_m, sigma8) at
    Omega_m = 0.3, sigma8 = 0.8, where that gradient is (4/3, 1).
    """
    covariance = np.array([[0.0009, -0.0024], [-0.0024, 0.01]])
    forecast = {
        "names": ["omegam", "sigma8"],
        "fiducial": {"omegam": 0.3, "sigma8": 0.8},
        "s8_gradient": np.array([0.8 / (2.0 * 0.3), 1.0]),
        "errors": {},
        "information": {"fisher_two_point": np.linalg.inv(covariance),
                        "fisher_joint": np.linalg.inv(0.25*covariance)},
    }
    result = amplitude_reports(forecast)
    # Var(S8)=.0016+.01-.0064=.0052. Dropping the cross term gives .0116.
    np.testing.assert_allclose(result[False, "two_point"]["sigma_s8"],
                               np.sqrt(0.0052), rtol=2e-14)
    ratio = result[False, "joint"]["omega_m_s8_fom"]
    ratio /= result[False, "two_point"]["omega_m_s8_fom"]
    np.testing.assert_allclose(ratio, 4., rtol=2e-14)


def test_derivative_change_preserves_small_aperture_units():
    """Execute the notebook's whitening assignments on a tiny aperture row."""
    source = next(cell for cell in notebook_cells("notes/intrinsic_alignments/EXAMPLE_FISHER_IA.ipynb")
                  if "whitened_change =" in cell)
    tree = ast.parse(source)
    statements = []
    # Only the pure numerical assignments from the physical-model loop are
    # evaluated. Their code remains the notebook's actual implementation.
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id in
                   ("factor", "whitened_change", "whitened_ref", "change")
                   for target in targets):
                statements.append(node)
    statements.sort(key=lambda node: node.lineno)
    code = compile(ast.Module(body=statements, type_ignores=[]), "notebook", "exec")
    for aperture_unit in (1e-8, 1.):
        namespace = dict(np=np, solve_triangular=solve_triangular,
                         covariance=np.diag([1., aperture_unit**2]),
                         dh=np.array([[1.], [aperture_unit]]),
                         dh2=np.array([[1.], [2*aperture_unit]]))
        exec(code, namespace)
        np.testing.assert_allclose(namespace["change"], [1/np.sqrt(5.)], rtol=2e-14)
