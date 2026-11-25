"""Official E5 hybrid-emulator comparison, following LSST Y1.

The numerical sweep runs in a separate process to isolate compiled
likelihood state. A saved Fisher reference can supply the contour. Missing model files skip
this test; installed but incomplete numerical pipelines fail it.
"""

import os
import sys
import unittest

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("COCOA_TEST_OMP_THREADS", "4"))

class TestHybridEmulator(unittest.TestCase):
    """E5: current hybrid matter emulator, with an optional saved Fisher overlay.

    This uses current example priors, not frozen E1--E4 reference vectors.
    It runs when the candidate code and networks are installed. Broad-prior
    accuracy exceedances and individual nonfinite candidate predictions are
    advisory. Missing CAMB predictions or a comparison with no valid vectors
    fail the test. Saved Fisher contours do not affect the numerical verdict.
    """

    def test_e5_hybrid_emulator_accuracy(self):
        """Compare the fixed broad and focused emulator accuracy points."""
        from datetime import datetime, timezone
        from pathlib import Path
        import subprocess

        project = Path(__file__).resolve().parents[2]
        root = project.parents[1]
        code = Path(os.environ.get(
            "EMULATOR_CODE_DIR", root / "external_modules/code/emulators/emultrf",
        ))
        data = Path(os.environ.get(
            "EMULATOR_DATA_DIR", root / "external_modules/data/emultrf",
        ))
        # The CLI uses the installed linear network and the selected hybrid
        # nonlinear network. Missing installations are a skip, while an
        # installed but broken numerical pipeline is a failed test.
        linear = root / "external_modules/data/emultrf/emulmps/npce_emul"
        nonlinear = data / "emulmps/w0wa_mead2020"
        required = [
            code / "emulmps/emulmps.py",
            code / "emulmps/emulmps_emul/emulmps_w0wa.py",
            linear / "emulator_npce.keras",
            linear / "metadata.joblib",
            nonlinear / "emulator_mead2020.keras",
            nonlinear / "metadata.joblib",
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            self.skipTest("Hybrid emulator inputs are not installed: " + ", ".join(missing))

        # Each invocation gets its own output folder. Earlier scientific
        # results are retained, including failed draws and their provenance.
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        output = project / "chains/hybrid_emulator_tests" / (stamp + "_" + str(os.getpid()))
        environment = os.environ.copy()
        environment["OPENBLAS_NUM_THREADS"] = "1"
        environment["MKL_NUM_THREADS"] = "1"
        environment["VECLIB_MAXIMUM_THREADS"] = "1"
        print("E5 hybrid emulator vector accuracy:", output, flush=True)
        frozen = project / "tests/fisher_reference"
        prior = frozen / "prior_50.txt"
        focused = frozen / "focused_25.txt"
        # A missing row changes the denominator even when CAMB itself succeeds.
        for path, count in ((prior, 50), (focused, 25)):
            rows = [s for s in path.read_text().splitlines() if s.strip() and not s.startswith("#")]
            self.assertEqual(len(rows), count, str(path))
        result = subprocess.run(
            [sys.executable, str(project / "tests/compare_emulator_camb.py"),
             "--points", str(prior), "--local-points", str(focused),
             "--advisory", "--output", str(output)],
            cwd=root,
            env=environment,
            check=False,
        )
        self.assertEqual(
            result.returncode, 0,
            "Emulator vector comparison failed; inspect " + str(output),
        )


class TestJointMetric(unittest.TestCase):
    """A change of xi/Map3 units must not spoil the inverse covariance."""

    def test_equilibrated_precision(self):
        from pathlib import Path
        import runpy
        from unittest.mock import patch
        import numpy as np

        project = Path(__file__).resolve().parents[2]
        driver = runpy.run_path(str(project / "tests/compare_emulator_camb.py"))
        factor = np.array([[1., 0., 0.], [.7, 1., 0.], [.4, -.3, 1.]])
        scale = np.array([1e-12, 1e-6, 1.])
        covariance = (factor @ factor.T) * np.outer(scale, scale)
        with patch.object(driver["ci"], "get_joint_covariance", return_value=covariance):
            precision = driver["get_precision"]()
        np.testing.assert_array_equal(precision, precision.T)
        whitened_metric = precision * np.outer(scale, scale)
        np.testing.assert_allclose((factor @ factor.T) @ whitened_metric,
                                   np.eye(3), atol=2e-15)
        vector = np.array([.2, -.4, .8])
        expected = np.sum(np.linalg.solve(factor, vector)**2)
        actual = (vector * scale) @ precision @ (vector * scale)
        self.assertAlmostEqual(actual, expected, places=14)

        # A genuinely asymmetric covariance still fails the normal check.
        covariance[0, 1] *= 1.1
        with patch.object(driver["ci"], "get_joint_covariance", return_value=covariance):
            with self.assertRaisesRegex(ValueError, "asymmetric"):
                driver["get_precision"]()


if __name__ == "__main__":
    unittest.main(verbosity=2)
