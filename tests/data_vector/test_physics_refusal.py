"""The physical model refuses curved backgrounds before preparing tables."""

from pathlib import Path
import subprocess
import sys


def invalid_curved_model():
    """Call the C constructor in a child process, where exit(1) is isolated."""
    project = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project / "interface"))
    import cosmolike_des_3pcf_interface as compiled
    from test_physics_model import physical_inputs

    inputs = physical_inputs()
    inputs["omega_de"] = 0.65
    compiled.PhysicsModel(**inputs)
    print("Curved model returned")


def test_curved_geometry_is_refused():
    """Require the intended C diagnostic, not an import or loader failure."""
    result = subprocess.run(
        args=[sys.executable, "-B", str(Path(__file__))],
        capture_output=True,
        text=True,
        check=False,
        timeout=20,
    )
    message = result.stdout + result.stderr
    assert result.returncode == 1
    assert "flat geometry requires abs(omega_m + omega_de - 1) <= 1e-8" in message
    assert "Traceback" not in message
    assert "runtime error:" not in message
    assert "Curved model returned" not in message


if __name__ == "__main__":
    invalid_curved_model()
