"""Draw the note's exact two-observable example; no simulation files are read."""
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt

# A Cholesky factor maps the unit circle into the constant-chi2 ellipse.
# This is a level set, not a claim of 68 percent enclosed probability.
covariance = np.array([[1.0, 0.8], [0.8, 1.0]])
angle = np.linspace(0.0, 2.0 * np.pi, 400)
circle = np.array([np.cos(angle), np.sin(angle)])
ellipse = np.linalg.cholesky(covariance) @ circle
with plt.rc_context({"font.family": "serif", "mathtext.fontset": "stix"}):
    fig, ax = plt.subplots(figsize=(6.6, 4.5), constrained_layout=True)
    ax.plot(*ellipse, color="#0072B2", label=r"Correlated: $\chi^2=1$")
    ax.plot(*circle, color="#E69F00", linestyle="--", label="Ignore correlation")
    ax.scatter([1.0, 1.0], [1.0, -1.0], color="black", marker="x")
    ax.annotate(r"$(1,1):\ \chi^2=1.11$", (1, 1), xytext=(0.4, 1.3))
    ax.annotate(r"$(1,-1):\ \chi^2=10$", (1, -1), xytext=(0.3, -1.4))
    ax.set(xlabel="First residual [reference units]", ylabel="Second residual [reference units]",
           xlim=(-1.7, 1.7), ylim=(-1.7, 1.7), aspect="equal")
    ax.legend(loc="upper left", fontsize=9)
    fig.savefig(Path(__file__).with_name("correlated_residuals.pdf"))
    plt.close(fig)
