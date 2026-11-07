import numpy as np
import matplotlib.pyplot as plt

from src.simulated_data import random_SPD, random_MVNs
from src.distance import dist2_d0, dist2_dF


def ellipse_curves(means: np.ndarray, covariances: np.ndarray, n_points: int = 100) -> np.ndarray:
    """
    Computes the 1-sigma ellipses {mu + L u : |u| = 1}, Sigma = L L^T, of a batch of 2D Gaussians.

    The square root is taken spectrally, L = U diag(sqrt(lambda)), from eigh(Sigma).

    Parameters
    ----------
    means : ndarray
        Means of shape (n_mvn, 2).
    covariances : ndarray
        SPD covariances of shape (n_mvn, 2, 2).
    n_points : int, default=100
        Number of points on each ellipse.

    Returns
    -------
    ndarray
        Ellipse points of shape (n_mvn, n_points, 2).
    """
    t = np.linspace(0.0, 2 * np.pi, n_points)
    circle = np.column_stack([np.cos(t), np.sin(t)])          # (n_points, 2)
    evals, U = np.linalg.eigh(covariances)                     # (n_mvn, 2), (n_mvn, 2, 2)
    L = U * np.sqrt(evals)[:, None, :]                         # columns scaled by sqrt(lambda)
    return means[:, None, :] + circle @ L.swapaxes(-1, -2)


def plot_marginal_ellipses(mu: np.ndarray, C: np.ndarray, means: np.ndarray,
                           covariances: np.ndarray, out_path: str) -> None:
    """
    Plots the 1-sigma ellipses of every 2D marginal (coordinates i < j) of the generated MVNs.

    The central MVN N(mu, C) is drawn in black, the generated MVNs in color,
    and their means as dots.

    Parameters
    ----------
    mu : ndarray
        Central mean of shape (n_features,).
    C : ndarray
        Central SPD covariance of shape (n_features, n_features).
    means : ndarray
        Generated means of shape (n_mvn, n_features).
    covariances : ndarray
        Generated SPD covariances of shape (n_mvn, n_features, n_features).
    out_path : str
        Path of the saved PNG.
    """
    n_features = mu.size
    pairs = [(i, j) for i in range(n_features) for j in range(i + 1, n_features)]
    fig, axes = plt.subplots(1, len(pairs), figsize=(4.5 * len(pairs), 4.5), squeeze=False)
    for ax, (i, j) in zip(axes[0], pairs):
        idx = [i, j]
        # The (i, j) marginal of N(mu, Sigma) is N(mu[idx], Sigma[idx, idx]).
        curves = ellipse_curves(means[:, idx], covariances[:, idx][:, :, idx])
        center = ellipse_curves(mu[None, idx], C[None][:, idx][:, :, idx])[0]
        for curve in curves:
            ax.plot(curve[:, 0], curve[:, 1], color="tab:blue", alpha=0.35, lw=0.8)
        ax.scatter(means[:, i], means[:, j], s=8, color="tab:blue")
        ax.plot(center[:, 0], center[:, 1], color="black", lw=2, label=r"center $\mathcal{N}(\mu, C)$")
        ax.scatter(mu[i], mu[j], color="black", marker="x", s=50)
        ax.set_xlabel(f"$x_{i + 1}$")
        ax.set_ylabel(f"$x_{j + 1}$")
        ax.set_aspect("equal", adjustable="datalim")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="best", fontsize=8)
    fig.suptitle(f"1-sigma ellipses of the 2D marginals of {means.shape[0]} generated MVNs")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def plot_distance_errors(d0: np.ndarray, dF: np.ndarray, out_path: str) -> None:
    """
    Plots, for each generated MVN, the relative error (d_0 - d_F) / d_F against d_F to the center.

    Parameters
    ----------
    d0, dF : ndarray
        Distances d_0 and d_F (not squared) of shape (n_mvn,).
    out_path : str
        Path of the saved PNG.
    """
    plt.figure()
    plt.scatter(dF, (d0 - dF) / dF)
    plt.yscale("log")
    plt.xlabel(r"$d_F$ to the center $\mathcal{N}(\mu, C)$")
    plt.ylabel(r"relative error $(d_0 - d_F)/d_F$")
    plt.title(r"Error of $d_0$ on the generated MVNs")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def main() -> None:
    """
    Generates MVNs with random_MVNs and visualizes them (marginal ellipses, distances to the center).
    """
    print("[run_visualize_MVNs] starting…")

    # ----- paramètres -----
    n_features = 3
    n_mvn = 30
    condition_number = 10.0
    radius = 0.3
    seed = 0

    np.random.seed(seed)

    # ----- génération -----
    mu = np.random.standard_normal(n_features)
    C = random_SPD(n_features, condition_number)
    means, covariances = random_MVNs(mu, C, n_mvn, radius)

    # ----- distances au centre (une optimisation BFGS par MVN pour d_F) -----
    d0 = np.sqrt([dist2_d0(mu, C, m, S) for m, S in zip(means, covariances)])
    dF = np.sqrt([dist2_dF(mu, C, m, S) for m, S in zip(means, covariances)])
    rel_error = (d0 - dF) / dF
    print(f"  d_F to center: min {dF.min():.3f}, median {np.median(dF):.3f}, max {dF.max():.3f}")
    print(f"  relative error of d_0: median {np.median(rel_error):.2e}, max {rel_error.max():.2e}")
    np.savetxt("output/visualize_MVNs_distances.csv", np.column_stack([d0, dF, rel_error]),
               delimiter=",", header="d0,dF,rel_error", comments="")

    # ----- figures -----
    plot_marginal_ellipses(mu, C, means, covariances, "output/visualize_MVNs_ellipses.png")
    plot_distance_errors(d0, dF, "output/visualize_MVNs_distances.png")
    print("[run_visualize_MVNs] saved figures -> output/visualize_MVNs_ellipses.png, "
          "output/visualize_MVNs_distances.png and data -> output/visualize_MVNs_distances.csv")

    plt.show(block=True)
    print("[run_visualize_MVNs] done.")


if __name__ == "__main__":
    main()
