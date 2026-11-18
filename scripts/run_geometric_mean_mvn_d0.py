import time

import numpy as np
import matplotlib.pyplot as plt

from src.simulated_data import random_SPD
from src.mean import frechet_mean
from src.distance import dist2_d0, dist2_dF
from scripts.run_geometric_mean_mvn import random_MVNs_exp


def mse_curve_d0_mean_mvn(
    n_features: int,
    Ns: tuple[int, ...],
    n_MC: int,
    condition_number: float,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Monte Carlo MSE curve of the d_0 Frechet mean of MVNs, on the same draws
    as run_geometric_mean_mvn.

    The random calls (central MVN, then the MVNs for each Monte Carlo run and
    each N) are made in the same order as in mse_curve_geometric_mvn, so with
    the same seed the MVNs are identical. The d_0 mean minimizes
    sum_k d_0^2(P, P_k) and is initialized at the true center, like the
    Fisher-Rao mean.

    Parameters
    ----------
    n_features : int
        Dimension of the MVNs.
    Ns : tuple of int
        Sample sizes N (number of MVNs in the mean).
    n_MC : int
        Number of Monte Carlo iterations.
    condition_number : float
        Condition number of the central covariance (for random_SPD).
    radius : float
        Scale of the tangent vectors around the central MVN.

    Returns
    -------
    Ns_array : ndarray
        Values of N.
    mse_dF2 : ndarray
        Mean over the n_MC iterations of d_F^2(true center, d_0 mean), for each N.
    mse_d02 : ndarray
        Mean over the n_MC iterations of d_0^2(true center, d_0 mean), for each N.
    runtime : float
        Total computation time of the d_0 means, in seconds.
    """
    Ns_array = np.array(Ns, dtype=int)
    mu = np.random.standard_normal(n_features)
    Sigma = random_SPD(n_features, condition_number)
    mse_dF2 = np.zeros(len(Ns_array))
    mse_d02 = np.zeros(len(Ns_array))
    runtime = 0.0
    for mc in range(n_MC):
        for i, N in enumerate(Ns_array):
            means, covariances = random_MVNs_exp(mu, Sigma, N, radius)
            start = time.perf_counter()
            mean_hat, Sigma_hat = frechet_mean(means, covariances, dist2_d0, init=(mu, Sigma))
            runtime += time.perf_counter() - start
            mse_dF2[i] += dist2_dF(mu, Sigma, mean_hat, Sigma_hat)
            mse_d02[i] += dist2_d0(mu, Sigma, mean_hat, Sigma_hat)
        print(f"[run_geometric_mean_mvn_d0] Monte Carlo {mc + 1}/{n_MC} done")
    return Ns_array, mse_dF2 / n_MC, mse_d02 / n_MC, runtime


def plot_mse_curves(
    Ns_array: np.ndarray,
    mse_dF_mean: np.ndarray,
    mse_d0_mean: np.ndarray,
    n_features: int,
    n_MC: int,
    radius: float,
    condition_number: float,
    out_path: str,
) -> None:
    """
    Plots a compact figure of the Monte Carlo mean of d_F^2(P_hat_N, P) in dB
    against N for the Fisher-Rao mean and the d_0 mean, and saves it.

    Parameters
    ----------
    Ns_array : ndarray
        Values of N.
    mse_dF_mean : ndarray
        MSE d_F^2 of the Fisher-Rao (Karcher) mean, for each N.
    mse_d0_mean : ndarray
        MSE d_F^2 of the d_0 Frechet mean, for each N.
    n_features, n_MC, radius, condition_number
        Experiment parameters, shown in the title.
    out_path : str
        Path of the saved PNG.
    """
    plt.figure(figsize=(4.5, 3.2))
    plt.plot(Ns_array, 10 * np.log10(mse_dF_mean), marker="o", label=r"moyenne $d_F$")
    plt.plot(Ns_array, 10 * np.log10(mse_d0_mean), marker="s", label=r"moyenne $d_0$")
    plt.xscale("log")
    plt.xlabel("N")
    plt.ylabel(r"MSE $d_F^2(\hat P_N, P)$ (dB)")
    plt.title(f"n={n_features}, n_MC={n_MC}, r={radius}, κ={condition_number}", fontsize=9)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)


def main() -> None:
    """
    Computes the d_0 mean on the MVNs of run_geometric_mean_mvn and compares
    its MSE with the stored Fisher-Rao mean results.
    """
    print("[run_geometric_mean_mvn_d0] starting…")

    # Mêmes paramètres et même graine que run_geometric_mean_mvn.
    n_features = 5
    Ns = (5, 10, 20, 50, 100, 200, 500)
    n_MC = 20
    condition_number = 10.0
    radius = 0.1
    seed = 0

    np.random.seed(seed)
    Ns_array, mse_dF2, mse_d02, runtime = mse_curve_d0_mean_mvn(
        n_features, Ns, n_MC, condition_number, radius)

    stored = np.loadtxt("output/geometric_mean_mvn_mse.csv", delimiter=",", skiprows=1)
    if not np.array_equal(stored[:, 0], Ns_array):
        raise ValueError("output/geometric_mean_mvn_mse.csv was computed with other values of N")
    mse_dF_mean, reference = stored[:, 1], stored[:, 2]

    print(f"  total time of the d_0 means: {runtime:.1f} s")
    print("N\tdF mean: d_F^2\td0 mean: d_F^2\td0 mean: d_0^2\treference")
    for row in zip(Ns_array, mse_dF_mean, mse_dF2, mse_d02, reference):
        print("{}\t{:.4e}\t{:.4e}\t{:.4e}\t{:.4e}".format(*row))

    csv_path = "output/geometric_mean_mvn_d0_mse.csv"
    np.savetxt(csv_path, np.column_stack([Ns_array, mse_dF2, mse_d02, reference]),
               delimiter=",", header="N,mse_dF2,mse_d02,reference", comments="")

    out_path = "output/geometric_mean_mvn_d0_mse.png"
    plot_mse_curves(Ns_array, mse_dF_mean, mse_dF2,
                    n_features, n_MC, radius, condition_number, out_path)
    print(f"[run_geometric_mean_mvn_d0] saved figure -> {out_path} and data -> {csv_path}")
    print("[run_geometric_mean_mvn_d0] done.")


if __name__ == "__main__":
    main()
