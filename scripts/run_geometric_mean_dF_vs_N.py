import json
import time

import numpy as np
import matplotlib.pyplot as plt

from scripts.run_geometric_mean_mvn import random_MVNs_exp
from src.simulated_data import random_SPD
from src.mean import geometric_mean_mvn
from src.distance import dist2_dF

N_FEATURES = 3
NS = (5, 10, 20, 50, 100, 200, 500)
N_MC = 10
CONDITION_NUMBER = 10.0
RADIUS = 0.3
SEED = 0
TOL = 1e-8        # tolérance de geometric_mean_mvn sur la norme du logarithme moyen
OUT_STEM = "output/geometric_mean_dF_vs_N"


def dF_curve(mu: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """
    Computes d_F(P_hat_N, P) for every N of NS on one nested sample around P = N(mu, Sigma).

    The MVNs P_k = Exp_P(RADIUS * xi_k) are drawn once for N = max(NS) with random_MVNs_exp,
    and P_hat_N is the Fisher-Rao geometric mean (geometric_mean_mvn, initialized at the
    arithmetic mean) of the first N of them.

    Parameters
    ----------
    mu : ndarray
        Central mean of shape (n_features,).
    Sigma : ndarray
        Central SPD covariance of shape (n_features, n_features).

    Returns
    -------
    ndarray
        d_F(P_hat_N, P) for each N of NS, of shape (len(NS),).
    """
    means, covariances = random_MVNs_exp(mu, Sigma, max(NS), RADIUS)
    distances = np.zeros(len(NS))
    for i, N in enumerate(NS):
        mean_hat, Sigma_hat = geometric_mean_mvn(means[:N], covariances[:N], tol=TOL)
        distances[i] = np.sqrt(dist2_dF(mu, Sigma, mean_hat, Sigma_hat))
    return distances


def main() -> None:
    """
    Plots the Monte Carlo mean of d_F(P_hat_N, P) against N, in dB, where P_hat_N is the
    Fisher-Rao geometric mean of N MVNs drawn by the exponential map around P.
    """
    print("[run_geometric_mean_dF_vs_N] starting...")
    np.random.seed(SEED)
    mu = np.random.standard_normal(N_FEATURES)
    Sigma = random_SPD(N_FEATURES, CONDITION_NUMBER)

    start = time.perf_counter()
    distances = np.zeros((N_MC, len(NS)))
    for mc in range(N_MC):
        distances[mc] = dF_curve(mu, Sigma)
        print(f"  Monte Carlo {mc + 1}/{N_MC}: d_F = {np.array2string(distances[mc], precision=4)}")
    runtime = time.perf_counter() - start
    mean_distance = distances.mean(axis=0)
    # Échelle en dB : x_dB = 10 log10(x), soit 10 dB par décade.
    mean_distance_dB = 10 * np.log10(mean_distance)

    print(f"  total time: {runtime:.1f} s")
    print("N\tmean d_F\tdB")
    for N, value, value_dB in zip(NS, mean_distance, mean_distance_dB):
        print(f"{N}\t{value:.4e}\t{value_dB:+.2f}")

    # ----- enregistrement, pour reproductibilité et pour retracer la figure -----
    header = "N,mean_dF,mean_dF_dB," + ",".join(f"dF_run{mc}" for mc in range(N_MC))
    np.savetxt(f"{OUT_STEM}.csv", np.column_stack([NS, mean_distance, mean_distance_dB, distances.T]),
               delimiter=",", header=header, comments="")
    settings = dict(
        n_features=N_FEATURES, Ns=NS, n_MC=N_MC, condition_number=CONDITION_NUMBER, radius=RADIUS,
        seed=SEED, tol=TOL, mu=mu.tolist(), Sigma=Sigma.tolist(),
        generation="P_k = Exp_P(radius * xi_k), random_MVNs_exp of run_geometric_mean_mvn, "
                   "nested samples: P_hat_N uses the first N MVNs of each Monte Carlo run",
        mean="geometric_mean_mvn, initialized at the arithmetic mean",
        error="d_F(P_hat_N, P) by dist2_dF, averaged over the Monte Carlo runs",
        runtime=runtime,
    )
    with open(f"{OUT_STEM}.json", "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)

    # ----- tracé : d_F(P_hat_N, P) en dB contre N -----
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(NS, mean_distance_dB, marker="o")
    ax.set_xscale("log")
    ax.set_xlabel("N (nombre de MVN dans la moyenne)")
    ax.set_ylabel(r"$d_F(\hat P_N, P)$ (dB)")
    ax.set_title(f"Moyenne géométrique de Fisher-Rao de N MVN\n"
                 f"n={N_FEATURES}, rayon={RADIUS}, kappa={CONDITION_NUMBER:g}, {N_MC} tirages",
                 fontsize=10)
    ax.grid(True, which="both", ls=":")
    fig.tight_layout()
    fig.savefig(f"{OUT_STEM}.png", dpi=150)
    print(f"[run_geometric_mean_dF_vs_N] saved {OUT_STEM}.{{png,csv,json}}")

    plt.show(block=True)
    print("[run_geometric_mean_dF_vs_N] done.")


if __name__ == "__main__":
    main()
