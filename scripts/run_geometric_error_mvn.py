import time

import numpy as np
import matplotlib.pyplot as plt

from src.simulated_data import random_SPD
from src.covariance import estimate_descriptors
from src.mean import geometric_mean_mvn
from src.mvn import fisher_norm2, log_map_mvn
from src.distance import dist2_dF


def objective_and_gradient(
    iterate_means: np.ndarray,
    iterate_covariances: np.ndarray,
    means: np.ndarray,
    covariances: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Evaluates the Karcher cost and the norm of its gradient along the iterates.

    Parameters
    ----------
    iterate_means : ndarray
        Means of the iterates P_k, of shape (n_iterates, n_features).
    iterate_covariances : ndarray
        Covariances of the iterates, of shape (n_iterates, n_features, n_features).
    means : ndarray
        Means of the data P_i, of shape (n_mvn, n_features).
    covariances : ndarray
        Covariances of the data, of shape (n_mvn, n_features, n_features).

    Returns
    -------
    F_values : ndarray
        F(P_k) = (1/K) sum_i d_F^2(P_k, P_i), of shape (n_iterates,).
    grad_norms : ndarray
        Fisher norm of the mean logarithm (1/K) sum_i Log_{P_k}(P_i), i.e. of
        minus the Riemannian gradient of F/2, of shape (n_iterates,).
    """
    F_values = np.zeros(len(iterate_means))
    grad_norms = np.zeros(len(iterate_means))
    for k, (m, S) in enumerate(zip(iterate_means, iterate_covariances)):
        F_values[k] = np.mean([dist2_dF(m, S, mu, C) for mu, C in zip(means, covariances)])
        logs = [log_map_mvn(m, S, mu, C) for mu, C in zip(means, covariances)]
        v = np.mean([log[0] for log in logs], axis=0)
        T = np.mean([log[1] for log in logs], axis=0)
        grad_norms[k] = np.sqrt(fisher_norm2(S, v, T))
    return F_values, grad_norms


def main() -> None:
    """
    Runs the Fisher-Rao geometric mean on MVNs estimated from simulated data
    and plots the convergence of the Karcher cost along the iterates.
    """
    print("[run_geometric_error_mvn] starting…")

    # ----- paramètres -----
    d = 10
    n_mvn = 20
    n_samples = 50
    condition_number = 50.0
    tol = 1e-10
    seed = 123

    np.random.seed(seed)
    mu_true = np.random.standard_normal(d)
    Sigma_true = random_SPD(d, condition_number=condition_number)

    # Données simulées X ~ N(mu_true, Sigma_true), shape (n_mvn, n_samples, d), et MVN estimées par bloc
    X = np.random.multivariate_normal(mu_true, Sigma_true, size=(n_mvn, n_samples))
    means, covariances = estimate_descriptors(X)

    start = time.perf_counter()
    mean_hat, Sigma_hat, iterate_means, iterate_covariances = geometric_mean_mvn(
        means, covariances, tol=tol, return_iterates=True)
    runtime = time.perf_counter() - start

    F_values, grad_norms = objective_and_gradient(iterate_means, iterate_covariances, means, covariances)
    n_iter = len(F_values) - 1

    print(f"  d={d}, n_mvn={n_mvn}, n_samples={n_samples}, kappa={condition_number}, seed={seed}")
    print(f"  {n_iter} iterations in {runtime:.2f} s (tol = {tol:.0e} on the Fisher norm of the mean log)")
    print(f"  d_F^2(mean, truth)       = {dist2_dF(mean_hat, Sigma_hat, mu_true, Sigma_true):.6f}")
    print(f"  d_F^2(init, truth)       = {dist2_dF(iterate_means[0], iterate_covariances[0], mu_true, Sigma_true):.6f}"
          "   (init = arithmetic mean)")

    # --------- Affichage de la liste des valeurs ---------
    print("k\tF(P_k)\t\t\tF(P_k) - F(P_last)\t|grad|")
    print("-" * 64)
    table = np.column_stack([np.arange(n_iter + 1), F_values, F_values - F_values[-1], grad_norms])
    np.savetxt("output/geometric_error_mvn.csv", table, delimiter=",",
               header="k,F,F_minus_F_last,grad_norm", comments="")
    for k, F, gap, grad in table:
        print(f"{int(k):3d}\t{F:.12e}\t{gap:.3e}\t\t{grad:.3e}")

    # --------- Tracé ---------
    fig, (ax_F, ax_gap) = plt.subplots(1, 2, figsize=(11, 4))
    ax_F.plot(F_values, marker="o")
    ax_F.set_xlabel("Itération k")
    ax_F.set_ylabel(r"$F(P_k) = \frac{1}{K}\sum_i d_F^2(P_k, P_i)$")
    ax_F.grid(True, ls=":")

    # Le dernier itéré sert de référence : l'écart y vaut 0, on ne le trace pas en log.
    ax_gap.semilogy(F_values[:-1] - F_values[-1], marker="o", label=r"$F(P_k) - F(P_{last})$")
    ax_gap.semilogy(grad_norms, marker="s", ms=3, label=r"$\|\frac{1}{K}\sum_i \mathrm{Log}_{P_k}(P_i)\|_F$")
    ax_gap.set_xlabel("Itération k")
    ax_gap.legend()
    ax_gap.grid(True, which="both", ls=":")

    fig.suptitle(f"Convergence de la moyenne géométrique de Fisher-Rao "
                 f"(d={d}, K={n_mvn}, n_samples={n_samples}, kappa={condition_number})")
    fig.tight_layout()

    out_path = "output/geometric_error_mvn.png"
    fig.savefig(out_path, dpi=150)
    print(f"[run_geometric_error_mvn] saved figure -> {out_path} and data -> output/geometric_error_mvn.csv")

    plt.show(block=True)
    print("[run_geometric_error_mvn] done.")


if __name__ == "__main__":
    main()
