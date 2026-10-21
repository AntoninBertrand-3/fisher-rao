import time

import numpy as np
import matplotlib.pyplot as plt

from src.simulated_data import random_SPD
from src.distance import dist2_d0, dist2_dF_iterates


def random_MVN_pair(
    n_features: int,
    condition_number: float,
    mean_scale: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Draws a random pair of MVNs N(0, Sigma0), N(mu1, Sigma1).

    Parameters
    ----------
    n_features : int
        Dimension of the MVNs.
    condition_number : float
        Condition number of Sigma0 and Sigma1 (for random_SPD).
    mean_scale : float
        Standard deviation of the entries of mu1.

    Returns
    -------
    mu0, Sigma0, mu1, Sigma1 : ndarray
        Means of shape (n_features,) and SPD covariances of shape (n_features, n_features).
    """
    mu0 = np.zeros(n_features)
    Sigma0 = random_SPD(n_features, condition_number)
    mu1 = mean_scale * np.random.standard_normal(n_features)
    Sigma1 = random_SPD(n_features, condition_number)
    return mu0, Sigma0, mu1, Sigma1


def main() -> None:
    """
    Computes d_F for one random MVN pair and plots the BFGS iterates J(K_k).
    """
    print("[run_dF_iterates] starting…")

    # ----- paramètres -----
    n_features = 10
    condition_number = 50.0
    mean_scale = 2.0
    tol = 1e-10
    max_iter = 1000
    seed = 123

    np.random.seed(seed)
    mu0, Sigma0, mu1, Sigma1 = random_MVN_pair(n_features, condition_number, mean_scale)

    # ----- calcul de d_0 et de d_F avec ses itérés -----
    start = time.perf_counter()
    d0_2 = dist2_d0(mu0, Sigma0, mu1, Sigma1)
    time_d0 = time.perf_counter() - start

    start = time.perf_counter()
    dF_2, K_iterates, J_values, grad_norms = dist2_dF_iterates(
        mu0, Sigma0, mu1, Sigma1, tol=tol, max_iter=max_iter)
    time_dF = time.perf_counter() - start

    d0, dF = np.sqrt(d0_2), np.sqrt(dF_2)
    n_iter = J_values.size - 1
    print(f"  n_features={n_features}, kappa={condition_number}, mean_scale={mean_scale}, seed={seed}")
    print(f"  d_0 = {d0:.10f}   ({time_d0 * 1e3:.2f} ms)")
    print(f"  d_F = {dF:.10f}   ({time_dF * 1e3:.2f} ms, {n_iter} BFGS iterations, "
          f"final |grad|_inf = {grad_norms[-1]:.2e}, tol = {tol:.0e})")
    print(f"  abs_error = {d0 - dF:.3e}   rel_error = {(d0 - dF) / dF:.3e}")
    print(f"  ||K_T||_F = {np.linalg.norm(K_iterates[-1]):.4f}")

    # ----- liste des itérés -----
    print("k\tJ(K_k)\t\tJ(K_k) - d_F^2\t|grad|_inf")
    print("-" * 56)
    table = np.column_stack([np.arange(n_iter + 1), J_values, J_values - dF_2, grad_norms])
    np.savetxt("output/dF_iterates.csv", table, delimiter=",",
               header="k,J,J_minus_dF2,grad_inf", comments="")
    for k, J, gap, grad in table:
        print(f"{int(k):3d}\t{J:.10e}\t{gap:.3e}\t{grad:.3e}")

    # ----- tracé : J(K_k) et écart à d_F^2 -----
    fig, (ax_J, ax_gap) = plt.subplots(1, 2, figsize=(11, 4))
    ax_J.plot(J_values, marker="o", label=r"$J(K_k)$")
    ax_J.axhline(d0_2, ls="--", c="gray", label=r"$d_0^2 = J(0)$")
    ax_J.axhline(dF_2, ls=":", c="k", label=r"$d_F^2 = J(K_T)$")
    ax_J.set_xlabel("Itération k")
    ax_J.set_ylabel(r"$J(K_k)$")
    ax_J.legend()
    ax_J.grid(True, ls=":")

    # Le dernier itéré est la référence : l'écart y vaut 0, on ne le trace pas en log.
    ax_gap.semilogy(J_values[:-1] - dF_2, marker="o", label=r"$J(K_k) - d_F^2$")
    ax_gap.semilogy(grad_norms, marker="s", ms=3, label=r"$\|\nabla J(K_k)\|_\infty$")
    ax_gap.set_xlabel("Itération k")
    ax_gap.legend()
    ax_gap.grid(True, which="both", ls=":")

    fig.suptitle(f"Convergence de BFGS pour $d_F$ (n={n_features}, kappa={condition_number}, "
                 f"mean_scale={mean_scale}, seed={seed})")
    fig.tight_layout()

    out_path = "output/dF_iterates.png"
    fig.savefig(out_path, dpi=150)
    print(f"[run_dF_iterates] saved figure -> {out_path} and data -> output/dF_iterates.csv")

    plt.show(block=True)
    print("[run_dF_iterates] done.")


if __name__ == "__main__":
    main()
