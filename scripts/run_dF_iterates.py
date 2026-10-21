import time

import numpy as np
import matplotlib.pyplot as plt

from src.distance import dist2_dF_iterates


def main() -> None:
    """
    Computes d_F between two chosen MVNs and plots the BFGS iterates J(K_k).
    """
    print("[run_dF_iterates] starting…")

    # ----- paramètres des deux MVN -----
    mu0 = np.array([0., 0., 0.])
    Sigma0 = np.array([[2.0, 0.5, 0.0],
                       [0.5, 1.0, 0.3],
                       [0.0, 0.3, 0.5]])
    mu1 = np.array([1.0, -2.0, 0.5])
    Sigma1 = np.array([[0.2, 0.0, 0.0],
                       [0.0, 3.0, 0.0],
                       [0.0, 0.0, 1.0]])

    # ----- paramètres de l'optimiseur -----
    tol = 1e-10
    max_iter = 1000

    # ----- calcul de d_F avec ses itérés -----
    start = time.perf_counter()
    dF_2, K_iterates, J_values, grad_norms = dist2_dF_iterates(
        mu0, Sigma0, mu1, Sigma1, tol=tol, max_iter=max_iter)
    time_dF = time.perf_counter() - start

    n_iter = J_values.size - 1
    print(f"  d_F = {np.sqrt(dF_2):.10f}   (d_F^2 = {dF_2:.10e})")
    print(f"  {n_iter} BFGS iterations in {time_dF * 1e3:.2f} ms, "
          f"final |grad|_inf = {grad_norms[-1]:.2e}, tol = {tol:.0e}")
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

    fig.suptitle(f"Convergence de BFGS pour $d_F$ (n={mu0.size}, d_F={np.sqrt(dF_2):.4f})")
    fig.tight_layout()

    out_path = "output/dF_iterates.png"
    fig.savefig(out_path, dpi=150)
    print(f"[run_dF_iterates] saved figure -> {out_path} and data -> output/dF_iterates.csv")

    plt.show(block=True)
    print("[run_dF_iterates] done.")


if __name__ == "__main__":
    main()
