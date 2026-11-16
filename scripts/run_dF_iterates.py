import json
import time

import numpy as np
import matplotlib.pyplot as plt

from src.distance import dist2_dF_iterates
from src.mvn import exp_map_mvn, fisher_norm2
from src.simulated_data import random_SPD, sym


def main() -> None:
    """
    Builds a pair of MVNs joined by a known geodesic, computes d_F by BFGS and
    compares it, together with d_0, to the length d_True of that geodesic.

    The first MVN N(mu0, Sigma0) is random, with cond(Sigma0) prescribed. A random
    tangent vector (v, T) is drawn with prescribed Fisher norms mean_speed and
    cov_speed of its mean and covariance parts, and
    N(mu1, Sigma1) = Exp_{N(mu0, Sigma0)}(v, T). The geodesic t -> Exp(t (v, T)),
    t in [0, 1], has length d_True = sqrt(mean_speed^2 + cov_speed^2) >= d_F;
    d_True is taken as the reference d_F,
    which assumes that this geodesic is minimizing (not guaranteed for large d_True).
    """
    print("[run_dF_iterates] starting...")

    # ----- paramètres du cas -----
    n_features = 3
    condition_number = 10.0   # cond(Sigma0)
    mean_speed = 1.0          # norme de Fisher de la partie moyenne : ||L^{-1} v||
    cov_speed = 1.0           # norme de Fisher de la partie covariance : ||L^{-1} T L^{-T}||_F / sqrt(2)
    seed = 0

    # ----- paramètres de l'optimiseur -----
    tol = 1e-10
    max_iter = 1000
    # Seuil relatif pour comparer d_F (BFGS) à d_True, au-dessus de l'erreur d'arrondi.
    agreement_rtol = 1e-6

    # ----- génération de la première MVN et du vecteur tangent -----
    # random_SPD utilise le générateur global de NumPy (via scipy.stats).
    np.random.seed(seed)
    mu0 = np.random.standard_normal(n_features)
    Sigma0 = random_SPD(n_features, condition_number)
    L = np.linalg.cholesky(Sigma0)

    # Vitesse blanchie (a, A) en N(0, I), normalisée selon la métrique de Fisher :
    # ||a||^2 = mean_speed^2 et Tr(A^2) / 2 = cov_speed^2.
    a = np.random.standard_normal(n_features)
    a *= mean_speed / np.linalg.norm(a)
    A = sym(np.random.standard_normal((n_features, n_features)))
    A *= cov_speed / np.sqrt(0.5 * np.sum(A**2))
    v, T = L @ a, L @ A @ L.T

    # Longueur de la géodésique t -> Exp(t (v, T)), t in [0, 1] :
    # d_True = ||(v, T)|| = sqrt(mean_speed^2 + cov_speed^2).
    d_True = np.sqrt(fisher_norm2(Sigma0, v, T))
    mu1, Sigma1 = exp_map_mvn(mu0, Sigma0, v, T)
    print(f"  n = {n_features}, cond(Sigma0) = {condition_number:g}, "
          f"mean_speed = {mean_speed:g}, cov_speed = {cov_speed:g}, seed = {seed}")
    print(f"  cond(Sigma1) = {np.linalg.cond(Sigma1):.3g}")

    # ----- calcul de d_F avec ses itérés -----
    start = time.perf_counter()
    dF_2, K_iterates, J_values, grad_norms = dist2_dF_iterates(
        mu0, Sigma0, mu1, Sigma1, tol=tol, max_iter=max_iter)
    time_dF = time.perf_counter() - start

    d_F = np.sqrt(dF_2)
    d_0 = np.sqrt(J_values[0])    # K_0 = 0, donc J(K_0) = d_0^2
    n_iter = J_values.size - 1
    rel_error_dF = (d_F - d_True) / d_True
    rel_error_d0 = (d_0 - d_True) / d_True

    print(f"  d_True     = {d_True:.10f}")
    print(f"  d_F (BFGS) = {d_F:.10f}   (d_F - d_True)/d_True = {rel_error_dF:+.3e}")
    print(f"  d_0        = {d_0:.10f}   (d_0 - d_True)/d_True = {rel_error_d0:+.3e}")
    print(f"  {n_iter} BFGS iterations in {time_dF * 1e3:.2f} ms, "
          f"final |grad|_inf = {grad_norms[-1]:.2e}, tol = {tol:.0e}")
    print(f"  ||K_T||_F = {np.linalg.norm(K_iterates[-1]):.4f}")

    # J(K) >= d_F^2 pour tout K et d_True >= d_F : les deux sont des majorants de d_F.
    if rel_error_dF > agreement_rtol:
        verdict = "d_F (BFGS) > d_True : BFGS s'arrête au-dessus de d_F (minimum local ou non convergé)."
    elif rel_error_dF < -agreement_rtol:
        verdict = "d_F (BFGS) < d_True : la géodésique générée n'est pas minimisante, d_True n'est pas d_F."
    else:
        verdict = "d_F (BFGS) = d_True à la tolérance près."
    print(f"  -> {verdict}")

    # ----- liste des itérés -----
    print("k\tJ(K_k)\t\tJ(K_k) - d_True^2\t|grad|_inf")
    print("-" * 64)
    table = np.column_stack([np.arange(n_iter + 1), J_values, J_values - d_True**2, grad_norms])
    np.savetxt("output/dF_iterates.csv", table, delimiter=",",
               header="k,J,J_minus_dTrue2,grad_inf", comments="")
    for k, J, gap, grad in table:
        print(f"{int(k):3d}\t{J:.10e}\t{gap:+.3e}\t\t{grad:.3e}")

    # ----- enregistrement du cas, pour reproductibilité -----
    record = dict(
        n_features=n_features, condition_number=condition_number, mean_speed=mean_speed,
        cov_speed=cov_speed, seed=seed, tol=tol, max_iter=max_iter,
        optimizer="BFGS, K_0 = 0, analytic gradient",
        mu0=mu0.tolist(), Sigma0=Sigma0.tolist(), v=v.tolist(), T=T.tolist(),
        mu1=mu1.tolist(), Sigma1=Sigma1.tolist(),
        d_True=d_True, d_F=d_F, d_0=d_0, rel_error_dF=rel_error_dF, rel_error_d0=rel_error_d0,
        n_iter=n_iter, final_grad_inf=grad_norms[-1], time_dF=time_dF, verdict=verdict,
    )
    with open("output/dF_iterates.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    # ----- tracé : J(K_k) et écart à d_True^2 -----
    fig, (ax_J, ax_gap) = plt.subplots(1, 2, figsize=(11, 4))
    ax_J.plot(J_values, marker="o", label=r"$J(K_k)$")
    ax_J.axhline(d_True**2, ls=":", c="k", label=r"$d_{True}^2$")
    ax_J.set_xlabel("Itération k")
    ax_J.set_ylabel(r"$J(K_k)$")
    ax_J.legend()
    ax_J.grid(True, ls=":")

    ax_gap.semilogy(np.abs(J_values - d_True**2), marker="o", label=r"$|J(K_k) - d_{True}^2|$")
    ax_gap.semilogy(grad_norms, marker="s", ms=3, label=r"$\|\nabla J(K_k)\|_\infty$")
    ax_gap.set_xlabel("Itération k")
    ax_gap.legend()
    ax_gap.grid(True, which="both", ls=":")

    fig.suptitle(f"BFGS pour $d_F$ (n={n_features}, $d_{{True}}$={d_True:.4f}, "
                 f"$d_F$={d_F:.4f}, $d_0$={d_0:.4f})")
    fig.tight_layout()

    out_path = "output/dF_iterates.png"
    fig.savefig(out_path, dpi=150)
    print(f"[run_dF_iterates] saved figure -> {out_path}, "
          f"data -> output/dF_iterates.csv and output/dF_iterates.json")

    plt.show(block=True)
    print("[run_dF_iterates] done.")


if __name__ == "__main__":
    main()
