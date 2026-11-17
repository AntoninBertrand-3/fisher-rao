import json
import time

import numpy as np
import matplotlib.pyplot as plt

from scripts.run_dF_random_geodesics import build_case
from src.distance import dist2_dF_iterates


# (n, ||a||, lambda, seed, préfixe des sorties)
CASES = (
    # Erreur de d_0 élevée (8 %), J(K_k) reste au-dessus de d_True^2 ; choisi par un balayage
    # de (n, ||a||, lambda, seed).
    (10, 7.0, 1000.0, 7, "output/dF_iterates"),
    # Paramètres de la dernière ligne du tableau de run_dF_random_geodesics, graine 18 au lieu
    # de 15 : avec 15, J(K_k) passe sous d_True^2 au niveau de l'erreur d'arrondi.
    (30, 10.0, 30.0, 18, "output/dF_iterates_n30_seed18"),
)
TOL = 1e-10
MAX_ITER = 1000
# Seuil relatif pour comparer d_F (BFGS) à d_True, au-dessus de l'erreur d'arrondi.
AGREEMENT_RTOL = 1e-6


def run_case(n_features: int, mean_norm: float, stretch: float, seed: int, out_stem: str) -> None:
    """
    Builds a pair of MVNs joined by a known geodesic, computes d_F by BFGS and
    compares it, together with d_0, to the length d_True of that geodesic.

    The pair is drawn exactly as in run_dF_random_geodesics (build_case):
    P0 = N(0, I_n), initial velocity (a, A) with ||a|| and the stretch lambda of A
    prescribed, and P1 = Exp_{P0}(a, A). The geodesic has length
    d_True^2 = ||a||^2 + Tr(A^2) / 2 >= d_F^2; d_True is taken as the reference d_F,
    which assumes that this geodesic is minimizing (not guaranteed).
    Saves the iterates (CSV), the case record (JSON) and the figure (PNG) under out_stem.

    Parameters
    ----------
    n_features : int
        Dimension n.
    mean_norm : float
        Norm ||a|| of the mean velocity.
    stretch : float
        lambda: eigenvalues of exp(A) log-spaced from 1/lambda to lambda.
    seed : int
        Seed of build_case.
    out_stem : str
        Output path without extension.
    """
    tol, max_iter, agreement_rtol = TOL, MAX_ITER, AGREEMENT_RTOL

    # ----- génération de la paire, comme dans run_dF_random_geodesics -----
    case = build_case(n_features, mean_norm, stretch, seed)
    mu0, Sigma0, mu1, Sigma1 = case["mu0"], case["Sigma0"], case["mu1"], case["Sigma1"]
    d_True = case["d_True"]
    print(f"  n = {n_features}, ||a|| = {mean_norm:g}, lambda = {stretch:g}, seed = {seed}")
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
    np.savetxt(f"{out_stem}.csv", table, delimiter=",",
               header="k,J,J_minus_dTrue2,grad_inf", comments="")
    for k, J, gap, grad in table:
        print(f"{int(k):3d}\t{J:.10e}\t{gap:+.3e}\t\t{grad:.3e}")

    # ----- enregistrement du cas, pour reproductibilité -----
    record = dict(
        n_features=n_features, mean_norm=mean_norm, stretch=stretch, seed=seed,
        construction="build_case of run_dF_random_geodesics", tol=tol, max_iter=max_iter,
        optimizer="BFGS, K_0 = 0, analytic gradient",
        a=case["a"].tolist(), eigenvalues_A=case["eigenvalues_A"].tolist(),
        mu1=mu1.tolist(), Sigma1=Sigma1.tolist(),
        d_True=d_True, d_F=d_F, d_0=d_0, rel_error_dF=rel_error_dF, rel_error_d0=rel_error_d0,
        n_iter=n_iter, final_grad_inf=grad_norms[-1], time_dF=time_dF, verdict=verdict,
    )
    with open(f"{out_stem}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    # ----- tracé : J(K_k) et écart à d_True^2 -----
    fig, (ax_J, ax_gap) = plt.subplots(1, 2, figsize=(11, 4))
    ax_J.plot(J_values, marker="o", ms=3, label=r"$J(K_k)$")
    ax_J.axhline(d_True**2, ls=":", c="k", label=r"$d_{True}^2$")
    ax_J.set_xlabel("Itération k")
    ax_J.set_ylabel(r"$J(K_k)$")
    ax_J.legend()
    ax_J.grid(True, ls=":")

    # Échelle en dB : x_dB = 10 log10(x), soit 10 dB par décade.
    ax_gap.plot(10 * np.log10(np.abs(J_values - d_True**2)), marker="o", ms=3,
                label=r"$|J(K_k) - d_{True}^2|$")
    ax_gap.plot(10 * np.log10(grad_norms), marker="s", ms=3, label=r"$\|\nabla J(K_k)\|_\infty$")
    ax_gap.set_xlabel("Itération k")
    ax_gap.set_ylabel("dB")
    ax_gap.legend()
    ax_gap.grid(True, ls=":")

    fig.suptitle(f"BFGS pour $d_F$ (n={n_features}, $d_{{True}}$={d_True:.4f}, "
                 f"$d_F$={d_F:.4f}, $d_0$={d_0:.4f})")
    fig.tight_layout()

    fig.savefig(f"{out_stem}.png", dpi=150)
    print(f"[run_dF_iterates] saved {out_stem}.{{png,csv,json}}")


def main() -> None:
    """
    Runs run_case on each case of CASES.
    """
    print("[run_dF_iterates] starting...")
    for n_features, mean_norm, stretch, seed, out_stem in CASES:
        run_case(n_features, mean_norm, stretch, seed, out_stem)
    plt.show(block=True)
    print("[run_dF_iterates] done.")


if __name__ == "__main__":
    main()
