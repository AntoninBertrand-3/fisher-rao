import json
import time

import numpy as np
import matplotlib.pyplot as plt

from scripts.run_dF_random_geodesics import build_case
from src.distance import dist2_dF_iterates
from src.mvn import _horizontal_generator, _lifted_endpoint, _project, _symmetric_function


# Cas de la figure de run_dF_iterates : (n, ||a||, lambda, seed).
CASE = (10, 7.0, 1000.0, 7)
OUT_STEM = "output/dF_iterates_horizontal_init"
TOL = 1e-10
MAX_ITER = 1000


def horizontal_init(nu: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Fiber parameter S_init obtained by shooting along the horizontal projection
    of the logarithm of the lift G_1(0) of the normalized endpoint N(nu, C).

    H_0 = log G_1(0) lies in the tangent space of M at I, whose elements are
    [[-A, a, W], [a^T, 0, -a^T], [-W, -a, A]] with W skew-symmetric; the
    horizontal part V(A, a) and the vertical part W are Frobenius-orthogonal,
    so the projection sets the (1,3) and (3,1) blocks to zero. The point
    G' = exp V(A, a) of M lies over some Gaussian N(nu', C') that is not the
    endpoint in general, and S_init is its fiber parameter (notation.md, section 7):
    G'_31 = (S - nu' nu'^T / 2) C'^{-1}, hence S_init = G'_31 C' + nu' nu'^T / 2.

    Parameters
    ----------
    nu : ndarray
        Normalized endpoint mean of shape (n_features,), the start being N(0, I).
    C : ndarray
        Normalized SPD endpoint covariance of shape (n_features, n_features).

    Returns
    -------
    S_init : ndarray
        Skew-symmetric fiber parameter of G', of shape (n_features, n_features).
    diagnostics : dict
        Norms of the vertical part of H_0, of the defects of H_0 in the tangent
        space of M, and of the skew-symmetry defect of S_init.
    """
    n = nu.size
    H0 = _symmetric_function(_lifted_endpoint(nu, C, np.zeros((n, n))), np.log)
    # Projection horizontale ; les moyennes symétrisent les blocs égaux en arithmétique exacte.
    a = 0.5 * (H0[:n, n] - H0[n + 1:, n])
    A = 0.5 * (H0[n + 1:, n + 1:] - H0[:n, :n])
    G_shot = _symmetric_function(_horizontal_generator(a, A), np.exp)
    nu_shot, C_shot = _project(G_shot, n)
    S = G_shot[n + 1:, :n] @ C_shot + 0.5 * np.outer(nu_shot, nu_shot)
    diagnostics = dict(
        norm_vertical_part=np.linalg.norm(H0[:n, n + 1:]),
        tangent_defect_a=np.linalg.norm(H0[:n, n] + H0[n + 1:, n]),
        tangent_defect_A=np.linalg.norm(H0[:n, :n] + H0[n + 1:, n + 1:]),
        skew_defect_S_init=np.linalg.norm(S + S.T),
    )
    return 0.5 * (S - S.T), diagnostics


def main() -> None:
    """
    Compares BFGS on J(K) started at K_0 = 0 and at K_0 = S_init / 2 on the case
    of run_dF_iterates, and saves the iterates (CSV), the record (JSON) and the figure (PNG).
    """
    print("[run_dF_iterates_horizontal_init] starting...")
    n_features, mean_norm, stretch, seed = CASE
    case = build_case(n_features, mean_norm, stretch, seed)
    mu0, Sigma0, mu1, Sigma1 = case["mu0"], case["Sigma0"], case["mu1"], case["Sigma1"]
    d_True = case["d_True"]

    # P0 = N(0, I) : la paire est déjà normalisée, (nu, C) = (mu1, Sigma1).
    start = time.perf_counter()
    S_init, diagnostics = horizontal_init(mu1, Sigma1)
    time_init = time.perf_counter() - start
    # Relation S = 2K entre paramètre de fibre et variable de J : observation numérique (test_mvn).
    K_init = 0.5 * S_init

    runs = {}
    for name, K0 in (("zero", None), ("horizontal", K_init)):
        start = time.perf_counter()
        dF_2, K_iterates, J_values, grad_norms = dist2_dF_iterates(
            mu0, Sigma0, mu1, Sigma1, tol=TOL, max_iter=MAX_ITER, K_init=K0)
        runtime = time.perf_counter() - start
        runs[name] = dict(J=J_values, grad=grad_norms, K_final=K_iterates[-1],
                          K_start=K_iterates[0], runtime=runtime, d_F=np.sqrt(dF_2))

    K_star = runs["zero"]["K_final"]
    print(f"  d_True = {d_True:.10f}, d_0 = {np.sqrt(runs['zero']['J'][0]):.10f}")
    for name, run in runs.items():
        run["n_iter"] = run["J"].size - 1
        run["dist_start_to_K_star"] = np.linalg.norm(run["K_start"] - K_star)
        run["rel_error_dF"] = (run["d_F"] - d_True) / d_True
        print(f"  K_0 = {name:10s}: J(K_0) = {run['J'][0]:.4f}, ||K_0 - K*||_F = "
              f"{run['dist_start_to_K_star']:.3f}, {run['n_iter']} iterations, "
              f"(d_F - d_True)/d_True = {run['rel_error_dF']:+.2e}, "
              f"final |grad|_inf = {run['grad'][-1]:.2e}")
    print(f"  ||K* (horizontal) - K* (zero)||_F = "
          f"{np.linalg.norm(runs['horizontal']['K_final'] - K_star):.2e}")
    print(f"  S_init computed in {time_init * 1e3:.2f} ms; diagnostics: {diagnostics}")

    # ----- sorties -----
    n_rows = max(run["J"].size for run in runs.values())
    table = np.full((n_rows, 5), np.nan)
    table[:, 0] = np.arange(n_rows)
    for column, name in ((1, "zero"), (3, "horizontal")):
        size = runs[name]["J"].size
        table[:size, column] = runs[name]["J"]
        table[:size, column + 1] = runs[name]["grad"]
    np.savetxt(f"{OUT_STEM}.csv", table, delimiter=",",
               header="k,J_zero,grad_inf_zero,J_horizontal,grad_inf_horizontal", comments="")

    record = dict(
        n_features=n_features, mean_norm=mean_norm, stretch=stretch, seed=seed,
        construction="build_case of run_dF_random_geodesics", tol=TOL, max_iter=MAX_ITER,
        optimizer="BFGS on J(K), analytic gradient",
        initializations=dict(
            zero="K_0 = 0",
            horizontal="K_0 = S_init / 2, S_init fiber parameter of exp of the horizontal "
                       "projection of log G_1(0); S = 2K is a numerical observation"),
        d_True=d_True, d_0=float(np.sqrt(runs["zero"]["J"][0])), time_S_init=time_init,
        horizontal_init_diagnostics=diagnostics,
        runs={name: dict(J_start=run["J"][0], n_iter=run["n_iter"], d_F=run["d_F"],
                         rel_error_dF=run["rel_error_dF"], final_grad_inf=run["grad"][-1],
                         dist_start_to_K_star=run["dist_start_to_K_star"],
                         runtime=run["runtime"])
              for name, run in runs.items()},
    )
    with open(f"{OUT_STEM}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)

    # ----- figure : même format que run_dF_iterates, deux initialisations -----
    labels = {"zero": r"$K_0 = 0$", "horizontal": r"$K_0 = S_{init}/2$"}
    fig, (ax_J, ax_gap) = plt.subplots(1, 2, figsize=(11, 4))
    for (name, run), color in zip(runs.items(), ("C0", "C3")):
        ax_J.plot(run["J"], marker="o", ms=3, c=color, label=labels[name])
        ax_gap.plot(10 * np.log10(np.abs(run["J"] - d_True**2)), marker="o", ms=3, c=color,
                    label=r"$|J(K_k) - d_{True}^2|$, " + labels[name])
        ax_gap.plot(10 * np.log10(run["grad"]), ls="--", marker="s", ms=2, c=color, alpha=0.6,
                    label=r"$\|\nabla J(K_k)\|_\infty$, " + labels[name])
    ax_J.axhline(d_True**2, ls=":", c="k", label=r"$d_{True}^2$")
    ax_J.set_xlabel("Itération k")
    ax_J.set_ylabel(r"$J(K_k)$")
    ax_J.legend()
    ax_J.grid(True, ls=":")
    ax_gap.set_xlabel("Itération k")
    ax_gap.set_ylabel("dB")
    ax_gap.legend(fontsize=8)
    ax_gap.grid(True, ls=":")
    fig.suptitle(f"BFGS pour $d_F$ : initialisation nulle vs horizontale "
                 f"(n={n_features}, $\\|a\\|$={mean_norm:g}, $\\lambda$={stretch:g}, graine {seed})")
    fig.tight_layout()
    fig.savefig(f"{OUT_STEM}.png", dpi=150)
    print(f"[run_dF_iterates_horizontal_init] saved {OUT_STEM}.{{png,csv,json}}")


if __name__ == "__main__":
    main()
