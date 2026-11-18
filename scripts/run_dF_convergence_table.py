import json
import time

import numpy as np
from scipy.optimize import minimize

from src.distance import _reduced_problem, _skew_objective, dist2_d0, dist2_dF
from src.simulated_data import random_SPD

TOL = 1e-10
MAX_ITER = 1000
BFGS_OPTIONS = dict(gtol=TOL, maxiter=MAX_ITER)
N_RANDOM_STARTS = 4
N_TIMING_REPEATS = 5
RESTART_SEED = 12345


def exact_case(name: str, mu0: np.ndarray, Sigma0: np.ndarray, mu1: np.ndarray,
               Sigma1: np.ndarray, d_exact: float | None = None, seed: int | None = None) -> dict:
    """
    Packs one Gaussian pair into a case record.

    Parameters
    ----------
    name : str
        Case identifier used in the table.
    mu0, mu1 : ndarray
        Means of shape (n,).
    Sigma0, Sigma1 : ndarray
        SPD covariances of shape (n, n).
    d_exact : float, optional
        Closed-form Fisher-Rao distance when one is known.
    seed : int, optional
        Seed of the global NumPy generator used to build the pair.

    Returns
    -------
    dict
        Case record.
    """
    return dict(name=name, mu0=mu0, Sigma0=Sigma0, mu1=mu1, Sigma1=Sigma1,
                d_exact=d_exact, seed=seed)


def random_case(name: str, n: int, condition_number: float, mean_norm: float, seed: int) -> dict:
    """
    Builds a random pair with both covariances drawn by random_SPD.

    The mean difference has a uniformly random direction and Mahalanobis norm
    mean_norm with respect to Sigma0, i.e. the normalized mean has norm mean_norm.

    Parameters
    ----------
    name : str
        Case identifier used in the table.
    n : int
        Dimension.
    condition_number : float
        Condition number passed to random_SPD for both covariances.
    mean_norm : float
        Norm of Sigma0^{-1/2} (mu1 - mu0).
    seed : int
        Seed of the global NumPy generator (random_SPD draws from it).

    Returns
    -------
    dict
        Case record.
    """
    np.random.seed(seed)
    Sigma0 = random_SPD(n, condition_number)
    Sigma1 = random_SPD(n, condition_number)
    mu0 = np.random.normal(size=n)
    direction = np.random.normal(size=n)
    direction /= np.linalg.norm(direction)
    mu1 = mu0 + mean_norm * np.linalg.cholesky(Sigma0) @ direction
    return exact_case(name, mu0, Sigma0, mu1, Sigma1, seed=seed)


def build_cases() -> list[dict]:
    """
    Returns the fixed list of Gaussian pairs of the convergence table.

    Returns
    -------
    list of dict
        Case records, exact cases first, then structured, random and stress cases.
    """
    cases = []

    Sigma = np.diag([4.0, 1.0, 0.25])
    cases.append(exact_case("equal means", np.zeros(3), np.eye(3), np.zeros(3), Sigma,
                            d_exact=np.linalg.norm(np.log(np.diag(Sigma))) / np.sqrt(2)))

    np.random.seed(1)
    Sigma = random_SPD(3, 10.0)
    delta = np.array([1.0, -2.0, 0.5])
    maha2 = delta @ np.linalg.solve(Sigma, delta)
    cases.append(exact_case("equal covariances", np.zeros(3), Sigma, delta, Sigma,
                            d_exact=np.sqrt(2) * np.arccosh(1 + maha2 / 4), seed=1))

    cases.append(exact_case("article example", np.zeros(2), np.eye(2), np.array([4.0, 2.0]),
                            np.diag([20.0, 1.0])))
    cases.append(exact_case("isotropic", np.zeros(3), np.eye(3), np.full(3, 2.0), 3.0 * np.eye(3)))
    cases.append(exact_case("diagonal, aligned mean", np.zeros(3), np.eye(3),
                            np.array([3.0, 0.0, 0.0]), np.diag([10.0, 1.0, 0.1])))
    cases.append(exact_case("diagonal, mixed mean", np.zeros(3), np.eye(3),
                            np.full(3, 2.0), np.diag([10.0, 1.0, 0.1])))
    cases.append(exact_case("dense, unnormalized",
                            np.zeros(3),
                            np.array([[2.0, 0.5, 0.0], [0.5, 1.0, 0.3], [0.0, 0.3, 0.5]]),
                            np.array([1.0, -2.0, 0.5]), np.diag([0.2, 3.0, 1.0])))

    cases.append(random_case("random", 5, 10.0, 1.0, seed=10))
    cases.append(random_case("random", 5, 100.0, 5.0, seed=11))
    cases.append(random_case("random, large mean", 5, 10.0, 30.0, seed=12))
    cases.append(random_case("random", 10, 100.0, 3.0, seed=13))
    cases.append(random_case("random", 20, 1000.0, 10.0, seed=14))
    cases.append(random_case("random", 30, 10.0, 3.0, seed=15))

    for L in (100.0, 1000.0):
        cases.append(exact_case(f"stress L={L:g}", np.zeros(2), np.eye(2),
                                np.array([1.0, np.sqrt(L)]), np.diag([1.0 / L, L])))
    return cases


def normalized_descriptors(case: dict) -> tuple[float, float]:
    """
    Computes the normalized mean norm and the condition number of the normalized covariance.

    Parameters
    ----------
    case : dict
        Case record.

    Returns
    -------
    mean_norm : float
        Norm of Sigma0^{-1/2} (mu1 - mu0).
    condition_number : float
        Condition number of Sigma0^{-1/2} Sigma1 Sigma0^{-1/2}.
    """
    L = np.linalg.cholesky(case["Sigma0"])
    mu = np.linalg.solve(L, case["mu1"] - case["mu0"])
    Sigma = np.linalg.solve(L, np.linalg.solve(L, case["Sigma1"]).T)
    eigenvalues = np.linalg.eigvalsh(Sigma)
    return float(np.linalg.norm(mu)), float(eigenvalues[-1] / eigenvalues[0])


def best_time(function, *args) -> float:
    """
    Returns the smallest wall-clock time over N_TIMING_REPEATS calls.

    Parameters
    ----------
    function : callable
        Function to time.
    *args
        Arguments passed to function.

    Returns
    -------
    float
        Runtime in seconds.
    """
    times = []
    for _ in range(N_TIMING_REPEATS):
        start = time.perf_counter()
        function(*args)
        times.append(time.perf_counter() - start)
    return min(times)


def evaluate_case(case: dict, rng: np.random.Generator) -> dict:
    """
    Runs BFGS from K = 0 and from random starts on one case and collects diagnostics.

    The reference d_F is the BFGS run started at K = 0, exactly as in dist2_dF.
    The random starts K ~ N(0, s^2) entrywise, s = ||Y(0)||_F / sqrt(n), only
    check whether a lower value of J exists; they do not certify a global minimum.

    Parameters
    ----------
    case : dict
        Case record.
    rng : numpy.random.Generator
        Generator for the random starts.

    Returns
    -------
    dict
        Result record.
    """
    args = (case["mu0"], case["Sigma0"], case["mu1"], case["Sigma1"])
    n = case["mu0"].size
    n_params = n * (n - 1) // 2
    W, Y = _reduced_problem(*args)
    mean_norm, condition_number = normalized_descriptors(case)
    d0 = np.sqrt(dist2_d0(*args))

    if n_params == 0:
        dF2, n_iter, n_eval, grad_inf, message = _skew_objective(np.zeros(0), W, Y)[0], 0, 1, 0.0, "n = 1"
        restart_values = []
    else:
        result = minimize(_skew_objective, np.zeros(n_params), args=(W, Y), jac=True,
                          method='BFGS', options=BFGS_OPTIONS)
        dF2, n_iter, n_eval = result.fun, result.nit, result.nfev
        grad_inf, message = float(np.max(np.abs(result.jac))), result.message
        scale = np.linalg.norm(Y) / np.sqrt(n)
        restart_values = [minimize(_skew_objective, scale * rng.normal(size=n_params), args=(W, Y),
                                   jac=True, method='BFGS', options=BFGS_OPTIONS).fun
                          for _ in range(N_RANDOM_STARTS)]

    dF = np.sqrt(dF2)
    restart_distances = np.sqrt(restart_values)
    return dict(
        name=case["name"], n=n, n_params=n_params, seed=case["seed"],
        mean_norm=mean_norm, condition_number=condition_number,
        d0=d0, dF=dF, d_exact=case["d_exact"],
        abs_error=abs(d0 - dF), rel_error=abs(d0 - dF) / dF,
        n_iter=int(n_iter), n_eval=int(n_eval), grad_inf=grad_inf, message=message,
        restart_max_deviation=float(np.max(np.abs(restart_distances - dF), initial=0.0)),
        restart_min_minus_dF=float(np.min(restart_distances - dF, initial=0.0)),
        time_dF=best_time(dist2_dF, *args), time_d0=best_time(dist2_d0, *args),
        mu0=case["mu0"].tolist(), Sigma0=case["Sigma0"].tolist(),
        mu1=case["mu1"].tolist(), Sigma1=case["Sigma1"].tolist(),
    )


def latex_table(results: list[dict]) -> str:
    """
    Formats the results as a LaTeX tabular for the article.

    Parameters
    ----------
    results : list of dict
        Result records.

    Returns
    -------
    str
        LaTeX source of the table.
    """
    lines = [
        r"\begin{tabular}{lrrrrrrrrrr}",
        r"\toprule",
        r"Case & $n$ & $\kappa(\Sigma)$ & $\|\mu\|$ & $d_0$ & $d_F$ & rel.\ err. & iter. & eval. "
        r"& $\|\nabla J\|_\infty$ & time (ms) \\",
        r"\midrule",
    ]
    for r in results:
        flag = "" if r["message"].startswith("Optimization terminated") or r["n_params"] == 0 else r"$^\dagger$"
        lines.append(
            f"{r['name']} & {r['n']} & {r['condition_number']:.3g} & {r['mean_norm']:.3g} & "
            f"{r['d0']:.6f} & {r['dF']:.6f} & {100 * r['rel_error']:.3f}\\% & "
            f"{r['n_iter']} & {r['n_eval']} & {r['grad_inf']:.1e}{flag} & {1e3 * r['time_dF']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    """
    Computes d_F and d_0 on the fixed case list and saves CSV, JSON and LaTeX tables.
    """
    print("[run_dF_convergence_table] starting…")
    rng = np.random.default_rng(RESTART_SEED)
    results = [evaluate_case(case, rng) for case in build_cases()]

    columns = ["name", "n", "n_params", "seed", "mean_norm", "condition_number", "d0", "dF",
               "d_exact", "abs_error", "rel_error", "n_iter", "n_eval", "grad_inf",
               "restart_max_deviation", "restart_min_minus_dF", "time_dF", "time_d0", "message"]
    with open("output/dF_convergence_table.csv", "w", encoding="utf-8") as file:
        file.write(",".join(columns) + "\n")
        for r in results:
            file.write(",".join(f'"{r[c]}"' if isinstance(r[c], str) else str(r[c]) for c in columns) + "\n")

    settings = dict(method="scipy.optimize.minimize BFGS, analytic gradient", gtol=TOL,
                    gtol_norm="sup norm on the n(n-1)/2 upper-triangular coordinates of K",
                    max_iter=MAX_ITER, initialization="K = 0",
                    random_starts=N_RANDOM_STARTS, restart_seed=RESTART_SEED,
                    restart_distribution="K_ij ~ N(0, (||Y(0)||_F / sqrt(n))^2)",
                    timing=f"min over {N_TIMING_REPEATS} calls of dist2_dF / dist2_d0")
    with open("output/dF_convergence_table.json", "w", encoding="utf-8") as file:
        json.dump(dict(settings=settings, results=results), file, indent=2)
    with open("output/dF_convergence_table.tex", "w", encoding="utf-8") as file:
        file.write(latex_table(results) + "\n")

    for r in results:
        exact = "" if r["d_exact"] is None else f"  |dF-exact|={abs(r['dF'] - r['d_exact']):.1e}"
        print(f"{r['name']:<24} n={r['n']:>2} kappa={r['condition_number']:>9.3g} |mu|={r['mean_norm']:>6.3g}"
              f"  d0={r['d0']:.6f} dF={r['dF']:.6f} rel={100 * r['rel_error']:.4f}%"
              f"  it={r['n_iter']:>3} ev={r['n_eval']:>3} g={r['grad_inf']:.1e}"
              f"  restarts dev={r['restart_max_deviation']:.1e} min-dF={r['restart_min_minus_dF']:.1e}"
              f"  t={1e3 * r['time_dF']:.2f}ms  [{r['message']}]{exact}")
    print("[run_dF_convergence_table] saved output/dF_convergence_table.{csv,json,tex}")


if __name__ == "__main__":
    main()
