import csv
import json
import time

import numpy as np
from scipy.stats import ortho_group

from src.distance import dist2_d0, dist2_dF_iterates
from src.mvn import exp_map_mvn, fisher_norm2

DIMENSIONS = (2, 5, 10, 30)
MEAN_NORMS = (0.5, 10.0)        # ||a||
STRETCHES = (3.0, 30.0)          # lambda: eigenvalues of exp(A) log-spaced from 1/lambda to lambda
BASE_SEED = 0
TOL = 1e-10
MAX_ITER = 1000


def build_case(n: int, mean_norm: float, stretch: float, seed: int) -> dict:
    """
    Builds P1 = Exp_{P0}(a, A) from P0 = N(0, I_n), with the random draw of run_d0_error_random
    applied to the initial velocity (a, A) instead of the endpoint (mu1, Sigma1).

    A = Q diag(l) Q^T with Q Haar-distributed and fixed eigenvalues l evenly spaced from
    -ln(lambda) to ln(lambda), so that exp(A) has eigenvalues log-spaced from 1/lambda to lambda.
    a = ||a|| u with u uniform on the unit sphere, drawn independently of Q.
    The geodesic length is d_True^2 = ||a||^2 + Tr(A^2) / 2.

    Parameters
    ----------
    n : int
        Dimension, n >= 2.
    mean_norm : float
        Norm of the mean velocity a.
    stretch : float
        lambda = lambda_max(exp(A)) = 1 / lambda_min(exp(A)) >= 1.
    seed : int
        Seed of the NumPy generator drawing Q and u.

    Returns
    -------
    dict
        Case record with the parameters, the eigenvalues of A, (a, A), d_True and (mu1, Sigma1).
    """
    rng = np.random.default_rng(seed)
    eigenvalues = np.linspace(-np.log(stretch), np.log(stretch), n)
    Q = ortho_group.rvs(n, random_state=rng)
    A = Q @ np.diag(eigenvalues) @ Q.T
    direction = rng.standard_normal(n)
    a = mean_norm * direction / np.linalg.norm(direction)

    mu0, Sigma0 = np.zeros(n), np.eye(n)
    d_True = np.sqrt(fisher_norm2(Sigma0, a, A))
    mu1, Sigma1 = exp_map_mvn(mu0, Sigma0, a, A)
    return dict(n=n, mean_norm=mean_norm, stretch=stretch, seed=seed,
                eigenvalues_A=eigenvalues, a=a, A=A, d_True=d_True,
                mu0=mu0, Sigma0=Sigma0, mu1=mu1, Sigma1=Sigma1)


def evaluate_case(case: dict) -> dict:
    """
    Computes d_F by BFGS (K_0 = 0) and d_0 for one case and compares them with d_True.

    Parameters
    ----------
    case : dict
        Case record from build_case.

    Returns
    -------
    dict
        Result record: parameters, d_True, d_F, d_0, signed relative errors with respect to
        d_True, absolute and relative error of d_0 with respect to d_F, BFGS diagnostics,
        runtimes and cond(Sigma1).
    """
    endpoints = (case["mu0"], case["Sigma0"], case["mu1"], case["Sigma1"])
    start = time.perf_counter()
    dF_2, _, J_values, grad_norms = dist2_dF_iterates(*endpoints, tol=TOL, max_iter=MAX_ITER)
    time_dF = time.perf_counter() - start
    start = time.perf_counter()
    d0_2 = dist2_d0(*endpoints)
    time_d0 = time.perf_counter() - start

    d_True, d_F, d_0 = case["d_True"], np.sqrt(dF_2), np.sqrt(d0_2)
    return dict(
        n=case["n"], mean_norm=case["mean_norm"], stretch=case["stretch"],
        seed=case["seed"], d_True=d_True, d_F=d_F, d_0=d_0,
        rel_error_dF_vs_dTrue=(d_F - d_True) / d_True,
        rel_error_d0_vs_dTrue=(d_0 - d_True) / d_True,
        abs_error=abs(d_0 - d_F), rel_error=abs(d_0 - d_F) / d_F,
        n_iter=J_values.size - 1, final_grad_inf=grad_norms[-1],
        hit_max_iter=bool(J_values.size - 1 >= MAX_ITER),
        cond_Sigma1=np.linalg.cond(case["Sigma1"]), time_dF=time_dF, time_d0=time_d0,
    )


def latex_sci(x: float) -> str:
    """
    Formats a number as $m\\text{e}k$ with two significant digits, for the LaTeX table.

    Parameters
    ----------
    x : float
        Number to format.

    Returns
    -------
    str
        LaTeX math string.
    """
    if x == 0:
        return "$0$"
    mantissa, exponent = f"{x:.1e}".split("e")
    return f"${mantissa}\\text{{e}}{int(exponent)}$".replace("e}-", "e-}")


def latex_table(results: list[dict]) -> str:
    """
    Builds the booktabs LaTeX table of the results, one block per dimension.

    Parameters
    ----------
    results : list of dict
        Result records from evaluate_case, ordered by dimension.

    Returns
    -------
    str
        LaTeX tabular environment.
    """
    lines = [r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$n$ & $\|a\|$ & $\lambda$ & $d_{\mathrm{True}}$ & $\varepsilon_F$ & $\varepsilon_0$ & it. \\",
             r"\midrule"]
    for index, r in enumerate(results):
        if index > 0 and r["n"] != results[index - 1]["n"]:
            lines.append(r"\midrule")
        iterations = f"${r['n_iter']}^\\dagger$" if r["hit_max_iter"] else str(r["n_iter"])
        lines.append(f"{r['n']} & {r['mean_norm']:g} & {r['stretch']:g} & {r['d_True']:.2f} & "
                     f"{latex_sci(r['rel_error_dF_vs_dTrue'])} & "
                     f"{latex_sci(r['rel_error_d0_vs_dTrue'])} & {iterations} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def main() -> None:
    """
    Runs d_F (BFGS) and d_0 on random geodesics from N(0, I) over a grid of
    (n, ||a||, lambda), one seed per configuration, and saves CSV, JSON and LaTeX tables.

    The reference d_True is the length of the generated geodesic; it equals d_F only if
    that geodesic is minimizing, which is assumed and not guaranteed.
    """
    print("[run_dF_random_geodesics] starting...")
    configurations = [(n, stretch, mean_norm) for n in DIMENSIONS
                      for stretch in STRETCHES for mean_norm in MEAN_NORMS]

    results, cases = [], []
    for index, (n, stretch, mean_norm) in enumerate(configurations):
        case = build_case(n, mean_norm, stretch, seed=BASE_SEED + index)
        result = evaluate_case(case)
        cases.append(case)
        results.append(result)
        print(f"  n={n:2d} |a|={mean_norm:g} lambda={stretch:4g}  d_True={result['d_True']:7.3f}  "
              f"eps_F={result['rel_error_dF_vs_dTrue']:+.1e}  eps_0={result['rel_error_d0_vs_dTrue']:+.1e}  "
              f"it={result['n_iter']:4d}  grad={result['final_grad_inf']:.0e}  "
              f"cond(Sigma1)={result['cond_Sigma1']:.1e}  t={result['time_dF']:.2f}s", flush=True)

    with open("output/dF_random_geodesics.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    settings = dict(
        P0="N(0, I_n)", dimensions=DIMENSIONS, mean_norms=MEAN_NORMS,
        stretches=STRETCHES, seeds="BASE_SEED + configuration index",
        construction="A = Q diag(l) Q^T, Q Haar (scipy ortho_group), l evenly spaced "
                     "from -ln(lambda) to ln(lambda); a = ||a|| u, u uniform on the sphere; "
                     "P1 = Exp_P0(a, A)",
        reference="d_True = Fisher length of the geodesic (equals d_F if it is minimizing)",
        optimizer="BFGS (scipy), K_0 = 0, analytic gradient", tol=TOL, max_iter=MAX_ITER,
    )
    records = [dict(result, eigenvalues_A=case["eigenvalues_A"].tolist(), a=case["a"].tolist(),
                    mu1=case["mu1"].tolist(), Sigma1=case["Sigma1"].tolist())
               for result, case in zip(results, cases)]
    with open("output/dF_random_geodesics.json", "w", encoding="utf-8") as file:
        json.dump(dict(settings=settings, results=records), file, indent=2)
    with open("output/dF_random_geodesics.tex", "w", encoding="utf-8") as file:
        file.write(latex_table(results) + "\n")
    print("[run_dF_random_geodesics] saved output/dF_random_geodesics.{csv,json,tex}")


if __name__ == "__main__":
    main()
