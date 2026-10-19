# Minimal numerical core

## Gaussian exponential, logarithm and mean

`exp_map_mvn` evaluates the full Fisher exponential. `log_map_mvn` returns
the initial tangent recovered by the horizontal fiber solver, together with
spectral, residual and endpoint reconstruction checks. Inspect `reliable`
before using it. `frechet_mean_mvn` averages qualified tangents and uses
backtracking; inspect `converged` and `message` on its result.

See [the map derivation and API examples](../../docs/mvn_maps.md).
Minimizing logarithms are a working assumption, and mean convergence does
not certify a global minimum. These are separate from the SPD-only exponential.

## Installation and distance usage

Install from the repository root with `pip install -r requirements.txt` and `pip install -e .`.
Run `python -m unittest discover -s test -t .` from the root. Runtime dependencies are NumPy and SciPy.

```python
import numpy as np
from src.distance import d_0, d_f_reference

args = (np.zeros(2), np.eye(2), np.array([2., 3.]), np.diag([.2, 5.]))
approximation = d_0(*args)
reference = d_f_reference(*args, n_starts=3, seed=0)
print(approximation, reference.distance)
for run in reference.runs:
    print(run.success, run.message, np.linalg.norm(run.jac), run.nit, run.nfev)
```

`skew_objective(mu, covariance, K)` takes an already normalized endpoint.
The distance functions take both original endpoints. A numerical reference is
the best finite converged restart, not a certified global minimum. If none
converges, the best finite fallback (including zero) has `reliable=False`.
Inspect `reliability_message` and every run's diagnostics before benchmarking.
For dimension one there are no coordinates and `runs` is empty.

## Formula and scaling checks

The implemented formula is exactly section 8 of `docs/notation.md`:
Y=(I+Sigma+mu mu.T/2)/2+K and J=2 sum(acosh(svdvals(Sigma^{-1/2}Y))²).
The matrix identity

    Y.T Sigma^{-1} Y - I
      = (Y-Sigma).T Sigma^{-1} (Y-Sigma) + mu mu.T/2

shows why every singular value must be at least one. Only roundoff-sized
violations are clipped. Poorly resolved spectra raise an error; there is no
covariance regularization. SVD avoids squaring the condition number by forming
Y.T Sigma^{-1} Y. The spectral derivative is
4 acosh(s)/sqrt(s²-1), with limit 4 at s=1; its skew-coordinate derivative
is gradient[i,j]-gradient[j,i]. Tests compare it with finite differences.

Special-case test values follow directly from the canonical metric:

- For n=1, Sigma=sigma² gives ds²=(dmu²+2 dsigma²)/sigma².
  Set y=sqrt(2)*sigma to obtain twice the upper-half-plane hyperbolic metric.
  Consequently d=sqrt(2)*acosh(((mu1-mu0)²+2*(sigma0²+sigma1²))/(4*sigma0*sigma1)).
- For equal means the covariance metric has factor 1/2, so the distance is
  the Euclidean norm of log generalized covariance eigenvalues divided by sqrt(2).
  Allowing the mean to vary cannot shorten it: dropping the mean velocity term
  gives the covariance-path lower bound, attained with constant mean.
- With equal isotropic covariances, rotation aligns the mean change with one
  coordinate; the univariate path attains the univariate marginal lower bound.

## Unresolved documentation issues

The mean geodesic equation in notation.md section 4 and state.md section 2.2
now has the minus sign consistent with its first integral. Differentiating
Sigma^{-1} mu_dot gives Sigma^{-1}(mu_ddot-Sigma_dot Sigma^{-1} mu_dot).
This code does not integrate that ODE.

The full reduced-distance identity is labeled established in state.md, but its
proof is still task A1 in open_problems.md. This implementation evaluates the
canonical objective without certifying that identity or global optimality.
Agreement on special cases is a check, not a general proof. Historical numerical
claims have not been reproduced. Very close endpoints can lose relative accuracy
when singular values round to one; the univariate case uses an equivalent stable
asinh formula. This initial core targets moderate conditioning.

## Horizontality experiment

`src.horizontality.HorizontalFiberObjective` evaluates
`R(S)=||(log G(S))_13||_F^2` and its analytic Frechet-log gradient for the
endpoint fiber in notation.md section 7. The lifted distance candidate is
`||log G(S)||_F/2`; its interpretation as the minimizing Fisher distance
assumes that a sufficiently horizontal lift projects to a minimizing geodesic.
It does not optimize `J(K)` or lifted distance.

The retired `experiments/horizontal_presentation_cases.py` (in
`archive/benchmarks_before_consolidation.zip`) was used to reuse the exact
fourteen saved pairs and the previous seeded initialization protocol. Install
`python -m pip install -e ".[test,experiments]"` for the experiment's additional
`threadpoolctl` dependency, used to fix BLAS to one thread. Existing result
files are preserved; outputs use the `horizontal_presentation_cases` prefix.
See `docs/horizontality_derivation.md` for the derivative, metric scaling,
and the distinction between identical S/K coordinate vectors and identical lifts.
