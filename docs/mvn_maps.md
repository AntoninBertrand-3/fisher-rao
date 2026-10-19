# Numerical Gaussian maps and Frechet means

The public functions `exp_map_mvn`, `log_map_mvn` and `frechet_mean_mvn`
are implemented in `src/mvn.py` and `src/mean.py`. They use the canonical Fisher
metric, including the covariance factor 1/2. The covariance-only
`geometry.exp_map_spd` remains a different function.

## Formulas and assumptions

**Literature-backed construction, implemented and numerically checked.**
For a Gaussian base P=(m,C), let Q=C^(1/2), W=C^(-1/2). A tangent (v,T)
normalizes to a=Wv, A=WTW. Form the horizontal generator

```text
V = [[-A, a,  0],
     [a.T,0,-a.T],
     [ 0,-a,  A]].
```

For G(t)=exp(tV), write Theta=G[:n,:n], delta=G[:n,n]. Projection gives
Sigma(t)=solve(Theta,I), mu(t)=solve(Theta,delta). The original-coordinate
endpoint is (m+Q mu(t), Q Sigma(t) Q). Thus differentiating at zero gives
the prescribed velocity (v,T), with squared speed a.T a + ||A||_F^2/2.
This is the horizontal construction from Kobayashi (2023), equations
(2.4)-(2.5) and section 2.2; see the bibliography and
`docs/horizontality_derivation.md` for the existing lift convention.

**Numerical logarithm under a working assumption.** Given a target,
`log_map_mvn` normalizes both endpoints and solves only
R(S)=||(log G_1(S))_13||_F^2, using analytic-gradient BFGS with zero and seeded
Gaussian starts. It does not optimize J(K), and assumes no S-to-K mapping.
From H=log G_1(S), it extracts a=H[:n,n] and A=-H[:n,:n]. The returned
original-coordinate tangent is (Qa, QAQ).

For an approximate horizontal solution, this extraction discards the
nonhorizontal part. The resulting horizontal generator may therefore miss
the target. Every candidate is checked by exponentiating (a,A) and measuring
both endpoint errors in normalized coordinates:

```text
mean_error = ||mu_reconstructed-mu|| / max(1, ||mu||)
covariance_error = ||Sigma_reconstructed-Sigma||_F / max(1, ||Sigma||_F).
```

Qualification requires the existing lift spectral screen, horizontal residual
<=1e-8 and both endpoint errors <=1e-7 by default. Thresholds are configurable;
they are numerical screens, not certified error bounds. Among qualified
candidates the smallest residual is selected, never the smallest distance.
If none qualifies, the best reconstructable, spectrally resolved candidate
is returned with `reliable=False`. If none can be reconstructed, the function
raises `RuntimeError`. Raw optimizer diagnostics are retained; BFGS success
alone neither qualifies nor disqualifies a tangent. The returned `distance`
is the Fisher norm of the extracted tangent.

Calling the result a minimizing logarithm assumes that the qualified
horizontal geodesic is minimizing, as in the existing horizontality benchmark.
Neither qualification nor endpoint reconstruction proves global minimality,
uniqueness, or the reduced-distance identity in open problem A1.

## Mean algorithm

For equally weighted observations P_i, the intended objective is
F(P)=(1/(2k)) sum_i d_F(P,P_i)^2. Wherever minimizing logarithms are smooth,
grad F(P)=-(1/k) sum_i Log_P(P_i). The implementation averages the qualified
candidate tangents and applies `exp_map_mvn` in that direction. Its numerical
objective uses the squared extracted tangent lengths.

An Armijo search starts from `step_size` and halves it until
F_new <= F_old - 1e-4 * step * ||average_tangent||_Fisher^2. Each trial
requires fresh qualified fiber solves. Accepted evaluations are reused for
the next iteration. Failure to find an acceptable step is reported as
nonconvergence, never success. Initialization defaults to componentwise
arithmetic averages of the input means and covariance matrices.

Convergence means that the Fisher norm of the average tangent meets `tol`.
This is numerical stationarity under the minimizing-logarithm assumption,
not a guarantee of a global Frechet mean. Inner tolerances must be sufficiently
tight for the requested outer tolerance; if numerical errors prevent descent,
the solver reports line-search failure. It does not silently loosen tolerances.
The implementation targets moderate dimension and conditioning; nested
restarts make it substantially more expensive than an SPD mean.

## Usage

```python
import numpy as np
from src.mvn import exp_map_mvn, log_map_mvn
from src.mean import frechet_mean_mvn

mu0, C0 = np.array([-2.]), np.array([[1.]])
mu1, C1 = np.array([2.]), np.array([[1.]])
log = log_map_mvn(mu0, C0, mu1, C1)
if not log.reliable:
    raise RuntimeError(str(log.diagnostics))
midpoint = exp_map_mvn(mu0, C0, log.mean_tangent,
                       log.covariance_tangent, time=0.5)

result = frechet_mean_mvn(
    np.stack([mu0, mu1]), np.stack([C0, C1]),
    return_iterates=True,
    log_options={"n_starts": 3, "seed": 0},
)
print(result.mean, result.covariance)  # approximately [0.], [[3.]]
print(result.converged, result.message)
```

`LogMapResult` contains original-coordinate tangent components, selected S
coordinates, candidate distance, qualification checks and all optimizer runs.
`FrechetMeanResult` contains the final Gaussian, convergence status, history
(objective, gradient norm, accepted step and maximum inner errors), and
optionally the accepted iterates. These functions do not overwrite experiments.

## Validation

**Numerical checks**, reproducible with `python -m unittest test.test_mvn test.test_mean`:

- Correct initial velocity by centered differences, and an independent
  covariance-only exponential identity.
- Independent integration of the Fisher geodesic ODE for a coupled 2D example.
  The mean equation is mu_ddot=Sigma_dot Sigma^(-1) mu_dot, consistent with
  the first integral in state.md; the opposite sign still printed in
  notation.md section 4 is an existing documentation inconsistency.
- A dense, noncommuting 2D endpoint pair: reconstruction, affine transformation
  of the recovered tangent, and agreement with the reduced numerical reference.
- Seeded moderate 3D and 4D round trips, recovering the original tangents.
- Exact univariate midpoint: N(-a,s²) and N(a,s²) give N(0,s²+a²/2).
- Equal-mean diagonal geometric covariance means and a noncommuting SPD midpoint.
- Dense three-Gaussian objective directional derivatives and mean affine
  equivariance, plus singleton and identical endpoints.
- Invalid inputs, unqualified inner solves, iteration exhaustion and rejected
  line searches.

These checks do not establish accuracy for extreme conditioning or certify
global minima. Existing benchmark data have not been recomputed or changed.
