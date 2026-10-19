# Horizontality benchmark: formula and analytic gradient

This historical experiment (scripts and results preserved in
`archive/benchmarks_before_consolidation.zip`) uses the saved endpoints in
`results/mixed_presentation_cases.json`, after the existing symmetric affine
normalization. It optimizes the endpoint-fiber parameter S of notation.md §7,
not the reduced parameter K. No equivalence between the two objectives is assumed.
The interpretation of a horizontal lift's length as the minimizing Fisher
distance is the user's explicit working assumption for this experiment.

## Objective and scaling

Write W=Sigma^(-1/2), Q=Sigma^(1/2), and

    M(S) = [[I, 0, 0], [mu.T, 1, 0], [S-mu mu.T/2, -mu, I]],
    D = diag(Sigma^(-1), 1, Sigma),
    L = M(S) diag(W, 1, Q),
    G = L L.T, H = log G, R(S) = ||H_13||_F^2.

S is skew and has n(n-1)/2 upper-triangle coordinates. The normalized initial
lift is I. At horizontality,

    H = [[-A, a, 0], [a.T, 0, -a.T], [0, -a, A]].

**Derived and checked:** ||H||_F^2=2||A||_F^2+4||a||^2. The canonical initial
Fisher speed squared is ||a||^2+||A||_F^2/2, so the distance candidate is
||H||_F/2. This conversion is checked against exact univariate, equal-mean and
mean-only distances. The block construction and horizontal generator follow
[Kobayashi (2023), equations (1.2), (2.4), (2.5) and section 2.2](https://arxiv.org/pdf/2304.12575).
The minimizing property is an assumption here, not an additional claim about
what that source proves.

## Frechet derivative and adjoint

For G=U diag(lambda) U.T, the spectral Frechet derivative is

    Dlog_G[E] = U (F * (U.T E U)) U.T,
    F_ij = (log(lambda_i)-log(lambda_j))/(lambda_i-lambda_j),
    F_ij = 1/lambda_i when lambda_i=lambda_j.

The repeated-eigenvalue expression is the continuous limit, including the
diagonal. `log1p` evaluates the divided differences without subtracting nearby
logarithms. The formula follows by differentiating the scalar logarithm's
resolvent integral: Dlog_G[E]=integral_0^infinity (G+tI)^(-1) E (G+tI)^(-1) dt.
In the eigenbasis the integral is F_ij. For a general treatment see
[Al-Mohy, Higham and Relton (2013)](https://eprints.maths.manchester.ac.uk/2015/).

Let E have blocks E_13=H_13, E_31=H_13.T, with all other blocks zero.
Then dR=<E,dH>_F. Since F is real symmetric, Dlog_G is self-adjoint, so

    T = Dlog_G[E],
    dR = <T,dG> = <2 T M D,dM>,
    grad_S R = (2 T M D)_31,
    dR/ds_ij = (grad_S R)_ij - (grad_S R)_ji, i<j.

**Equivalent implementation:** obtain U and lambda=s^2 from the standard SVD
L=U diag(s) V.T, so H=U diag(2 log s) U.T. This avoids explicitly squaring the
factor's condition number in an eigendecomposition of L L.T. The factor adjoint is

    grad_L R = 2 U (F * (U.T E U)) diag(s) V.T,
    grad_S R = (grad_L R)_31 W.

One SVD and matrix products produce the entire analytic coordinate gradient;
no finite differences are used by BFGS. Finite differences and independent
matrix-log evaluations are used only for validation.

## Identical inputs and initialization protocol

The saved endpoint arrays are loaded directly without regenerating or rounding.
Primary BFGS uses five starts, gtol=1e-6 and maxiter=2000, as before: zero followed
by four `default_rng(saved_seed).standard_normal(n*(n-1)//2)` vectors. Strict
checks use two starts, gtol=1e-8, seed+1000; reversed endpoints use three starts,
gtol=1e-6, seed+2000. All initial vectors and final coordinates are saved.
Identical numerical coordinates in S and K do not assert identical lifted
points; no S-to-K map has been assumed. Restart selection uses the smallest
horizontal residual among spectrally resolved candidates, never the smallest
distance. BFGS convergence and small horizontal residual are recorded separately.

## Spectral failure and supplemental solves

**Numerical observation:** some random starts escape to extremely large S.
At the original gtol=1e-6, a D07 restart even reports convergence with residual
6.92e-8 while ||H+JHJ|| is about 322 and trace(H) is about 1008. Its computed
length 132.040111 is invalid: the logarithm no longer resolves the exact
symmetric-space constraints. This is not a counterexample to the user's
minimizing-geodesic assumption.

The report retains every raw outcome, including this invalid apparent minimum.
It screens candidate spectra using eps*cond(L)<=1e-6, ||H+JHJ||<=1e-7 and
|trace(H)|<=1e-7. These are diagnostic thresholds, not error certificates.
The well-resolved final solutions additionally pass an independent LAPACK SVD
driver check and are compared with initial Fisher speed. Moderate-condition
examples also pass a direct scipy.linalg.logm check on G.

The original stopping tolerance does not guarantee a small residual even when
the spectrum is resolved. Nine cases therefore receive additional five-start
forward and three-start reverse BFGS solves at gtol=1e-11, maxiter=5000, each
starting again from the original respective seed and vectors. No distance
objective, preconditioning, or warm start is substituted. The original run
diagnostics, tight runs and their costs are saved separately.

The numerical unit tests cover centered coordinate gradients, repeated and
near-repeated eigenvalues via an independent block-log Frechet identity,
exact distances and metric scaling, endpoint projection, affine invariance,
reversal, and rejection of spectral qualification for an enormous fiber point.
`experiments/validate_horizontal_benchmark.py` validates all fourteen stored
solutions and confirms exact equality of the source endpoints and seeded starts.
