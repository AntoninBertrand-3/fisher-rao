# Directed triangle divergences in the MDM benchmark

Source: Collas, Bouchard, Ginolhac, Breloy, Ren and Ovarlez,
*On the Use of Geodesic Triangles between Gaussian Distributions for
Classification Problems*, ICASSP 2022,
[DOI 10.1109/ICASSP43922.2022.9747872](https://doi.org/10.1109/ICASSP43922.2022.9747872).
[Author PDF](https://www.antoinecollas.fr/assets/pdf/papers/2022_ICASSP.pdf).

## Published definitions

Let P=(mu,C), Q=(nu,S), h=nu-mu, q=h^T C^{-1}h,
ell=(logdet(S)-logdet(C))/n and c=exp(ell). Write d_SPD for the
trace-normalized SPD distance in notation section 11. Corollaries 1 and 2 give

    delta_c^2(P,Q) = 2 acosh((c+1+q/2)/(2 sqrt(c)))^2
                    + (n-1) ell^2/2 + d_SPD(c C,S)^2/2,
    delta_perp^2(P,Q) = acosh(1+q)^2/2
                       + d_SPD(C+h h^T/2,S)^2/2.

Equation (7) uses the repository's Fisher metric, including the covariance
factor 1/2. No normalization conversion is needed. Equation (18) minimizes
mean(delta^2(center, observation))/2. We use center-first order at prediction
too. These are generally asymmetric divergences, not exact Fisher distances.
The sum of squared legs is not the square of their summed lengths; no Fisher
upper-bound claim is inferred. This is distinct from the geometric-mean
covariance heuristic previously discussed in the repository.

## Project derivation of analytic gradients

The following derivatives are derived and checked in this project. Matrix
gradients use the Frobenius pairing on symmetric matrices. For fixed S, let

    H(T,S) = S^{-1/2} [log(Z) Z^{-1}] S^{-1/2},
    Z = S^{-1/2} T S^{-1/2}.

This is the gradient of d_SPD(T,S)^2/2 in T: differentiate the spectral
trace sum(log(lambda)^2)/2. The scalar derivative is log(lambda)/lambda,
so the implementation needs only the symmetric eigendecomposition of Z.

For the c divergence, the covariance terms simplify to
d_SPD(C,S)^2/2 - ell^2/2. Indeed, log generalized eigenvalues shift by ell,
and their sum is n*ell. Set x=cosh(ell/2)+q exp(-ell/2)/4 and
r=acosh(x)/sqrt(x^2-1), with continuous value one at x=1. Then

    a = r exp(-ell/2),
    b = 4 r [sinh(ell/2)/2 - q exp(-ell/2)/8] - ell,
    grad_mu = -2 a C^{-1}h,
    grad_C = H(C,S) - a (C^{-1}h)(C^{-1}h)^T - (b/n) C^{-1}.

For the perpendicular divergence put T=C+h h^T/2 and
a=acosh(1+q)/sqrt(q(q+2)), again with continuous value one at zero:

    grad_mu = -2 a C^{-1}h - H(T,S) h,
    grad_C = H(T,S) - a (C^{-1}h)(C^{-1}h)^T.

The extra mean term follows from dT=(dh h^T+h dh^T)/2 and dh=-dmu.
All gradients are averaged and divided by two for the center objective.

## Numerical implementation and qualification

Targets' whitening matrices and log determinants are cached. Distances and
gradients over observations use batched NumPy eigendecompositions. The scalar
acosh is evaluated as 2 asinh(sqrt((x-1)/2)); the c offset is formed as
2 sinh(ell/4)^2 + q exp(-ell/2)/4 to avoid cancellation near identical laws.

For fitting, training data are affinely normalized using their arithmetic
mean and covariance. SciPy BFGS optimizes mean and lower Cholesky entries,
with logarithmic diagonal entries to preserve SPD. If C=L L^T, the chain
rule gives grad_L=2 grad_C L; diagonal coordinates get an extra L_ii factor.
This solves the published objective with a different optimizer from the paper.
Starts use the arithmetic covariance and the first training covariance,
both at the arithmetic mean. No test data select starts or tolerances.

Optimizer gtol is 1e-8, maximum iterations 500. Numerical qualification
requires both coordinate infinity-gradient norm and intrinsic Fisher gradient
norm at most 1e-7. For objective gradients (g_mu,G_C), the latter squared is
g_mu^T C g_mu + 2 ||L^T G_C L||_F^2. Raw optimizer success flags and all
restart diagnostics are retained. Selection uses the lowest objective found;
a nonqualified last iterate remains flagged. Two starts do not prove global
optimality or uniqueness.

## Reproduction

For the fixed dimension probe used to test whether larger n alone increases
discrepancies, run `python experiments/compare_triangle_dimension.py`.
It saves dimensions 3,5,10 and four controlled covariance/mean families in
`results/triangle_dimension_probe.{json,csv,md}`. The family includes condition
numbers 100, 10,000 and 1,000,000; it is a fixed probe, not an adaptive search.

For a parent-based MDM stress test with a large endpoint dF/delta_c gap, run
`python experiments/compare_triangle_parent_stress.py`. It samples both
classes from fixed A=N(0,I3) and B=N((1,10,0),diag(.01,100,1)), with exact-law
and batch-size controls. No test descriptor is selected from a disagreement.

For the saved partial 10D benchmark, run
`python experiments/compare_mdm_components.py --triangles --input results/mdm_sampled_10d.json --output results/mdm_triangles_10d`.
This evaluates only its 37 saved trials, retaining the source's partial status
and explicit per-setting seed counts. It does not resume the Fisher run.

Run `python experiments/compare_mdm_components.py --triangles`.
The source remains `results/mdm_sampled_3d.json`; outputs use
`results/mdm_triangles_3d.{json,csv,md}`. Fisher/d0 predictions are reused
exactly. Covariance-only and mean-only fits are recomputed on the same splits.
Sample covariances retain the existing unbiased denominator m-1, rather than
switching to the paper's maximum-likelihood denominator m. This compares the
methods on our fixed synthetic benchmark, not a reproduction of Breizhcrops.

Tests independently evaluate the literal formulas and finite-difference
gradients in dimensions 1,3,5,10, check affine invariance, center equivariance,
identical/equal-mean inputs, anisotropy, asymmetry, and exact c-divergence
special cases (univariate or equal covariance).
