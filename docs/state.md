# Current research state

Last updated: 2026-09-05.

This document is the persistent research memory of the project. It should be updated after substantial theoretical or numerical progress.

---

## 1. Central question

The central problem is the general Fisher–Rao distance between arbitrary multivariate normal distributions

\[
P_0=\mathcal N(\mu_0,\Sigma_0),
\qquad
P_1=\mathcal N(\mu_1,\Sigma_1).
\]

The initial-value Fisher geodesic equations are known. The unresolved difficulty is the two-point boundary-value problem: determine the correct initial tangent direction, or equivalently the correct horizontal lift, from the endpoints.

The practical objective is not only theoretical. We want a distance accurate enough to replace expensive Fisher geodesic calculations inside classification algorithms.

---

## 2. Established geometric facts used by the project

### 2.1 Affine normalization

The Fisher geometry is invariant under invertible affine transformations of the observation space.

Therefore the two-endpoint problem can be reduced to

\[
P_0=\mathcal N(0,I),
\qquad
P_1=\mathcal N(\mu,\Sigma).
\]

This reduction should be performed before most theoretical or numerical work.

### 2.2 Geodesic ODE and first integrals

The Fisher geodesic equations are

\[
\ddot\Sigma+\dot\mu\dot\mu^\top
-\dot\Sigma\Sigma^{-1}\dot\Sigma=0,
\]

\[
\ddot\mu+\dot\Sigma\Sigma^{-1}\dot\mu=0.
\]

The second equation implies

\[
\Sigma^{-1}\dot\mu=a_0
\]

with constant \(a_0\), and the first then yields

\[
\Sigma^{-1}\dot\Sigma+a_0\mu^\top=A_0
\]

with constant symmetric \(A_0\).

This makes the IVP explicit; it does not by itself solve the BVP.

### 2.3 Kobayashi lift

Kobayashi's 2023 construction realizes Fisher geodesics as projections of horizontal geodesics in

\[
\mathcal M
=
\mathrm{SO}(n+1,n)/
\mathrm{S}(\mathrm O(n+1)\times\mathrm O(n)),
\]

embedded as a totally geodesic submanifold of determinant-one SPD matrices of size \(2n+1\).

The projection to the Gaussian manifold is a Riemannian submersion.

At the normalized identity, horizontal geodesics have generators

\[
V=
\begin{pmatrix}
-A_0&a_0&0\\
a_0^\top&0&-a_0^\top\\
0&-a_0&A_0
\end{pmatrix}.
\]

This is the structural reason the BVP can be reinterpreted as a problem of selecting the correct point in the endpoint fiber.

---

## 3. Boundary-value reformulation developed in the project

The endpoint fiber is parameterized by a skew-symmetric matrix \(S\).

For normalized endpoint \(P_1=(\nu,C)\),

\[
G_1(S)
=
M_{\nu,S}
D_C
M_{\nu,S}^\top,
\]

with

\[
D_C=\operatorname{diag}(C^{-1},1,C),
\]

\[
M_{\nu,S}
=
\begin{pmatrix}
I&0&0\\
\nu^\top&1&0\\
S-\frac12\nu\nu^\top&-\nu&I
\end{pmatrix},
\qquad S^\top=-S.
\]

The correct endpoint lift is the one for which the connecting ambient geodesic is horizontal.

At the normalized initial point, horizontality is encoded by the vanishing of the \((1,3)\) block of the logarithmic generator. This motivates the residual

\[
R(S)
=
\|
(\log G_1(S))_{13}
\|_F^2.
\]

This is conceptually useful because it converts the continuous BVP into a finite-dimensional skew-symmetric problem.

---

## 4. Exact reduced distance formulation

A further reduction gives the exact finite-dimensional optimization currently used in the project.

For

\[
P_0=\mathcal N(0,I),
\qquad
P_1=\mathcal N(\mu,\Sigma),
\]

define

\[
B
=
I+\Sigma+\frac12\mu\mu^\top,
\]

and, for \(K^\top=-K\),

\[
Y(K)=\frac B2+K,
\]

\[
C(K)
=
\left(
Y(K)^\top\Sigma^{-1}Y(K)
\right)^{1/2},
\]

\[
J(K)
=
2\,\operatorname{tr}
\left[
\operatorname{arcosh}^2 C(K)
\right].
\]

Then the project uses

\[
\boxed{
d_F^2
=
\min_{K^\top=-K}J(K).
}
\]

Equivalently,

\[
J(K)
=
2
\sum_i
\operatorname{arcosh}^2
\left(
\sigma_i(\Sigma^{-1/2}Y(K))
\right).
\]

This is the preferred exact reference formulation for the current line of work.

### Status

Treat this formula as an established project result, but when converting it into a theorem for publication, reconstruct the derivation cleanly from the Kobayashi submersion/fiber argument and check every metric scaling constant.

---

## 5. The \(d_0\) approximation

The simplest candidate is obtained by choosing

\[
K=0.
\]

Thus

\[
d_0^2=J(0).
\]

No skew optimization is required. The value is computed spectrally using singular values/eigenvalues and inverse hyperbolic cosine.

This is why \(d_0\) is computationally attractive.

### Main empirical observation

For many moderate and even fairly extreme configurations, \(d_0\) is very close to \(d_F\).

A previous numerical suite contained 47 configurations with dimensions

\[
n\in\{8,10,12,15\},
\]

condition ranges approximately controlled by

\[
\Lambda\in\{2,5,10\},
\]

mean norms

\[
\|\mu\|\in\{1,3,10,20\},
\]

and random / spread / concentrated spectra with several mean orientations.

One recorded example was

\[
n=8,\quad \Lambda=10,\quad \|\mu\|=10,
\]

with approximately

\[
d_0\approx7.56223012,
\qquad
d_F\approx7.46373091,
\]

giving

\[
|d_0-d_F|\approx0.0985
\]

or about \(1.3\%\) relative error.

These numbers came from prior exploratory computations and must be reproduced from code before being cited externally.

---

## 6. \(d_0\) is not uniformly excellent

A key discovery is that the apparently excellent generic behaviour of \(d_0\) does not imply a small universal relative-error bound.

An asymptotic anisotropic family studied in the project is

\[
\Sigma_L
=
\operatorname{diag}(L^{-1},L),
\]

with mean of the form

\[
\mu_L=(u,v\sqrt L).
\]

The asymptotic behaviour found was

\[
d_0\sim\sqrt2\,\log L,
\qquad
d_F\sim\log L,
\]

so that the relative excess tends to

\[
\sqrt2-1
\approx0.41421356.
\]

Thus the previously discussed “0.41 error” is a **relative** asymptotic effect, not an absolute error of \(0.41\).

The same mechanism can be embedded in dimensions \(3\) and \(4\), for example with

\[
\Sigma_L
=
\operatorname{diag}(L^{-1},L,1,\ldots),
\]

\[
\mu_L
=
(1,\sqrt L,0,\ldots).
\]

Extremely large-\(L\) experiments were used only to probe the asymptotic regime. They should not be confused with numerically stable practical benchmarks.

### Consequence

Do not attempt to prove a global statement of the form

\[
\frac{d_0-d_F}{d_F}\le 10^{-2}
\]

without additional hypotheses.

Any useful small-error theorem for \(d_0\) will need structural or compactness assumptions, or a bound depending on anisotropy / mean size.

---

## 7. Restricted approximations \(d_m\)

To improve \(d_0\) without optimizing over all

\[
\frac{n(n-1)}2
\]

skew degrees of freedom, the project restricts \(K\) to a Krylov-generated subspace.

Let

\[
A=\log\Sigma,
\]

\[
\mathcal K_m
=
\operatorname{span}
\{
\mu,A\mu,\ldots,A^{m-1}\mu
\}.
\]

Then

\[
d_m^2
=
\min_{K\in\Lambda^2\mathcal K_m}
J(K).
\]

The motivation is that the interaction between the mean direction and covariance anisotropy is generated by the orbit of \(\mu\) under \(A=\log\Sigma\). The full \(n(n-1)/2\)-dimensional skew problem may therefore have a much smaller effective dimension.

### Numerical evidence

Previous exploratory tests found extremely fast improvement in many cases.

In the 47-case suite mentioned above, a sufficiently large restricted approximation (recorded as `d5` in the earlier experiment) matched the reference distance to about numerical precision:

\[
|d_5-d_F|<1.6\times10^{-8}
\]

in that suite.

Again, this needs to be reproduced from version-controlled scripts before external use.

Some extreme examples previously recorded:

- embedded 2D mechanism in 4D: \(d_0=d_1\) had about \(30.7\%\) relative excess while \(d_2\) was essentially exact;
- a 3D example with mean approximately \((1,1,1000)\): \(d_0\) about \(27.4\%\) high, \(d_2\) about \(0.087\%\) high, \(d_3\) essentially exact;
- a 3D example with mean approximately \((1,100,1000)\): \(d_0\) about \(9.49\%\) high, \(d_2\) still about \(8.00\%\) high, while \(d_3\) was essentially exact.

These examples suggest that the Krylov dimension, not simply \(n\), controls the difficulty.

They are numerical observations, not theorems.

---

## 8. Structural interpretation currently favoured

The project increasingly views the problem as having two layers.

### Spectral core

For fixed skew correction \(K\), the distance is reduced to the singular values of

\[
\Sigma^{-1/2}Y(K),
\]

followed by

\[
\operatorname{arcosh}.
\]

Thus the expensive Riemannian path problem becomes a spectral matrix problem.

### Low-dimensional geometric shell

The nontrivial choice of \(K\) appears to be driven mainly by how the mean direction interacts with eigenspaces of

\[
A=\log\Sigma.
\]

This motivates Krylov spaces

\[
\operatorname{span}\{
\mu,A\mu,A^2\mu,\ldots
\}.
\]

A major theoretical goal is to convert this empirical/effective low-dimensionality into a rigorous reduction or error theorem.

---

## 9. Special cases that should always be used as tests

Every implementation or proposed formula should be checked against cases where the geometry simplifies.

### Same mean

When

\[
\mu_0=\mu_1,
\]

the problem reduces to covariance geometry and has the standard SPD-type closed form with the appropriate Fisher scaling.

### One-dimensional Gaussian family

For \(n=1\), the Gaussian manifold is a scaled hyperbolic upper half-plane and the Fisher–Rao distance has an exact closed form.

### Proportional / commuting covariances

These cases often reduce strongly and should be used to test matrix formulas.

### Mean aligned with an invariant covariance direction

When \(\mu\) lies in a small invariant subspace of \(A=\log\Sigma\), the Krylov space may have very low dimension. These are especially important for testing the \(d_m\) theory.

---

## 10. Alternative distances / approximations being tracked

### 10.1 Calvo–Oller SPD embedding

Calvo and Oller embed the Gaussian manifold isometrically as a submanifold of `SPD(n+1)` with a scaled trace metric.

Because the embedded Gaussian submanifold is not totally geodesic, the ambient SPD geodesic distance gives a tractable lower bound rather than the general Fisher–Rao distance itself.

This is a useful cheap baseline.

### 10.2 Nielsen curve-discretization approximation

A tractable curve between two Gaussians may be discretized and the local Fisher length approximated using the square root of Jeffreys divergence.

This is useful as a numerical baseline but is conceptually different from the Kobayashi fiber optimization.

### 10.3 Geodesic triangle

Triangle constructions change covariance and mean in separate legs and use tractable distances / upper bounds.

The project additionally proposes a more symmetric heuristic:

\[
\Sigma_{\mathrm{mid}}
=
\Sigma_1\#\Sigma_2
\]

with

\[
\Sigma_1\#\Sigma_2
=
\Sigma_1^{1/2}
(
\Sigma_1^{-1/2}
\Sigma_2
\Sigma_1^{-1/2}
)^{1/2}
\Sigma_1^{1/2},
\]

and uses this geometric mean instead of only \(\Sigma_1\) or \(\Sigma_2\) in the term weighting the difference of the means.

This modification is currently a heuristic candidate, not a proven Fisher formula.

### 10.4 Pullback SPD-cone distances

Nielsen and related work propose computationally light distances obtained by embedding the Gaussian manifold into SPD cones and pulling back tractable cone distances.

These should be kept as application-oriented baselines.

---

## 11. Application pipeline

The application target is acoustic signal classification.

Current intended pipeline:

1. start from a temporal signal;
2. apply a sliding short-time Fourier transform (STFT);
3. extract local spectral feature vectors;
4. summarize a window / segment by an MVN \((\mu,\Sigma)\);
5. compute pairwise geometric distances between MVNs;
6. classify using a method such as k-nearest neighbours.

The geometry work and application work should remain separated in code: distance implementations belong in `src/`; dataset/STFT/classification experiments belong in `experiments/` or a future application package.

---

## 12. What is proved vs. what is not

### Established / literature-backed

- standard Fisher metric for MVNs;
- affine invariance;
- Fisher geodesic ODEs and IVP first integrals;
- explicit IVP solutions in the literature;
- Kobayashi's Riemannian submersion interpretation;
- Calvo–Oller embedding and associated lower-bound idea;
- absence of a simple known general two-endpoint Fisher–Rao closed form in the standard literature.

### Established within the current project, but should be rewritten cleanly for publication

- skew-fiber formulation of the normalized BVP;
- reduced exact optimization
  \[
  d_F^2=\min_{K^\top=-K}J(K);
  \]
- definition of \(d_0\);
- restricted Krylov hierarchy \(d_m\).

### Numerical / experimental

- strong generic accuracy of \(d_0\);
- rapid convergence of low-dimensional \(d_m\) in many tests;
- asymptotic / extreme examples showing large relative \(d_0\) error;
- effective low-dimensionality suggested by Krylov structure.

### Heuristic / open

- a useful general small-error bound for \(d_0\);
- a rigorous convergence/error theory for \(d_m\);
- whether the Krylov restriction captures the exact optimizer at a predictable finite order;
- geometric-mean modification of the triangle approximation;
- possibility of a more economical SPD embedding/submersion than the \(2n+1\) Kobayashi construction.

---

## 13. Immediate next research tasks

Highest-value tasks:

1. implement one canonical, tested version of \(J(K)\), \(d_0\), and full skew optimization;
2. reconstruct the previous numerical suites in version-controlled scripts;
3. implement \(d_m\) with explicit Krylov bases and stable optimization coordinates;
4. verify all scaling constants against exact special cases;
5. characterize the low-dimensional subspace containing or approximating the optimal \(K\);
6. turn the asymptotic bad family for \(d_0\) into a clean proposition with proof;
7. benchmark \(d_0,d_m\), Calvo–Oller, triangle methods, and a high-accuracy reference solver;
8. only then integrate the preferred distances into the STFT + KNN application pipeline.

---

## 14. Reproducibility warning

A significant amount of numerical knowledge currently comes from exploratory conversations rather than committed scripts.

Therefore:

> no numerical value in this file should be treated as publication-ready until the corresponding experiment is reconstructed under `experiments/`, with fixed seeds/parameters and a clearly defined reference solver.

Updating this state from conversational memory into reproducible code is one of the first goals of the repository.
