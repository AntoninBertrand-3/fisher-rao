# Canonical notation

Last updated: 2026-09-05.

This file defines the notation that should be used throughout the project. Agents should not silently introduce incompatible conventions.

---

## 1. Gaussian manifold

The manifold of non-degenerate \(n\)-variate Gaussian distributions is

\[
\mathcal N_n
=
\left\{
(\mu,\Sigma):
\mu\in\mathbb R^n,\;
\Sigma\in\mathrm{SPD}(n)
\right\}.
\]

We write

\[
P=\mathcal N(\mu,\Sigma)
\]

interchangeably with \(P=(\mu,\Sigma)\).

`SPD(n)` denotes the cone of real symmetric positive-definite \(n\times n\) matrices.

A tangent vector at \(P=(\mu,\Sigma)\) is written

\[
(\dot\mu,\dot\Sigma)
\in
\mathbb R^n\times\mathrm{Sym}(n).
\]

---

## 2. Fisher metric: canonical convention

The standard Fisher information metric for MVNs is

\[
g_{(\mu,\Sigma)}
\big(
(\delta\mu,\delta\Sigma),
(\delta\mu',\delta\Sigma')
\big)
=
\delta\mu^\top\Sigma^{-1}\delta\mu'
+
\frac12
\operatorname{tr}
\left(
\Sigma^{-1}\delta\Sigma
\Sigma^{-1}\delta\Sigma'
\right).
\]

Hence

\[
ds^2
=
d\mu^\top\Sigma^{-1}d\mu
+
\frac12
\operatorname{tr}
\left[
(\Sigma^{-1}d\Sigma)^2
\right].
\]

The corresponding geodesic distance is denoted

\[
d_F(P_0,P_1)
\quad\text{or}\quad
d_{\mathrm{FR}}(P_0,P_1).
\]

### Convention warning

Some earlier project notes wrote the covariance term without the factor \(1/2\):

\[
d\mu^\top\Sigma^{-1}d\mu
+
\operatorname{tr}
\left[
(\Sigma^{-1}d\Sigma)^2
\right].
\]

That is **not** the canonical Fisher metric used in this repository unless explicitly stated. Any old derivation or numerical script inherited from previous notes must be checked for this factor before results are compared.

---

## 3. Affine invariance and normalization

For an invertible affine transformation

\[
x\mapsto Ax+b,
\]

an MVN transforms as

\[
(\mu,\Sigma)
\mapsto
(A\mu+b,\;A\Sigma A^\top).
\]

The Fisher metric and Fisher–Rao distance are invariant under this action.

Therefore a pair

\[
P_0=(\mu_0,\Sigma_0),
\qquad
P_1=(\mu_1,\Sigma_1)
\]

may be normalized to

\[
\widehat P_0=(0,I),
\qquad
\widehat P_1=(\mu,\Sigma),
\]

with, for example,

\[
\mu
=
\Sigma_0^{-1/2}(\mu_1-\mu_0),
\qquad
\Sigma
=
\Sigma_0^{-1/2}
\Sigma_1
\Sigma_0^{-1/2}.
\]

A symmetric positive-definite square root should be used unless a different factorization is explicitly required.

Most theoretical derivations below assume this normalized setting:

\[
P_0=\mathcal N(0,I),
\qquad
P_1=\mathcal N(\mu,\Sigma).
\]

---

## 4. Fisher geodesic equations

For a Fisher geodesic

\[
t\mapsto(\mu(t),\Sigma(t)),
\]

the second-order equations are

\[
\ddot\Sigma
+
\dot\mu\dot\mu^\top
-
\dot\Sigma\Sigma^{-1}\dot\Sigma
=
0,
\]

\[
\ddot\mu
+
\dot\Sigma\Sigma^{-1}\dot\mu
=
0.
\]

The second equation gives immediately

\[
\frac{d}{dt}
\left(
\Sigma^{-1}\dot\mu
\right)=0,
\]

hence

\[
\Sigma^{-1}\dot\mu=a_0
\]

for a constant \(a_0\in\mathbb R^n\).

Using this in the first equation gives the second first integral

\[
\Sigma^{-1}\dot\Sigma+a_0\mu^\top=A_0,
\]

where \(A_0\in\mathrm{Sym}(n)\) is constant.

The initial-value problem is therefore explicit once \((A_0,a_0)\) is known.

The hard problem for the distance is the **boundary-value problem**: determine the correct initial direction \((A_0,a_0)\) from the two endpoints.

---

## 5. Precision / natural variables

Following the Kobayashi / Eriksen formulation, define

\[
\Theta=\Sigma^{-1},
\qquad
\delta=\Sigma^{-1}\mu.
\]

Then the Gaussian is represented by the \((n+1)\times(n+1)\) SPD matrix

\[
H(\Theta,\delta)
=
\begin{pmatrix}
\Theta & \delta\\
\delta^\top &
1+\delta^\top\Theta^{-1}\delta
\end{pmatrix}.
\]

This representation is closely related to the inverse of the classical Calvo–Oller embedding.

---

## 6. Kobayashi ambient symmetric space

Define

\[
\mathcal L
=
\mathrm{SPD}_*(2n+1)
=
\left\{
G\in\mathrm{SPD}(2n+1):
\det G=1
\right\}.
\]

Let

\[
J=
\begin{pmatrix}
0&0&I_n\\
0&1&0\\
I_n&0&0
\end{pmatrix}.
\]

The relevant symmetric subspace is

\[
\mathcal M
=
\left\{
G\in\mathcal L:
JG^{-1}J=G
\right\}.
\]

Kobayashi identifies \(\mathcal M\) with the symmetric space

\[
\mathrm{SO}(n+1,n)/
\mathrm{S}(\mathrm O(n+1)\times\mathrm O(n)).
\]

It is a totally geodesic submanifold of \(\mathcal L\).

The projection

\[
\pi:\mathcal M\to\mathcal N_n
\]

is obtained from the first \((n+1)\times(n+1)\) principal block and is a Riemannian submersion.

A Fisher geodesic in \(\mathcal N_n\) is the projection of a horizontal geodesic in \(\mathcal M\).

At the identity, a horizontal generator has the form

\[
V(A_0,a_0)
=
\begin{pmatrix}
-A_0 & a_0 & 0\\
a_0^\top & 0 & -a_0^\top\\
0 & -a_0 & A_0
\end{pmatrix},
\]

and the lifted geodesic is

\[
G(t)=\exp(tV).
\]

The vanishing \((1,3)\) block in \(V\) is the horizontal condition at the normalized base point.

---

## 7. Fiber parameterization used in this project

After normalization to \(P_0=(0,I)\), a lift of the endpoint \(P_1=(\nu,C)\) is parameterized by a skew-symmetric matrix

\[
S^\top=-S.
\]

The project uses

\[
D_C
=
\operatorname{diag}(C^{-1},1,C),
\]

and

\[
M_{\nu,S}
=
\begin{pmatrix}
I&0&0\\
\nu^\top&1&0\\
S-\frac12\nu\nu^\top&-\nu&I
\end{pmatrix}.
\]

The lifted endpoint is

\[
G_1(S)
=
M_{\nu,S}
D_C
M_{\nu,S}^\top.
\]

The choice of \(S\) moves inside the fiber over the same Gaussian endpoint.

The exact boundary-value problem may therefore be interpreted as:

> choose the endpoint lift such that the ambient geodesic from the normalized initial lift is horizontal.

At the normalized initial lift \(G_0=I\), a practical horizontal-residual objective used in the project is

\[
R(S)
=
\left\|
\left(
\log G_1(S)
\right)_{13}
\right\|_F^2.
\]

More generally, before normalization the residual is written using

\[
\log(G_0^{-1}G_1(S))
\]

with the appropriate invariant/symmetric representation.

Do **not** assume that minimizing this residual and minimizing distance are automatically identical unless the precise equivalence being used has been established in the derivation at hand.

---

## 8. Exact reduced skew-symmetric optimization

A second, algebraically reduced formulation is the current canonical expression for the exact distance in the project.

Assume

\[
P_0=\mathcal N(0,I),
\qquad
P_1=\mathcal N(\mu,\Sigma).
\]

Define

\[
A=\log\Sigma,
\]

and, separately,

\[
B
=
I+\Sigma+\frac12\mu\mu^\top.
\]

For a skew-symmetric matrix

\[
K^\top=-K,
\]

define

\[
Y(K)
=
\frac{B}{2}+K,
\]

\[
C(K)
=
\left(
Y(K)^\top
\Sigma^{-1}
Y(K)
\right)^{1/2}.
\]

The project uses the objective

\[
J(K)
=
2\,\operatorname{tr}
\left[
\operatorname{arcosh}^2 C(K)
\right].
\]

Equivalently, if

\[
c_i(K)
=
\sigma_i
\left(
\Sigma^{-1/2}Y(K)
\right)
\]

are the singular values, then

\[
J(K)
=
2
\sum_{i=1}^n
\operatorname{arcosh}^2(c_i(K)).
\]

The exact reduced boundary-value formulation is

\[
\boxed{
d_F^2
=
\min_{K^\top=-K}
J(K).
}
\]

This is the formulation that should be used as the reference when discussing \(d_0,d_m\).

### Important notation warning

Older discussions also used another matrix called \(Y_u(K)\), leading to a formula written directly in terms of the eigenvalues of a symmetric matrix. That representation may be algebraically equivalent after transformations, but it should **not** be mixed with the present definition of \(Y(K)=B/2+K\) without an explicit derivation of equivalence.

---

## 9. Zeroth approximation \(d_0\)

The zeroth approximation is obtained by taking

\[
K=0.
\]

Thus

\[
\boxed{
d_0^2=J(0).
}
\]

Explicitly,

\[
Y(0)
=
\frac12
\left(
I+\Sigma+\frac12\mu\mu^\top
\right),
\]

\[
C(0)
=
\left(
Y(0)\Sigma^{-1}Y(0)
\right)^{1/2},
\]

because \(Y(0)\) is symmetric, and

\[
d_0^2
=
2\,\operatorname{tr}
\left[
\operatorname{arcosh}^2 C(0)
\right].
\]

Equivalently,

\[
d_0^2
=
2
\sum_i
\operatorname{arcosh}^2
\left(
\sigma_i\left(\Sigma^{-1/2}Y(0)\right)
\right).
\]

This spectral formula is why \(d_0\) is much cheaper to compute than solving the full boundary-value problem: no optimization over skew matrices or geodesic shooting is required.

---

## 10. Restricted approximations \(d_m\)

Let

\[
A=\log\Sigma.
\]

Define the Krylov subspace

\[
\mathcal K_m
=
\operatorname{span}
\left\{
\mu,
A\mu,
A^2\mu,
\ldots,
A^{m-1}\mu
\right\}.
\]

The skew search space is restricted to

\[
\Lambda^2\mathcal K_m,
\]

i.e. skew-symmetric operators supported on this low-dimensional subspace.

The project defines

\[
\boxed{
d_m^2
=
\min_{K\in\Lambda^2\mathcal K_m}
J(K).
}
\]

This gives a hierarchy between the zero-skew approximation and the full skew optimization.

### Indexing warning

With the literal definition above,

\[
\dim\mathcal K_1\le1
\]

and hence

\[
\Lambda^2\mathcal K_1=\{0\},
\]

so \(d_1=d_0\).

If implementation code adopts shifted indexing, document the shift explicitly rather than changing the mathematical definition silently.

---

## 11. SPD affine-invariant geometry

For \(P,Q\in\mathrm{SPD}(n)\), the affine-invariant Riemannian distance associated with

\[
g_P(U,V)
=
\operatorname{tr}
(P^{-1}UP^{-1}V)
\]

is

\[
d_{\mathrm{SPD}}(P,Q)
=
\left\|
\log
\left(
P^{-1/2}QP^{-1/2}
\right)
\right\|_F.
\]

If the metric is scaled by \(1/2\), the distance is scaled by \(1/\sqrt2\).

The affine-invariant geometric mean is

\[
P\#Q
=
P^{1/2}
\left(
P^{-1/2}QP^{-1/2}
\right)^{1/2}
P^{1/2}.
\]

This object appears in one project-specific modification of the geodesic-triangle approximation.

---

## 12. Project-specific triangle heuristic

A known elementary strategy is to connect two MVNs through an intermediate point while changing mean and covariance in separate legs, then use the triangle inequality / tractable submodel distances.

The project is investigating a symmetric modification:

> wherever the chosen triangle formula weights the mean displacement using only one endpoint covariance, replace that covariance by the affine-invariant geometric mean
>
> \[
> \Sigma_1\#\Sigma_2.
> \]

This is currently a **heuristic / candidate approximation**, not an established identity for the Fisher–Rao distance.

Any implementation should keep the original triangle formula and the geometric-mean modification as distinct named methods.

---

## 13. Numerical-error notation

For an approximation \(d_{\mathrm{approx}}\), use

\[
e_{\mathrm{abs}}
=
|d_{\mathrm{approx}}-d_F|,
\]

\[
e_{\mathrm{rel}}
=
\frac{|d_{\mathrm{approx}}-d_F|}{d_F}.
\]

When the approximation is known to be an upper bound in a specific formulation, the signed relative excess may be recorded as

\[
e_+
=
\frac{d_{\mathrm{approx}}}{d_F}-1.
\]

Never write “error \(=0.41\)” without stating whether it is absolute or relative.