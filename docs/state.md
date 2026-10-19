# Current research state

## Moderate parent-gap MDM control (2026-10-19)

**Numerical observation.** After the extreme-parent stress test exposed Fisher
qualification failures, `compare_triangle_parent_gap.py` uses a fixed parent
pair with covariance condition number 50.36 and endpoint values
dF=2.586830, forward delta_c=3.454785 (+33.6%). The parent mean is
(3.106933,.597730,-.543625), and its dense covariance is saved in JSON.
Five seeds, 12 training and 40 test descriptors per class are generated from
these parents; no test law is selected by a disagreement. Fisher coverage is
100% for every setting. Mean accuracies Fisher/d0/delta_c/delta_perp are:

    m=5:  98.25/98.25/98.50/98.25%;
    m=10: 100/100/100/100%;
    m=20: 100/100/100/100%;
    exact: 100/100/100/100%.

Thus a large endpoint divergence gap is numerically clean but does not produce
a useful classification gap here: the parents are too well separated. The
next unbiased design should predeclare a catalogue of moderate-condition
parent pairs and evaluate all of them, rather than tuning one pair to a test
disagreement. Results are in `results/mdm_parent_gap_3d`.

## Parent-stress MDM: large pairwise dF/delta_c gap (2026-10-19)

**Numerical observation.** `experiments/compare_triangle_parent_stress.py`
uses one fixed, non-adversarial parent pair in 3D:

    A=N(0,I), B=N((1,10,0),diag(.01,100,1)).

At the exact parent endpoints, dF=5.413124 while forward delta_c=7.251064.
The MDM then samples independent finite batches from those parents; no test
law is selected using a classifier disagreement. With five seeds, 12 training
and 40 test descriptors per class, Fisher/d0/triangle-c/triangle-perp mean
accuracies for batch sizes 5,10,20 are respectively:

    m=5:  Fisher reported 38% coverage-adjusted accuracy, d0/c/perp 100/100/100%;
    m=10: Fisher 55%, d0/c/perp 100/100/100%;
    m=20: Fisher 67%, d0/c/perp 100/100/100%;
    exact laws: 100/100/100/100 percent.

Here d0 uses the fitted Fisher prototypes but is a prediction approximation.
The displayed Fisher percentages count unqualified predictions as errors:
mean prediction coverage is 38%,55%,67% for m=5,10,20, while accuracy among
qualified Fisher predictions is 100% in every seed and batch. The extreme
parent pair therefore exposes a numerical qualification problem, not a clean
classification failure of dF. It is still a useful warning: a large endpoint
discrepancy does not automatically make dF practically better when finite
sample covariances make the reference solver unreliable.
It also shows why future comparisons should report both exact-law and finite-
sample controls, covariance conditioning, prototype qualification, and all
method-specific diagnostics. These results do not imply triangle divergences
are generally superior. Full saved data are in `results/mdm_parent_stress_3d`.

## Targeted class construction for a Fisher-versus-triangle gap (2026-10-19)

**Numerical observation.** A fixed search over 45 candidate/prototype
configurations found two nearest-prototype disagreements between numerical
dF and directed delta_c. The clearest pair uses class prototype A

    mu_A=(1,1,1), Sigma_A=diag(.25,1,4)

and class prototype B

    mu_B=(1,3,1), Sigma_B=diag(.1,1,10).

For the fixed candidate

    mu_X=(.137319,5.102396,-3.891670),

    Sigma_X=[[.322599,-1.563191,1.556193],
             [-1.563191,15.291704,-14.349658],
             [1.556193,-14.349658,13.810271]],

dF(A,X)=3.607409 < dF(B,X)=4.070281, so Fisher assigns X to A. The
published directed triangle values are delta_c(A,X)=4.841675 and
delta_c(B,X)=4.563450, so delta_c assigns X to B. The candidate covariance
has eigenvalues (.1142,.2216,29.0888) and condition number about 255.

This gives a concrete way to design a targeted MDM stress test: place class A
around the A prototype while including a controlled test cloud around X,
labelled A, and class B around B. Fisher should classify that cloud correctly
where delta_c is deliberately pushed toward B. The construction is adversarial
by design: it demonstrates possibility, not expected performance on random or
real data. A neutral benchmark and this targeted stress benchmark must remain
separate. All 25 candidates in the identity/stress pair search had no
disagreement; the mismatch appeared after adding mixed anisotropic prototype
pairs. Full fixed-search output is `results/triangle_classification_gap`.

## Dimension probe for pairwise triangle divergences (2026-10-19)

**Numerical observation.** `experiments/compare_triangle_dimension.py` evaluates
four fixed families in dimensions 3, 5 and 10: isotropic mean change,
balanced anisotropy (condition number 100), mixed mean/anisotropy (condition
number 10,000), and a stronger mixed family (condition number 10^6). All 12
reduced references qualify. The probe is fixed in advance; it is not a search
for a worst case.

Dimension increases absolute distances: for the strong mixed family,
dF is 7.7290, 8.7608 and 11.0530 in dimensions 3,5,10. It does not force a
larger relative discrepancy. In this family, forward d0 excess is about 1.2%,
1.0%,1.0%, while reverse delta_c excess is 52.1%,40.7%,27.8% in dimensions
3,5,10. Thus extra dimensions offer more directions in which to search for
large discrepancies, but anisotropy, mean orientation and divergence direction
matter more than n alone. The complete table, endpoint constructions and
optimizer diagnostics are in `results/triangle_dimension_probe.{json,csv,md}`.

## Pairwise 3D triangle values versus Fisher (2026-10-19)

**Numerical observation.** `experiments/compare_triangle_pairs.py` compares
eight fixed illustrative pairs: identical, equal means, equal covariances,
isotropic, mixed diagonal, anisotropic, the embedded L=100 stress example,
and the benchmark's s=.8 parent pair. Both orientations and squared divergence
values are saved; the table displays roots on the distance scale. All eight
reduced references qualify and agree with forward/reverse horizontal solves
within 1.24e-14; affine checks for d0 and triangle roots agree within 6.28e-13.
Three independent exact checks pass. These are numerical checks, not global
optimality certificates or a worst-case search.

The triangles can lie below dF and be asymmetric. For N(0,I) versus
N((1,1,1),diag(.25,1,4)), dF=2.210160, d0=2.219585, delta_c=2.146508 forward
and 2.511750 reverse. On the embedded L=100 case, dF=5.413124,
d0=5.912835, delta_c=7.251064, delta_perp=5.222309. These numerical values
are roots, not the published squared divergences. Full endpoints, solver
settings, diagnostics and signed relative discrepancies are in
`results/triangle_pairs_3d.{json,csv,md}`. Classification accuracy and fidelity
to Fisher distance are distinct criteria.

## Triangle divergences on the saved 10D trials (2026-10-19)

**Partial numerical observation.** Both triangle classifiers and both component
classifiers were evaluated on the 37 previously saved 10D trials, without
recomputing Fisher/d0 or generating additional data. Source SHA256 and all
Fisher/d0 predictions are preserved. Results are in `results/mdm_triangles_10d`.
All 74 centers per triangle method qualify, as do all covariance-only fits;
all predictions are finite. The 73-test suite passes, now including independent
triangle formula, gradient and affine-invariance checks in dimension 10.

At fixed-parent s=.8,m=15, Fisher/d0/c/perp score 91.5/91.25/93/92.25 percent;
mean-only scores 93.25 percent. At s=.35,m=60 the corresponding full-Gaussian
scores are 90.5/90.5/91.25/90.25 percent. No uniform winner is established.
The hierarchy has only five seeds at s=.35,m=15 and two at s=.8,m=15;
other hierarchical batches and the exact-law control remain absent.
As before, gradient qualification and agreement between starts do not certify
global center optimality. The original Fisher qualifications still apply.

## Published triangle divergences on the same 3D benchmark (2026-10-19)

**Numerical observation.** `compare_mdm_components.py --triangles` evaluates
both directed divergences of Collas et al. (ICASSP 2022, Corollaries 1 and 2)
on all 70 saved 3D trials. Each has its own center minimizing equation (18),
with center-first order for training and prediction. The source SHA256 and
all four existing classifiers' predictions match the component-only results.
See `docs/triangle_divergences.md` for formulas and analytic gradient derivations.

On fixed parents at s=.8,m=5, delta_c/delta_perp achieve 81.75%/79.5%, versus
78.75% Fisher/d0. On hierarchical observations at s=.35,m=5 they achieve
58.5%/56%, versus 54% Fisher/d0; at s=.35,m=20, delta_c drops to 69.25%
versus 71.5% Fisher. Thus neither new method is uniformly best. Full per-seed
results and SDs are in `results/mdm_triangles_3d.{json,csv,md}`.

Of 140 centers per method, 139 delta_c and all 140 delta_perp centers satisfy
the 1e-7 coordinate/intrinsic gradient checks. One delta_c last iterate has
intrinsic norm 1.28e-7 and is retained and flagged. All predictions are finite.
Two deterministic starts agree in objective within 1.20e-14 across all centers;
this does not certify global minima. The five previous covariance-only
qualification failures remain visible. All 71 tests pass, including independent
formula/gradient checks, affine invariance/equivariance, special cases and
training-test isolation. Existing Fisher minimizing-geodesic assumptions remain.

Measured one-thread training/prediction totals are 2.34/.253 s for delta_c
and 1.81/.244 s for delta_perp. Fisher's saved run used 114.59/84.86 s;
d0 reused that Fisher training and took 4.28 s for prediction. These compare
current implementations on separate runs; the new divergence evaluations
are batched, unlike the earlier per-pair Fisher/d0 prediction loop.

## 3D component-only MDM comparison (2026-10-19)

**Numerical observation.** `experiments/compare_mdm_components.py` reuses all
70 saved 3D trials and Fisher/d0 predictions, adding covariance-only MDM
(affine-invariant SPD mean/distance) and mean-only MDM (arithmetic/Euclidean).
At hierarchical separation .8 and batch 20, accuracies are Fisher/d0 92.5%,
covariance-only 84.25%, mean-only 81%. At separation .35 and batch 5,
mean-only reaches 58.25% versus Fisher/d0 54%; full geometry is not uniformly
better on this design. Tables, SDs and diagnostics are saved under
`results/mdm_components_3d`. Five of 140 covariance prototypes fail the strict
1e-8 gradient threshold (norms 1.16e-8 to 1.98e-7); last iterates are retained
and explicitly flagged. All other fits converge. The 59-test suite passes,
including exact commuting means, noncommuting geometric midpoint, congruence
invariance and component/training-test isolation. Existing Fisher minimizing-
geodesic assumptions remain unchanged.

## Dimension-configurable MDM (2026-10-19)

**Partial numerical observation.** The 10D run was stopped at the user's
request after 37/70 trials. All 30 fixed-parent trials and five hierarchical
trials at s=.35,m=15 are complete; s=.8,m=15 has two seeds only. Other
hierarchical settings and exact-law controls remain uncomputed. All saved
Fisher fits and predictions qualify; 74 reduced-J checks agree within
8.79e-14. d0 differs on 5/2960 predictions, with maximum relative distance
discrepancy 1.9990%. See `results/mdm_sampled_10d.md` for the partial tables.
These observations retain the minimizing-geodesic assumption and do not
establish superiority across scenarios; the two-seed hierarchical setting
actually has lower Fisher than naive accuracy.

The MDM experiment now accepts `--dimension` and `--batch-sizes`. The 10D
design uses batches 15,30,60, with unchanged classifier and solver settings.
Outputs are separate from the original 3D results. The explicit parent
extension and dimension-dependent hierarchical noise normalization are
documented in `docs/mdm_benchmark.md`; this is not a dimension-only comparison.
The 56-test suite passes, including independent sample-covariance and matrix
exponential checks in 10D, SPD/rank checks and shared latent laws across batches.
One saved baseline and one hierarchical-exact 3D training split were reproduced
bit for bit by the generalized generators.

Documentation inconsistency noted during inspection: `docs/notation.md`
section 4 has a plus sign in the mean geodesic equation, whereas section 2.2
below and `docs/mvn_maps.md` use a minus sign. The claimed first integral
Sigma^{-1} dot(mu)=constant requires the minus sign. The benchmark extension
does not change the geometric solvers or rely on that conflicting equation.

Current benchmark inventory (2026-10-18): only `experiments/compare_d0_df.py`
(38 fixed 3D pairs) and `experiments/compare_mdm_sampled.py` (sampled 3D MDM)
remain active. Earlier benchmark scripts and results referenced below are
historical and preserved with their original paths inside
`archive/benchmarks_before_consolidation.zip`. This consolidation does not
resolve the theoretical qualifications recorded below.

## Hierarchical 3D MDM and shared-prototype comparison (2026-10-18)

The active MDM benchmark now retains the fixed-parent baseline with smaller
batches (5, 10, 20) and adds per-observation latent Gaussian laws, evaluated
both exactly and after small-batch estimation. Independent latent/sampling
RNG streams preserve identical latent laws across these observation settings.
Defaults tau_mu=tau_Sigma=.35 are fixed before accuracy evaluation.
See `docs/mdm_benchmark.md` for the precise symmetric-noise distribution,
covariance construction, paired design and numerical qualifications.

Four classifiers compare arithmetic/Euclidean MDM, Fisher MDM, and d0 or
Euclidean prediction distances on the identical fitted Fisher prototypes.
The latter isolate prediction distance; no d0-specific mean is implemented.
Training costs are shared, and failures are retained with explicit paired
denominators. Earlier sampled-parent outputs described in the next section
are preserved in `archive/mdm_before_hierarchical.zip`; the active results
prefix now refers to this expanded experiment.

**Numerical observation.** The full 70-trial run (five seeds, 12 training and
40 test laws per class) qualified all Fisher fits and predictions. On the
same Fisher prototypes, d0 changed 6/5600 predictions (0.1071%); maximum
relative pairwise distance discrepancy was 2.9207%. Total prediction times
were 84.86 s for qualified horizontal d_F and 4.28 s for d0 (19.84x ratio,
one BLAS thread), excluding the shared Fisher prototype-fitting cost.
All 140 reduced-J spot checks were reliable and agreed within 8.72e-15.
Observed descriptor covariance condition numbers reached 21,622 with no
regularization. These findings apply only to the fixed tested design and
retain the minimizing-horizontal-geodesic assumption.

The 51-test suite passes, including independent matrix-exponential checks,
zero-spread collapse, exact raw-sample reconstruction, latent-stream reuse,
shared prototypes, training/test separation and qualification-mask handling.
A saved-data audit checks all 70 trials, SPD descriptors, identical latent
laws across views, prototype equality and paired disagreement denominators.

## MDM from sampled 3D Gaussian parent laws (2026-10-15)

**Numerical observation.** `experiments/compare_mdm_sampled.py` extends the
complete-classifier comparison to fitted descriptors: each class has a fixed
3D Gaussian parent and each datapoint is the sample mean and unbiased sample
covariance of a fresh independent batch. Two fixed parent separations (.35,
.8) change both means and noncommuting covariances; raw batch sizes are
10, 30 and 100. Five seeds use 12 train and 40 test descriptors per class.
No shrinkage, regularization, test tuning or seed filtering is used.

Naive/Fisher mean accuracies for separation .35 are 67.75%/75.50%,
76.75%/86.75%, 94.75%/98.50% as batch size increases. At separation .8,
they are 87.50%/95.50%, 98%/100%, 100%/100%. This uses ordinary Gaussian
sampling rather than descriptors generated via the Fisher exponential.
Larger batches reduce descriptor estimation noise without changing the
number of training descriptors. The findings are specific to these parent
laws and scales; they do not establish universal superiority or isolate
mean construction from distance choice.

All 30 trials have converged Fisher prototypes and qualified predictions;
60 independent reduced-distance checks agree within 5.11e-14. Total Fisher
training/prediction times are 60.1/86.7 s, versus .004/.116 s for naive,
with one BLAS thread. Existing minimizing-geodesic assumptions still apply.
Full parent parameters, generated descriptors, predictions and diagnostics
are in `results/mdm_sampled_3d.json`; the companion CSV/Markdown include
across-seed SDs. `experiments/plot_mdm_sampled.py` regenerates PNG/SVG plots.
The 49-test suite passes, including sample-covariance checks against np.cov
and exact reconstruction of the seeded raw-sampling process.

## Synthetic complete-MDM comparison (2026-10-15)

**Numerical observation.** At the user's request, `experiments/compare_mdm.py`
compares separate arithmetic Gaussian prototypes with Euclidean parameter
distance against estimated Fisher Frechet prototypes with qualified horizontal
distance estimates. This changes both prototype and metric; it does not isolate
the contribution of either. Five fixed seeds use 12 training and 40 held-out
descriptors per class, two balanced classes. Gaussian parameters themselves
are simulated; descriptor estimation from raw signals is not included.

Mean accuracies (naive / Fisher): tight 2D Fisher-generated clusters
96.75% / 99.25%; broad clusters 67.50% / 80.25%; additive parameter-space
clouds 91.50% / 84.75%; deliberately multimodal coupled 1D classes
45.00% / 51.75%. Thus improvement is not universal. The tangent-generated
families favor intrinsic geometry by construction, while both single-prototype
classifiers perform poorly on the multimodal example.

Refitting after the fixed affine feature transform B=diag(3,.3), b=(1,-2)
changes 65/400 naive predictions and 0/400 Fisher predictions on the broad
samples; accuracies become 63.75% / 80.25%. All 25 trials have converged
Fisher prototypes and qualified prediction distances. Fifty independent
reduced-objective spot checks agree within 4.11e-15. These are numerical
checks under the existing minimizing-horizontal-geodesic assumption.
Total measured training/prediction times across all trials were 56.8/37.1 s
for Fisher versus 0.006/0.108 s for naive, with one BLAS thread.

`results/mdm_comparison.md`, `.csv`, and `.json` preserve the design,
across-seed variability, exact endpoints, predictions, mean histories and
distance diagnostics. `experiments/plot_mdm.py` regenerates PNG/SVG figures
from the CSV and requires matplotlib. The 47-test suite passes, including
simulation reproducibility, SPD validity and exclusion of test data from fits.
This small synthetic study does not establish performance on real data,
higher-dimensional descriptors or estimated covariances from finite samples.

## Gaussian maps and numerical Frechet means (2026-10-15)

**Implemented and numerically checked under the existing minimizing-horizontal-
geodesic assumption.** `src/fisher_rao/mvn.py` exposes `exp_map_mvn`,
`log_map_mvn` and `frechet_mean_mvn`. The logarithm solves for S, extracts
the initial tangent and qualifies it using spectral, horizontal residual and
exponential endpoint reconstruction checks. It does not use an unverified
S-to-K conversion. The mean uses qualified tangent averaging and Armijo
backtracking, retaining explicit convergence status and numerical history.

Tests cover independent geodesic ODE integration, exact special cases, seeded
3D/4D round trips, dense affine-equivariance checks and numerical first
variations of the mean objective. The symmetric univariate pair with means
-2 and 2 and variance 1 yields midpoint variance 3. These checks neither
close A1 nor certify minimizing logarithms, uniqueness or global means.
See `docs/mvn_maps.md` for formulas, reproducible tests and limitations.
Existing benchmark outputs and distance definitions are unchanged.

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
\ddot\mu-\dot\Sigma\Sigma^{-1}\dot\mu=0.
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

## 15. Compact presentation benchmark (2026-09-10)

**Numerical observation.** `experiments/presentation_cases.py` now evaluates a
fixed suite of 16 normalized Gaussian pairs, without filtering by observed error.
The table is saved in `results/presentation_cases.md`; full endpoints, seeds,
absolute/relative errors, runtimes and optimizer diagnostics are in the companion CSV.

The first 14 cases cover dimensions 1, 3, 5, 12, 20 and 30, mean norms up to 30,
isotropic expansion/contraction, aligned and mixed means, and covariance condition
numbers up to 10,000. Their maximum relative discrepancy between d_0 and the
numerical reduced-objective reference is 0.7727%. Two additional deliberate stress
cases, Sigma=diag(1/L,L), mu=(1,sqrt(L)), give 9.2315% at L=100 and 16.5318% at
L=1000. These are finite numerical observations, not a universal error bound or
a verification of the historical asymptotic claim.

Five-start BFGS at gtol=1e-8 reports convergence for 10/16 cases. The six remaining
cases encounter precision loss; five-start reruns at gtol=1e-6 converge and change
distances by at most 1.25e-12. All pairs were also solved with reversed endpoints
(three starts), with maximum distance discrepancy 1.95e-14. Strict failures are
retained in the CSV rather than relabeled as converged. These checks qualify the
displayed precision empirically; they do not certify global optimality or prove
the reduced Fisher-distance identity.

`results/presentation_validation.json` additionally records agreement with Brent
scalar minimization in both 2D stress cases and an affine-invariance check using
noncommuting endpoint covariances (distance discrepancies below 1e-12).
The existing numerical-core tests pass (23 tests). Earlier implementation tasks
in this document therefore predate the numerical core now present in `src/fisher_rao`.

### Revised table with explicit dense inputs (2026-09-10)

**Numerical observation.** `experiments/mixed_presentation_cases.py` retains seven
old examples (including three simple checks and the L=100 stress family) and adds
seven fixed dense-covariance examples in dimensions 2, 5, 12, 20 and 30. Two new
pairs have independently generated dense covariances at both endpoints; their
commutator norms are nonzero. These new examples are not common rotations of the
previous endpoints. The table and CSV are `results/mixed_presentation_cases.md`
and `.csv`; the companion `.json` contains every endpoint matrix and mean,
construction seeds, runtime environment and all optimizer diagnostics.

The seven new cases have relative discrepancies ranging from 0.00731% to 1.96529%.
The maximum among the thirteen non-stress rows is 1.96529%; the retained stress
row gives 9.23146%. All fourteen pass the recorded numerical checks: converged
five-start primary and three-start reversed solves at gtol=1e-6, agreement with
two-start strict solves at gtol=1e-8 within 1e-8 in distance, available exact checks,
and the diagonal-path upper bound after normalization. The largest strict/reversal
distance discrepancy is below 5.1e-12. As before, this does not certify global
optimality or independently establish the reduced distance identity.

## 16. Same fourteen pairs with horizontality BFGS (2026-09-11)

**Numerical observation under an explicit working assumption.** The user asked
to assume that a horizontal lifted geodesic projects to a minimizing MVN
geodesic. `experiments/horizontal_presentation_cases.py` loads the exact endpoint
arrays of the fourteen-case mixed benchmark and minimizes
R(S)=||(log G_1(S))_13||_F^2 using analytic-gradient BFGS. It does not minimize
lifted distance or J(K). The gradient follows the spectral Frechet derivative
of log and its adjoint through G=M D M.T; the derivation and canonical distance
scaling ||log G||_F/2 are recorded in `experiments/horizontality_derivation.md`.

The primary settings retain five starts (zero plus the same seeded Gaussian
coordinate vectors), gtol=1e-6 and maxiter=2000. The coordinates now represent
S, not K; no mapping between the two is assumed. Original strict and reverse
checks are retained. Nine cases additionally use gtol=1e-11, maxiter=5000,
with the original five forward and three reverse starts, because gradient
termination alone does not ensure a small horizontal residual.

All fourteen final rows pass convergence, spectral, residual, reversal,
available exact-distance and diagonal-path-upper-bound checks. The maximum
selected forward residual is 8.62e-10, maximum forward/reverse (or strict)
distance discrepancy is 9.24e-14, and maximum discrepancy with the previous
reduced-objective reference is 9.86e-13. Thus the displayed comparison table
is unchanged: maximum non-stress d_0 error 1.9653%, retained stress error 9.2315%.
This is numerical agreement between objectives under the stated assumption,
not a proof of their equivalence or of the reduced-distance identity.

**Numerical reliability finding.** Random restarts can escape to large fiber
parameters. One D07 restart reports success and a small residual after severe
spectral breakdown (||H+JHJ|| about 322, trace(H) about 1008); its apparent
distance 132.040111 is invalid. All raw runs are preserved and unresolved
spectra are excluded from qualified selection. Even resolved primary results
can have residuals near 1e-3. Final selection minimizes residual among resolved
candidates, never distance, and requires forward/reverse residuals <=1e-7.

Outputs: `results/horizontal_presentation_cases.md`, `.csv`, `.json`, and
`results/horizontal_validation.json`. The validation script checks exact saved
endpoints/restart vectors, independent SVD spectra, exact special cases and
directional derivatives across the suite. Its maximum scaled directional
gradient discrepancy is 6.19e-10. The numerical core now has 32 passing tests.
The existing mixed/presentation/initial result files were preserved.
