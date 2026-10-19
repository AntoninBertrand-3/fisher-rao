# Gaussian-law MDM benchmark (3D and 10D)

## Published triangle divergences (2026-10-19)

Run `python experiments/compare_mdm_components.py --triangles` to add the
two Collas et al. divergences, each with its own fitted center, to the same
70 saved 3D trials. Results use `results/mdm_triangles_3d.{json,csv,md}`.
See [formulas, gradients, argument order and solver qualification](triangle_divergences.md).
Fisher/d0 results are reused; no Gaussian data or estimation conventions change.

## Component-only comparison (2026-10-19)

`python experiments/compare_mdm_components.py` reuses the saved 70-trial 3D
dataset and Fisher/d0 predictions, and fits two additional classifiers using
the identical training splits. Covariance-only MDM uses the affine-invariant
SPD Frechet mean and trace-metric distance from notation section 11.
Mean-only MDM uses the arithmetic mean vector and Euclidean distance.
The covariance mean minimizes mean(d_SPD**2)/2: its negative gradient in
whitened coordinates is mean(log(P^{-1/2} C_i P^{-1/2})). Updates use
P^{1/2} exp(step * mean_log) P^{1/2}, with Armijo backtracking.
This manifold iteration preserves SPD; it uses standard NumPy eigensolvers.
The stopping tolerance is 1e-8, with at most 200 updates. Nonconverged
last iterates remain visible and are flagged per scenario and seed.
Scaling the covariance metric by 1/2 changes neither means nor predictions.

Outputs use `results/mdm_components_3d.{json,csv,md}` and record a SHA256 of
the source dataset, prototypes, predictions, distances and convergence
diagnostics. Fisher/d0 timings come from the earlier run; component timings
come from this run, both with one BLAS thread. This compares complete MDM
pipelines, not four distances on a common prototype. d0 continues to reuse
the Fisher prototype and does not define a separate d0 Frechet mean.

## Dimension-10 extension (2026-10-19)

Run `python experiments/compare_mdm_sampled.py --dimension 10 --batch-sizes 15 30 60`,
then `python experiments/plot_mdm_sampled.py --input results/mdm_sampled_10d.csv`.
The dimension is saved in both CSV and JSON; outputs use a separate 10D prefix.
The default invocation still reproduces the 3D design below.

The 10D run retains the four classifiers, two separations, five seeds, 12
training and 40 test descriptors per class, and all seven observation settings.
Unbiased sample covariances require m>n; the selected batches are 15,30,60,
without regularization. There are still 70 trials.

For n other than 3, extend the parent construction as follows: cycle the
diagonal (1,.6,1.8) to length n; cycle the mean direction (.8,-.5,.6) and
rescale to its original Euclidean norm. Generate an n-by-n iid standard
normal matrix with NumPy seed 20261019, symmetrize it, and rescale to the
original B's Frobenius norm. Use the same C1=Q exp(s B) Q formula.
This distributes class differences over all coordinates, with fixed total
perturbation norms; it does not prescribe equal Fisher separation across n.
All parent arrays are saved in JSON.

The hierarchical noise generalizes to z~N(0,I_n) and
H=(G+G.T)/sqrt(2*n). Diagonal and off-diagonal entry variances are 2/n
and 1/n, respectively. Thus E[||H||_F^2]=n+1, while E[||z||^2]=n.
The spreads remain .35: total within-class variation grows with dimension.
This is an explicit modelling convention, not a controlled experiment
isolating dimension alone; batch sizes and parent construction also change.
At n=3, the previous generator and its RNG sequence are preserved.

The following sections describe the original 3D settings; classifier and
qualification rules also apply in 10D.

This is a fixed synthetic experiment, not a claim about real acoustic data.
Run `python experiments/compare_mdm_sampled.py`, then
`python experiments/plot_mdm_sampled.py`. Reports and figures use the
`results/mdm_sampled_3d` prefix. The earlier fixed-parent experiment with
batches 10, 30 and 100 is saved in `archive/mdm_before_hierarchical.zip`.

## Class centres

Both classes have dimension three. The centres are unchanged:

\[
\mu_0=0,\quad C_0=\operatorname{diag}(1,.6,1.8),\qquad
\mu_1=s(.8,-.5,.6)^T,\quad C_1=Q\exp(sB)Q,
\]

where \(Q=C_0^{1/2}\) and

\[
B=\begin{pmatrix}.7&.35&-.2\\.35&-.6&.25\\-.2&.25&.2\end{pmatrix}.
\]

The separation settings are s=.35 and .8. These are generator parameters,
not prescribed Fisher distances. Full matrices are saved in JSON.

## Observation generators

**Fixed-parent baseline.** Each observation is a Gaussian descriptor fitted
to m independent vectors from N(mu_c,C_c), with m=5,10,20. Mean and covariance
use the sample average and denominator m-1, without regularization.

**Hierarchical model (a modelling choice).** Each observation has its own
latent Gaussian law. For L_c=chol(C_c), generate independently

\[
z_i\sim N(0,I_3),\qquad (G_i)_{jk}\overset{iid}{\sim}N(0,1),
\qquad H_i=(G_i+G_i^T)/\sqrt{6},
\]

and define

\[
\mu_i=\mu_c+\tau_\mu L_cz_i,\qquad
\Sigma_i=L_c\exp(\tau_\Sigma H_i)L_c^T.
\]

The six independent symmetric entries of H have diagonal variance 2/3
and off-diagonal variance 1/3. Defaults tau_mu=tau_Sigma=.35 are fixed
before evaluating accuracy; CLI overrides are recorded in metadata.
The covariance exponential uses batched symmetric eigendecomposition.
Symmetry makes exp(tau_Sigma H) positive definite; congruence by invertible L
preserves that property. Numerical SPD checks still apply.

Two views of this same model are evaluated:

- `hierarchical_exact`: use (mu_i,Sigma_i) directly as the observation.
- `hierarchical_sampled`: draw m=5,10,20 vectors from each latent law and
  fit a descriptor, adding estimation noise to between-law variation.

Class centres are construction parameters, not asserted population Fisher
means. In particular, E[exp(tau_Sigma H)] need not equal I. The generator
does not use the full Gaussian Fisher exponential. Setting both spreads to
zero collapses latent laws to class centres, which is independently tested.

## Paired design and reproducibility

Five seeds (20261016+100*k, k=0,...,4) each produce 12 training and 40 test
laws per class. Training uses the seed; testing uses seed+1. There are 70
trials: two separations, seven observation settings and five seeds.

For hierarchical data, SeedSequence(seed).spawn(2) creates independent latent
and sampling streams. Latent laws are identical across exact and sampled
views and all batch sizes for the same split and seed. Raw batches at different
sizes are not claimed to be nested. Scenarios reuse seeds, so cross-scenario
results are dependent, not independent evidence. Test observations are never
used to fit prototypes or select generator/solver settings.

JSON saves all latent laws, observed descriptors, parents, seeds, versions,
fit histories, distance diagnostics, predictions and qualification masks.
Raw vectors are reproducible from the recorded RNG design. Covariance
spectra and within-class dispersion can be recovered from these arrays.

## What each comparison measures

| Method | Prototype | Prediction distance |
|---|---|---|
| naive | Separate arithmetic means of mu and Sigma | Euclidean parameter norm |
| fisher | Estimated Fisher Frechet mean | Qualified numerical d_F |
| d0 | The identical fitted Fisher prototype | Canonical d_0=sqrt(J(0)) |
| euclidean_fisher | The identical fitted Fisher prototype | Euclidean parameter norm |

The Euclidean parameter norm is sqrt(||delta_mu||^2+||delta_Sigma||_F^2).
Naive versus Fisher compares complete pipelines, changing prototype and
distance together. Fisher versus d0 or euclidean_fisher isolates prediction
distance on fixed prototypes. This does not implement a d0-specific Frechet
mean or assert d0 is a metric. Distances are evaluated prototype-to-test.

Fisher fits use arithmetic initialization, 60 iterations and tolerance 1e-6.
Horizontal logarithms use two starts, seed 1701, gtol=1e-11, maxiter=2000,
residual tolerance 1e-8 and endpoint tolerance 1e-7. Spectral, residual and
endpoint checks qualify candidates under the existing minimizing-horizontal-
geodesic assumption; global optimality remains unproved. One test-to-prototype
distance per class is cross-checked against three-start reduced-J optimization.

## Reporting rules

- Identical splits for all methods; ties go to class 0.
- Unqualified predictions receive -1 and count as accuracy errors.
- Nonconverged prototype fits retain last-iterate results, explicitly flagged.
  Paired disagreement/error summaries exclude those fits.
- Disagreement counts require both predictions qualified. Distance errors
  require both distances qualified; relative errors also require d_F>1e-7.
  Counts and denominators are explicit.
- Reported SDs are across seed accuracies, not confidence intervals.
- Fit and prediction timings are separate, with one BLAS thread. Shared
  methods incur the same Fisher prototype cost, reused rather than refitted.
  Numerical d_F timings include qualification checks: this is measured
  implementation cost, not an intrinsic complexity comparison.

Small batches deliberately increase variability and can produce ill-conditioned
covariances. Failures remain visible; no automatic shrinkage, seed rejection
or adjustment based on observed accuracy is performed.
