 Fisher–Rao distance for multivariate normal distributions

This repository studies the Fisher–Rao geometry of multivariate normal distributions (MVNs), with a particular focus on the general boundary-value problem and on accurate, computationally cheap approximations of the Fisher–Rao distance.

A Gaussian is represented by
\[
P=(\mu,\Sigma),\qquad \mu\in\mathbb R^n,\quad \Sigma\in\mathrm{SPD}(n).
\]

## Main scientific problem

Given
\[
P_0=(\mu_0,\Sigma_0),\qquad P_1=(\mu_1,\Sigma_1),
\]
compute or approximate the Riemannian distance induced by the Fisher information metric.

For arbitrary MVNs, the initial-value geodesic equations are known explicitly, but the two-point boundary-value problem is not known in a simple closed form in general. This distinction is central to the project.

The current main direction uses Shimpei Kobayashi's description of Fisher geodesics as projections of horizontal geodesics in a higher-dimensional symmetric space. After affine normalization, the boundary-value problem can be reduced to an optimization over a skew-symmetric matrix. This yields:

- an exact finite-dimensional formulation of the Fisher–Rao distance;
- a very cheap zeroth approximation `d0`, obtained by setting the skew parameter to zero;
- restricted approximations `dm` based on low-dimensional Krylov subspaces;
- numerical experiments designed to understand when `d0` is accurate, when it fails, and how quickly `dm` converges.

## Secondary directions

The repository also tracks alternative geometric distances and approximations, including:

- the Calvo–Oller embedding into `SPD(n+1)`;
- geodesic-triangle upper bounds / approximations;
- a project-specific triangle heuristic in which the covariance weighting the mean displacement is replaced by the affine-invariant geometric mean of the two endpoint covariances;
- pullback distances in SPD cones;
- numerical geodesic shooting / high-accuracy reference solvers.

## Intended application

The intended application is classification of acoustic signals.

A temporal signal is transformed into local frequency-domain feature vectors using a sliding STFT. A window or collection of local feature vectors is then modeled by a multivariate normal distribution. Distances between the resulting MVNs can be used in geometric classifiers such as k-nearest neighbours.

The methodological goal is to obtain distances that are:

1. geometrically meaningful;
2. accurate with respect to Fisher–Rao;
3. cheap enough for repeated use in classification.

## Code layout

The repository follows the layout of the earlier `Simulated_data` project:

- `src/distance.py` — squared distances: `dist2_L2` (Euclidean, between MVNs),
  `dist2_Fisher` (affine-invariant, between SPD matrices), `dist2_d0` (closed-form
  zeroth approximation) and `dist2_dF` (Fisher–Rao, by BFGS minimization of `J(K)`
  over skew-symmetric `K`, started at `K=0`);
- `src/mean.py` — `arithmetic_mean` of MVNs, SPD `geometric_mean` (fixed point),
  and `frechet_mean(means, covariances, dist2)` for the `d_0` and `d_F` geometric means;
- `src/simulated_data.py` — `random_SPD`, SPD exponential map, `random_MVNs` around a central MVN;
- `src/covariance.py` — sample estimators of Gaussian parameters.

Experiments are scripts `scripts/run_*.py`, run from the repository root as modules,
e.g. `python -m scripts.run_MVN_means`; figures are written to `output/`.

Install with `pip install -r requirements.txt` and `pip install -e .`; run the tests
from the repository root with `python -m unittest discover -s test -t .`.

The former code (horizontal fiber solver, exponential/logarithm maps, MDM classifiers,
triangle divergences) is available in the git history before the simplification.
The former `d_0`/`d_F`, MDM and triangle benchmarks, with all their results, are
preserved in `archive/experiments_before_mean_focus.zip`; earlier retired
benchmarks are in `archive/benchmarks_before_consolidation.zip`. Historical
research notes refer to those archived files.

## Repository directories

- `docs/notation.md` — canonical notation and formulas.
- `docs/state.md` — current state of the research, including proved results, numerical evidence, and unresolved points.
- `docs/open_problems.md` — active research questions, ordered by priority.
- `docs/bibliography.md` — references and why they matter.
- `papers/` — local copies of relevant papers.
- `src/` — reusable, tested mathematical/numerical code, as flat modules imported with `from src.<module> import ...`.
- `scripts/` — experiment scripts `run_*.py`, run from the repository root.
- `output/` — figures and data written by the scripts.
- `test/` — automated `unittest` tests, one `test_<module>.py` per module.
- `archive/` — retired benchmark scripts and their results.
- `paper/` — article/report sources.
- `slides/` — presentation sources.

## Working rules

Always distinguish explicitly between:

- **proved** statements;
- **derived but not yet independently checked** statements;
- **numerical observations**;
- **heuristics / conjectures**.

Never promote a numerical pattern to a theorem without proof.

Whenever a formula depends on a metric normalization, state the convention explicitly. In particular, older project notes occasionally omitted the standard factor `1/2` in the covariance part of the Fisher metric; this must be audited before comparing formulas or numerical values.

Numerical experiments should be reproducible and should record:

- dimension;
- endpoint parameters;
- normalization used;
- solver tolerance;
- definition of the reference distance;
- absolute and relative errors.

See `docs/state.md` before starting new work.
