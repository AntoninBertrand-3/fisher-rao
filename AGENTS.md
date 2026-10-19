# AGENTS.md

## Purpose

This repository is a mathematical research project on the Fisher–Rao distance between multivariate normal distributions (MVNs).

The long-term goal is to understand, compute, and approximate the general Fisher–Rao distance.

## Current project phase

The **immediate objective of the repository** is narrower:

1. implement a reliable numerical computation of the reference Fisher–Rao distance `d_F`;
2. implement the cheap approximation `d_0`;
3. generate varied pairs of multivariate normal distributions;
4. compute `d_F` and `d_0` on those cases;
5. measure and analyze the absolute and relative error of `d_0`;
6. identify configurations in which `d_0` is especially accurate or inaccurate.

This objective has priority over later work on `d_m`, acoustic classification, alternative distances, or speculative theoretical extensions unless the user explicitly asks for them.

The repository should therefore first become a clean, readable and efficient numerical laboratory for comparing `d_0` with a trusted numerical value of `d_F`.

This is a research repository, not only a software project. Mathematical correctness, numerical reliability, readability, efficiency and traceability are all first-class requirements.

---

# 1. Read this before doing substantial work

Before starting any nontrivial task, inspect the relevant project context.

At minimum, read:

1. `README.md`
2. `docs/notation.md`
3. `docs/state.md`
4. `docs/open_problems.md`

For literature-related work, also read:

5. `docs/bibliography.md`

Do not rely on assumptions from the repository name or from previous model knowledge when the project files already define the relevant conventions.

If a prompt conflicts with the current repository documentation, follow the user's explicit prompt, but point out the inconsistency.

---

# 2. Mathematical conventions are authoritative

`docs/notation.md` is the canonical source for notation and metric conventions.

In particular, the repository uses the standard Fisher metric

g_{(mu,Sigma)}((delta mu,delta Sigma),(delta mu',delta Sigma'))
=
delta mu^T Sigma^{-1} delta mu'
+
1/2 Tr(Sigma^{-1} delta Sigma Sigma^{-1} delta Sigma').

Do not silently change:

- metric normalization;
- definitions of d_F, d0, dm;
- normalization conventions;
- indexing of the Krylov hierarchy;
- definitions of Y(K), C(K), J(K);
- definitions of the Kobayashi lift or fiber parameters.

If an external paper uses a different normalization, explicitly derive the conversion before comparing formulas or numerical values.

If two project files appear inconsistent, stop treating the conflicting formula as established and report the inconsistency.

---

# 3. Research-status discipline

Every mathematical claim must be mentally classified as one of:

- **proved / literature-backed**;
- **derived in this project and checked**;
- **numerical observation**;
- **heuristic / conjecture**;
- **unverified idea**.

Never upgrade a claim from one category to a stronger category without justification.

In particular:

- numerical agreement is not a proof;
- symbolic manipulation is not automatically a proof;
- a result from one random experiment is not evidence of a general bound;
- asymptotic experiments with very ill-conditioned matrices require numerical-stability checks;
- a formula suggested by a special case must be tested against noncommuting and anisotropic cases.

When writing research notes, use explicit labels such as:

`Proposition`, `Conjecture`, `Numerical observation`, `Heuristic`, `Open question`.

---

# 4. Preferred research workflow

For a substantial mathematical problem, use the following loop.

## Step A — identify the exact question

Restate the mathematical target precisely.

Determine:

- assumptions;
- variables;
- normalization;
- quantity to prove or compute;
- whether the target is exact, asymptotic, numerical, or heuristic.

## Step B — inspect existing work

Search the repository before deriving something from scratch.

Check:

- `docs/state.md`;
- `docs/open_problems.md`;
- relevant code;
- existing experiments;
- local papers if available.

Do not duplicate an experiment or derivation without first checking whether it already exists.

## Step C — derive before implementing

For mathematical code, write down the formula being implemented and identify the source or derivation.

Do not create opaque numerical code whose mathematical objective is unclear.

## Step D — test aggressively

Whenever possible, verify a new formula using:

- exact special cases;
- low-dimensional cases;
- random numerical checks;
- symmetry/invariance properties;
- independent implementations;
- limiting regimes.

## Step E — attempt to falsify

Before presenting a conjecture as plausible, actively search for counterexamples.

For bounds or approximations, test:

- high condition numbers;
- large mean norms;
- noncommuting configurations;
- dimensions above 2;
- mean directions that mix covariance eigenspaces;
- asymptotic scaling families.

## Step F — record durable progress

If a result is substantial and verified, update the relevant project documentation.

Normally:

- stable notation -> `docs/notation.md`;
- research progress -> `docs/state.md`;
- newly opened/closed questions -> `docs/open_problems.md`;
- verified references -> `docs/bibliography.md`.

Do not add speculative conclusions to `docs/state.md` as established facts.

---

# 5. Exact special cases are mandatory tests

Any implementation or proposed general formula for the Fisher–Rao distance should be tested against known simplifications.

At minimum consider:

## Equal means

When mu_0 = mu_1, the distance reduces to covariance-only Fisher geometry.

## One-dimensional Gaussians

For n = 1, the Gaussian manifold has a known hyperbolic closed form.

## Equal covariances

Use cases where only the mean changes.

## Commuting / diagonal covariances

These are useful for checking matrix-ordering issues.

## Isotropic covariances

Test Sigma = c I.

## Affine invariance

Numerically verify, where appropriate, that the distance is unchanged by invertible affine transformations.

A candidate formula that fails one of these tests should not be trusted.

---

# 6. Kobayashi / reduced-skew work

The current central exact formulation is the reduced optimization

d_F^2 = min_{K^T=-K} J(K).

Before modifying this formulation, read the relevant sections of:

- `docs/notation.md`;
- `docs/state.md`;
- Kobayashi (2023);
- Calvo–Oller when relevant.

Important distinctions:

- S is used for the endpoint-fiber parameter in the lifted formulation;
- K is used for the reduced skew optimization;
- the relationship between S and K must not be guessed;
- horizontality residual minimization and distance minimization must not be declared equivalent unless the equivalence has been derived for the formulation being used.

Do not casually replace matrix logarithms, square roots, eigenvalues, and singular values with one another. State the spectral identity being used.

---

# 7. d0 and dm

The canonical definitions are in `docs/notation.md`.

Remember:

d0^2 = J(0).

For the Krylov hierarchy,

K_m = span{mu, A mu, ..., A^{m-1} mu},
A = log Sigma,

and

dm^2 = min_{K in wedge^2 K_m} J(K).

With this literal indexing, d1 = d0 because dim(K_1) <= 1.

Do not silently shift the indexing in code.

If a different implementation convention is useful, document it explicitly in the function name/docstring and in any result table.

---

# 8. Code readability and documentation

Human readability is a primary design criterion.

The code should be easy to review by a mathematically trained reader who knows the general context of the project but did not write the implementation.

## General style

Prefer:

- small functions with one clear mathematical or computational role;
- descriptive names;
- explicit intermediate quantities when they clarify the mathematics;
- code whose structure follows the equations in `docs/notation.md`;
- straightforward control flow;
- type hints on function signatures;
- comments only where they explain a non-obvious mathematical or numerical choice.

Avoid:

- clever one-liners that hide the mathematics;
- deeply nested control flow;
- large functions mixing generation, optimization, evaluation and plotting;
- unnecessary abstraction layers;
- unexplained constants;
- duplicated mathematical formulas in several files;
- code golf;
- premature micro-optimization that makes the code hard to audit.

Mathematical variable names such as `mu`, `Sigma`, `K`, `Y` are appropriate when they match the notation documentation. Outside direct mathematical formulas, prefer descriptive names.

## Docstrings

Every function, including internal helper functions, must have a docstring immediately after its definition.

Use the NumPy-style format used in the user's existing Python repositories:

```python
def example_distance(
    mu1: np.ndarray,
    Sigma1: np.ndarray,
    mu2: np.ndarray,
    Sigma2: np.ndarray,
) -> float:
    """
    Computes the distance between two Gaussian descriptors.

    Parameters
    ----------
    mu1 : ndarray
        First mean vector.
    Sigma1 : ndarray
        First SPD covariance matrix.
    mu2 : ndarray
        Second mean vector.
    Sigma2 : ndarray
        Second SPD covariance matrix.

    Returns
    -------
    float
        Computed distance.
    """
```

Rules:

- begin with one concise sentence describing what the function does;
- include `Parameters` when the function takes meaningful arguments;
- include `Returns` when it returns a value;
- include `Raises` when user-visible validation errors are part of the function contract;
- state array shapes when they matter;
- state SPD, symmetry, normalization or skew-symmetry assumptions when they matter;
- do not put implementation trivia in the docstring;
- keep docstrings concise enough that the mathematical purpose remains immediately visible.

The public `AntoninBertrand-3/Inaturalist` repository is a visible style reference for this NumPy-style documentation.

---

# 9. Numerical implementation and performance rules

Numerical efficiency matters because the project will evaluate many Gaussian pairs and repeatedly solve optimization problems.

Use optimized NumPy/SciPy linear-algebra routines whenever an equivalent optimized routine exists.

## Vectorization

Avoid unnecessary Python loops.

In particular:

- do not loop element-by-element over vectors or matrices when NumPy broadcasting or vectorized operations express the same computation;
- do not manually implement matrix products, norms, traces, pairwise algebra, decompositions or reductions;
- prefer batched/vectorized operations when several independent cases can be handled cleanly together;
- avoid list comprehensions containing expensive numerical kernels when a vectorized or batched implementation is practical.

A Python loop is acceptable when:

- iterations have genuine sequential dependencies;
- the loop is over optimization restarts, experiment configurations or objects of incompatible sizes;
- vectorization would cause excessive memory use;
- the vectorized alternative would substantially reduce readability without a meaningful performance benefit.

When a loop is necessary in a performance-critical path, keep the expensive work inside optimized NumPy/SciPy calls.

## Linear algebra

Prefer optimized, structure-aware routines:

- `np.linalg.solve` / `scipy.linalg.solve` instead of explicitly forming an inverse;
- `np.linalg.eigh` or `eigvalsh` for symmetric matrices;
- Cholesky factorization for SPD systems when appropriate;
- SVD / singular-value routines when the formula is singular-value based;
- optimized matrix multiplication with `@`;
- SciPy optimization routines rather than hand-written generic optimizers.

Do not implement custom matrix square roots, logarithms, eigensolvers, SVDs or generic optimization algorithms unless there is a specific mathematical reason.

## Avoid repeated work

Do not recompute a costly quantity if it is unchanged inside a loop or optimizer.

Examples of candidates for precomputation include:

- `log(Sigma)`;
- Cholesky factors;
- eigendecompositions;
- `Sigma^{-1/2}`;
- fixed Krylov bases;
- fixed matrices entering repeated evaluations of `J(K)`.

When designing an objective passed to an optimizer, precompute endpoint-dependent constants outside the objective whenever possible.

## Readability versus performance

Performance improvements must preserve auditability.

Prefer a clear vectorized implementation using standard NumPy/SciPy primitives over a shorter but opaque trick.

For nontrivial performance changes, measure runtime rather than assuming an optimization is faster.

---

# 10. Numerical correctness and SPD safety

Reusable mathematical implementations belong in `src/`.

Exploratory work belongs in `scripts/run_*.py`.

Do not move exploratory code into `src/` until the mathematical definition is stable and the implementation has tests.

Prefer Python with NumPy/SciPy unless the repository later specifies otherwise.

## SPD matrices

Never assume an input matrix is SPD merely because it is intended to be one.

Where appropriate:

- check symmetry within tolerance;
- inspect the smallest eigenvalue;
- use symmetric eigendecompositions (`eigh`) for symmetric matrices;
- avoid generic `eig` when a matrix is mathematically symmetric;
- do not hide invalid inputs by silently adding regularization.

If regularization is required, expose it as an explicit parameter and record it in experiment output.

## Matrix functions

For SPD matrices, prefer spectral implementations for:

- square roots;
- inverse square roots;
- logarithms.

Avoid forming explicit matrix inverses when a linear solve is numerically preferable.

Use singular values when the formula is genuinely singular-value based.

## arcosh

Near x = 1, numerical evaluation can be delicate.

Check that quantities theoretically satisfying x >= 1 only fall below 1 by numerical roundoff before clipping.

Do not mask a mathematically invalid value with unconditional clipping.

## Skew parameterization

Parameterize an n x n skew-symmetric matrix using exactly n(n-1)/2 independent coordinates.

For restricted dm, parameterize the skew matrix directly in an orthonormal basis of the relevant Krylov subspace.

---

# 11. Reproducible experiments

## Current experiment objective

The current experiment layer is specifically designed to compare `d_0` with a trusted numerical reference value of `d_F`.

For every Gaussian pair, the preferred result record should contain at least:

- dimension `n`;
- endpoint means and covariance construction;
- normalized mean norm when relevant;
- covariance eigenvalues or condition number;
- case/family identifier;
- random seed for random cases;
- computed `d_F`;
- computed `d_0`;
- absolute error

  `abs_error = abs(d_0 - d_F)`;

- relative error

  `rel_error = abs(d_0 - d_F) / d_F`;

- runtime for `d_F`;
- runtime for `d_0`;
- optimizer diagnostics used for the reference `d_F`.

The first goal is breadth and interpretability of test cases, not maximum dataset size.

The experiment suite should include a mixture of:

- exact/sanity-check cases;
- isotropic covariances;
- diagonal and commuting covariances;
- random SPD matrices;
- controlled condition numbers;
- small and large mean norms;
- means aligned with covariance eigenvectors;
- means mixing several covariance eigendirections;
- deliberately anisotropic/extreme cases.

Keep case generation separate from distance computation so that exactly the same cases can be reused when implementations change.


Every serious experiment should record enough information to reproduce it.

At minimum:

- random seed;
- dimension n;
- mu;
- Sigma or the exact construction of Sigma;
- condition number or spectral range;
- algorithm/method;
- optimizer and initialization;
- stopping tolerances;
- reference-distance definition;
- absolute error;
- relative error;
- runtime when relevant.

Save figures and machine-readable results under `output/`, in a documented format such as PNG/CSV/JSON.

Plots must be regenerable from stored data.

Do not report a “worst case” unless the search domain and search procedure are specified.

---

# 12. Optimization and reference distances

A numerical optimizer result is not automatically the exact Fisher distance.

When using full skew optimization as the reference:

- use multiple initializations when the objective may be difficult;
- inspect gradient norm or optimizer termination diagnostics;
- compare independent parameterizations when possible;
- test exact special cases;
- distinguish `best value found` from `proven global minimum`.

When a quantity is used as a benchmark reference, document why it is trusted.

---

# 13. Literature work

For bibliographic tasks, prefer primary sources.

When making a claim from a paper, record when practical:

- authors;
- title;
- year;
- DOI/arXiv;
- theorem/equation/section;
- what exactly the source establishes.

Distinguish clearly between:

- exact Fisher distance;
- upper bound;
- lower bound;
- approximation;
- pullback metric;
- alternative divergence.

Do not call an alternative Gaussian distance “the Fisher–Rao distance” unless it is actually equal to it under the stated assumptions.

If a citation cannot be identified confidently, mark it unresolved rather than inventing metadata.

---

# 14. Code organization

Use the following separation.

## `src/`

Stable reusable implementations.

Examples:

- affine normalization;
- Fisher special-case distances;
- SPD matrix utilities;
- J(K);
- full skew optimization;
- d0;
- Krylov basis construction;
- dm.

## `scripts/`

Research scripts named `run_*.py`, run from the repository root; outputs go to `output/`.

Examples:

- extreme-case search;
- asymptotic families;
- random benchmarks;
- approximation comparisons;
- classification experiments.

## `test/`

`unittest` tests for stable code, one `test_<module>.py` per `src/` module.

Tests should prefer mathematical invariants and exact special cases over tests that merely duplicate implementation details.

## `docs/`

Durable mathematical/research state.

Do not use documentation as a dumping ground for raw terminal logs.

---

# 15. Testing

When Python code exists, run the relevant tests after modifications.

Command, from the repository root:

```bash
python -m unittest discover -s test -t .
```

If only a small part of the codebase was changed, run targeted tests first, then the broader suite when practical.

For numerical tests:

- use justified tolerances;
- do not weaken tolerances merely to make a failing test pass;
- investigate whether the failure is mathematical, algorithmic, or floating-point.

If there is no test yet for a stable mathematical function being added, add one.

---

# 16. Git discipline

Do not commit, push, rewrite history, create branches, or delete user work unless explicitly asked.

Before making broad modifications, inspect:

```bash
git status
```

Do not overwrite unrelated uncommitted changes.

After substantial modifications, summarize:

- files changed;
- tests/experiments run;
- unresolved warnings.

The human user decides when the work is ready to commit unless explicitly delegating that decision.

---

# 17. Editing discipline

Prefer small, interpretable changes.

Do not rewrite a correct derivation merely for stylistic uniformity unless asked.

Do not rename established mathematical objects casually.

When correcting a mathematical error:

1. identify the previous statement;
2. explain why it is wrong;
3. replace it consistently everywhere affected;
4. update tests or experiments if needed.

If a change affects a formula used in previous numerical results, flag those results as requiring recomputation.

---

# 18. Communication style for research tasks

Be precise and concise.

For a mathematical derivation:

- define objects before using them;
- do not introduce unexplained matrices or transformations;
- show the key algebraic step rather than saying “it follows” when that step is nontrivial;
- separate intuition from proof.

For numerical conclusions:

- state the tested domain;
- state whether the conclusion is empirical;
- give enough numbers to judge magnitude;
- avoid sweeping language such as “always”, “exact”, or “negligible” without justification.

When stuck, report the exact obstruction rather than filling the gap with plausible prose.

---

# 19. Long research tasks

For a large exploratory request, do not immediately edit the canonical documentation.

First:

1. investigate;
2. derive/test;
3. produce a concise result;
4. only update canonical files once the result is sufficiently verified.

It is acceptable to create a temporary experiment or research note when needed.

Do not let temporary exploratory notation leak into `docs/notation.md`.

---

# 20. Application-layer rules

The acoustic-classification application is downstream of the geometry.

Do not alter the mathematical distance definition to improve classifier performance without explicitly treating the altered quantity as a different method.

The intended pipeline is broadly:

signal -> sliding STFT -> feature vectors -> MVN -> geometric distance -> KNN/classifier.

Keep preprocessing, statistical estimation, geometry, and classification modular so that distance methods can be compared fairly.

---

# 21. Priority order when the task is underspecified

If asked simply to “continue the project” or “work on the main problem”:

1. read `docs/state.md`;
2. read `docs/open_problems.md`;
3. select the highest-priority unresolved problem that can be advanced concretely;
4. prefer verification/reproducibility gaps before inventing a new speculative direction;
5. explain what was attempted and what changed.

At the current project state, the default priority is:

1. obtain a reliable numerical `d_F` reference implementation;
2. obtain a clear and efficient `d_0` implementation;
3. generate varied, reproducible Gaussian test cases;
4. measure and analyze `d_0` error;
5. improve numerical reliability and speed;
6. only then expand toward `d_m`, alternative distances or the application layer unless explicitly requested.

---

# 22. Final check before presenting substantial work

Before claiming completion, ask:

- Did I use the repository's metric convention?
- Did I distinguish proof from numerics?
- Did I check relevant special cases?
- Did I search for counterexamples where appropriate?
- Did I avoid confusing an optimizer's best value with a proven global optimum?
- Did I preserve affine invariance when expected?
- Are numerical results reproducible?
- Did I update durable documentation only when justified?
- Did I avoid modifying unrelated user work?
- Is the implementation easy for a human reader to audit?
- Did I avoid unnecessary Python loops and repeated decompositions?
- Did I use optimized NumPy/SciPy routines where appropriate?
- Does every function have a NumPy-style docstring?
- Does this work advance the current `d_F` versus `d_0` comparison objective?
- Did I state remaining uncertainty explicitly?

If any answer is no, address it before presenting the result as complete.
