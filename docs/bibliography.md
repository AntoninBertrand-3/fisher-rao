# Bibliography and reading map

Last updated: 2026-09-05.

This file is not just a citation list. Each entry states why it matters to the current project.

---

## Core references

### Skovgaard (1981)

L. T. Skovgaard, **A Riemannian Geometry of the Multivariate Normal Model**, Stanford University Department of Statistics, Technical Report EFS NSF 167, 1981.

Role in project:

- classical differential geometry of the MVN Fisher manifold;
- source for the geodesic equations used by later papers;
- useful for checking conventions and special submanifolds.

Stanford record:
https://statistics.stanford.edu/technical-reports/riemannian-geometry-multivariate-normal-model

---

### Eriksen (1986/1987)

P. S. Eriksen, **Geodesics Connected with the Fisher Metric on the Multivariate Normal Manifold**, Technical Report 86-13, Institute of Electronic Systems, Aalborg University, 1986.

Conference version:

P. S. Eriksen, **Geodesics connected with the Fisher metric on the multivariate normal manifold**, Proceedings of the GST Workshop, Lancaster, 1987, pp. 225–229.

Role in project:

- early explicit construction of Fisher geodesics for MVNs;
- fundamental source for the IVP formulation;
- repeatedly cited by Kobayashi and later numerical work.

---

### Calvo & Oller (1990)

M. Calvo and J. M. Oller, **A distance between multivariate normal distributions based in an embedding into the Siegel group**, *Journal of Multivariate Analysis* 35(2), 223–242, 1990.

DOI:
https://doi.org/10.1016/0047-259X(90)90026-E

Role in project:

- isometric embedding of the Gaussian Fisher manifold into a higher-dimensional SPD/Siegel geometry;
- produces a tractable lower bound because the embedded Gaussian submanifold is not totally geodesic;
- important alternative to the Kobayashi construction;
- baseline distance for applications.

---

### Calvo & Oller (1991)

M. Calvo and J. M. Oller, **An Explicit Solution of Information Geodesic Equations for the Multivariate Normal Model**, *Statistics & Risk Modeling* 9(1–2), 119–138, 1991.

DOI:
https://doi.org/10.1524/strm.1991.9.12.119

Role in project:

- explicit solution of the Fisher geodesic initial-value equations;
- reinforces the key distinction:
  IVP is explicit, BVP is the difficult part.

---

### Kobayashi (2023)

Shimpei Kobayashi, **Geodesics of multivariate normal distributions and a Toda lattice type Lax pair**, *Physica Scripta* 98(11), 115241, 2023.

DOI:
https://doi.org/10.1088/1402-4896/ad0087

arXiv:
https://arxiv.org/abs/2304.12575

Role in project:

- central geometric reference;
- reinterprets the MVN Fisher geodesic construction using block Cholesky decomposition;
- introduces the natural Riemannian submersion
  \[
  \pi:\mathcal M\to\mathcal N;
  \]
- realizes \(\mathcal M\) as a totally geodesic symmetric subspace inside determinant-one `SPD(2n+1)`;
- proves that Fisher geodesics are projections of horizontal geodesics;
- conceptual basis for the project's endpoint-fiber / skew-optimization approach.

Most relevant parts:

- equations (2.1)–(2.5): Fisher geodesics and block Cholesky construction;
- Section 2.2: Riemannian submersion;
- Theorem A: every Fisher geodesic is the projection of a horizontal geodesic.

---

## Reviews / special cases / numerical baselines

### Pinele, Strapasson & Costa (2020)

J. Pinele, J. E. Strapasson and S. I. R. Costa, **The Fisher–Rao Distance between Multivariate Normal Distributions: Special Cases, Bounds and Applications**, *Entropy* 22(4), 404, 2020.

DOI:
https://doi.org/10.3390/e22040404

Role in project:

- convenient review of known special cases and bounds;
- emphasizes that a general closed form is unavailable;
- useful source for affine invariance and closed forms on special configurations;
- application perspective for Gaussian comparison.

---

### Nielsen (2023), approximation by tractable curves

Frank Nielsen, **A Simple Approximation Method for the Fisher–Rao Distance between Multivariate Normal Distributions**, *Entropy* 25(4), 654, 2023.

DOI:
https://doi.org/10.3390/e25040654

arXiv:
https://arxiv.org/abs/2302.08175

Role in project:

- clearly states the IVP/BVP distinction;
- approximates Fisher lengths by discretizing tractable curves;
- uses local approximation by the square root of Jeffreys divergence;
- compares ordinary, natural, expectation-parameter and Calvo–Oller-derived curves;
- useful benchmark against the Kobayashi-based approach;
- contains a useful overview of triangle bounds and SPD embeddings.

---

### Nielsen (2023), pullback Hilbert cone distance

Frank Nielsen, **Fisher-Rao and pullback Hilbert cone distances on the multivariate Gaussian manifold with applications to simplification and quantization of mixtures**, Proceedings of the 2nd Annual Workshop on Topology, Algebra, and Geometry in Machine Learning, PMLR 221, 488–504, 2023.

Paper:
https://proceedings.mlr.press/v221/nielsen23b.html

Related arXiv:
https://arxiv.org/abs/2307.10644

Role in project:

- alternative computationally cheap distances on embedded Gaussian manifolds;
- relevant for application benchmarking;
- illustrates the difference between exact Fisher geometry and useful pullback metrics.

---

### Nielsen (2024), general approximation/bounding review

Frank Nielsen, **Approximation and bounding techniques for the Fisher-Rao distances between parametric statistical models**, *Handbook of Statistics*, Vol. 51, 67–116, 2024.

DOI:
https://doi.org/10.1016/bs.host.2024.06.003

arXiv:
https://arxiv.org/abs/2403.10089

Role in project:

- general toolbox for bounds and approximations;
- useful context for positioning any new \(d_0/d_m\) approximation theorem;
- useful when deciding what kind of theoretical guarantee would be meaningful.

---

## Related symmetric-space geometry

### Lovrić, Min-Oo & Ruh (2000)

M. Lovrić, M. Min-Oo and E. A. Ruh, **Multivariate Normal Distributions Parametrized as a Riemannian Symmetric Space**, *Journal of Multivariate Analysis* 74(1), 36–48, 2000.

DOI:
https://doi.org/10.1006/jmva.1999.1853

Role in project:

- studies a symmetric-space parametrization of MVNs;
- uses a metric different from the Fisher metric;
- important warning that a convenient symmetric-space geometry can approximate or resemble Fisher geometry without being Fisher-isometric.

---

### Globke & Quiroga-Barranco

W. Globke and R. Quiroga-Barranco, **Information geometry and asymptotic geodesics on the space of normal distributions**.

arXiv:
https://arxiv.org/abs/1904.12921

Role in project:

- compares Fisher geometry with the symmetric-space/Killing-metric viewpoint;
- useful background for understanding when alternative symmetric metrics approximate Fisher geometry.

---

## General information geometry

### Amari & Nagaoka

S.-I. Amari and H. Nagaoka, **Methods of Information Geometry**, AMS/Oxford University Press, 2000.

Role:

- general Fisher geometry, dual connections, invariance principles;
- background reference, not the main technical source for the BVP.

---

### Amari

S.-I. Amari, **Information Geometry and Its Applications**, Springer, 2016.

Role:

- modern general reference for information geometry;
- useful for terminology and conceptual framing.

---

## Project-specific alternative: geodesic triangle with geometric-mean covariance

The project currently studies a modification of a triangle-based Gaussian distance approximation.

The original triangle idea is literature-backed: connect endpoint Gaussians by tractable legs that change covariance and mean separately, and use the triangle inequality / tractable submodel distances.

The project-specific modification is:

\[
\Sigma_{\mathrm{GM}}
=
\Sigma_1\#\Sigma_2
=
\Sigma_1^{1/2}
\left(
\Sigma_1^{-1/2}
\Sigma_2
\Sigma_1^{-1/2}
\right)^{1/2}
\Sigma_1^{1/2},
\]

and replace the single endpoint covariance used to weight the mean difference in the chosen triangle formula by \(\Sigma_{\mathrm{GM}}\).

Status:

- **not** currently a published Fisher–Rao identity;
- treat as a proposed symmetric heuristic;
- the exact original `Delta_c` formula used in the slides should be copied into the repository once its source/notation is reconciled, rather than reconstructed from memory.

---

## Bibliographic item still to resolve

Earlier presentation notes referred informally to a “Bouchard” / “Ces Distribution F. Bouchard” reference.

Its exact bibliographic identity has not yet been recovered reliably.

Do not invent a citation. When the PDF or title is found, add:

- full authors;
- exact title;
- year/journal;
- DOI/arXiv;
- which formula/result is used in the project.

---

## Recommended reading order for a new agent

1. `docs/notation.md`.
2. Kobayashi (2023), especially Sections 2.1–2.2.
3. Calvo & Oller (1990, 1991).
4. Nielsen (2023, Entropy) for the modern statement of the BVP difficulty and numerical alternatives.
5. Pinele et al. (2020) for special cases and bounds.
6. `docs/state.md` for the project's own reduced optimization and \(d_m\) program.

---

## Citation discipline

When an agent makes a literature claim, it should record:

- exact source;
- theorem/equation/section when practical;
- whether the source proves an identity, bound, approximation, or only reports numerical evidence.

Do not cite secondary summaries for a theorem when the primary source is available.