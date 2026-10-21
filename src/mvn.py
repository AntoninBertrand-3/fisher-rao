from typing import Callable

import numpy as np
from scipy.linalg import solve_triangular

from src.distance import dist2_dF_with_minimizer


def _symmetric_function(X: np.ndarray, f: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
    """
    Applies a scalar function to a symmetric matrix through its eigendecomposition.

    Parameters
    ----------
    X : ndarray
        Symmetric matrix of shape (m, m).
    f : callable
        Vectorized scalar function, applied to the eigenvalues of X.

    Returns
    -------
    ndarray
        Symmetric matrix f(X) of shape (m, m).
    """
    vals, vecs = np.linalg.eigh(0.5 * (X + X.T))
    return (vecs * f(vals)) @ vecs.T


def _whiten(L: np.ndarray, v: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Applies the affine map x -> L^{-1} x to a vector and a symmetric matrix.

    Parameters
    ----------
    L : ndarray
        Lower-triangular Cholesky factor of shape (n_features, n_features).
    v : ndarray
        Vector of shape (n_features,).
    T : ndarray
        Symmetric matrix of shape (n_features, n_features).

    Returns
    -------
    ndarray
        L^{-1} v, of shape (n_features,).
    ndarray
        L^{-1} T L^{-T}, of shape (n_features, n_features).
    """
    a = solve_triangular(L, v, lower=True)
    A = solve_triangular(L, solve_triangular(L, T, lower=True).T, lower=True)
    return a, 0.5 * (A + A.T)


def _lifted_endpoint(nu: np.ndarray, C: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Lift G_1(S) of the normalized endpoint N(nu, C) in the Kobayashi space.

    G_1(S) = M D M^T with D = diag(C^{-1}, 1, C) and
    M = [[I, 0, 0], [nu^T, 1, 0], [S - nu nu^T / 2, -nu, I]] (notation.md, section 7).
    Every skew-symmetric S gives a lift of the same Gaussian.

    Parameters
    ----------
    nu : ndarray
        Normalized mean of shape (n_features,).
    C : ndarray
        Normalized SPD covariance of shape (n_features, n_features).
    S : ndarray
        Skew-symmetric fiber parameter of shape (n_features, n_features).

    Returns
    -------
    ndarray
        SPD matrix G_1(S) of shape (2 n_features + 1, 2 n_features + 1).
    """
    n = nu.size
    M = np.eye(2 * n + 1)
    M[n, :n] = nu
    M[n + 1:, :n] = S - 0.5 * np.outer(nu, nu)
    M[n + 1:, n] = -nu
    D = np.zeros((2 * n + 1, 2 * n + 1))
    D[:n, :n] = np.linalg.inv(C)
    D[n, n] = 1.0
    D[n + 1:, n + 1:] = C
    return M @ D @ M.T


def _horizontal_generator(a: np.ndarray, A: np.ndarray) -> np.ndarray:
    """
    Horizontal generator V(A, a) at the identity of the Kobayashi space.

    V = [[-A, a, 0], [a^T, 0, -a^T], [0, -a, A]] (notation.md, section 6);
    the projection of exp(t V) is the Fisher geodesic leaving N(0, I) with
    velocity (a, A).

    Parameters
    ----------
    a : ndarray
        Mean velocity of shape (n_features,).
    A : ndarray
        Symmetric covariance velocity of shape (n_features, n_features).

    Returns
    -------
    ndarray
        Symmetric matrix V of shape (2 n_features + 1, 2 n_features + 1).
    """
    n = a.size
    V = np.zeros((2 * n + 1, 2 * n + 1))
    V[:n, :n] = -A
    V[:n, n] = a
    V[n, :n] = a
    V[n, n + 1:] = -a
    V[n + 1:, n] = -a
    V[n + 1:, n + 1:] = A
    return V


def _project(G: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Projects a point of the Kobayashi space onto a Gaussian.

    With Theta = G[:n, :n] and delta = G[:n, n], the Gaussian is
    Sigma = Theta^{-1}, mu = Theta^{-1} delta.

    Parameters
    ----------
    G : ndarray
        SPD matrix of shape (2 n_features + 1, 2 n_features + 1).
    n_features : int
        Dimension of the Gaussian.

    Returns
    -------
    mu : ndarray
        Mean of shape (n_features,).
    Sigma : ndarray
        SPD covariance of shape (n_features, n_features).
    """
    Theta, delta = G[:n_features, :n_features], G[:n_features, n_features]
    solution = np.linalg.solve(Theta, np.column_stack([np.eye(n_features), delta]))
    Sigma = solution[:, :n_features]
    return solution[:, n_features], 0.5 * (Sigma + Sigma.T)


def fisher_norm2(Sigma: np.ndarray, v: np.ndarray, T: np.ndarray) -> float:
    """
    Squared Fisher norm of a tangent vector (v, T) at a Gaussian of covariance Sigma.

    g((v, T), (v, T)) = v^T Sigma^{-1} v + 1/2 Tr((Sigma^{-1} T)^2), the
    canonical metric of notation.md; it does not depend on the mean.

    Parameters
    ----------
    Sigma : ndarray
        SPD covariance of the base point, of shape (n_features, n_features).
    v : ndarray
        Mean velocity of shape (n_features,).
    T : ndarray
        Symmetric covariance velocity of shape (n_features, n_features).

    Returns
    -------
    float
        Squared Fisher norm of (v, T).
    """
    a, A = _whiten(np.linalg.cholesky(Sigma), v, T)
    return a @ a + 0.5 * np.sum(A**2)


def exp_map_mvn(mu: np.ndarray, Sigma: np.ndarray, v: np.ndarray, T: np.ndarray,
                t: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Fisher-Rao exponential map: point at time t of the geodesic from N(mu, Sigma)
    with initial velocity (v, T).

    With Sigma = L L^T, the velocity is whitened to (a, A) = (L^{-1} v, L^{-1} T L^{-T}),
    the geodesic from N(0, I) is the projection of exp(t V(A, a)), and the
    result is mapped back by x -> L x + mu.

    Parameters
    ----------
    mu : ndarray
        Base mean of shape (n_features,).
    Sigma : ndarray
        Base SPD covariance of shape (n_features, n_features).
    v : ndarray
        Mean velocity of shape (n_features,).
    T : ndarray
        Symmetric covariance velocity of shape (n_features, n_features).
    t : float, optional
        Time along the geodesic (default 1).

    Returns
    -------
    mean : ndarray
        Mean of the endpoint, of shape (n_features,).
    covariance : ndarray
        SPD covariance of the endpoint, of shape (n_features, n_features).
    """
    L = np.linalg.cholesky(Sigma)
    a, A = _whiten(L, v, T)
    G = _symmetric_function(t * _horizontal_generator(a, A), np.exp)
    nu, C = _project(G, mu.size)
    covariance = L @ C @ L.T
    return mu + L @ nu, 0.5 * (covariance + covariance.T)


def log_map_mvn(mu0: np.ndarray, Sigma0: np.ndarray, mu1: np.ndarray, Sigma1: np.ndarray,
                K_init: np.ndarray | None = None, tol: float = 1e-10, max_iter: int = 1000,
                return_K: bool = False,
                ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fisher-Rao logarithm map: initial velocity at N(mu0, Sigma0) of the geodesic
    reaching N(mu1, Sigma1) at time 1.

    The pair is normalized to N(0, I), N(nu, C) with Sigma0 = L L^T, and K* is
    the BFGS minimizer of J(K) (dist2_dF_with_minimizer). The lift G_1(S) with
    S = 2 K* is the closest point of the fiber over N(nu, C); its logarithm
    H = log G_1(2 K*) is horizontal and gives the normalized velocity
    (a, A) = (H[:n, n], -H[:n, :n]), mapped back to (L a, L A L^T).

    The relation S = 2K between the fiber parameter and the skew variable of J
    is a numerical observation (checked in test_mvn), not yet derived. If K* is
    only a local minimizer of J, the result is the velocity of a geodesic
    reaching the endpoint that need not be minimizing.

    Parameters
    ----------
    mu0, mu1 : ndarray
        Means of shape (n_features,).
    Sigma0, Sigma1 : ndarray
        SPD covariance matrices of shape (n_features, n_features).
    K_init : ndarray, optional
        Initial skew-symmetric matrix for BFGS, of shape (n_features, n_features),
        by default 0.
    tol : float, optional
        Gradient tolerance of BFGS (default 1e-10).
    max_iter : int, optional
        Maximum number of BFGS iterations (default 1000).
    return_K : bool, optional
        If True, also returns K* (for warm starts), by default False.

    Returns
    -------
    v : ndarray
        Mean velocity of shape (n_features,).
    T : ndarray
        Symmetric covariance velocity of shape (n_features, n_features).
    K : ndarray
        Minimizer K* of shape (n_features, n_features), only if return_K is True.
    """
    _, K = dist2_dF_with_minimizer(mu0, Sigma0, mu1, Sigma1, K_init=K_init,
                                   tol=tol, max_iter=max_iter)
    L = np.linalg.cholesky(Sigma0)
    nu, C = _whiten(L, mu1 - mu0, Sigma1)
    n = mu0.size
    H = _symmetric_function(_lifted_endpoint(nu, C, 2 * K), np.log)
    a, A = H[:n, n], -H[:n, :n]
    v, T = L @ a, L @ A @ L.T
    if return_K:
        return v, 0.5 * (T + T.T), K
    return v, 0.5 * (T + T.T)
