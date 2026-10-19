from typing import Callable

import numpy as np
from scipy.optimize import minimize


def arithmetic_mean(means: np.ndarray, covariances: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the arithmetic mean of a set of MVNs, parameter by parameter.

    Parameters
    ----------
    means : ndarray
        Means of shape (n_mvn, n_features).
    covariances : ndarray
        SPD covariances of shape (n_mvn, n_features, n_features).

    Returns
    -------
    mean : ndarray
        Arithmetic mean of the means, of shape (n_features,).
    covariance : ndarray
        Arithmetic mean of the covariances, of shape (n_features, n_features).
    """
    return means.mean(axis=0), covariances.mean(axis=0)


def geometric_mean(
    matrices: np.ndarray,
    init: np.ndarray | None = None,
    max_iter: int = 50,
    tol: float = 1e-6,
) -> np.ndarray:
    """
    Computes the geometric mean of a set of SPD matrices under the
    affine-invariant Riemannian metric (fixed-point algorithm).

    Parameters
    ----------
    matrices : ndarray
        Set of SPD matrices of shape (n_matrices, n_features, n_features).
    init : ndarray, optional
        Initial SPD matrix of shape (n_features, n_features),
        by default the arithmetic mean.
    max_iter : int, optional
        Maximum number of iterations (default 50).
    tol : float, optional
        Relative tolerance on the Frobenius norm for the stopping criterion
        (default 1e-6).

    Returns
    -------
    ndarray
        Estimated geometric mean of shape (n_features, n_features).
    """
    M = matrices.mean(axis=0) if init is None else init
    for _ in range(max_iter):
        L = np.linalg.cholesky(M)
        iL = np.linalg.inv(L)
        # Mean of the logarithms of the whitened matrices L^{-1} C_k L^{-T}
        vals, vecs = np.linalg.eigh(iL @ matrices @ iL.T)
        S = np.mean((vecs * np.log(vals)[:, None, :]) @ vecs.swapaxes(-1, -2), axis=0)
        lam, U = np.linalg.eigh(0.5 * (S + S.T))
        M_new = L @ ((U * np.exp(lam)) @ U.T) @ L.T
        if np.linalg.norm(M_new - M, 'fro') < tol * np.linalg.norm(M, 'fro'):
            return M_new
        M = M_new
    return M


def _unpack(params: np.ndarray, n_features: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Builds an MVN from unconstrained parameters.

    Parameters
    ----------
    params : ndarray
        Mean, then lower-triangular entries of the Cholesky factor of the
        covariance with a logarithmic diagonal, of shape
        (n_features + n_features * (n_features + 1) / 2,).
    n_features : int
        Dimension of the MVN.

    Returns
    -------
    mean : ndarray
        Mean of shape (n_features,).
    covariance : ndarray
        SPD covariance of shape (n_features, n_features).
    """
    rows, cols = np.tril_indices(n_features)
    L = np.zeros((n_features, n_features))
    L[rows, cols] = params[n_features:]
    L[np.diag_indices(n_features)] = np.exp(np.diag(L))
    return params[:n_features], L @ L.T


def _pack(mean: np.ndarray, covariance: np.ndarray) -> np.ndarray:
    """
    Inverse of _unpack.

    Parameters
    ----------
    mean : ndarray
        Mean of shape (n_features,).
    covariance : ndarray
        SPD covariance of shape (n_features, n_features).

    Returns
    -------
    ndarray
        Unconstrained parameters, see _unpack.
    """
    L = np.linalg.cholesky(covariance)
    L[np.diag_indices(mean.size)] = np.log(np.diag(L))
    return np.concatenate([mean, L[np.tril_indices(mean.size)]])


def frechet_mean(
    means: np.ndarray,
    covariances: np.ndarray,
    dist2: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float],
    init: tuple[np.ndarray, np.ndarray] | None = None,
    tol: float = 1e-12,
    max_iter: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the Frechet mean of a set of MVNs for a given squared distance,
    i.e. the minimizer of sum_k dist2(mean, covariance, mu_k, C_k).

    The covariance is parametrized by its Cholesky factor with a logarithmic
    diagonal so that it stays SPD; the minimization uses L-BFGS-B with
    finite-difference gradients. Use dist2_d0 or dist2_dF from
    src.distance for the geometric means.

    Parameters
    ----------
    means : ndarray
        Means of shape (n_mvn, n_features).
    covariances : ndarray
        SPD covariances of shape (n_mvn, n_features, n_features).
    dist2 : callable
        Squared distance dist2(mu0, C0, mu1, C1) between two MVNs.
    init : tuple of ndarray, optional
        Initial (mean, covariance), by default the arithmetic mean.
    tol : float, optional
        Tolerance of L-BFGS-B on the objective (default 1e-12). Because
        the gradient is computed by finite differences, the attainable
        accuracy on the parameters is about 1e-7.
    max_iter : int, optional
        Maximum number of L-BFGS-B iterations (default 500).

    Returns
    -------
    mean : ndarray
        Frechet mean of the means, of shape (n_features,).
    covariance : ndarray
        Frechet mean of the covariances, of shape (n_features, n_features).
    """
    n_features = means.shape[1]
    if init is None:
        init = arithmetic_mean(means, covariances)

    def objective(params: np.ndarray) -> float:
        mean, covariance = _unpack(params, n_features)
        return sum(dist2(mean, covariance, mu, C) for mu, C in zip(means, covariances))

    result = minimize(objective, _pack(*init), method='L-BFGS-B',
                      options=dict(ftol=tol, gtol=tol, maxiter=max_iter))
    return _unpack(result.x, n_features)
