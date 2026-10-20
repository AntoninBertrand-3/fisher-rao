import numpy as np
from scipy.optimize import minimize


def dist2_L2(mu0: np.ndarray, Sigma0: np.ndarray, mu1: np.ndarray, Sigma1: np.ndarray) -> float:
    """
    Squared Euclidean distance between two MVNs.

    Parameters
    ----------
    mu0, mu1 : ndarray
        Means of shape (n_features,).
    Sigma0, Sigma1 : ndarray
        SPD covariance matrices of shape (n_features, n_features).

    Returns
    -------
    float
        ||mu0 - mu1||^2 + ||Sigma0 - Sigma1||_F^2.
    """
    return np.sum((mu0 - mu1)**2) + np.sum((Sigma0 - Sigma1)**2)


def dist2_Fisher(A: np.ndarray, B: np.ndarray) -> float:
    """
    Squared affine-invariant (Fisher) distance between two SPD matrices.

    Parameters
    ----------
    A, B : ndarray
        SPD matrices of shape (n_features, n_features).

    Returns
    -------
    float
        Sum of the squared logarithms of the eigenvalues of B^{-1} A.
    """
    iL = np.linalg.inv(np.linalg.cholesky(B))
    return np.sum(np.log(np.linalg.eigvalsh(iL @ A @ iL.T))**2)


def dist2_d0(mu0: np.ndarray, Sigma0: np.ndarray, mu1: np.ndarray, Sigma1: np.ndarray) -> float:
    """
    Squared zeroth approximation d_0 of the Fisher-Rao distance between two MVNs.

    The pair is first normalized so that the first MVN is N(0, I):
    mu = L^{-1}(mu1 - mu0) and S = L^{-1} Sigma1 L^{-T}, with Sigma0 = L L^T. Then
    d_0^2 = 2 sum_i arccosh^2(c_i), where c_i are the singular values of
    S^{-1/2} Y and Y = (I + S + mu mu^T / 2) / 2.

    Parameters
    ----------
    mu0, mu1 : ndarray
        Means of shape (n_features,).
    Sigma0, Sigma1 : ndarray
        SPD covariance matrices of shape (n_features, n_features).

    Returns
    -------
    float
        Squared d_0 distance between N(mu0, Sigma0) and N(mu1, Sigma1).
    """
    iL = np.linalg.inv(np.linalg.cholesky(Sigma0))
    mu = iL @ (mu1 - mu0)
    S = iL @ Sigma1 @ iL.T
    Y = 0.5 * (np.eye(mu.size) + S + 0.5 * np.outer(mu, mu))
    iLS = np.linalg.inv(np.linalg.cholesky(S))
    c = np.linalg.svd(iLS @ Y, compute_uv=False)
    return 2 * np.sum(np.arccosh(np.maximum(c, 1.0))**2)


def _skew_objective(params: np.ndarray, W: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Objective J(K) of the reduced Fisher-Rao problem and its gradient.

    Parameters
    ----------
    params : ndarray
        Upper-triangular entries of the skew-symmetric matrix K,
        of shape (n_features * (n_features - 1) / 2,).
    W : ndarray
        Whitening matrix with W^T W = S^{-1}, of shape (n_features, n_features).
    Y : ndarray
        Matrix Y(0) of the normalized pair, of shape (n_features, n_features).

    Returns
    -------
    float
        J(K) = 2 sum_i arccosh^2(c_i), c_i singular values of W (Y + K).
    ndarray
        Gradient of J with respect to params.
    """
    n = Y.shape[0]
    rows, cols = np.triu_indices(n, 1)
    K = np.zeros((n, n))
    K[rows, cols] = params
    K[cols, rows] = -params
    U, c, Vt = np.linalg.svd(W @ (Y + K))
    c = np.maximum(c, 1.0)
    a = np.arccosh(c)
    # d(2 arccosh^2(c))/dc tends to 4 when c tends to 1.
    weights = np.full_like(c, 4.0)
    np.divide(4 * a, np.sqrt(c**2 - 1), out=weights, where=c > 1)
    G = W.T @ (U * weights) @ Vt
    return 2 * np.sum(a**2), G[rows, cols] - G[cols, rows]


def dist2_dF(mu0: np.ndarray, Sigma0: np.ndarray, mu1: np.ndarray, Sigma1: np.ndarray,
             tol: float = 1e-10, max_iter: int = 1000) -> float:
    """
    Squared Fisher-Rao distance between two MVNs, computed by optimization.

    The pair is normalized as in dist2_d0, then
    d_F^2 = min over skew-symmetric K of 2 sum_i arccosh^2(c_i(K)), where
    c_i(K) are the singular values of S^{-1/2} (Y + K). The minimization
    uses BFGS started at K = 0, so that d_F <= d_0; the result is a local
    minimum.

    Parameters
    ----------
    mu0, mu1 : ndarray
        Means of shape (n_features,).
    Sigma0, Sigma1 : ndarray
        SPD covariance matrices of shape (n_features, n_features).
    tol : float, optional
        Gradient tolerance of BFGS (default 1e-10).
    max_iter : int, optional
        Maximum number of BFGS iterations (default 1000).

    Returns
    -------
    float
        Squared Fisher-Rao distance between N(mu0, Sigma0) and N(mu1, Sigma1).
    """
    iL = np.linalg.inv(np.linalg.cholesky(Sigma0))
    mu = iL @ (mu1 - mu0)
    S = iL @ Sigma1 @ iL.T
    Y = 0.5 * (np.eye(mu.size) + S + 0.5 * np.outer(mu, mu))
    W = np.linalg.inv(np.linalg.cholesky(S))
    if mu.size == 1:
        return _skew_objective(np.zeros(0), W, Y)[0]
    result = minimize(_skew_objective, np.zeros(mu.size * (mu.size - 1) // 2), args=(W, Y),
                      jac=True, method='BFGS', options=dict(gtol=tol, maxiter=max_iter))
    return result.fun
