from typing import Callable

import numpy as np
from scipy.optimize import OptimizeResult, minimize


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
    mu = L^{-1}(mu1 - mu0) and Sigma = L^{-1} Sigma1 L^{-T}, with Sigma0 = L L^T. Then
    d_0^2 = 2 sum_i arccosh^2(c_i), where c_i are the singular values of
    Sigma^{-1/2} Y and Y = (I + Sigma + mu mu^T / 2) / 2.

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
    Sigma = iL @ Sigma1 @ iL.T
    Y = 0.5 * (np.eye(mu.size) + Sigma + 0.5 * np.outer(mu, mu))
    iL_Sigma = np.linalg.inv(np.linalg.cholesky(Sigma))
    c = np.linalg.svd(iL_Sigma @ Y, compute_uv=False)
    return 2 * np.sum(np.arccosh(np.maximum(c, 1.0))**2)


def _reduced_problem(mu0: np.ndarray, Sigma0: np.ndarray, mu1: np.ndarray,
                     Sigma1: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalizes a pair of MVNs and returns the fixed matrices of J(K).

    The pair is normalized so that the first MVN is N(0, I), as in dist2_d0.

    Parameters
    ----------
    mu0, mu1 : ndarray
        Means of shape (n_features,).
    Sigma0, Sigma1 : ndarray
        SPD covariance matrices of shape (n_features, n_features).

    Returns
    -------
    W : ndarray
        Whitening matrix with W^T W = Sigma^{-1}, of shape (n_features, n_features).
    Y : ndarray
        Matrix Y(0) = (I + Sigma + mu mu^T / 2) / 2 of the normalized pair,
        of shape (n_features, n_features).
    """
    iL = np.linalg.inv(np.linalg.cholesky(Sigma0))
    mu = iL @ (mu1 - mu0)
    Sigma = iL @ Sigma1 @ iL.T
    Y = 0.5 * (np.eye(mu.size) + Sigma + 0.5 * np.outer(mu, mu))
    W = np.linalg.inv(np.linalg.cholesky(Sigma))
    return W, Y


def _skew_from_params(params: np.ndarray, n_features: int) -> np.ndarray:
    """
    Builds the skew-symmetric matrix K from its upper-triangular entries.

    Parameters
    ----------
    params : ndarray
        Upper-triangular entries of K, of shape (n_features * (n_features - 1) / 2,).
    n_features : int
        Size of K.

    Returns
    -------
    ndarray
        Skew-symmetric matrix K of shape (n_features, n_features).
    """
    rows, cols = np.triu_indices(n_features, 1)
    K = np.zeros((n_features, n_features))
    K[rows, cols] = params
    K[cols, rows] = -params
    return K


def _skew_objective(params: np.ndarray, W: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray]:
    """
    Objective J(K) of the reduced Fisher-Rao problem and its gradient.

    Parameters
    ----------
    params : ndarray
        Upper-triangular entries of the skew-symmetric matrix K,
        of shape (n_features * (n_features - 1) / 2,).
    W : ndarray
        Whitening matrix with W^T W = Sigma^{-1}, of shape (n_features, n_features).
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
    K = _skew_from_params(params, n)
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
    c_i(K) are the singular values of Sigma^{-1/2} (Y + K). The minimization
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
    W, Y = _reduced_problem(mu0, Sigma0, mu1, Sigma1)
    n = Y.shape[0]
    if n == 1:
        return _skew_objective(np.zeros(0), W, Y)[0]
    return _minimize_J(W, Y, np.zeros(n * (n - 1) // 2), tol, max_iter).fun


def _minimize_J(W: np.ndarray, Y: np.ndarray, params_init: np.ndarray, tol: float,
                max_iter: int, callback: Callable | None = None) -> OptimizeResult:
    """
    Minimizes J(K) over skew-symmetric K with BFGS and the analytic gradient.

    Parameters
    ----------
    W : ndarray
        Whitening matrix with W^T W = Sigma^{-1}, of shape (n_features, n_features).
    Y : ndarray
        Matrix Y(0) of the normalized pair, of shape (n_features, n_features).
    params_init : ndarray
        Upper-triangular entries of the initial K,
        of shape (n_features * (n_features - 1) / 2,).
    tol : float
        Gradient tolerance of BFGS.
    max_iter : int
        Maximum number of BFGS iterations.
    callback : callable, optional
        SciPy callback called after each BFGS iteration.

    Returns
    -------
    OptimizeResult
        SciPy result; fun is J at the final iterate x.
    """
    return minimize(_skew_objective, params_init, args=(W, Y), jac=True, method='BFGS',
                    callback=callback, options=dict(gtol=tol, maxiter=max_iter))


def dist2_dF_with_minimizer(mu0: np.ndarray, Sigma0: np.ndarray, mu1: np.ndarray,
                            Sigma1: np.ndarray, K_init: np.ndarray | None = None,
                            tol: float = 1e-10, max_iter: int = 1000,
                            ) -> tuple[float, np.ndarray]:
    """
    Squared Fisher-Rao distance as in dist2_dF, also returning the minimizer K*.

    K* is expressed in the coordinates of the pair normalized by the Cholesky
    factor of Sigma0 (see dist2_d0). Starting from K_init instead of K = 0 is
    useful as a warm start when the pair changes slightly; the bound
    d_F <= d_0 is then no longer guaranteed by construction.

    Parameters
    ----------
    mu0, mu1 : ndarray
        Means of shape (n_features,).
    Sigma0, Sigma1 : ndarray
        SPD covariance matrices of shape (n_features, n_features).
    K_init : ndarray, optional
        Initial skew-symmetric matrix of shape (n_features, n_features),
        by default 0.
    tol : float, optional
        Gradient tolerance of BFGS (default 1e-10).
    max_iter : int, optional
        Maximum number of BFGS iterations (default 1000).

    Returns
    -------
    dist2 : float
        Squared Fisher-Rao distance J(K*).
    K : ndarray
        Skew-symmetric minimizer K* of shape (n_features, n_features).
    """
    W, Y = _reduced_problem(mu0, Sigma0, mu1, Sigma1)
    n = Y.shape[0]
    if n == 1:
        return _skew_objective(np.zeros(0), W, Y)[0], np.zeros((1, 1))
    rows, cols = np.triu_indices(n, 1)
    params_init = np.zeros(rows.size) if K_init is None else K_init[rows, cols]
    result = _minimize_J(W, Y, params_init, tol, max_iter)
    return result.fun, _skew_from_params(result.x, n)


def dist2_dF_iterates(mu0: np.ndarray, Sigma0: np.ndarray, mu1: np.ndarray, Sigma1: np.ndarray,
                      tol: float = 1e-10, max_iter: int = 1000,
                      ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Squared Fisher-Rao distance as in dist2_dF, also returning the BFGS iterates.

    The iterates K_0 = 0, K_1, ..., K_T are those of the BFGS minimization of
    J(K) used by dist2_dF, so that J(K_0) = d_0^2 and J(K_T) = d_F^2.

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
    dist2 : float
        Squared Fisher-Rao distance J(K_T).
    K_iterates : ndarray
        Skew-symmetric iterates K_k, of shape (T + 1, n_features, n_features).
    J_values : ndarray
        Objective values J(K_k), of shape (T + 1,).
    grad_norms : ndarray
        Sup norm of the gradient of J with respect to the upper-triangular
        entries of K_k (the quantity tested against tol by BFGS), of shape (T + 1,).
    """
    W, Y = _reduced_problem(mu0, Sigma0, mu1, Sigma1)
    n = Y.shape[0]
    params_iterates = [np.zeros(n * (n - 1) // 2)]

    def record_iterate(intermediate_result) -> None:
        """
        Stores the current BFGS iterate (SciPy callback).

        Parameters
        ----------
        intermediate_result : OptimizeResult
            Current state of the optimizer, with the iterate in its attribute x.
        """
        params_iterates.append(intermediate_result.x.copy())

    if n > 1:
        _minimize_J(W, Y, params_iterates[0], tol, max_iter, callback=record_iterate)
    objectives = [_skew_objective(params, W, Y) for params in params_iterates]
    J_values = np.array([value for value, _ in objectives])
    grad_norms = np.array([np.max(np.abs(grad), initial=0.0) for _, grad in objectives])
    K_iterates = np.stack([_skew_from_params(params, n) for params in params_iterates])
    return J_values[-1], K_iterates, J_values, grad_norms
