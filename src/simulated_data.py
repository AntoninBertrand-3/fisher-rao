import numpy as np
from scipy.stats import ortho_group, uniform


def sym(X: np.ndarray) -> np.ndarray:
    """
    Returns the symmetrical part of its argument, i.e., :math: `sym(X)=(X+X^{T})/2`.

    Parameters
    ----------
    X : ndarray
        square matrices of shape (..., n_features, n_features).

    Returns
    -------
    ndarray
        symmetrical part of X, same shape as X.
    """
    return 0.5 * (X + X.swapaxes(-1, -2))


def random_SPD(n_features: int, condition_number: float = 10.0) -> np.ndarray:
    """
    Generate a random symmetric positive definite (SPD) matrix
    with a prescribed condition number.

    Parameters
    ----------
    n_features : int
        Dimension of the SPD matrix.
    condition_number : float, default=10.0
        Desired condition number (ratio of max to min eigenvalue).

    Returns
    -------
    ndarray
        Randomly generated SPD matrix, shape (n_features, n_features).
    """
    if condition_number < 1:
        raise ValueError("condition_number must be >= 1")
    if n_features == 1:
        return np.array([[1.0]])

    # Random orthogonal matrix
    U = ortho_group.rvs(dim=n_features)

    # Random eigenvalues in [1/sqrt(kappa), sqrt(kappa)], extremes included
    lam_min, lam_max = 1.0 / np.sqrt(condition_number), np.sqrt(condition_number)
    d = uniform.rvs(loc=lam_min, scale=lam_max - lam_min, size=n_features)
    d[0], d[-1] = lam_min, lam_max

    return U @ np.diag(d) @ U.T


def exp_map_spd(point: np.ndarray, tangents: np.ndarray) -> np.ndarray:
    """
    Riemannian exponential map on the SPD manifold under the
    affine-invariant metric (batch version).

    Given a reference SPD matrix ``point`` and a batch of tangent matrices
    ``tangents`` at this point, this function returns the corresponding
    SPD matrices obtained by applying the exponential map at ``point`` to
    each tangent matrix.

    Parameters
    ----------
    point : ndarray
        Reference SPD matrix of shape ``(n_features, n_features)``.
    tangents : ndarray
        Batch of tangent matrices at ``point``, of shape
        ``(n_matrices, n_features, n_features)``.

    Returns
    -------
    ndarray
        Batch of SPD matrices obtained by applying the exponential map
        at ``point`` to each tangent vector, of shape
        ``(n_matrices, n_features, n_features)``.
    """
    # G^{1/2} and G^{-1/2}
    evals, U = np.linalg.eigh(point)
    sqrtG = (U * np.sqrt(evals)) @ U.T
    inv_sqrtG = (U / np.sqrt(evals)) @ U.T

    # S_k = G^{-1/2} eta_k G^{-1/2}, then C_k = G^{1/2} exp(S_k) G^{1/2}
    S = sym(inv_sqrtG @ tangents @ inv_sqrtG)
    lam, V = np.linalg.eigh(S)
    expS = (V * np.exp(lam)[:, None, :]) @ V.swapaxes(-1, -2)
    return sym(sqrtG @ expS @ sqrtG)


def random_MVNs(mu: np.ndarray, C: np.ndarray, n_mvn: int, radius: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate random MVNs scattered around a central MVN N(mu, C).

    The means are mu_k = mu + radius * L z_k with C = L L^T and z_k ~ N(0, I);
    the covariances are C_k = Exp_C(radius * eta_k), with eta_k random
    symmetric matrices (symmetrical part of a standard Gaussian matrix).

    Parameters
    ----------
    mu : ndarray
        Central mean of shape (n_features,).
    C : ndarray
        Central SPD covariance of shape (n_features, n_features).
    n_mvn : int
        Number of MVNs to generate.
    radius : float, default=0.1
        Spread of the MVNs around N(mu, C).

    Returns
    -------
    means : ndarray
        Means of shape (n_mvn, n_features).
    covariances : ndarray
        SPD covariances of shape (n_mvn, n_features, n_features).
    """
    n_features = mu.size
    L = np.linalg.cholesky(C)
    means = mu + radius * np.random.standard_normal((n_mvn, n_features)) @ L.T
    tangents = radius * sym(np.random.standard_normal((n_mvn, n_features, n_features)))
    covariances = exp_map_spd(C, tangents)
    return means, covariances
