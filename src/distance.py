import numpy as np


def dist2_L2(mu0: np.ndarray, C0: np.ndarray, mu1: np.ndarray, C1: np.ndarray) -> float:
    """
    Squared Euclidean distance between two MVNs.

    Parameters
    ----------
    mu0, mu1 : ndarray
        Means of shape (n_features,).
    C0, C1 : ndarray
        SPD covariance matrices of shape (n_features, n_features).

    Returns
    -------
    float
        ||mu0 - mu1||^2 + ||C0 - C1||_F^2.
    """
    return np.sum((mu0 - mu1)**2) + np.sum((C0 - C1)**2)


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


def dist2_d0(mu0: np.ndarray, C0: np.ndarray, mu1: np.ndarray, C1: np.ndarray) -> float:
    """
    Squared zeroth approximation d_0 of the Fisher-Rao distance between two MVNs.

    The pair is first normalized so that the first MVN is N(0, I):
    mu = L^{-1}(mu1 - mu0) and S = L^{-1} C1 L^{-T}, with C0 = L L^T. Then
    d_0^2 = 2 sum_i arccosh^2(c_i), where c_i are the singular values of
    S^{-1/2} Y and Y = (I + S + mu mu^T / 2) / 2.

    Parameters
    ----------
    mu0, mu1 : ndarray
        Means of shape (n_features,).
    C0, C1 : ndarray
        SPD covariance matrices of shape (n_features, n_features).

    Returns
    -------
    float
        Squared d_0 distance between N(mu0, C0) and N(mu1, C1).
    """
    iL = np.linalg.inv(np.linalg.cholesky(C0))
    mu = iL @ (mu1 - mu0)
    S = iL @ C1 @ iL.T
    Y = 0.5 * (np.eye(mu.size) + S + 0.5 * np.outer(mu, mu))
    iLS = np.linalg.inv(np.linalg.cholesky(S))
    c = np.linalg.svd(iLS @ Y, compute_uv=False)
    return 2 * np.sum(np.arccosh(np.maximum(c, 1.0))**2)
