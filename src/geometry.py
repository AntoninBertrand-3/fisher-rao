"""Validation and affine coordinates for real nondegenerate Gaussians."""

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.linalg import cholesky, eigh

FloatArray = NDArray[np.float64]


def validate_mvn(mu: ArrayLike, covariance: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Validate a real mean and SPD covariance without regularization.

    Parameters
    ----------
    mu : array_like, shape (n,)
        Finite mean, with n positive.
    covariance : array_like, shape (n, n)
        Finite covariance, symmetric to relative tolerance 1e-12.

    Returns
    -------
    mu, covariance : ndarray
        Float copies; accepted roundoff asymmetry is averaged.

    Raises
    ------
    ValueError
        If dimensions, reality, finiteness, symmetry or positivity fail.
    """
    if np.iscomplexobj(mu) or np.iscomplexobj(covariance):
        raise ValueError("Mean and covariance must be real.")
    mean = np.array(mu, dtype=float, copy=True)
    matrix = np.array(covariance, dtype=float, copy=True)
    if mean.ndim != 1 or mean.size == 0 or matrix.shape != (mean.size, mean.size):
        raise ValueError("Expected mean (n,) and covariance (n, n), n >= 1.")
    if not np.isfinite(mean).all() or not np.isfinite(matrix).all():
        raise ValueError("Mean and covariance must be finite.")
    scale = np.max(np.abs(matrix))
    if np.max(np.abs(matrix - matrix.T)) > 1e-12 * scale:
        raise ValueError("Covariance must be symmetric.")
    matrix = 0.5 * matrix + 0.5 * matrix.T
    try:
        cholesky(matrix, lower=True)
    except np.linalg.LinAlgError as error:
        raise ValueError("Covariance must be positive definite.") from error
    return mean, matrix


def normalize_pair(
    mu0: ArrayLike, covariance0: ArrayLike,
    mu1: ArrayLike, covariance1: ArrayLike,
) -> tuple[FloatArray, FloatArray]:
    """Map a Gaussian pair to (0, I) and (mu, Sigma) by affine invariance.

    Parameters
    ----------
    mu0, mu1 : array_like, shape (n,)
        Endpoint means.
    covariance0, covariance1 : array_like, shape (n, n)
        Endpoint SPD covariances.

    Returns
    -------
    mu, Sigma : ndarray
        Second endpoint whitened by the symmetric inverse square root of covariance0.

    Raises
    ------
    ValueError
        If endpoints are invalid or their dimensions differ.
    """
    mean0, matrix0 = validate_mvn(mu0, covariance0)
    mean1, matrix1 = validate_mvn(mu1, covariance1)
    if mean0.shape != mean1.shape:
        raise ValueError("Endpoint dimensions must agree.")
    if np.array_equal(mean0, mean1) and np.array_equal(matrix0, matrix1):
        return np.zeros_like(mean0), np.eye(mean0.size)
    eigenvalues, eigenvectors = eigh(matrix0)
    if np.any(eigenvalues <= 0):
        raise ValueError("Covariance is not numerically positive definite.")
    whitening = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
    normalized_covariance = whitening @ matrix1 @ whitening
    normalized_covariance = (normalized_covariance + normalized_covariance.T) / 2
    return validate_mvn(whitening @ (mean1 - mean0), normalized_covariance)


def skew_from_parameters(parameters: ArrayLike, dimension: int) -> FloatArray:
    """Fill the strict upper triangle of a skew matrix in row-major order.

    Parameters
    ----------
    parameters : array_like, shape (n*(n-1)//2,)
        Finite real upper-triangle entries.
    dimension : int
        Positive matrix dimension n.

    Returns
    -------
    ndarray, shape (n, n)
        Matrix K satisfying K.T = -K.

    Raises
    ------
    ValueError
        If dimension or parameters are invalid.
    """
    if not isinstance(dimension, (int, np.integer)) or isinstance(dimension, bool) or dimension < 1:
        raise ValueError("Dimension must be a positive integer.")
    if np.iscomplexobj(parameters):
        raise ValueError("Skew parameters must be real.")
    values = np.asarray(parameters, dtype=float)
    if values.shape != (dimension * (dimension - 1) // 2,) or not np.isfinite(values).all():
        raise ValueError("Invalid skew parameter shape or nonfinite entries.")
    matrix = np.zeros((dimension, dimension))
    rows, columns = np.triu_indices(dimension, 1)
    matrix[rows, columns] = values
    matrix[columns, rows] = -values
    return matrix
