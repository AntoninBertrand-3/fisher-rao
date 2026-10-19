"""Sample estimators of Gaussian parameters."""

import numpy as np


def estimate_descriptors(batches: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fit sample means and unbiased covariances to independent observation batches.

    Parameters
    ----------
    batches : ndarray, shape (k,m,n)
        k independent raw observation batches, each with m>n vectors.

    Returns
    -------
    means, covariances : ndarray, shapes (k,n), (k,n,n)
        Sample means and symmetric sample covariances using denominator m-1.

    Raises
    ------
    ValueError
        If shape, batch size, finiteness or numerical positive definiteness fails.
    """
    batches = np.asarray(batches, dtype=float)
    if batches.ndim != 3 or batches.shape[2] < 1 or batches.shape[1] <= batches.shape[2] or batches.shape[0] == 0:
        raise ValueError('Expected a nonempty (k,m,n) batch with m>n.')
    if not np.isfinite(batches).all():
        raise ValueError('Raw observations must be finite.')
    means = batches.mean(axis=1)
    centered = batches - means[:, None, :]
    covariances = centered.transpose(0, 2, 1) @ centered / (batches.shape[1] - 1)
    if np.any(np.linalg.eigvalsh(covariances) <= 0):
        raise ValueError('A sample covariance is not numerically positive definite; no regularization applied.')
    return means, covariances
