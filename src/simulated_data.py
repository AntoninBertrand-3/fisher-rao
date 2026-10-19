"""Simulated SPD matrices."""

import numpy as np


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
    # Eigendecomposition of G = point.
    evals, U = np.linalg.eigh(point)
    sqrt_evals = np.sqrt(evals)
    inv_sqrt_evals = 1.0 / sqrt_evals

    # G^{1/2} and G^{-1/2}.
    sqrtG = (U * sqrt_evals[None, :]) @ U.T
    inv_sqrtG = (U * inv_sqrt_evals[None, :]) @ U.T

    # S_k = G^{-1/2} eta_k G^{-1/2} for the full batch.
    S = np.einsum('ij,kjl,ln->kin', inv_sqrtG, tangents, inv_sqrtG.T)
    lam, V = np.linalg.eigh(S)
    exp_lam = np.exp(lam)

    # exp(S_k) = V_k diag(exp(lam_k)) V_k^T.
    expS = (V * exp_lam[:, None, :]) @ V.transpose(0, 2, 1)

    # Exp_G(eta_k) = G^{1/2} exp(S_k) G^{1/2}.
    Cs = np.einsum('ij,kjl,ln->kin', sqrtG, expS, sqrtG.T)
    return Cs
