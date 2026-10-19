"""Endpoint-fiber horizontality and its analytic matrix-logarithm derivative.

See experiments/horizontality_derivation.md for the adjoint derivation and
metric scaling. This module does not minimize lifted or reduced distance.
"""

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigh, svd

from src.geometry import FloatArray, skew_from_parameters, validate_mvn


def lift_spectrum_resolved(lift_condition: float, structure_error: float, log_determinant: float) -> bool:
    """Screen a lifted logarithm for conditioning and structural breakdown.

    Parameters
    ----------
    lift_condition : float
        Estimated condition number of G=L L.T.
    structure_error : float
        Frobenius norm of H+J H J, theoretically zero.
    log_determinant : float
        Trace of H, theoretically zero.

    Returns
    -------
    bool
        Whether eps*cond(L)<=1e-6 and both structural errors are <=1e-7.
        These are screening criteria, not certified spectral error bounds.
    """
    return bool(np.isfinite(lift_condition)
                and np.finfo(float).eps * np.sqrt(lift_condition) <= 1e-6
                and structure_error <= 1e-7 and abs(log_determinant) <= 1e-7)


def log_divided_differences(eigenvalues: FloatArray) -> FloatArray:
    """Compute the positive spectral coefficients of the logarithm derivative.

    Parameters
    ----------
    eigenvalues : ndarray, shape (p,)
        Positive eigenvalues of an SPD matrix.

    Returns
    -------
    ndarray, shape (p, p)
        (log(x)-log(y))/(x-y), with continuous value 1/x at x=y.

    Raises
    ------
    ValueError
        If eigenvalues are not a finite positive vector.
    """
    values = np.asarray(eigenvalues, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or np.any(values <= 0):
        raise ValueError("Logarithm eigenvalues must be finite and positive.")
    low = np.minimum(values[:, None], values[None, :])
    gap = np.abs(values[:, None] - values[None, :])
    coefficients = 1 / low
    np.divide(np.log1p(gap / low), gap, out=coefficients, where=gap != 0)
    return coefficients


class HorizontalFiberObjective:
    """Prepared residual ||(log G(S))_13||_F squared for a normalized endpoint.

    Parameters
    ----------
    mu : array_like, shape (n,)
        Normalized endpoint mean.
    covariance : array_like, shape (n, n)
        Normalized SPD covariance.
    """

    def __init__(self, mu: ArrayLike, covariance: ArrayLike) -> None:
        """Prepare the fixed blocks of the endpoint's lifted SPD factor.

        Parameters
        ----------
        mu : array_like, shape (n,)
            Normalized endpoint mean.
        covariance : array_like, shape (n, n)
            Normalized SPD covariance.
        """
        self.mu, self.covariance = validate_mvn(mu, covariance)
        self.n = self.mu.size
        values, vectors = eigh(self.covariance)
        if np.any(values <= 0):
            raise ValueError("Covariance is not numerically positive definite.")
        self.whitening = (vectors / np.sqrt(values)) @ vectors.T
        square_root = (vectors * np.sqrt(values)) @ vectors.T
        self.base_factor = np.zeros((2 * self.n + 1, 2 * self.n + 1))
        self.base_factor[:self.n, :self.n] = self.whitening
        self.base_factor[self.n, :self.n] = self.mu @ self.whitening
        self.base_factor[self.n, self.n] = 1
        self.base_factor[self.n + 1:, :self.n] = -0.5 * np.outer(self.mu, self.mu) @ self.whitening
        self.base_factor[self.n + 1:, self.n] = -self.mu
        self.base_factor[self.n + 1:, self.n + 1:] = square_root
        self.rows, self.columns = np.triu_indices(self.n, 1)
        self.parameter_count = self.rows.size

    def factor(self, parameters: FloatArray) -> FloatArray:
        """Construct L=M(S) diag(Sigma^-1/2,1,Sigma^1/2), with G=L L.T.

        Parameters
        ----------
        parameters : ndarray, shape (n*(n-1)//2,)
            Upper-triangle coordinates of skew-symmetric S.

        Returns
        -------
        ndarray, shape (2*n+1, 2*n+1)
            Invertible factor of the lifted endpoint.
        """
        skew = skew_from_parameters(parameters, self.n)
        factor = self.base_factor.copy()
        factor[self.n + 1:, :self.n] += skew @ self.whitening
        return factor

    def spectral_log(self, parameters: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
        """Evaluate log G from an SVD of its factor without forming L L.T.

        Parameters
        ----------
        parameters : ndarray, shape (n*(n-1)//2,)
            Independent skew coordinates.

        Returns
        -------
        logarithm, left, singular, right : ndarray
            Symmetric log G and the SVD factors L=left diag(singular) right.

        Raises
        ------
        FloatingPointError
            If the factor's positive spectrum cannot be resolved.
        """
        left, singular, right = svd(self.factor(parameters), full_matrices=False)
        if not np.isfinite(singular).all() or np.any(singular <= 0):
            raise FloatingPointError("Lifted factor has an unresolved spectrum.")
        logarithm = (left * (2 * np.log(singular))) @ left.T
        return logarithm, left, singular, right

    def value_and_gradient(self, parameters: FloatArray) -> tuple[float, FloatArray]:
        """Evaluate horizontality and its analytic Frechet-log skew gradient.

        Parameters
        ----------
        parameters : ndarray, shape (n*(n-1)//2,)
            Upper-triangle coordinates of S.

        Returns
        -------
        value, gradient : float, ndarray
            Squared Frobenius residual and its coordinate gradient.
        """
        logarithm, left, singular, right = self.spectral_log(parameters)
        residual = logarithm[:self.n, self.n + 1:]
        top, bottom = left[:self.n], left[self.n + 1:]
        # E_13=H_13 and E_31=H_13.T give dR=<E,dH> for symmetric dH.
        half_adjoint = top.T @ residual @ bottom
        spectral_adjoint = half_adjoint + half_adjoint.T
        coefficients = log_divided_differences(singular**2)
        # Adjoint through G=L L.T: grad_L=2 L_log(G,E) L.
        weighted_adjoint = (coefficients * spectral_adjoint) * singular
        matrix_gradient = 2 * bottom @ weighted_adjoint @ (right[:, :self.n] @ self.whitening)
        gradient = matrix_gradient[self.rows, self.columns] - matrix_gradient[self.columns, self.rows]
        return float(np.sum(residual**2)), gradient

    def initial_tangent(self, parameters: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Extract the normalized initial mean and covariance velocities.

        Parameters
        ----------
        parameters : ndarray, shape (n*(n-1)//2,)
            Fiber coordinates; horizontality must be checked separately.

        Returns
        -------
        a, A : ndarray
            Vector H[:n,n] and symmetric matrix -H[:n,:n], where H=log G.
            These define a horizontal generator even for an inexact solve;
            its projected endpoint must then be checked.
        """
        logarithm = self.spectral_log(parameters)[0]
        return logarithm[:self.n, self.n].copy(), -logarithm[:self.n, :self.n].copy()

    def diagnostics(self, parameters: FloatArray) -> dict:
        """Measure the horizontal residual, ambient length and structural errors.

        Parameters
        ----------
        parameters : ndarray, shape (n*(n-1)//2,)
            Candidate endpoint-fiber coordinates.

        Returns
        -------
        dict
            Distance candidate, residual, spectrum and horizontal-block checks.
            A nonhorizontal candidate is not qualified as a Fisher distance.
            spectrally_resolved=False invalidates even a small computed residual.
        """
        logarithm, _, singular, _ = self.spectral_log(parameters)
        residual = logarithm[:self.n, self.n + 1:]
        reversal = np.r_[np.arange(self.n + 1, 2 * self.n + 1), self.n, np.arange(self.n)]
        a0 = logarithm[:self.n, self.n]
        A0 = -logarithm[:self.n, :self.n]
        structure_error = float(np.linalg.norm(logarithm + logarithm[np.ix_(reversal, reversal)]))
        log_determinant = float(2 * np.sum(np.log(singular)))
        lift_condition = float((singular[0] / singular[-1])**2)
        return dict(distance=float(np.linalg.norm(np.log(singular))),
                    horizontal_residual=float(np.linalg.norm(residual)),
                    initial_fisher_speed=float(np.sqrt(a0 @ a0 + 0.5 * np.sum(A0**2))),
                    log_structure_error=structure_error,
                    log_determinant=log_determinant, lift_condition=lift_condition,
                    spectrally_resolved=lift_spectrum_resolved(lift_condition, structure_error, log_determinant))
