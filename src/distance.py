"""Canonical J(K), d_0 and local numerical minimization (notation.md, section 8)."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eigh, eigvalsh, svd
from scipy.optimize import OptimizeResult, minimize

from src.geometry import FloatArray, normalize_pair, validate_mvn


@dataclass
class ReferenceResult:
    """Best value found, its coordinates, and unmodified termination diagnostics.

    Attributes
    ----------
    distance : float
        Square root of the best converged J, or the best finite fallback.
    parameters : ndarray
        Upper-triangle coordinates of the selected K.
    runs : tuple of OptimizeResult
        Every BFGS run, including success, message, jac, nit and nfev.
    used_initial_value : bool
        Whether the zero initialization was retained instead of a solver endpoint.
    reliable : bool
        Whether a finite converged run was selected, or no optimization was needed.
        This convergence qualification never certifies a global minimum.
    reliability_message : str
        Reason for the convergence qualification or unqualified fallback.
    """

    distance: float
    parameters: FloatArray
    runs: tuple[OptimizeResult, ...]
    used_initial_value: bool
    reliable: bool
    reliability_message: str


def _univariate_objective(mu: float, variance: float) -> float:
    """Evaluate scalar J(0) without subtractive loss near the identity.

    Parameters
    ----------
    mu : float
        Normalized mean.
    variance : float
        Positive normalized variance.

    Returns
    -------
    float
        Squared univariate distance under the canonical metric.
    """
    sigma = np.sqrt(variance)
    sigma_difference = (variance - 1) / (sigma + 1)
    argument = np.hypot(mu, np.sqrt(2) * sigma_difference) / np.sqrt(8 * sigma)
    return float(8 * np.arcsinh(argument) ** 2)


def _prepare(mu: ArrayLike, covariance: ArrayLike) -> tuple[FloatArray, FloatArray]:
    """Precompute the fixed spectral whitening and Y(0).

    Parameters
    ----------
    mu : array_like, shape (n,)
        Normalized endpoint mean.
    covariance : array_like, shape (n, n)
        Normalized SPD covariance.

    Returns
    -------
    whitening, base : ndarray
        Sigma^{-1/2} and (I + Sigma + mu mu.T / 2) / 2.

    Raises
    ------
    ValueError
        If the Gaussian is invalid or numerically nonpositive.
    """
    mean, matrix = validate_mvn(mu, covariance)
    values, vectors = eigh(matrix)
    if np.any(values <= 0):
        raise ValueError("Covariance is not numerically positive definite.")
    whitening = (vectors / np.sqrt(values)) @ vectors.T
    base = 0.5 * (np.eye(mean.size) + matrix) + 0.25 * np.outer(mean, mean)
    return whitening, base


def _evaluate(whitening: FloatArray, base: FloatArray, skew: FloatArray) -> tuple[float, FloatArray]:
    """Evaluate J and its ambient matrix gradient using singular values.

    Parameters
    ----------
    whitening, base, skew : ndarray, shape (n, n)
        Fixed Sigma^{-1/2}, Y(0), and skew-symmetric K.

    Returns
    -------
    value : float
        Canonical J(K).
    gradient : ndarray, shape (n, n)
        Gradient with respect to entries of K before skew restriction.

    Raises
    ------
    FloatingPointError
        If singular values fall below one beyond roundoff.
    """
    left, singular, right = svd(whitening @ (base + skew), full_matrices=False)
    tolerance = 32 * np.finfo(float).eps * base.shape[0] * max(1.0, singular[0])
    if tolerance > 1e-8 or np.any(singular < 1.0 - tolerance):
        raise FloatingPointError("Cannot reliably evaluate arcosh: spectral domain or conditioning failure.")
    singular = np.maximum(singular, 1.0)
    angles = np.arccosh(singular)
    # d[2 acosh(s)^2]/ds tends to 4 as s tends to 1.
    weights = np.full_like(singular, 4.0)
    denominator = np.sqrt(singular - 1) * np.sqrt(singular + 1)
    np.divide(4 * angles, denominator, out=weights, where=singular > 1)
    gradient = whitening.T @ ((left * weights) @ right)
    return float(2 * (angles @ angles)), gradient


def skew_objective(mu: ArrayLike, covariance: ArrayLike, K: ArrayLike) -> float:
    """Compute canonical J(K) for the normalized pair (0, I), (mu, Sigma).

    Parameters
    ----------
    mu : array_like, shape (n,)
        Normalized mean.
    covariance : array_like, shape (n, n)
        Normalized SPD covariance Sigma.
    K : array_like, shape (n, n)
        Finite real skew-symmetric matrix.

    Returns
    -------
    float
        2 times the sum of squared acosh singular values of Sigma^{-1/2} Y(K).

    Raises
    ------
    ValueError
        If the Gaussian or K is invalid.
    FloatingPointError
        If the spectral domain cannot be resolved reliably.
    """
    whitening, base = _prepare(mu, covariance)
    if np.iscomplexobj(K):
        raise ValueError("K must be real.")
    skew = np.asarray(K, dtype=float)
    if skew.shape != base.shape or not np.isfinite(skew).all():
        raise ValueError("K must be finite with shape (n, n).")
    if not np.allclose(skew + skew.T, 0, rtol=0, atol=1e-12):
        raise ValueError("K must be skew-symmetric.")
    if base.shape == (1, 1):
        return _univariate_objective(float(np.asarray(mu)[0]), float(np.asarray(covariance)[0, 0]))
    return _evaluate(whitening, base, 0.5 * (skew - skew.T))[0]


def d_0(mu0: ArrayLike, covariance0: ArrayLike, mu1: ArrayLike, covariance1: ArrayLike) -> float:
    """Compute the zero-skew approximation after symmetric affine normalization.

    Parameters
    ----------
    mu0, mu1 : array_like, shape (n,)
        Endpoint means.
    covariance0, covariance1 : array_like, shape (n, n)
        Endpoint SPD covariances.

    Returns
    -------
    float
        Square root of J(0).

    Raises
    ------
    ValueError
        If endpoints are invalid.
    FloatingPointError
        If the spectral domain cannot be resolved reliably.
    """
    mean, matrix = normalize_pair(mu0, covariance0, mu1, covariance1)
    return float(np.sqrt(skew_objective(mean, matrix, np.zeros_like(matrix))))


def d_f_reference(
    mu0: ArrayLike, covariance0: ArrayLike, mu1: ArrayLike, covariance1: ArrayLike,
    *, n_starts: int = 3, seed: int = 0, gtol: float = 1e-8, maxiter: int = 1000,
) -> ReferenceResult:
    """Minimize canonical J with reproducible BFGS restarts and analytic gradients.

    Parameters
    ----------
    mu0, mu1 : array_like, shape (n,)
        Endpoint means.
    covariance0, covariance1 : array_like, shape (n, n)
        Endpoint SPD covariances.
    n_starts : int, optional
        Positive restart count, including zero; other starts are standard normal.
    seed : int, optional
        Local random generator seed.
    gtol : float, optional
        Positive BFGS gradient tolerance.
    maxiter : int, optional
        Positive iteration limit per run.

    Returns
    -------
    ReferenceResult
        Best converged distance, or an explicitly unqualified finite fallback.
        All solver diagnostics are retained. No global optimality
        certificate is implied. For n=1, runs is empty: no optimization is needed.
        Interpretation as Fisher distance depends on the documented reduction.

    Raises
    ------
    ValueError
        If endpoints or optimization controls are invalid.
    FloatingPointError
        If the spectral domain cannot be resolved reliably.
    """
    if not isinstance(n_starts, int) or n_starts < 1 or not isinstance(maxiter, int) or maxiter < 1:
        raise ValueError("n_starts and maxiter must be positive integers.")
    if not np.isfinite(gtol) or gtol <= 0:
        raise ValueError("gtol must be finite and positive.")
    mean, matrix = normalize_pair(mu0, covariance0, mu1, covariance1)
    whitening, base = _prepare(mean, matrix)
    rows, columns = np.triu_indices(mean.size, 1)
    parameter_count = rows.size

    def objective(parameters: FloatArray) -> tuple[float, FloatArray]:
        """Evaluate J and its gradient in independent skew coordinates.

        Parameters
        ----------
        parameters : ndarray, shape (n*(n-1)//2,)
            Upper-triangle entries of K.

        Returns
        -------
        value, gradient : float, ndarray
            Objective and coordinate gradient for BFGS.
        """
        skew = np.zeros_like(base)
        skew[rows, columns] = parameters
        skew[columns, rows] = -parameters
        value, gradient = _evaluate(whitening, base, skew)
        return value, gradient[rows, columns] - gradient[columns, rows]

    best_parameters = np.zeros(parameter_count)
    best_value = objective(best_parameters)[0]
    runs: list[OptimizeResult] = []
    used_initial_value = True
    reliable = parameter_count == 0
    reliability_message = "No skew degrees of freedom; no optimization required."
    if not parameter_count:
        best_value = _univariate_objective(float(mean[0]), float(matrix[0, 0]))
    if parameter_count:
        generator = np.random.default_rng(seed)
        for index in range(n_starts):
            initial = np.zeros(parameter_count) if index == 0 else generator.standard_normal(parameter_count)
            result = minimize(objective, initial, jac=True, method="BFGS",
                              options={"gtol": gtol, "maxiter": maxiter})
            runs.append(result)
            if np.isfinite(result.fun) and result.fun <= best_value:
                best_value = float(result.fun)
                best_parameters = result.x.copy()
                used_initial_value = False
        converged = [run for run in runs if run.success and np.isfinite(run.fun) and np.isfinite(run.x).all()]
        if converged:
            selected = min(converged, key=lambda run: run.fun)
            best_value = float(selected.fun)
            best_parameters = selected.x.copy()
            used_initial_value = False
            reliable = True
            reliability_message = "Finite converged restart selected; global optimality is not certified."
        else:
            reliability_message = "No restart converged: finite fallback is not qualified as reliable."
    return ReferenceResult(float(np.sqrt(best_value)), best_parameters, tuple(runs),
                           used_initial_value, reliable, reliability_message)


def covariance_distance(first: np.ndarray, second: np.ndarray) -> float:
    """Compute the affine-invariant SPD distance with trace metric normalization.

    Parameters
    ----------
    first, second : ndarray, shape (n,n)
        Symmetric positive-definite matrices.

    Returns
    -------
    float
        sqrt(sum(log(lambda_i)**2)), for generalized eigenvalues of (second,first).

    Raises
    ------
    ValueError
        If inputs are not finite SPD matrices of the same shape.
    """
    first = np.asarray(first)
    if first.ndim != 2:
        raise ValueError('Expected a square SPD matrix.')
    zero = np.zeros(first.shape[0])
    _, first = validate_mvn(zero, first)
    _, second = validate_mvn(zero, second)
    return float(np.linalg.norm(np.log(eigvalsh(second, first))))
