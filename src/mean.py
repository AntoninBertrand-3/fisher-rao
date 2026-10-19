"""Frechet means of Gaussians and of SPD covariance matrices.

The Gaussian mean assumes that qualified horizontal geodesics minimize
length. Numerical qualification does not prove this assumption or uniqueness.
See docs/mvn_maps.md for the formulas and numerical contracts.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from src.geometry import FloatArray, validate_mvn
from src.mvn import _frame, exp_map_mvn, log_map_mvn


@dataclass
class FrechetMeanResult:
    """Store a numerical Gaussian mean and its convergence history.

    Attributes
    ----------
    mean, covariance : ndarray
        Final Gaussian, shapes (n,) and (n,n).
    converged : bool
        Whether the Fisher norm of the average qualified tangent met tol.
    message : str
        Termination reason; no global minimum or uniqueness is certified.
    history : tuple of dict
        Initial and accepted objectives, gradient norms, steps and inner errors.
    iterates : tuple
        Copies of accepted Gaussian pairs, including initialization, if requested.
    """

    mean: FloatArray
    covariance: FloatArray
    converged: bool
    message: str
    history: tuple[dict, ...]
    iterates: tuple[tuple[FloatArray, FloatArray], ...]


def _mean_evaluation(mean: FloatArray, covariance: FloatArray,
                     means: FloatArray, covariances: FloatArray,
                     log_options: dict) -> tuple[float, FloatArray, FloatArray, dict]:
    """Evaluate qualified logarithms and their average at a common base.

    Parameters
    ----------
    mean, covariance : ndarray
        Current Gaussian.
    means, covariances : ndarray
        Nonempty validated observations.
    log_options : dict
        Keyword arguments for log_map_mvn.

    Returns
    -------
    objective, v, T, diagnostics : tuple
        Half the average squared length, average tangent and convergence data.

    Raises
    ------
    RuntimeError
        If any logarithm fails numerical qualification.
    """
    logs = []
    # Each endpoint requires an independent nonlinear fiber solve.
    for target_mean, target_covariance in zip(means, covariances):
        result = log_map_mvn(mean, covariance, target_mean, target_covariance, **log_options)
        if not result.reliable:
            raise RuntimeError("An inner logarithm failed residual or endpoint qualification.")
        logs.append(result)
    v = np.mean(np.stack([result.mean_tangent for result in logs]), axis=0)
    T = np.mean(np.stack([result.covariance_tangent for result in logs]), axis=0)
    _, whitening = _frame(covariance)
    a, A = whitening @ v, whitening @ T @ whitening
    norm = float(np.sqrt(a @ a + .5 * np.sum(A * A)))
    objective = float(np.mean([result.distance**2 for result in logs]) / 2)
    checks = dict(objective=objective, gradient_norm=norm,
                  max_horizontal_residual=max(result.diagnostics['horizontal_residual'] for result in logs),
                  max_endpoint_error=max(max(result.diagnostics['mean_endpoint_error'],
                                             result.diagnostics['covariance_endpoint_error']) for result in logs))
    return objective, v, T, checks


def frechet_mean_mvn(means: ArrayLike, covariances: ArrayLike, *,
                     init: tuple[ArrayLike, ArrayLike] | None = None,
                     max_iter: int = 50, tol: float = 1e-6, step_size: float = 1.,
                     max_backtracks: int = 20, return_iterates: bool = False,
                     log_options: dict | None = None) -> FrechetMeanResult:
    """Estimate a local Fisher Frechet mean using qualified tangent averaging.

    Parameters
    ----------
    means, covariances : array_like, shapes (k,n), (k,n,n)
        Nonempty equally weighted Gaussian observations with SPD covariances.
    init : tuple of array_like, optional
        Initial Gaussian; defaults to arithmetic averages of the parameters.
    max_iter : int, optional
        Maximum accepted updates; zero returns the evaluated initialization.
    tol : float, optional
        Positive stopping threshold on the Fisher norm of the average tangent.
    step_size : float, optional
        Positive initial step for each Armijo backtracking search.
    max_backtracks : int, optional
        Maximum step halvings per iteration, nonnegative.
    return_iterates : bool, optional
        Whether to retain copies of the initial and accepted Gaussians.
    log_options : dict, optional
        Controls forwarded to log_map_mvn, including restart seed and tolerances.

    Returns
    -------
    FrechetMeanResult
        Numerical estimate and explicit termination status. Convergence is
        stationarity under the minimizing-logarithm assumption, not globality.

    Raises
    ------
    ValueError
        If inputs, initialization or controls are invalid.
    RuntimeError
        If logarithms at the initialization cannot be qualified.
    """
    if np.iscomplexobj(means) or np.iscomplexobj(covariances):
        raise ValueError("Gaussian observations must be real.")
    means, covariances = np.asarray(means, dtype=float), np.asarray(covariances, dtype=float)
    if means.ndim != 2 or min(means.shape) == 0 or covariances.shape != (len(means), means.shape[1], means.shape[1]):
        raise ValueError("Expected nonempty shapes (k,n) and (k,n,n).")
    pairs = [validate_mvn(mu, Sigma) for mu, Sigma in zip(means, covariances)]
    means, covariances = np.stack([pair[0] for pair in pairs]), np.stack([pair[1] for pair in pairs])
    for value in (max_iter, max_backtracks):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
            raise ValueError("Iteration limits must be nonnegative integers.")
    if any(not np.isfinite(value) or value <= 0 for value in (tol, step_size)):
        raise ValueError("tol and step_size must be finite and positive.")
    mean, covariance = validate_mvn(*(init if init is not None else (means.mean(axis=0), covariances.mean(axis=0))))
    if mean.shape != means.shape[1:]:
        raise ValueError("Initialization dimension must match observations.")
    options = dict(log_options or {})
    objective, v, T, checks = _mean_evaluation(mean, covariance, means, covariances, options)
    history = [dict(checks, step_size=0.)]
    iterates = [(mean.copy(), covariance.copy())] if return_iterates else []
    message = 'Maximum iteration count reached.'
    for _ in range(max_iter):
        if checks['gradient_norm'] <= tol:
            break
        step, accepted = step_size, False
        for _ in range(max_backtracks + 1):
            try:
                candidate = exp_map_mvn(mean, covariance, v, T, time=step)
                evaluation = _mean_evaluation(*candidate, means, covariances, options)
                if evaluation[0] <= objective - 1e-4 * step * checks['gradient_norm']**2:
                    accepted = True
                    break
            except (RuntimeError, FloatingPointError, np.linalg.LinAlgError):
                pass
            step *= .5
        if not accepted:
            message = 'Backtracking failed to find a qualified decreasing step.'
            break
        mean, covariance = candidate
        objective, v, T, checks = evaluation
        history.append(dict(checks, step_size=step))
        if return_iterates:
            iterates.append((mean.copy(), covariance.copy()))
    converged = checks['gradient_norm'] <= tol
    if converged:
        message = 'Average tangent meets the Fisher-norm tolerance; global optimality is not certified.'
    return FrechetMeanResult(mean, covariance, converged, message, tuple(history), tuple(iterates))


def _mean_state(point: np.ndarray, matrices: np.ndarray) -> tuple:
    """Evaluate the squared-distance objective and whitened negative gradient.

    Parameters
    ----------
    point : ndarray, shape (n,n)
        Current SPD mean candidate.
    matrices : ndarray, shape (k,n,n)
        Validated SPD observations.

    Returns
    -------
    tuple
        Objective mean(d**2)/2, mean whitened logarithm, and point square root.
    """
    values, vectors = np.linalg.eigh(point)
    root = (vectors * np.sqrt(values)) @ vectors.T
    whitening = (vectors / np.sqrt(values)) @ vectors.T
    normalized = whitening @ matrices @ whitening
    values, vectors = np.linalg.eigh((normalized + normalized.swapaxes(-1, -2)) / 2)
    if np.any(values <= 0):
        raise ValueError('Whitened covariance lost positive definiteness.')
    logarithms = np.log(values)
    tangent = ((vectors * logarithms[:, None, :]) @ vectors.swapaxes(-1, -2)).mean(axis=0)
    return float(np.mean(np.sum(logarithms**2, axis=1)) / 2), tangent, root


def covariance_mean(matrices: np.ndarray, *, tol: float = 1e-8, maxiter: int = 200) -> dict:
    """Estimate the affine-invariant Frechet mean by Riemannian gradient descent.

    Parameters
    ----------
    matrices : ndarray, shape (k,n,n)
        Nonempty equally weighted sample of SPD matrices.
    tol : float
        Positive stopping tolerance on the whitened mean-log Frobenius norm.
    maxiter : int
        Maximum number of Riemannian updates with Armijo backtracking.

    Returns
    -------
    dict
        Matrix, convergence flag, gradient norm, iteration count and objective.
        A nonconverged last iterate is returned explicitly.

    Raises
    ------
    ValueError
        If inputs are invalid or a normalized covariance loses definiteness.
    """
    matrices = np.asarray(matrices)
    if matrices.ndim != 3 or len(matrices) == 0:
        raise ValueError('Expected a nonempty (k,n,n) SPD sample.')
    if not np.isfinite(tol) or tol <= 0 or maxiter < 1:
        raise ValueError('Require positive tolerance and iteration limit.')
    zero = np.zeros(matrices.shape[1])
    matrices = np.stack([validate_mvn(zero, matrix)[1] for matrix in matrices])
    point = matrices.mean(axis=0)
    for iteration in range(maxiter + 1):
        objective, tangent, root = _mean_state(point, matrices)
        norm = float(np.linalg.norm(tangent))
        if norm <= tol or iteration == maxiter:
            break
        values, vectors = np.linalg.eigh(tangent)
        step = 1.
        # This is the SPD exponential update, not Euclidean matrix descent.
        for _ in range(40):
            candidate = root @ ((vectors * np.exp(step * values)) @ vectors.T) @ root
            candidate = (candidate + candidate.T) / 2
            trial, _, _ = _mean_state(candidate, matrices)
            roundoff = 16 * np.finfo(float).eps * max(1., objective)
            if trial <= objective - 1e-4 * step * norm**2 + roundoff:
                point = candidate
                break
            step /= 2
        else:
            break
    return dict(matrix=point, converged=norm <= tol, gradient_norm=norm,
                iterations=iteration, objective=objective)
