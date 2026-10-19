"""Fisher exponential and qualified fiber logarithm for Gaussians.

The logarithm and mean assume that qualified horizontal geodesics minimize
length. Numerical qualification does not prove this assumption or uniqueness.
See docs/mvn_maps.md for the formulas and numerical contracts.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import expm, solve
from scipy.optimize import OptimizeResult, minimize

from src.geometry import FloatArray, validate_mvn
from src.horizontality import HorizontalFiberObjective


@dataclass
class LogMapResult:
    """Store a candidate tangent and its numerical qualification.

    Attributes
    ----------
    mean_tangent, covariance_tangent : ndarray
        Initial velocities in the original coordinates, shapes (n,) and (n,n).
    distance : float
        Fisher norm of the extracted tangent, not a global distance certificate.
    parameters : ndarray
        Selected normalized fiber coordinates S (not reduced coordinates K).
    reliable : bool
        Whether spectral, residual and endpoint checks passed.
    diagnostics : dict
        Selected spectral/residual checks and normalized endpoint errors.
    runs : tuple of OptimizeResult
        All restarts, including raw termination flags and qualification checks.
    """

    mean_tangent: FloatArray
    covariance_tangent: FloatArray
    distance: float
    parameters: FloatArray
    reliable: bool
    diagnostics: dict
    runs: tuple[OptimizeResult, ...]


def _frame(covariance: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Compute symmetric affine normalization factors.

    Parameters
    ----------
    covariance : ndarray, shape (n,n)
        Validated SPD matrix.

    Returns
    -------
    root, inverse_root : ndarray
        Symmetric square root and inverse square root.

    Raises
    ------
    ValueError
        If the spectral decomposition cannot resolve positive eigenvalues.
    """
    values, vectors = np.linalg.eigh(covariance)
    if np.any(values <= 0):
        raise ValueError("Covariance is not numerically positive definite.")
    return ((vectors * np.sqrt(values)) @ vectors.T,
            (vectors / np.sqrt(values)) @ vectors.T)


def _tangent(mean_tangent: ArrayLike, covariance_tangent: ArrayLike,
             n: int) -> tuple[FloatArray, FloatArray]:
    """Validate a real finite vector and symmetric covariance velocity.

    Parameters
    ----------
    mean_tangent, covariance_tangent : array_like
        Tangent components with shapes (n,) and (n,n).
    n : int
        Positive Gaussian dimension.

    Returns
    -------
    v, T : ndarray
        Validated tangent; roundoff asymmetry is averaged.

    Raises
    ------
    ValueError
        If shapes, finiteness, reality or symmetry fail.
    """
    if np.iscomplexobj(mean_tangent) or np.iscomplexobj(covariance_tangent):
        raise ValueError("Tangents must be real.")
    v = np.asarray(mean_tangent, dtype=float)
    T = np.asarray(covariance_tangent, dtype=float)
    if v.shape != (n,) or T.shape != (n, n):
        raise ValueError("Expected tangent shapes (n,) and (n,n).")
    if not np.isfinite(v).all() or not np.isfinite(T).all():
        raise ValueError("Tangents must be finite.")
    if np.max(np.abs(T - T.T)) > 1e-12 * max(1., np.max(np.abs(T))):
        raise ValueError("Covariance tangent must be symmetric.")
    return v, (T + T.T) / 2


def _exp_normalized(a: FloatArray, A: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Project the exponential of a normalized horizontal generator.

    Parameters
    ----------
    a, A : ndarray
        Mean and symmetric covariance velocities at (0,I).

    Returns
    -------
    mean, covariance : ndarray
        Gaussian endpoint at time one.

    Raises
    ------
    FloatingPointError
        If the lifted exponential or precision block is poorly resolved.
    """
    n = a.size
    V = np.zeros((2 * n + 1, 2 * n + 1))
    V[:n, :n], V[n + 1:, n + 1:] = -A, A
    V[:n, n] = V[n, :n] = a
    V[n + 1:, n] = V[n, n + 1:] = -a
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        lifted = expm(V)
    if not np.isfinite(lifted).all():
        raise FloatingPointError("Nonfinite lifted exponential.")
    precision = (lifted[:n, :n] + lifted[:n, :n].T) / 2
    spectrum = np.linalg.eigvalsh(precision)
    if spectrum[0] <= 0 or np.finfo(float).eps * spectrum[-1] / spectrum[0] > 1e-8:
        raise FloatingPointError("Unresolved lifted precision block.")
    projected = solve(precision, np.column_stack((lifted[:n, n], np.eye(n))), assume_a='pos')
    return validate_mvn(projected[:, 0], (projected[:, 1:] + projected[:, 1:].T) / 2)


def exp_map_mvn(mean: ArrayLike, covariance: ArrayLike,
                mean_tangent: ArrayLike, covariance_tangent: ArrayLike,
                *, time: float = 1.) -> tuple[FloatArray, FloatArray]:
    """Evaluate the Fisher exponential of a Gaussian tangent at the given time.

    Parameters
    ----------
    mean, covariance : array_like, shapes (n,), (n,n)
        Base Gaussian with SPD covariance.
    mean_tangent, covariance_tangent : array_like, shapes (n,), (n,n)
        Initial velocities; the covariance velocity must be symmetric.
    time : float, optional
        Finite geodesic time, including zero and negative values.

    Returns
    -------
    mean, covariance : ndarray
        Endpoint Gaussian. This is the full Fisher exponential, not an
        independent Euclidean mean update and SPD covariance exponential.

    Raises
    ------
    ValueError
        If the Gaussian, tangent or time is invalid.
    FloatingPointError
        If numerical range or conditioning prevents evaluation.
    """
    mean, covariance = validate_mvn(mean, covariance)
    v, T = _tangent(mean_tangent, covariance_tangent, mean.size)
    if not np.isfinite(time):
        raise ValueError("time must be finite.")
    if time == 0 or (not np.any(v) and not np.any(T)):
        return mean, covariance
    root, whitening = _frame(covariance)
    a, A = whitening @ v, whitening @ T @ whitening
    endpoint_mean, endpoint_covariance = _exp_normalized(time * a, time * A)
    matrix = root @ endpoint_covariance @ root
    return validate_mvn(mean + root @ endpoint_mean, (matrix + matrix.T) / 2)


def log_map_mvn(mean: ArrayLike, covariance: ArrayLike,
                target_mean: ArrayLike, target_covariance: ArrayLike,
                *, n_starts: int = 3, seed: int = 0, gtol: float = 1e-11,
                maxiter: int = 2000, residual_tol: float = 1e-8,
                endpoint_tol: float = 1e-7) -> LogMapResult:
    """Estimate a Fisher logarithm by solving endpoint-fiber horizontality.

    Parameters
    ----------
    mean, covariance, target_mean, target_covariance : array_like
        Base and target Gaussian parameters, shapes (n,) and (n,n).
    n_starts : int, optional
        Positive BFGS restart count, including zero then seeded normal starts.
    seed : int, optional
        Local random seed.
    gtol : float, optional
        Positive BFGS gradient tolerance.
    maxiter : int, optional
        Positive iteration limit per restart.
    residual_tol, endpoint_tol : float, optional
        Positive horizontal residual and normalized relative endpoint thresholds.

    Returns
    -------
    LogMapResult
        Tangent and diagnostics. Always inspect reliable. Qualification uses
        residual, spectrum and reconstruction, independently of BFGS success.
        Interpreting this as a minimizing logarithm remains an assumption.

    Raises
    ------
    ValueError
        If endpoints or solver controls are invalid.
    RuntimeError
        If every restart fails spectral or endpoint evaluation.
    """
    mean, covariance = validate_mvn(mean, covariance)
    target_mean, target_covariance = validate_mvn(target_mean, target_covariance)
    if mean.shape != target_mean.shape:
        raise ValueError("Endpoint dimensions must agree.")
    for value in (n_starts, maxiter):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError("n_starts and maxiter must be positive integers.")
    if any(not np.isfinite(value) or value <= 0 for value in (gtol, residual_tol, endpoint_tol)):
        raise ValueError("Solver tolerances must be finite and positive.")
    root, whitening = _frame(covariance)
    mu = whitening @ (target_mean - mean)
    Sigma = whitening @ target_covariance @ whitening
    Sigma = (Sigma + Sigma.T) / 2
    objective = HorizontalFiberObjective(mu, Sigma)
    rng = np.random.default_rng(seed)
    runs, candidates = [], []
    for index in range(n_starts if objective.parameter_count else 1):
        initial = (np.zeros(objective.parameter_count) if index == 0
                   else rng.standard_normal(objective.parameter_count))
        run = None
        try:
            if objective.parameter_count:
                run = minimize(objective.value_and_gradient, initial, jac=True, method='BFGS',
                               options=dict(gtol=gtol, maxiter=maxiter))
            else:
                value, gradient = objective.value_and_gradient(initial)
                run = OptimizeResult(x=initial, fun=value, jac=gradient, success=True,
                                     message='No skew coordinates.', nit=0, nfev=1)
            runs.append(run)
            run.initial_parameters = initial.copy()
            checks = objective.diagnostics(run.x)
            run.diagnostics = checks
            if not checks['spectrally_resolved']:
                continue
            a, A = objective.initial_tangent(run.x)
            recovered_mean, recovered_covariance = _exp_normalized(a, A)
            checks['mean_endpoint_error'] = float(np.linalg.norm(recovered_mean - mu) / max(1., np.linalg.norm(mu)))
            checks['covariance_endpoint_error'] = float(np.linalg.norm(recovered_covariance - Sigma) / max(1., np.linalg.norm(Sigma)))
            checks['qualified'] = bool(checks['horizontal_residual'] <= residual_tol
                                       and checks['mean_endpoint_error'] <= endpoint_tol
                                       and checks['covariance_endpoint_error'] <= endpoint_tol)
            candidates.append((run, a, A))
        except (FloatingPointError, np.linalg.LinAlgError, ValueError) as error:
            if run is None:
                runs.append(OptimizeResult(x=initial, success=False, message=str(error),
                                           initial_parameters=initial.copy()))
            else:
                run.qualification_error = str(error)
    if not candidates:
        raise RuntimeError("No spectrally resolved fiber tangent could be reconstructed: "
                           + '; '.join(str(run.message) for run in runs))
    selected, a, A = min(candidates, key=lambda item: (
        not item[0].diagnostics['qualified'], item[0].diagnostics['horizontal_residual']))
    T = root @ A @ root
    return LogMapResult(root @ a, (T + T.T) / 2, float(np.sqrt(a @ a + .5 * np.sum(A * A))),
                        selected.x.copy(), selected.diagnostics['qualified'],
                        selected.diagnostics.copy(), tuple(runs))
