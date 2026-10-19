"""Published directed triangle divergences (Collas et al., ICASSP 2022).

Implements Corollaries 1 and 2 literally, not Fisher distance or a triangle
length upper bound. See docs/triangle_divergences.md for formulas and gradients.
"""

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

from src.geometry import validate_mvn


def _acosh_terms(offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate acosh(1+t)^2 and acosh(1+t)/sqrt(t*(t+2)) continuously at zero.

    Parameters
    ----------
    offset : ndarray
        Nonnegative offsets t.

    Returns
    -------
    tuple of ndarray
        Squared acosh and ratio, whose limiting value at zero is one.

    Raises
    ------
    ValueError
        If an offset is negative or nonfinite.
    """
    if np.any(offset < 0) or not np.isfinite(offset).all():
        raise ValueError('Invalid acosh offset.')
    angle = 2 * np.arcsinh(np.sqrt(offset / 2))
    denominator = np.sqrt(offset) * np.sqrt(offset + 2)
    ratio = np.divide(angle, denominator, out=np.ones_like(angle), where=offset > 0)
    return angle**2, ratio


class TriangleObjective:
    """Cache fixed target covariances for directed divergence evaluations."""

    def __init__(self, means: np.ndarray, covariances: np.ndarray, kind: str) -> None:
        """Validate targets and precompute their whitening matrices and log determinants.

        Parameters
        ----------
        means, covariances : ndarray, shapes (k,n), (k,n,n)
            Nonempty collection of target Gaussians with SPD covariances.
        kind : {'c', 'perp'}
            Published divergence to evaluate.

        Raises
        ------
        ValueError
            If kind, dimensions or Gaussian parameters are invalid.
        """
        means, covariances = np.asarray(means), np.asarray(covariances)
        if kind not in ('c', 'perp'):
            raise ValueError('Expected kind c or perp.')
        if means.ndim != 2 or len(means) == 0 or covariances.shape != (len(means), means.shape[1], means.shape[1]):
            raise ValueError('Expected means (k,n), covariances (k,n,n).')
        pairs = [validate_mvn(mu, covariance) for mu, covariance in zip(means, covariances)]
        self.means = np.stack([pair[0] for pair in pairs])
        self.covariances = np.stack([pair[1] for pair in pairs])
        values, vectors = np.linalg.eigh(self.covariances)
        self.whitening = (vectors / np.sqrt(values)[:, None, :]) @ vectors.swapaxes(-1, -2)
        self.logdet = np.log(values).sum(axis=1)
        self.kind = kind

    def evaluate(self, mean: np.ndarray, covariance: np.ndarray) -> tuple:
        """Evaluate all squared divergences and their Euclidean first-argument gradients.

        Parameters
        ----------
        mean, covariance : ndarray, shapes (n,), (n,n)
            Common first argument, with SPD covariance.

        Returns
        -------
        tuple
            Divergences (k,), mean gradients (k,n), symmetric covariance
            gradients (k,n,n), with Frobenius inner product convention.
        """
        mean, covariance = validate_mvn(mean, covariance)
        if mean.shape != self.means.shape[1:]:
            raise ValueError('Endpoint dimensions differ.')
        factor = cho_factor(covariance, lower=True)
        delta = self.means - mean
        solved = cho_solve(factor, delta.T).T
        q = np.einsum('ki,ki->k', delta, solved)
        outer = delta[:, :, None] * delta[:, None, :]
        intermediate = covariance if self.kind == 'c' else covariance + outer / 2
        normalized = self.whitening @ intermediate @ self.whitening
        values, vectors = np.linalg.eigh((normalized + normalized.swapaxes(-1, -2)) / 2)
        if np.any(values <= 0):
            raise ValueError('Whitened covariance lost positive definiteness.')
        logs = np.log(values)
        covariance_term = np.sum(logs**2, axis=1) / 2
        gradient = self.whitening @ ((vectors * (logs / values)[:, None, :]) @ vectors.swapaxes(-1, -2)) @ self.whitening
        if self.kind == 'c':
            n = len(mean)
            logdet = 2 * np.log(np.diag(factor[0])).sum()
            ell = (self.logdet - logdet) / n
            scale = np.exp(-ell / 2)
            offset = 2 * np.sinh(ell / 4)**2 + q * scale / 4
            squared, ratio = _acosh_terms(offset)
            divergences = 2 * squared + covariance_term - ell**2 / 2
            weight_q = ratio * scale
            weight_ell = 4 * ratio * (np.sinh(ell / 2) / 2 - q * scale / 8) - ell
            inverse = cho_solve(factor, np.eye(n))
            gradient -= (weight_ell / n)[:, None, None] * inverse
            mean_gradient = -2 * weight_q[:, None] * solved
        else:
            squared, weight_q = _acosh_terms(q)
            divergences = squared / 2 + covariance_term
            mean_gradient = -2 * weight_q[:, None] * solved - (gradient @ delta[..., None])[..., 0]
        gradient -= weight_q[:, None, None] * solved[:, :, None] * solved[:, None, :]
        return divergences, mean_gradient, gradient


def triangle_divergence_squared(mu0: np.ndarray, covariance0: np.ndarray,
                                mu1: np.ndarray, covariance1: np.ndarray, *, kind: str) -> float:
    """Compute a published directed triangle divergence from first to second Gaussian.

    Parameters
    ----------
    mu0, mu1 : ndarray, shape (n,)
        Endpoint means.
    covariance0, covariance1 : ndarray, shape (n,n)
        Endpoint SPD covariances.
    kind : {'c', 'perp'}
        Corollary 1 or 2, respectively, without symmetrization.

    Returns
    -------
    float
        Squared divergence, not asserted equal to squared Fisher distance.
    """
    mu1, covariance1 = validate_mvn(mu1, covariance1)
    objective = TriangleObjective(mu1[None], covariance1[None], kind)
    return float(objective.evaluate(mu0, covariance0)[0][0])


def triangle_center(means: np.ndarray, covariances: np.ndarray, *, kind: str,
                    gtol: float = 1e-7, maxiter: int = 500) -> dict:
    """Fit a directed divergence center with two deterministic BFGS starts.

    Parameters
    ----------
    means, covariances : ndarray, shapes (k,n), (k,n,n)
        Nonempty training sample with SPD covariances.
    kind : {'c', 'perp'}
        Divergence minimized as mean(delta_squared(center, sample))/2.
    gtol : float
        Positive coordinate and intrinsic gradient qualification threshold.
    maxiter : int
        Maximum BFGS iterations per start.

    Returns
    -------
    dict
        Mean, covariance, qualification, objective and restart diagnostics.
        Best numerical stationary point found, not a certified global minimum.
    """
    original = TriangleObjective(means, covariances, kind)
    if gtol <= 0 or not np.isfinite(gtol) or maxiter < 1:
        raise ValueError('Require positive gtol and maxiter.')
    origin = original.means.mean(axis=0)
    values, vectors = np.linalg.eigh(original.covariances.mean(axis=0))
    root = (vectors * np.sqrt(values)) @ vectors.T
    whitening = (vectors / np.sqrt(values)) @ vectors.T
    normalized_means = (original.means - origin) @ whitening
    normalized_covariances = whitening @ original.covariances @ whitening
    objective = TriangleObjective(normalized_means, normalized_covariances, kind)
    n = len(origin)
    rows, columns = np.tril_indices(n)
    diagonal = rows == columns

    def unpack(parameters: np.ndarray) -> tuple:
        """Decode a mean and log-diagonal Cholesky covariance.

        Parameters
        ----------
        parameters : ndarray
            n mean coordinates followed by n(n+1)/2 Cholesky coordinates.

        Returns
        -------
        tuple
            Mean vector and lower triangular Cholesky factor.
        """
        lower = np.zeros((n, n))
        lower[rows, columns] = parameters[n:]
        lower[np.diag_indices(n)] = np.exp(parameters[n:][diagonal])
        return parameters[:n], lower

    def value_gradient(parameters: np.ndarray) -> tuple:
        """Evaluate the center objective and analytic Cholesky-coordinate gradient.

        Parameters
        ----------
        parameters : ndarray
            Unconstrained mean and log-Cholesky coordinates.

        Returns
        -------
        tuple
            Scalar objective and gradient vector.
        """
        mean, lower = unpack(parameters)
        divergences, mean_gradient, matrix_gradient = objective.evaluate(mean, lower @ lower.T)
        cholesky_gradient = (matrix_gradient.mean(axis=0) @ lower)[rows, columns]
        cholesky_gradient[diagonal] *= np.diag(lower)
        return float(divergences.mean() / 2), np.concatenate((mean_gradient.mean(axis=0) / 2, cholesky_gradient))

    starts = []
    for covariance in (np.eye(n), normalized_covariances[0]):
        lower = np.linalg.cholesky(covariance)
        coordinates = lower[rows, columns].copy()
        coordinates[diagonal] = np.log(np.diag(lower))
        starts.append(np.concatenate((np.zeros(n), coordinates)))
    runs = []
    candidates = []
    for start in starts:
        result = minimize(value_gradient, start, jac=True, method='BFGS',
                          options=dict(gtol=gtol / 10, maxiter=maxiter))
        mean, lower = unpack(result.x)
        covariance = lower @ lower.T
        _, mean_gradient, matrix_gradient = objective.evaluate(mean, covariance)
        gm, gc = mean_gradient.mean(axis=0) / 2, matrix_gradient.mean(axis=0) / 2
        intrinsic_norm = float(np.sqrt(gm @ covariance @ gm + 2 * np.sum((lower.T @ gc @ lower)**2)))
        coordinate_norm = float(np.linalg.norm(result.jac, ord=np.inf))
        qualified = intrinsic_norm <= gtol and coordinate_norm <= gtol
        runs.append(dict(success=bool(result.success), message=str(result.message),
                         iterations=int(result.nit), evaluations=int(result.nfev),
                         objective=float(result.fun), intrinsic_gradient_norm=intrinsic_norm,
                         coordinate_gradient_norm=coordinate_norm, qualified=qualified,
                         initial_parameters=start, final_parameters=result.x))
        candidates.append((mean, covariance))
    selected = min(range(len(runs)), key=lambda index: runs[index]['objective'])
    mean, covariance = candidates[selected]
    return dict(mean=origin + root @ mean, covariance=root @ covariance @ root,
                converged=runs[selected]['qualified'], objective=runs[selected]['objective'],
                selected_start=selected, runs=runs)
