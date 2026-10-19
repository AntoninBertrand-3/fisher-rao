"""Independent formula, derivative and invariance checks for triangle divergences."""

import unittest

import numpy as np
from scipy.linalg import eigvalsh

from src.triangles import TriangleObjective, triangle_center, triangle_divergence_squared


class TestTriangleObjective(unittest.TestCase):
    def test_formula_gradient_and_affine_invariance(self) -> None:
        """Check literal formulas independently, directional derivatives and affine invariance."""
        cases = [(kind, dimension) for dimension in [1, 3, 5, 10] for kind in ['c', 'perp']]
        for kind, dimension in cases:
            with self.subTest(kind=kind, dimension=dimension):
                rng = np.random.default_rng(940 + dimension)
                raw = rng.normal(size=(4, dimension, dimension))
                covariances = raw @ raw.swapaxes(-1, -2) + np.eye(dimension)
                means = rng.normal(size=(4, dimension))
                base_mean, base_covariance = means[0], covariances[0]
                objective = TriangleObjective(means[1:], covariances[1:], kind)
                values, gm, gc = objective.evaluate(base_mean, base_covariance)
                for index in range(3):
                    delta = means[index+1] - base_mean
                    q = delta @ np.linalg.solve(base_covariance, delta)
                    if kind == 'c':
                        ell = (np.linalg.slogdet(covariances[index+1])[1] - np.linalg.slogdet(base_covariance)[1]) / dimension
                        c = np.exp(ell)
                        expected = (2*np.arccosh((c+1+q/2)/(2*np.sqrt(c)))**2
                                    +(dimension-1)*ell**2/2
                                    +np.sum(np.log(eigvalsh(c*base_covariance, covariances[index+1]))**2)/2)
                    else:
                        expected = (np.arccosh(1+q)**2
                                    +np.sum(np.log(eigvalsh(base_covariance+np.outer(delta,delta)/2,
                                                            covariances[index+1]))**2))/2
                    np.testing.assert_allclose(values[index], expected, rtol=1e-12, atol=1e-12)
                direction_mean = rng.normal(size=dimension)
                raw_direction = rng.normal(size=(dimension, dimension))
                direction_covariance = (raw_direction + raw_direction.T)/2
                step = 1e-5
                numerical = (objective.evaluate(base_mean+step*direction_mean, base_covariance+step*direction_covariance)[0]
                             -objective.evaluate(base_mean-step*direction_mean, base_covariance-step*direction_covariance)[0])/(2*step)
                analytic = gm @ direction_mean + np.sum(gc*direction_covariance, axis=(1,2))
                np.testing.assert_allclose(numerical, analytic, rtol=2e-6, atol=2e-8)
                transform = rng.normal(size=(dimension, dimension)) + 3*np.eye(dimension)
                offset = rng.normal(size=dimension)
                transformed = TriangleObjective(means[1:] @ transform.T + offset,
                                                transform @ covariances[1:] @ transform.T, kind)
                np.testing.assert_allclose(transformed.evaluate(transform@base_mean+offset,
                                                                transform@base_covariance@transform.T)[0], values,
                                           rtol=1e-10, atol=1e-10)


class TestTriangleDivergenceSquared(unittest.TestCase):
    def test_special_cases_and_direction(self) -> None:
        """Check identical, equal-mean, anisotropic and genuinely asymmetric inputs."""
        for kind in ['c', 'perp']:
            with self.subTest(kind=kind):
                covariance = np.diag([.001, 1., 100.])
                zero = np.zeros(3)
                np.testing.assert_allclose(triangle_divergence_squared(zero, covariance, zero, covariance, kind=kind), 0, atol=1e-25)
                np.testing.assert_allclose(triangle_divergence_squared(zero, covariance, zero, np.eye(3), kind=kind),
                                           np.sum(np.log(np.diag(covariance))**2)/2, atol=1e-12)
                other = np.diag([.5, 2., 4.])
                forward = triangle_divergence_squared(zero, np.eye(3), np.ones(3), other, kind=kind)
                reverse = triangle_divergence_squared(np.ones(3), other, zero, np.eye(3), kind=kind)
                self.assertTrue(abs(forward-reverse) > 1e-3)
                with self.assertRaises(ValueError):
                    triangle_divergence_squared(zero, -np.eye(3), zero, np.eye(3), kind=kind)

    def test_univariate_and_equal_covariance_c(self) -> None:
        """Recover exact Fisher squared distances for the c construction in exact cases."""
        np.testing.assert_allclose(triangle_divergence_squared(np.array([0.]), np.array([[2.]]),
                                                              np.array([3.]), np.array([[4.]]), kind='c'),
                                   2*np.arccosh((2+4+9/2)/(2*np.sqrt(8)))**2, atol=1e-12)
        q = 3.
        np.testing.assert_allclose(triangle_divergence_squared(np.zeros(3), np.eye(3),
                                                              np.ones(3), np.eye(3), kind='c'),
                                   2*np.arccosh(1+q/4)**2, atol=1e-12)


class TestTriangleCenter(unittest.TestCase):
    def test_center_identity_and_affine_equivariance(self) -> None:
        """Recover identical samples and check fitted centers under dense affine transforms."""
        for kind in ['c', 'perp']:
            with self.subTest(kind=kind):
                mean = np.array([.3, -.2])
                covariance = np.array([[2., .2], [.2, 1.]])
                identical = triangle_center(np.repeat(mean[None], 3, axis=0),
                                            np.repeat(covariance[None], 3, axis=0), kind=kind)
                self.assertTrue(identical['converged'])
                np.testing.assert_allclose(identical['mean'], mean, atol=1e-10)
                np.testing.assert_allclose(identical['covariance'], covariance, atol=1e-10)
                means = np.array([[0., .1], [.4, -.2], [-.2, .3]])
                covariances = np.array([covariance, np.eye(2), np.diag([.8, 1.5])])
                transform = np.array([[1.5, .3], [-.4, .7]])
                shift = np.array([2., -1.])
                original = triangle_center(means, covariances, kind=kind)
                changed = triangle_center(means@transform.T+shift, transform@covariances@transform.T, kind=kind)
                self.assertTrue(original['converged'] and changed['converged'])
                np.testing.assert_allclose(changed['mean'], transform@original['mean']+shift, atol=1e-6)
                np.testing.assert_allclose(changed['covariance'], transform@original['covariance']@transform.T, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
