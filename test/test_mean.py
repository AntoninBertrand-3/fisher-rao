"""Exact cases, invariance and failure contracts for Fisher and covariance-only means."""

import unittest

import numpy as np
from scipy.linalg import expm, fractional_matrix_power, logm

from src.distance import covariance_distance
from src.mean import covariance_mean, frechet_mean_mvn
from src.mvn import exp_map_mvn, log_map_mvn


class TestFrechetMeanMvn(unittest.TestCase):
    def test_univariate_midpoint_and_mean(self) -> None:
        """Check the exact midpoint of N(-a,s²) and N(a,s²), including inflation."""
        result = log_map_mvn([-2.], [[1.]], [2.], [[1.]])
        self.assertTrue(result.reliable)
        midpoint = exp_map_mvn([-2.], [[1.]], result.mean_tangent, result.covariance_tangent, time=.5)
        np.testing.assert_allclose(midpoint[0], [0.], atol=1e-12)
        np.testing.assert_allclose(midpoint[1], [[3.]], atol=1e-12)
        mean = frechet_mean_mvn([[-2.], [2.]], [[[1.]], [[1.]]], return_iterates=True)
        self.assertTrue(mean.converged, mean.message)
        np.testing.assert_allclose(mean.mean, [0.], atol=1e-10)
        np.testing.assert_allclose(mean.covariance, [[3.]], atol=3e-6)
        self.assertTrue(np.all(np.diff([row['objective'] for row in mean.history]) <= 0))
        self.assertEqual(len(mean.iterates), len(mean.history))

    def test_equal_mean_diagonal_covariances(self) -> None:
        """Recover componentwise geometric covariance means and fixed mean vectors."""
        covariances = np.array([np.diag([1., 4.]), np.diag([4., 1.]), np.diag([2., 2.])])
        result = frechet_mean_mvn(np.tile([1., -1.], (3, 1)), covariances)
        self.assertTrue(result.converged, result.message)
        np.testing.assert_allclose(result.mean, [1., -1.], atol=1e-12)
        np.testing.assert_allclose(result.covariance, np.diag([2., 2.]), atol=1e-9)

    def test_singleton_and_identical_inputs(self) -> None:
        """Check exact zero tangent behavior and already-stationary means."""
        mu, C = np.array([1., 2.]), np.array([[2., .3], [.3, 1.]])
        result = log_map_mvn(mu, C, mu, C)
        self.assertTrue(result.reliable)
        self.assertLess(result.distance, 1e-12)
        mean = frechet_mean_mvn(mu[None], C[None])
        self.assertTrue(mean.converged)
        np.testing.assert_allclose(mean.mean, mu, atol=1e-12)
        np.testing.assert_allclose(mean.covariance, C, atol=1e-12)

    def test_failure_and_validation_contracts(self) -> None:
        """Reject invalid inputs and retain explicit nonconvergence status."""
        with self.assertRaises(ValueError):
            exp_map_mvn([0., 0.], np.eye(2), [1., 0.], [[0., 1.], [0., 0.]])
        with self.assertRaises(ValueError):
            log_map_mvn([0.], [[1.]], [1.], [[-1.]])
        with self.assertRaises(ValueError):
            frechet_mean_mvn(np.empty((0, 2)), np.empty((0, 2, 2)))
        result = log_map_mvn([0., 0.], np.eye(2), [2., 3.], np.diag([.2, 5.]),
                             n_starts=1, maxiter=1)
        self.assertFalse(result.reliable)
        with self.assertRaisesRegex(RuntimeError, 'qualification'):
            frechet_mean_mvn([[0., 0.], [2., 3.]], [np.eye(2), np.diag([.2, 5.])],
                             init=(np.zeros(2), np.eye(2)), log_options=dict(n_starts=1, maxiter=1))
        result = frechet_mean_mvn([[-2.], [2.]], [[[1.]], [[1.]]], max_iter=0)
        self.assertFalse(result.converged)
        self.assertIn('Maximum iteration', result.message)

    def test_average_tangent_is_descent_direction(self) -> None:
        """Check the first variation of squared lengths by centered differences."""
        mean, covariance = np.array([.1, -.2]), np.array([[1.3, .1], [.1, .9]])
        means = np.array([[.4, .2], [-.3, .5], [.7, -.1]])
        covariances = np.array([[[1., .2], [.2, 2.]], [[2., -.3], [-.3, 1.]], [[.8, .1], [.1, 1.1]]])
        logs = [log_map_mvn(mean, covariance, mu, C) for mu, C in zip(means, covariances)]
        self.assertTrue(all(result.reliable for result in logs))
        v = np.mean([result.mean_tangent for result in logs], axis=0)
        T = np.mean([result.covariance_tangent for result in logs], axis=0)
        squared_norm = v @ np.linalg.solve(covariance, v) + .5 * np.trace(np.linalg.solve(covariance, T) @ np.linalg.solve(covariance, T))
        values = []
        for time in [-1e-4, 1e-4]:
            base = exp_map_mvn(mean, covariance, v, T, time=time)
            results = [log_map_mvn(*base, mu, C) for mu, C in zip(means, covariances)]
            self.assertTrue(all(result.reliable for result in results))
            values.append(np.mean([result.distance**2 for result in results]) / 2)
        np.testing.assert_allclose((values[1] - values[0]) / 2e-4, -squared_norm, rtol=1e-5, atol=1e-8)
        result = frechet_mean_mvn(means, covariances)
        self.assertTrue(result.converged, result.message)
        B, shift = np.array([[1.1, .2], [-.3, .9]]), np.array([1., -2.])
        transformed = frechet_mean_mvn(means @ B.T + shift, B @ covariances @ B.T)
        self.assertTrue(transformed.converged, transformed.message)
        np.testing.assert_allclose(transformed.mean, B @ result.mean + shift, atol=2e-6)
        np.testing.assert_allclose(transformed.covariance, B @ result.covariance @ B.T, atol=2e-6)

    def test_dense_covariance_midpoint_and_backtracking_failure(self) -> None:
        """Check a noncommuting SPD midpoint and explicit rejected-step termination."""
        C0, C1 = np.array([[2., .4], [.4, 1.]]), np.array([[1., -.2], [-.2, 3.]])
        expected = C0 @ expm(.5 * logm(np.linalg.solve(C0, C1)))
        result = frechet_mean_mvn([[0., 0.], [0., 0.]], [C0, C1])
        self.assertTrue(result.converged, result.message)
        np.testing.assert_allclose(result.covariance, expected, atol=2e-6)
        rejected = frechet_mean_mvn([[-2.], [2.]], [[[1.]], [[1.]]],
                                    step_size=10., max_backtracks=0, return_iterates=True)
        self.assertFalse(rejected.converged)
        self.assertIn('Backtracking failed', rejected.message)
        self.assertEqual(len(rejected.iterates), 1)


class TestCovarianceMean(unittest.TestCase):
    def test_commuting_mean_and_distance(self) -> None:
        """Recover coordinatewise geometric means and the logarithmic distance."""
        diagonals = np.array([[1., 2., 4.], [3., 5., 2.], [7., 1., 9.]])
        matrices = np.array([np.diag(row) for row in diagonals])
        result = covariance_mean(matrices)
        self.assertTrue(result['converged'])
        np.testing.assert_allclose(result['matrix'], np.diag(np.exp(np.log(diagonals).mean(axis=0))), atol=1e-9)
        np.testing.assert_allclose(covariance_distance(matrices[0], matrices[1]),
                                   np.linalg.norm(np.log(diagonals[1] / diagonals[0])), atol=1e-12)

    def test_noncommuting_midpoint_and_congruence(self) -> None:
        """Check the exact two-point geometric mean and general congruence invariance."""
        first = np.diag([1., 2., 5.])
        factor = np.array([[1., .2, -.1], [.4, 1., .3], [.1, -.2, 1.]])
        second = factor @ np.diag([3., 1., 2.]) @ factor.T
        root = fractional_matrix_power(first, .5)
        inverse_root = fractional_matrix_power(first, -.5)
        expected = root @ fractional_matrix_power(inverse_root @ second @ inverse_root, .5) @ root
        result = covariance_mean(np.array([first, second]))
        self.assertTrue(result['converged'])
        np.testing.assert_allclose(result['matrix'], expected, atol=1e-8)
        matrices = np.array([first, second, np.eye(3)])
        original = covariance_mean(matrices)
        transformed = covariance_mean(factor @ matrices @ factor.T)
        self.assertTrue(original['converged'] and transformed['converged'])
        np.testing.assert_allclose(transformed['matrix'], factor @ original['matrix'] @ factor.T, atol=1e-7)
        np.testing.assert_allclose(covariance_distance(first, second),
                                   covariance_distance(factor @ first @ factor.T, factor @ second @ factor.T), atol=1e-12)
        with self.assertRaises(ValueError):
            covariance_mean(np.array([np.diag([1., 0., 1.])]))


if __name__ == "__main__":
    unittest.main()
