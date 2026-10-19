"""Independent differential, lift and metric checks for horizontality."""

import unittest

import numpy as np
from scipy.linalg import expm, logm
from scipy.optimize import minimize

from src.geometry import normalize_pair
from src.horizontality import HorizontalFiberObjective, log_divided_differences


class TestHorizontalFiberObjective(unittest.TestCase):
    def test_unresolved_lift_is_not_qualified(self) -> None:
        """Reject spectral qualification of an enormous, unresolved fiber parameter."""
        objective = HorizontalFiberObjective([1., 2.], np.diag([.2, 5.]))
        self.assertTrue(objective.diagnostics(np.zeros(1))['spectrally_resolved'])
        self.assertFalse(objective.diagnostics(np.array([1e14]))['spectrally_resolved'])

    def test_fiber_projection_logarithm_and_gradient(self) -> None:
        """Check the endpoint, symmetric-space constraint and centered derivatives."""
        mean = np.array([1., -0.5, 2.])
        covariance = np.array([[2., .3, .1], [.3, 1., .2], [.1, .2, 3.]])
        objective = HorizontalFiberObjective(mean, covariance)
        parameters = np.array([.2, -.3, .1])
        factor = objective.factor(parameters)
        lifted = factor @ factor.T
        logarithm = objective.spectral_log(parameters)[0]
        np.testing.assert_allclose(logarithm, logm(lifted), atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(expm(logarithm), lifted, atol=2e-12, rtol=1e-12)
        recovered_covariance = np.linalg.solve(lifted[:3, :3], np.eye(3))
        recovered_mean = np.linalg.solve(lifted[:3, :3], lifted[:3, 3])
        np.testing.assert_allclose(recovered_mean, mean, atol=1e-13)
        np.testing.assert_allclose(recovered_covariance, covariance, atol=1e-13)
        self.assertTrue(objective.diagnostics(parameters)['log_structure_error'] < 1e-12)
        _, gradient = objective.value_and_gradient(parameters)
        epsilon = 1e-5
        numerical = np.empty(3)
        for index in range(3):
            direction = np.eye(3)[index]
            numerical[index] = (objective.value_and_gradient(parameters + epsilon * direction)[0]
                                - objective.value_and_gradient(parameters - epsilon * direction)[0]) / (2 * epsilon)
        np.testing.assert_allclose(gradient, numerical, atol=2e-10, rtol=2e-8)


class TestLogDividedDifferences(unittest.TestCase):
    def test_frechet_log_against_block_logarithm(self) -> None:
        """Compare the spectral derivative to the independent block-log identity."""
        rng = np.random.default_rng(441)
        basis, _ = np.linalg.qr(rng.standard_normal((4, 4)))
        values = np.array([0.3, 1., 1., 3.])
        matrix = (basis * values) @ basis.T
        direction = rng.standard_normal((4, 4))
        direction += direction.T
        expected = logm(np.block([[matrix, direction], [np.zeros_like(matrix), matrix]]))[:4, 4:]
        actual = basis @ (log_divided_differences(values) * (basis.T @ direction @ basis)) @ basis.T
        np.testing.assert_allclose(actual, expected, atol=2e-13, rtol=2e-13)
        np.testing.assert_allclose(log_divided_differences(np.array([1., 1. + 1e-14])), 1., atol=2e-14)


class TestHorizontalMetric(unittest.TestCase):
    def test_horizontal_metric_special_cases(self) -> None:
        """Check canonical scaling in exact cases at the zero fiber coordinate."""
        cases = [
            ([3.], [[4.]], np.sqrt(2) * np.arccosh(19 / 8)),
            ([0., 0., 0.], np.diag([.1, 1., 10.]), np.log(10)),
            ([2., -1., 3.], np.eye(3), np.sqrt(2) * np.arccosh(1 + 14 / 4)),
            ([2., 0.], np.diag([4., .5]), np.sqrt(2 * np.arccosh(14 / 8)**2 + .5 * np.log(.5)**2)),
            ([0., 0.], np.eye(2), 0.),
        ]
        for mean, covariance, expected in cases:
            with self.subTest(mean=mean, covariance=covariance, expected=expected):
                objective = HorizontalFiberObjective(mean, covariance)
                diagnostics = objective.diagnostics(np.zeros(objective.parameter_count))
                self.assertTrue(diagnostics['horizontal_residual'] < 1e-12)
                np.testing.assert_allclose(diagnostics['distance'], expected, rtol=0, atol=2e-12)
                np.testing.assert_allclose(diagnostics['initial_fisher_speed'], expected, rtol=0, atol=2e-12)

    def test_horizontal_solve_affine_invariance_and_reversal(self) -> None:
        """Check nontrivial residual solving under affine transformation and reversal."""
        mu0, mu1 = np.array([.2, -.3]), np.array([2., 3.])
        cov0, cov1 = np.eye(2), np.diag([.2, 5.])
        transform = np.array([[2., .3], [-.4, 1.]])
        shift = np.array([3., -2.])
        pairs = [(mu0, cov0, mu1, cov1), (mu1, cov1, mu0, cov0),
                 (transform @ mu0 + shift, transform @ cov0 @ transform.T,
                  transform @ mu1 + shift, transform @ cov1 @ transform.T)]
        distances = []
        for pair in pairs:
            objective = HorizontalFiberObjective(*normalize_pair(*pair))
            result = minimize(objective.value_and_gradient, np.zeros(1), jac=True,
                              method='BFGS', options={'gtol': 1e-10})
            diagnostics = objective.diagnostics(result.x)
            self.assertTrue(result.success)
            self.assertTrue(diagnostics['horizontal_residual'] < 1e-8)
            distances.append(diagnostics['distance'])
        np.testing.assert_allclose(distances, distances[0], atol=1e-10, rtol=0)


if __name__ == "__main__":
    unittest.main()
