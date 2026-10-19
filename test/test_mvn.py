"""Independent special cases and reconstruction checks for the MVN exponential and logarithm."""

import unittest

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from src.distance import d_f_reference
from src.mvn import exp_map_mvn, log_map_mvn


class TestExpMapMvn(unittest.TestCase):
    def test_exponential_initial_velocity_and_covariance_case(self) -> None:
        """Check initial derivatives and the exact covariance-only exponential."""
        mean = np.array([.3, -.2])
        covariance = np.array([[2., .3], [.3, 1.]])
        v, T = np.array([.4, -.1]), np.array([[.2, .3], [.3, -.2]])
        plus = exp_map_mvn(mean, covariance, v, T, time=1e-5)
        minus = exp_map_mvn(mean, covariance, v, T, time=-1e-5)
        np.testing.assert_allclose((plus[0] - minus[0]) / 2e-5, v, atol=1e-10)
        np.testing.assert_allclose((plus[1] - minus[1]) / 2e-5, T, atol=1e-10)
        actual = exp_map_mvn(mean, covariance, np.zeros(2), T)
        np.testing.assert_allclose(actual[0], mean, atol=1e-14)
        np.testing.assert_allclose(actual[1], covariance @ expm(np.linalg.solve(covariance, T)), atol=1e-12)

    def test_exponential_against_geodesic_ode(self) -> None:
        """Compare the lifted construction with independent Fisher ODE integration."""
        a = np.array([.4, -.3])
        A = np.array([[.2, .1], [.1, -.2]])

        def rhs(time: float, state: np.ndarray) -> np.ndarray:
            """Evaluate Fisher geodesic equations with the mean minus-sign convention.

            Parameters
            ----------
            time : float
                Integration time, unused in this autonomous system.
            state : ndarray
                Flattened mean, covariance and their velocities.

            Returns
            -------
            ndarray
                Flattened first and second derivatives.
            """
            C, v, T = state[2:6].reshape(2, 2), state[6:8], state[8:].reshape(2, 2)
            return np.r_[v, T.ravel(), T @ np.linalg.solve(C, v),
                         (T @ np.linalg.solve(C, T) - np.outer(v, v)).ravel()]

        initial = np.r_[np.zeros(2), np.eye(2).ravel(), a, A.ravel()]
        solution = solve_ivp(rhs, (0., 1.), initial, rtol=1e-11, atol=1e-13)
        self.assertTrue(solution.success)
        mean, covariance = exp_map_mvn(np.zeros(2), np.eye(2), a, A)
        np.testing.assert_allclose(mean, solution.y[:2, -1], atol=2e-11)
        np.testing.assert_allclose(covariance, solution.y[2:6, -1].reshape(2, 2), atol=2e-11)


class TestLogMapMvn(unittest.TestCase):
    def test_log_reconstruction_affine_equivariance_and_distance(self) -> None:
        """Check a noncommuting pair, tangent transformation, and independent distance."""
        mu0, C0 = np.array([.2, -.3]), np.array([[1.5, .2], [.2, .8]])
        mu1, C1 = np.array([1., .7]), np.array([[.8, -.1], [-.1, 2.]])
        result = log_map_mvn(mu0, C0, mu1, C1)
        self.assertTrue(result.reliable)
        endpoint = exp_map_mvn(mu0, C0, result.mean_tangent, result.covariance_tangent)
        np.testing.assert_allclose(endpoint[0], mu1, atol=2e-8)
        np.testing.assert_allclose(endpoint[1], C1, atol=2e-8)
        np.testing.assert_allclose(result.distance, d_f_reference(mu0, C0, mu1, C1).distance, rtol=0, atol=1e-8)
        B, b = np.array([[1.2, .3], [-.2, .9]]), np.array([2., -1.])
        transformed = log_map_mvn(B @ mu0 + b, B @ C0 @ B.T, B @ mu1 + b, B @ C1 @ B.T)
        self.assertTrue(transformed.reliable)
        np.testing.assert_allclose(transformed.mean_tangent, B @ result.mean_tangent, atol=2e-8)
        np.testing.assert_allclose(transformed.covariance_tangent, B @ result.covariance_tangent @ B.T, atol=2e-8)

    def test_generated_geodesic_round_trip(self) -> None:
        """Recover moderate seeded tangents in dimensions above two."""
        cases = [3, 4]
        for dimension in cases:
            with self.subTest(dimension=dimension):
                rng = np.random.default_rng(503 + dimension)
                a = rng.normal(size=dimension) * .25
                raw = rng.normal(size=(dimension, dimension)) * .15
                A = (raw + raw.T) / 2
                endpoint = exp_map_mvn(np.zeros(dimension), np.eye(dimension), a, A)
                result = log_map_mvn(np.zeros(dimension), np.eye(dimension), *endpoint)
                self.assertTrue(result.reliable)
                np.testing.assert_allclose(result.mean_tangent, a, atol=1e-7)
                np.testing.assert_allclose(result.covariance_tangent, A, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
