import unittest

import numpy as np
from scipy.linalg import eigvalsh
from scipy.optimize import check_grad

from src.distance import _evaluate, _prepare, d_0, d_f_reference, skew_objective
from src.geometry import skew_from_parameters


class TestSkewObjective(unittest.TestCase):
    def test_objective_identity_and_gradient(self) -> None:
        """Check J(0)=d_0 squared and independently finite-difference the gradient."""
        mean = np.array([1., 2., -.5])
        covariance = np.array([[2., .3, .1], [.3, 1., .2], [.1, .2, 3.]])
        value = skew_objective(mean, covariance, np.zeros((3, 3)))
        np.testing.assert_allclose(value, d_0(np.zeros(3), np.eye(3), mean, covariance) ** 2,
                                   rtol=1e-6, atol=1e-12)
        whitening, base = _prepare(mean, covariance)
        rows, columns = np.triu_indices(3, 1)

        def value_and_gradient(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            """Evaluate independent skew coordinates for a gradient check.

            Parameters
            ----------
            parameters : ndarray
                Three independent skew entries.

            Returns
            -------
            value, gradient : float, ndarray
                Objective and coordinate derivatives.
            """
            value, gradient = _evaluate(whitening, base, skew_from_parameters(parameters, 3))
            return value, gradient[rows, columns] - gradient[columns, rows]

        error = check_grad(lambda x: value_and_gradient(x)[0],
                           lambda x: value_and_gradient(x)[1], np.array([.2, -.3, .1]), epsilon=1e-6)
        self.assertLess(error, 1e-5)
        with self.assertRaises(ValueError):
            skew_objective(mean, covariance, np.eye(3))


class TestExactDistances(unittest.TestCase):
    def test_univariate_metric(self) -> None:
        """Check the hyperbolic formula with metric ds²=(dmu²+2 dsigma²)/sigma²."""
        cases = [(0., 1., 0., 1.), (1., 4., 3., 9.), (0., 1., 2., 1.), (0., 1., 0., 4.)]
        for mean0, variance0, mean1, variance1 in cases:
            with self.subTest(mean0=mean0, variance0=variance0, mean1=mean1, variance1=variance1):
                argument = ((mean1 - mean0)**2 + 2 * (variance0 + variance1)) / (4 * np.sqrt(variance0 * variance1))
                expected = np.sqrt(2) * np.arccosh(argument)
                args = ([mean0], [[variance0]], [mean1], [[variance1]])
                result = d_f_reference(*args)
                np.testing.assert_allclose(result.distance, expected, rtol=0, atol=1e-14)
                self.assertEqual(result.distance, d_0(*args))
                self.assertEqual(result.parameters.size, 0)
                self.assertEqual(result.runs, ())

    def test_equal_means_metric_factor(self) -> None:
        """Check covariance distance ||log generalized eigenvalues|| / sqrt(2)."""
        covariance0 = np.array([[2., .4], [.4, 1.]])
        covariance1 = np.array([[1., -.2], [-.2, 3.]])
        expected = np.linalg.norm(np.log(eigvalsh(covariance1, covariance0))) / np.sqrt(2)
        args = ([1., 2.], covariance0, [1., 2.], covariance1)
        np.testing.assert_allclose(d_0(*args), expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(d_f_reference(*args).distance, expected, rtol=0, atol=1e-12)

    def test_equal_isotropic_covariances(self) -> None:
        """Check a mean-only change reducible by rotation to one hyperbolic factor."""
        mean = np.array([1., 2., -1.])
        variance = 2.
        expected = np.sqrt(2) * np.arccosh(1 + (mean @ mean) / (4 * variance))
        args = (np.zeros(3), variance * np.eye(3), mean, variance * np.eye(3))
        np.testing.assert_allclose(d_0(*args), expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(d_f_reference(*args).distance, expected, rtol=0, atol=1e-12)

    def test_identical_distributions(self) -> None:
        """Check numerical zero for identical nontrivial multivariate endpoints."""
        mean = np.array([1., -2.])
        covariance = np.array([[2., .4], [.4, 1.]])
        result = d_f_reference(mean, covariance, mean, covariance)
        np.testing.assert_allclose(result.distance, 0., rtol=0, atol=1e-14)
        np.testing.assert_allclose(d_0(mean, covariance, mean, covariance), 0., rtol=0, atol=1e-14)
        self.assertTrue(result.reliable)

    def test_univariate_near_identity(self) -> None:
        """Compare near-identity distances with the stable hyperbolic atanh formula."""
        mean = 1e-8
        variance = 1. + 2e-8
        sigma = np.sqrt(variance)
        difference = (variance - 1) / (sigma + 1)
        # Hyperbolic distance: 2 sqrt(2) atanh(|z1-z0| / |z1-conj(z0)|).
        ratio = np.hypot(mean, np.sqrt(2) * difference) / np.hypot(mean, np.sqrt(2) * (sigma + 1))
        expected = 2 * np.sqrt(2) * np.arctanh(ratio)
        args = ([0.], [[1.]], [mean], [[variance]])
        result = d_f_reference(*args)
        np.testing.assert_allclose(result.distance, expected, rtol=1e-12, atol=0.)
        np.testing.assert_allclose(d_0(*args), expected, rtol=1e-12, atol=0.)
        self.assertTrue(result.reliable)


class TestReferenceDistance(unittest.TestCase):
    def test_reference_improves_and_preserves_affine_invariance(self) -> None:
        """Check genuine improvement, affine invariance, reversal and diagnostics."""
        mean0, mean1 = np.array([.2, -.3]), np.array([2., 3.])
        covariance0, covariance1 = np.eye(2), np.diag([.2, 5.])
        args = (mean0, covariance0, mean1, covariance1)
        result = d_f_reference(*args)
        self.assertLess(result.distance, d_0(*args) - 1e-5)
        self.assertEqual(len(result.runs), 3)
        self.assertTrue(any(run.success for run in result.runs))
        self.assertTrue(result.reliable)
        np.testing.assert_allclose(result.distance**2, min(run.fun for run in result.runs if run.success),
                                   rtol=1e-6, atol=1e-12)
        self.assertTrue(all("jac" in run and "nfev" in run for run in result.runs))
        transform = np.array([[2., .3], [-.4, 1.]])
        shift = np.array([3., -2.])
        transformed = (transform @ mean0 + shift, transform @ covariance0 @ transform.T,
                       transform @ mean1 + shift, transform @ covariance1 @ transform.T)
        np.testing.assert_allclose(d_0(*transformed), d_0(*args), rtol=0, atol=1e-10)
        np.testing.assert_allclose(d_f_reference(*transformed).distance, result.distance, rtol=0, atol=1e-9)
        np.testing.assert_allclose(d_f_reference(mean1, covariance1, mean0, covariance0).distance,
                                   result.distance, rtol=0, atol=1e-9)

    def test_failure_diagnostics_and_domain_guard(self) -> None:
        """Preserve iteration-limit failures and reject a materially invalid spectrum."""
        result = d_f_reference([0., 0.], np.eye(2), [2., 3.], np.diag([.2, 5.]), maxiter=1)
        self.assertFalse(any(run.success for run in result.runs))
        self.assertFalse(result.reliable)
        self.assertIn("not qualified as reliable", result.reliability_message)
        self.assertLessEqual(result.distance, d_0([0., 0.], np.eye(2), [2., 3.], np.diag([.2, 5.])))
        with self.assertRaises(FloatingPointError):
            _evaluate(np.eye(2), .9 * np.eye(2), np.zeros((2, 2)))
        value, gradient = _evaluate(np.eye(2), np.eye(2), np.zeros((2, 2)))
        self.assertEqual(value, 0)
        np.testing.assert_allclose(gradient, 4 * np.eye(2))


if __name__ == "__main__":
    unittest.main()
