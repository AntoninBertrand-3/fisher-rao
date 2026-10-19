"""Validate the SPD exponential map and raw-sample-to-Gaussian simulation."""

import unittest

import numpy as np
from scipy.linalg import expm

from src.simulated_data import exp_map_spd


class TestExpMapSPD(unittest.TestCase):
    def test_exact_cases(self) -> None:
        """Check zero tangents, diagonal formulas, scalar inputs and empty batches."""
        point = np.diag([2., 3.])
        tangents = np.array([np.zeros((2, 2)), np.diag([1., -2.])])
        expected = np.array([point, np.diag([2 * np.exp(.5), 3 * np.exp(-2 / 3)])])
        np.testing.assert_allclose(exp_map_spd(point, tangents), expected, rtol=1e-13, atol=1e-14)
        np.testing.assert_allclose(exp_map_spd(np.array([[2.]]), np.array([[[1.]]])),
                                   [[[2 * np.exp(.5)]]], rtol=1e-13)
        self.assertEqual(exp_map_spd(point, np.empty((0, 2, 2))).shape, (0, 2, 2))

    def test_noncommuting_batch(self) -> None:
        """Compare with a general matrix exponential and check affine equivariance."""
        point = np.array([[2., .4], [.4, 1.]])
        tangents = np.array([[[.3, .2], [.2, -.1]], [[-.2, .5], [.5, .4]]])
        result = exp_map_spd(point, tangents)
        # Equivalent identity Exp_G(eta) = G exp(G^{-1} eta).
        for tangent, actual in zip(tangents, result):
            expected = point @ expm(np.linalg.solve(point, tangent))
            np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
        np.testing.assert_allclose(result, result.transpose(0, 2, 1), atol=1e-14)
        self.assertTrue(np.all(np.linalg.eigvalsh(result) > 0))
        transform = np.array([[1.2, .3], [-.2, .8]])
        transformed = exp_map_spd(transform @ point @ transform.T,
                                  transform @ tangents @ transform.T)
        np.testing.assert_allclose(transformed, transform @ result @ transform.T,
                                   rtol=1e-12, atol=1e-13)


if __name__ == "__main__":
    unittest.main()
