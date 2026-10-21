import unittest

import numpy as np
from scipy.linalg import expm

from src.simulated_data import (exp_map_spd, project_to_spd, projection, random_MVNs,
                                random_SPD, random_tangent_vector, sym)


class TestRandomSPD(unittest.TestCase):
    def test_sym(self):
        X = np.arange(9.).reshape(3, 3)
        self.assertTrue(np.allclose(sym(X), sym(X).T))
        self.assertTrue(np.allclose(sym(X) + 0.5 * (X - X.T), X))

    def test_condition_number_and_spd(self):
        np.random.seed(0)
        for d in [2, 5, 10]:
            for kappa in [1.0, 10.0, 1000.0]:
                A = random_SPD(d, kappa)
                vals = np.linalg.eigvalsh(A)
                self.assertTrue(np.allclose(A, A.T, atol=1e-12))
                self.assertTrue((vals > 0).all())
                self.assertAlmostEqual(vals[-1] / vals[0], kappa, places=6)
        with self.assertRaises(ValueError):
            random_SPD(3, 0.5)


class TestRandomMVNs(unittest.TestCase):
    def test_shapes_and_spd(self):
        np.random.seed(0)
        mu, C = np.array([1., 2., 3.]), random_SPD(3, 10.0)
        means, covs = random_MVNs(mu, C, 50, radius=0.3)
        self.assertEqual(means.shape, (50, 3))
        self.assertEqual(covs.shape, (50, 3, 3))
        self.assertTrue(np.allclose(covs, covs.transpose(0, 2, 1), atol=1e-12))
        self.assertTrue((np.linalg.eigvalsh(covs) > 0).all())

    def test_zero_radius(self):
        mu, C = np.array([1., -1.]), np.array([[2., .3], [.3, 1.]])
        means, covs = random_MVNs(mu, C, 4, radius=0.0)
        self.assertTrue(np.allclose(means, mu))
        self.assertTrue(np.allclose(covs, C))


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


class TestTangentVectors(unittest.TestCase):
    def test_projection(self):
        point = np.eye(3)
        X = np.arange(9.).reshape(3, 3)
        np.testing.assert_array_equal(projection(point, X), sym(X))
        np.testing.assert_array_equal(projection(point, X + 2j * X), sym(X))

    def test_random_tangent_vector(self):
        np.random.seed(0)
        eta = random_tangent_vector(4, np.eye(4))
        self.assertEqual(eta.shape, (4, 4))
        np.testing.assert_array_equal(eta, eta.T)


class TestProjectToSPD(unittest.TestCase):
    def test_spd_matrix_unchanged(self):
        A = np.array([[2., .4], [.4, 1.]])
        np.testing.assert_allclose(project_to_spd(A), A, rtol=1e-13, atol=1e-14)

    def test_eigenvalue_clipping(self):
        # Valeurs propres -1, 0.5, 20 dans une base non canonique.
        Q = np.linalg.qr(np.array([[1., 2., 0.], [0., 1., 3.], [1., 0., 1.]]))[0]
        A = (Q * np.array([-1., .5, 20.])) @ Q.T
        np.testing.assert_allclose(np.linalg.eigvalsh(project_to_spd(A, eps=1e-3)),
                                   [1e-3, .5, 20.], rtol=1e-12)
        np.testing.assert_allclose(np.linalg.eigvalsh(project_to_spd(A, eps=1e-3, max_eig=10.)),
                                   [1e-3, .5, 10.], rtol=1e-12)

    def test_nonsymmetric_input(self):
        A = np.array([[1., 2.], [0., 1.]])
        np.testing.assert_allclose(project_to_spd(A), project_to_spd(sym(A)), atol=0)


if __name__ == "__main__":
    unittest.main()
