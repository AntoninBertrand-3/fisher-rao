import unittest
import numpy as np
from scipy.linalg import sqrtm
from src.distance import dist2_L2, dist2_d0, dist2_dF
from src.mean import arithmetic_mean, geometric_mean, frechet_mean
from src.simulated_data import random_SPD, random_MVNs


class TestMeans(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.mu = np.array([1., -0.5, 2.])
        self.C = random_SPD(3, 10.0)
        self.means, self.covs = random_MVNs(self.mu, self.C, 10, radius=0.5)
        A3 = np.array([[3., 1., 0.],
                       [1., 2., 1.],
                       [0., 1., 2.]])
        B3 = np.array([[2., 0., 1.],
                       [0., 2., 0.5],
                       [1., 0.5, 3.]])
        self.A3, self.B3 = A3, B3

    # ---------- Moyenne arithmétique ----------
    def test_arithmetic_mean(self):
        m, S = arithmetic_mean(self.means, self.covs)
        self.assertTrue(np.allclose(m, self.means.mean(axis=0)))
        self.assertTrue(np.allclose(S, self.covs.mean(axis=0)))

    # ---------- Moyenne géométrique SPD ----------
    def test_geometric_two_matrix_closed_form(self):
        G_impl = geometric_mean(np.stack([self.A3, self.B3]), max_iter=300, tol=1e-12)
        A_half = sqrtm(self.A3)
        A_mhalf = np.linalg.inv(A_half)
        G_closed = A_half @ sqrtm(A_mhalf @ self.B3 @ A_mhalf) @ A_half
        self.assertTrue(np.allclose(G_impl, G_closed, atol=1e-8))

    def test_geometric_commuting_case(self):
        # Matrices diagonales : moyenne géométrique coefficient par coefficient.
        diagonals = np.array([[1., 2., 4.], [3., 5., 2.], [7., 1., 9.]])
        G = geometric_mean(np.array([np.diag(d) for d in diagonals]), max_iter=200, tol=1e-12)
        self.assertTrue(np.allclose(G, np.diag(np.exp(np.log(diagonals).mean(axis=0))), atol=1e-10))

    # ---------- Moyenne de Fréchet ----------
    def test_frechet_L2_is_arithmetic(self):
        # Pour la distance euclidienne, la moyenne de Fréchet est la moyenne arithmétique.
        m, S = frechet_mean(self.means, self.covs, dist2_L2)
        am, aS = arithmetic_mean(self.means, self.covs)
        self.assertTrue(np.allclose(m, am, atol=1e-6))
        self.assertTrue(np.allclose(S, aS, atol=1e-6))

    def test_frechet_single_mvn(self):
        m, S = frechet_mean(self.means[:1], self.covs[:1], dist2_d0)
        self.assertTrue(np.allclose(m, self.means[0], atol=1e-5))
        self.assertTrue(np.allclose(S, self.covs[0], atol=1e-5))

    def test_frechet_d0_equal_means(self):
        # Moyennes égales : d_0^2 = dist2_Fisher / 2, donc la covariance est la moyenne géométrique SPD.
        same = np.repeat(self.mu[None], len(self.covs), axis=0)
        m, S = frechet_mean(same, self.covs, dist2_d0)
        G = geometric_mean(self.covs, max_iter=200, tol=1e-12)
        self.assertTrue(np.allclose(m, self.mu, atol=1e-5))
        self.assertTrue(np.allclose(S, G, atol=1e-5))

    def test_frechet_dF_equal_means(self):
        # Même propriété pour d_F, sur un petit exemple (le calcul est plus coûteux).
        same = np.zeros((3, 2))
        covs = np.stack([self.A3[:2, :2], self.B3[:2, :2], np.diag([0.5, 4.])])
        m, S = frechet_mean(same, covs, dist2_dF)
        G = geometric_mean(covs, max_iter=200, tol=1e-12)
        self.assertTrue(np.allclose(m, 0, atol=1e-5))
        self.assertTrue(np.allclose(S, G, atol=1e-5))

    def test_frechet_outputs_spd(self):
        for dist2 in (dist2_d0, dist2_dF):
            m, S = frechet_mean(self.means[:4, :2], self.covs[:4, :2, :2], dist2)
            self.assertTrue(np.allclose(S, S.T, atol=1e-12))
            self.assertTrue((np.linalg.eigvalsh(S) > 0).all())


if __name__ == "__main__":
    unittest.main()
