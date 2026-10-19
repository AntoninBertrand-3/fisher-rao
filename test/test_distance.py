import unittest
import numpy as np
from src.distance import dist2_L2, dist2_Fisher, dist2_d0, dist2_dF


class TestDistances(unittest.TestCase):
    def setUp(self):
        self.mu0 = np.array([1., -2.])
        self.mu1 = np.array([.5, 3.])
        self.A = np.array([[2., 0.5],
                           [0.5, 1.5]])
        self.B = np.array([[1.5, 0.2],
                           [0.2, 2.0]])
        self.I = np.eye(2)
        rng = np.random.default_rng(0)
        self.random_pairs = []
        for d in [1, 3, 5, 10]:
            X, Z = rng.normal(size=(2, d, d))
            self.random_pairs.append((rng.normal(size=d), X @ X.T + 0.1 * np.eye(d),
                                      rng.normal(size=d), Z @ Z.T + 0.1 * np.eye(d)))

    # ---------- 1. Distance nulle entre MVN ou matrices identiques ----------
    def test_zero_distance_same_point(self):
        self.assertAlmostEqual(dist2_L2(self.mu0, self.A, self.mu0, self.A), 0.0, places=12)
        self.assertAlmostEqual(dist2_Fisher(self.A, self.A), 0.0, places=12)
        self.assertAlmostEqual(dist2_d0(self.mu0, self.A, self.mu0, self.A), 0.0, places=12)

    # ---------- 2. Symétrie ----------
    def test_symmetry(self):
        for mu0, C0, mu1, C1 in self.random_pairs:
            self.assertAlmostEqual(dist2_L2(mu0, C0, mu1, C1), dist2_L2(mu1, C1, mu0, C0), places=10)
            self.assertAlmostEqual(dist2_Fisher(C0, C1), dist2_Fisher(C1, C0), places=8)
            self.assertAlmostEqual(dist2_d0(mu0, C0, mu1, C1), dist2_d0(mu1, C1, mu0, C0), places=8)

    # ---------- 3. Positivité ----------
    def test_positive(self):
        self.assertGreater(dist2_L2(self.mu0, self.A, self.mu1, self.B), 0.0)
        self.assertGreater(dist2_Fisher(self.A, self.B), 0.0)
        self.assertGreater(dist2_d0(self.mu0, self.A, self.mu1, self.B), 0.0)

    # ---------- 4. Cas connus ----------
    def test_known_identity_case(self):
        mu = np.zeros(2)
        self.assertAlmostEqual(dist2_L2(mu, self.I, mu, 2 * self.I), 2.0, places=12)
        self.assertAlmostEqual(dist2_Fisher(self.I, 2 * self.I), 2 * np.log(2)**2, places=10)

    def test_d0_equal_means(self):
        # Moyennes égales : d_0^2 = dist2_Fisher / 2 (facteur 1/2 de la métrique de Fisher).
        self.assertAlmostEqual(dist2_d0(self.mu0, self.A, self.mu0, self.B),
                               dist2_Fisher(self.A, self.B) / 2, places=10)

    def test_d0_univariate(self):
        # Distance hyperbolique exacte pour n = 1.
        for m0, v0, m1, v1 in [(1., 4., 3., 9.), (0., 1., 2., 1.), (0., 1., 0., 4.)]:
            expected = 2 * np.arccosh(((m1 - m0)**2 + 2 * (v0 + v1)) / (4 * np.sqrt(v0 * v1)))**2
            self.assertAlmostEqual(dist2_d0(np.array([m0]), np.array([[v0]]),
                                            np.array([m1]), np.array([[v1]])), expected, places=10)

    def test_d0_equal_isotropic_covariances(self):
        # Covariances isotropes égales : réduction à un seul facteur hyperbolique.
        mu = np.array([1., 2., -1.])
        expected = 2 * np.arccosh(1 + (mu @ mu) / 8)**2
        self.assertAlmostEqual(dist2_d0(np.zeros(3), 2 * np.eye(3), mu, 2 * np.eye(3)), expected, places=10)

    # ---------- 5. Invariance affine de d_0 ----------
    def test_d0_affine_invariance(self):
        for mu0, C0, mu1, C1 in self.random_pairs:
            d = mu0.size
            T = np.eye(d) + 0.3 * np.triu(np.ones((d, d)), 1)
            b = np.arange(d, dtype=float)
            self.assertAlmostEqual(dist2_d0(T @ mu0 + b, T @ C0 @ T.T, T @ mu1 + b, T @ C1 @ T.T),
                                   dist2_d0(mu0, C0, mu1, C1), places=8)

    # ---------- 6. d_F par optimisation ----------
    def test_dF_below_d0(self):
        for mu0, C0, mu1, C1 in self.random_pairs:
            self.assertLessEqual(dist2_dF(mu0, C0, mu1, C1), dist2_d0(mu0, C0, mu1, C1) + 1e-12)

    def test_dF_exact_cases(self):
        # n = 1 et moyennes égales : d_0 est exact.
        args = (np.array([1.]), np.array([[4.]]), np.array([3.]), np.array([[9.]]))
        self.assertAlmostEqual(dist2_dF(*args), dist2_d0(*args), places=12)
        self.assertAlmostEqual(dist2_dF(self.mu0, self.A, self.mu0, self.B),
                               dist2_Fisher(self.A, self.B) / 2, places=10)

    def test_dF_improves_on_d0(self):
        # Exemple où la correction antisymétrique diminue strictement la distance.
        args = (np.array([.2, -.3]), np.eye(2), np.array([2., 3.]), np.diag([.2, 5.]))
        self.assertLess(dist2_dF(*args), dist2_d0(*args) - 1e-5)

    def test_dF_symmetry_and_affine_invariance(self):
        for mu0, C0, mu1, C1 in self.random_pairs:
            d = mu0.size
            T = np.eye(d) + 0.3 * np.triu(np.ones((d, d)), 1)
            b = np.arange(d, dtype=float)
            reference = dist2_dF(mu0, C0, mu1, C1)
            self.assertAlmostEqual(dist2_dF(mu1, C1, mu0, C0), reference, places=8)
            self.assertAlmostEqual(dist2_dF(T @ mu0 + b, T @ C0 @ T.T, T @ mu1 + b, T @ C1 @ T.T),
                                   reference, places=8)


if __name__ == "__main__":
    unittest.main()
