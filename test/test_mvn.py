import unittest

import numpy as np

from src.distance import _reduced_problem, _skew_objective, dist2_d0, dist2_dF
from src.mvn import (_lifted_endpoint, _symmetric_function, _whiten, exp_map_mvn,
                     fisher_norm2, log_map_mvn)
from src.simulated_data import exp_map_spd


class TestMVNMaps(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.rng = rng
        self.random_pairs = []
        for d in [1, 2, 3, 5]:
            X, Z = rng.normal(size=(2, d, d))
            self.random_pairs.append((rng.normal(size=d), X @ X.T + 0.3 * np.eye(d),
                                      2 * rng.normal(size=d), Z @ Z.T + 0.3 * np.eye(d)))
        self.T_affine = np.array([[1.2, .3, 0.], [-.2, .8, .5], [.1, 0., 1.5]])
        self.b_affine = np.array([1., -2., .5])

    # ---------- 1. Lien entre J(K) et le relèvement de Kobayashi ----------
    def test_J_is_ambient_distance_with_S_equal_2K(self):
        # Observation numérique : J(K) = ||log G_1(2K)||_F^2 / 4 pour tout K antisymétrique.
        for mu0, C0, mu1, C1 in self.random_pairs:
            d = mu0.size
            W, Y = _reduced_problem(mu0, C0, mu1, C1)
            nu, C = _whiten(np.linalg.cholesky(C0), mu1 - mu0, C1)
            rows, cols = np.triu_indices(d, 1)
            for _ in range(3):
                K = np.zeros((d, d))
                K[rows, cols] = 2 * self.rng.normal(size=rows.size)
                K -= K.T
                J = _skew_objective(K[rows, cols], W, Y)[0]
                H = _symmetric_function(_lifted_endpoint(nu, C, 2 * K), np.log)
                self.assertAlmostEqual(0.25 * np.sum(H**2) / J, 1.0, places=9)

    def test_log_is_horizontal(self):
        # Au minimiseur, le bloc (1,3) de log G_1(2K*) s'annule (géodésique horizontale).
        for mu0, C0, mu1, C1 in self.random_pairs:
            d = mu0.size
            _, _, K = log_map_mvn(mu0, C0, mu1, C1, return_K=True)
            nu, C = _whiten(np.linalg.cholesky(C0), mu1 - mu0, C1)
            H = _symmetric_function(_lifted_endpoint(nu, C, 2 * K), np.log)
            self.assertLess(np.linalg.norm(H[:d, d + 1:]), 1e-6)

    # ---------- 2. Logarithme ----------
    def test_log_norm_is_distance(self):
        for mu0, C0, mu1, C1 in self.random_pairs:
            v, T = log_map_mvn(mu0, C0, mu1, C1)
            self.assertAlmostEqual(fisher_norm2(C0, v, T) / dist2_dF(mu0, C0, mu1, C1), 1.0, places=9)

    def test_round_trip(self):
        for mu0, C0, mu1, C1 in self.random_pairs:
            v, T = log_map_mvn(mu0, C0, mu1, C1)
            mean, covariance = exp_map_mvn(mu0, C0, v, T)
            np.testing.assert_allclose(mean, mu1, rtol=1e-7, atol=1e-7)
            np.testing.assert_allclose(covariance, C1, rtol=1e-7, atol=1e-7)

    def test_log_same_point(self):
        mu0, C0, _, _ = self.random_pairs[2]
        v, T = log_map_mvn(mu0, C0, mu0, C0)
        np.testing.assert_allclose(v, 0, atol=1e-12)
        np.testing.assert_allclose(T, 0, atol=1e-12)

    def test_log_equal_means(self):
        # Moyennes égales : logarithme SPD C0^{1/2} log(C0^{-1/2} C1 C0^{-1/2}) C0^{1/2}.
        _, C0, _, C1 = self.random_pairs[2]
        mu = np.array([1., 2., 3.])
        sqrt_C0 = _symmetric_function(C0, np.sqrt)
        inv_sqrt_C0 = np.linalg.inv(sqrt_C0)
        expected = sqrt_C0 @ _symmetric_function(inv_sqrt_C0 @ C1 @ inv_sqrt_C0, np.log) @ sqrt_C0
        v, T = log_map_mvn(mu, C0, mu, C1)
        np.testing.assert_allclose(v, 0, atol=1e-10)
        np.testing.assert_allclose(T, expected, rtol=1e-9, atol=1e-10)

    def test_univariate_midpoint(self):
        # n = 1 : le milieu de N(-a, s^2) et N(a, s^2) est N(0, s^2 + a^2 / 2).
        a, s2 = 2., 1.5
        v, T = log_map_mvn(np.array([-a]), np.array([[s2]]), np.array([a]), np.array([[s2]]))
        exact = dist2_d0(np.array([-a]), np.array([[s2]]), np.array([a]), np.array([[s2]]))
        self.assertAlmostEqual(fisher_norm2(np.array([[s2]]), v, T) / exact, 1.0, places=11)
        mean, covariance = exp_map_mvn(np.array([-a]), np.array([[s2]]), v, T, t=0.5)
        np.testing.assert_allclose(mean, [0.], atol=1e-12)
        np.testing.assert_allclose(covariance, [[s2 + a**2 / 2]], rtol=1e-12)

    # ---------- 3. Exponentielle ----------
    def test_exp_initial_point_and_velocity(self):
        mu, C, v, T = self.random_pairs[2]
        T = 0.5 * (T + T.T) / 4
        mean0, covariance0 = exp_map_mvn(mu, C, v, T, t=0.)
        np.testing.assert_allclose(mean0, mu, atol=1e-12)
        np.testing.assert_allclose(covariance0, C, atol=1e-12)
        h = 1e-5
        mean_p, covariance_p = exp_map_mvn(mu, C, v, T, t=h)
        mean_m, covariance_m = exp_map_mvn(mu, C, v, T, t=-h)
        np.testing.assert_allclose((mean_p - mean_m) / (2 * h), v, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose((covariance_p - covariance_m) / (2 * h), T, rtol=1e-7, atol=1e-7)

    def test_exp_equal_means(self):
        # Vitesse moyenne nulle : la moyenne reste fixe et la covariance suit la géodésique SPD.
        mu, C, _, T = self.random_pairs[2]
        T = 0.5 * (T + T.T) / 4
        mean, covariance = exp_map_mvn(mu, C, np.zeros(3), T)
        np.testing.assert_allclose(mean, mu, atol=1e-12)
        np.testing.assert_allclose(covariance, exp_map_spd(C, T[None])[0], rtol=1e-10, atol=1e-12)

    # ---------- 4. Invariance affine ----------
    def test_affine_equivariance(self):
        A, b = self.T_affine, self.b_affine
        mu0, C0, mu1, C1 = self.random_pairs[2]
        v, T = log_map_mvn(mu0, C0, mu1, C1)
        v_t, T_t = log_map_mvn(A @ mu0 + b, A @ C0 @ A.T, A @ mu1 + b, A @ C1 @ A.T)
        np.testing.assert_allclose(v_t, A @ v, rtol=1e-7, atol=1e-7)
        np.testing.assert_allclose(T_t, A @ T @ A.T, rtol=1e-7, atol=1e-7)
        mean, covariance = exp_map_mvn(mu0, C0, v, T, t=0.3)
        mean_t, covariance_t = exp_map_mvn(A @ mu0 + b, A @ C0 @ A.T, A @ v, A @ T @ A.T, t=0.3)
        np.testing.assert_allclose(mean_t, A @ mean + b, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(covariance_t, A @ covariance @ A.T, rtol=1e-10, atol=1e-10)

    def test_fisher_norm2(self):
        # Sigma = 2I : v^T v / 2 + Tr((T / 2)^2) / 2.
        v, T = np.array([1., 0., 2.]), np.diag([1., 2., 0.])
        self.assertAlmostEqual(fisher_norm2(2 * np.eye(3), v, T), 5 / 2 + 5 / 8, places=14)


if __name__ == "__main__":
    unittest.main()
