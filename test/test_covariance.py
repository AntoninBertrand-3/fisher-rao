"""Validate covariance conventions of the descriptor estimator."""

import unittest

import numpy as np

from src.covariance import estimate_descriptors


class TestEstimateDescriptors(unittest.TestCase):
    def test_sample_estimates_match_numpy(self) -> None:
        """Check vectorized unbiased covariance estimates against independent np.cov."""
        cases = [(3, 10), (10, 15), (10, 30), (10, 60)]
        for dimension, batch_size in cases:
            with self.subTest(dimension=dimension, batch_size=batch_size):
                batches = np.random.default_rng(73).normal(size=(5, batch_size, dimension))
                means, covariances = estimate_descriptors(batches)
                for index, batch in enumerate(batches):
                    np.testing.assert_allclose(means[index], np.mean(batch, axis=0), atol=1e-14)
                    np.testing.assert_allclose(covariances[index], np.cov(batch, rowvar=False, ddof=1), atol=1e-14)
                with self.assertRaises(ValueError):
                    estimate_descriptors(np.zeros((2, 10, 3)))
                with self.assertRaises(ValueError):
                    estimate_descriptors(np.ones((2, 3, 3)))


if __name__ == "__main__":
    unittest.main()
