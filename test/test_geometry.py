import unittest

import numpy as np

from src.geometry import normalize_pair, skew_from_parameters, validate_mvn


class TestValidateMVN(unittest.TestCase):
    def test_invalid_covariance(self) -> None:
        """Reject invalid covariances."""
        cases = [np.zeros((2, 2)), np.diag([1., -1.]), [[1., .2], [0., 1.]],
                 [[np.nan, 0], [0, 1]], np.eye(3), np.eye(2) * 1j]
        for covariance in cases:
            with self.subTest(covariance=covariance):
                with self.assertRaises(ValueError):
                    validate_mvn(np.zeros(2), covariance)

    def test_invalid_mean(self) -> None:
        """Reject invalid means."""
        for mean in [[], [[0., 1.]], [np.inf, 0.], [1j, 0.]]:
            with self.subTest(mean=mean):
                with self.assertRaises(ValueError):
                    validate_mvn(mean, np.eye(2))

    def test_validation_and_normalization(self) -> None:
        """Check SPD preservation, input ownership, and symmetric whitening."""
        covariance = np.array([[2., .4], [.4, 1.]])
        mean = np.array([1., 2.])
        validated_mean, validated_covariance = validate_mvn(mean, covariance)
        self.assertFalse(np.shares_memory(mean, validated_mean))
        self.assertFalse(np.shares_memory(covariance, validated_covariance))
        normalized_mean, normalized_covariance = normalize_pair(mean, covariance, mean, covariance)
        np.testing.assert_allclose(normalized_mean, 0, atol=1e-14)
        np.testing.assert_allclose(normalized_covariance, np.eye(2), atol=1e-14)
        with self.assertRaises(ValueError):
            normalize_pair([0], [[1]], [0, 0], np.eye(2))


class TestSkewFromParameters(unittest.TestCase):
    def test_skew_coordinates(self) -> None:
        """Check coordinate ordering, dimension one, and invalid coordinate counts."""
        np.testing.assert_array_equal(skew_from_parameters([1, 2, 3], 3),
                                      [[0, 1, 2], [-1, 0, 3], [-2, -3, 0]])
        np.testing.assert_array_equal(skew_from_parameters([], 1), [[0]])
        with self.assertRaises(ValueError):
            skew_from_parameters([1, 2], 3)
        with self.assertRaises(ValueError):
            skew_from_parameters([], 0)


if __name__ == "__main__":
    unittest.main()
