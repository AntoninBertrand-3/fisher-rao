"""Reproducibility and information-flow checks for the MDM classifiers."""

import unittest

import numpy as np

from src.covariance import estimate_descriptors
from src.distance import d_0
from src.mdm import compare_shared_predictions, evaluate_components, evaluate_trial, evaluate_triangles
from src.triangles import triangle_divergence_squared


def _two_class_sample(count: int, seed: int, batch_size: int) -> dict:
    """Fit Gaussian descriptors to batches drawn from two fixed 3D Gaussians.

    Parameters
    ----------
    count : int
        Number of descriptors per class.
    seed : int
        Seed of the NumPy random generator.
    batch_size : int
        Number of raw vectors per descriptor, greater than 3.

    Returns
    -------
    dict
        Fitted means of shape (2*count, 3), covariances of shape
        (2*count, 3, 3) and balanced labels.
    """
    rng = np.random.default_rng(seed)
    means = np.array([[0., 0., 0.], [.8, -.5, .6]])
    covariances = np.array([np.diag([1., .6, 1.8]),
                            [[1.4, .3, -.2], [.3, .5, .1], [-.2, .1, 2.]]])
    roots = np.linalg.cholesky(covariances)
    batches = rng.standard_normal((2, count, batch_size, 3)) @ roots[:, None].swapaxes(-1, -2) + means[:, None, None]
    fitted_means, fitted_covariances = estimate_descriptors(batches.reshape(2 * count, batch_size, 3))
    return dict(means=fitted_means, covariances=fitted_covariances, labels=np.repeat([0, 1], count))


class TestEvaluateTrial(unittest.TestCase):
    def test_prototypes_do_not_depend_on_test_data(self) -> None:
        """Check both fits use training alone and verify the exact naive distance."""
        train = _two_class_sample(3, 12, 100)
        test = _two_class_sample(2, 13, 100)
        other = _two_class_sample(2, 14, 100)
        first = evaluate_trial('sampled_3d', 12, train, test)
        second = evaluate_trial('sampled_3d', 12, train, other)
        for method in ('naive', 'fisher'):
            for label in range(2):
                prototype = first['methods'][method]['prototypes'][label]
                self.assertIsNotNone(prototype)
                for value, repeated in zip(prototype, second['methods'][method]['prototypes'][label]):
                    np.testing.assert_array_equal(value, repeated)
        for label, (mu, C) in enumerate(first['methods']['naive']['prototypes']):
            selected = train['labels'] == label
            np.testing.assert_allclose(mu, train['means'][selected].mean(axis=0))
            np.testing.assert_allclose(C, train['covariances'][selected].mean(axis=0))
            expected = np.sqrt(np.sum((test['means'] - mu)**2, axis=1)
                               + np.sum((test['covariances'] - C)**2, axis=(1, 2)))
            np.testing.assert_allclose(first['methods']['naive']['distances'][:, label], expected)
        for method in ('d0', 'euclidean_fisher'):
            self.assertIs(first['methods'][method]['prototypes'], first['methods']['fisher']['prototypes'])
            self.assertTrue(first['methods'][method]['training_reused'])
            for label, prototype in enumerate(first['methods']['fisher']['prototypes']):
                target = (test['means'][0], test['covariances'][0])
                expected = (d_0(*prototype, *target) if method == 'd0' else
                            np.sqrt(np.sum((prototype[0]-target[0])**2) + np.sum((prototype[1]-target[1])**2)))
                np.testing.assert_allclose(first['methods'][method]['distances'][0, label], expected)


class TestCompareSharedPredictions(unittest.TestCase):
    def test_unqualified_predictions_are_excluded_from_paired_comparison(self) -> None:
        """Keep failure counts separate from genuine changes in class predictions."""
        reference = dict(predictions=np.array([0, 1, -1]), all_fits_converged=True,
                         qualified_distances=np.array([[True, True], [True, True], [False, False]]),
                         distances=np.array([[1., 2.], [2., 1.], [np.nan, np.nan]]))
        candidate = dict(predictions=np.array([1, -1, 0]),
                         qualified_distances=np.array([[True, True], [False, False], [True, True]]),
                         distances=np.array([[2., 1.], [np.nan, np.nan], [1., 2.]]))
        fits = dict(fisher=reference, d0=candidate, euclidean_fisher=candidate)
        paired = compare_shared_predictions(fits)['d0']
        self.assertEqual(paired['paired_count'], 1)
        self.assertEqual(paired['disagreements'], 1)
        self.assertEqual(paired['distance_count'], 2)
        reference['all_fits_converged'] = False
        paired = compare_shared_predictions(fits)['d0']
        self.assertEqual(paired['paired_count'], 0)
        self.assertIsNone(paired['maximum_relative_error'])


class TestEvaluateTriangles(unittest.TestCase):
    def test_triangle_classifier_training_and_orientation(self) -> None:
        """Check test-independent centers and explicit center-to-observation predictions."""
        train = _two_class_sample(3, 44, 20)
        test = _two_class_sample(2, 45, 20)
        other = _two_class_sample(2, 46, 20)
        result = evaluate_triangles(train, test)
        changed = evaluate_triangles(train, other)
        for kind in ('c', 'perp'):
            method = f'triangle_{kind}'
            for label, prototype in enumerate(result[method]['prototypes']):
                for value, repeated in zip(prototype, changed[method]['prototypes'][label]):
                    np.testing.assert_array_equal(value, repeated)
                expected = triangle_divergence_squared(*prototype, test['means'][0],
                                                       test['covariances'][0], kind=kind)
                np.testing.assert_allclose(result[method]['squared_divergences'][0, label], expected, atol=1e-12)


class TestEvaluateComponents(unittest.TestCase):
    def test_component_information_is_separated(self) -> None:
        """Ensure covariance and mean classifiers ignore the other component and test fitting."""
        train = dict(means=np.array([[0., 0.], [1., 0.], [3., 0.], [4., 0.]]),
                     covariances=np.array([np.eye(2), 2*np.eye(2), 3*np.eye(2), 4*np.eye(2)]),
                     labels=np.array([0, 0, 1, 1]))
        test = dict(means=np.array([[.2, 0.], [3.5, 0.]]),
                    covariances=np.array([np.eye(2), 4*np.eye(2)]), labels=np.array([0, 1]))
        original = evaluate_components(train, test)
        changed_mean = evaluate_components(dict(train, means=train['means'] + 100),
                                           dict(test, means=-test['means']))
        changed_cov = evaluate_components(dict(train, covariances=train['covariances'] * 7),
                                          dict(test, covariances=test['covariances'] * 3))
        np.testing.assert_array_equal(original['covariance_only']['distances'], changed_mean['covariance_only']['distances'])
        np.testing.assert_array_equal(original['mean_only']['distances'], changed_cov['mean_only']['distances'])
        altered_test = evaluate_components(train, dict(test, means=test['means'] + 20, covariances=test['covariances'] * 2))
        for method in original:
            np.testing.assert_array_equal(original[method]['prototypes'], altered_test[method]['prototypes'])


if __name__ == "__main__":
    unittest.main()
