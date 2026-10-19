"""Minimum-distance-to-mean (MDM) classifiers on Gaussian descriptors."""

from time import perf_counter

import numpy as np

from src.distance import covariance_distance, d_0, d_f_reference
from src.mean import covariance_mean, frechet_mean_mvn
from src.mvn import log_map_mvn
from src.triangles import TriangleObjective, triangle_center

LOG_OPTIONS = dict(n_starts=2, seed=1701, gtol=1e-11, maxiter=2000,
                   residual_tol=1e-8, endpoint_tol=1e-7)

METHODS = ('naive', 'fisher', 'd0', 'euclidean_fisher')


def distance_record(base: tuple, target: tuple) -> dict:
    """Compute a qualified numerical Fisher length and retain solver diagnostics.

    Parameters
    ----------
    base, target : tuple
        Gaussian parameters (mean, covariance).

    Returns
    -------
    dict
        Candidate distance, qualification, spectral checks and restart diagnostics.
        Failed solves are retained with a null distance.
    """
    try:
        result = log_map_mvn(*base, *target, **LOG_OPTIONS)
        return dict(distance=result.distance, reliable=result.reliable,
                    diagnostics=result.diagnostics, parameters=result.parameters,
                    runs=[dict(success=bool(run.success), message=str(run.message),
                               nit=int(run.get('nit', 0)), nfev=int(run.get('nfev', 0)),
                               diagnostics=run.get('diagnostics'),
                               qualification_error=run.get('qualification_error')) for run in result.runs])
    except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
        return dict(distance=None, reliable=False, error=str(error))


def evaluate_trial(family: str, seed: int, train: dict, test: dict) -> dict:
    """Fit two prototype sets and compare four classifiers on the same test data.

    Parameters
    ----------
    family : str
        Fixed generator and batch-size identifier.
    seed : int
        Replicate identifier.
    train, test : dict
        Independent splits, each with means, covariances and balanced labels.

    Returns
    -------
    dict
        Endpoints, prototypes, predictions, confusion matrices, timings and checks.
        Unqualified test predictions count as errors rather than being dropped.
    """
    fits, summaries = {}, {}
    for method in METHODS:
        start = perf_counter()
        prototypes, fit_details = [], []
        shared = method in ('d0', 'euclidean_fisher')
        if shared:
            prototypes = fits['fisher']['prototypes']
            fit_details = fits['fisher']['fit_details']
        for label in (() if shared else range(2)):
            selected = train['labels'] == label
            means, covariances = train['means'][selected], train['covariances'][selected]
            initial = (means.mean(axis=0), covariances.mean(axis=0))
            if method == 'naive':
                prototypes.append(initial)
                fit_details.append(dict(converged=True))
            else:
                try:
                    result = frechet_mean_mvn(means, covariances, init=initial,
                                              max_iter=60, tol=1e-6, log_options=LOG_OPTIONS)
                    prototypes.append((result.mean, result.covariance))
                    fit_details.append(dict(converged=result.converged, message=result.message,
                                            history=result.history))
                except (RuntimeError, ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    prototypes.append(None)
                    fit_details.append(dict(converged=False, error=str(error)))
        train_seconds = fits['fisher']['train_seconds'] if shared else perf_counter() - start
        start = perf_counter()
        distances = np.full((len(test['labels']), 2), np.nan)
        qualified = np.zeros_like(distances, dtype=bool)
        records = []
        for index, target in enumerate(zip(test['means'], test['covariances'])):
            row = []
            for label, prototype in enumerate(prototypes):
                if prototype is None:
                    row.append(dict(reliable=False, error='Prototype fit failed.'))
                elif method in ('naive', 'euclidean_fisher'):
                    distances[index, label] = np.sqrt(np.sum((target[0] - prototype[0])**2)
                                                       + np.sum((target[1] - prototype[1])**2))
                    qualified[index, label] = True
                elif method == 'd0':
                    try:
                        distances[index, label] = d_0(*prototype, *target)
                        qualified[index, label] = np.isfinite(distances[index, label])
                        row.append(dict(reliable=bool(qualified[index, label])))
                    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                        row.append(dict(reliable=False, error=str(error)))
                else:
                    record = distance_record(prototype, target)
                    row.append(record)
                    if record['distance'] is not None:
                        distances[index, label] = record['distance']
                    qualified[index, label] = record['reliable']
            if row:
                records.append(row)
        predict_seconds = perf_counter() - start
        valid = qualified.all(axis=1)
        predictions = np.full(len(valid), -1)
        predictions[valid] = np.argmin(distances[valid], axis=1)
        confusion = np.zeros((2, 3), dtype=int)
        np.add.at(confusion, (test['labels'], np.where(predictions == -1, 2, predictions)), 1)
        summary = dict(accuracy=float(np.mean(predictions == test['labels'])),
                       prediction_coverage=float(valid.mean()),
                       all_fits_converged=all(item['converged'] for item in fit_details),
                       train_seconds=train_seconds, predict_seconds=predict_seconds,
                       confusion=confusion)
        summaries[method] = summary
        fits[method] = dict(prototypes=prototypes, fit_details=fit_details,
                            predictions=predictions, distances=distances, distance_records=records,
                            qualified_distances=qualified,
                            prototype_method='arithmetic' if method == 'naive' else 'fisher',
                            training_reused=shared, **summary)
    # Independently check one test-to-prototype distance per class using reduced J.
    validation = []
    for label, prototype in enumerate(fits['fisher']['prototypes']):
        if prototype is not None:
            index = int(np.flatnonzero(test['labels'] == label)[0])
            target = (test['means'][index], test['covariances'][index])
            try:
                reference = d_f_reference(*prototype, *target, n_starts=3, gtol=1e-6)
                validation.append(dict(reference_reliable=reference.reliable,
                                       difference=abs(reference.distance - fits['fisher']['distances'][index, label])))
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                validation.append(dict(reference_reliable=False, error=str(error)))
    return dict(family=family, seed=seed, train=train, test=test, methods=fits,
                independent_distance_checks=validation,
                shared_comparison=compare_shared_predictions(fits),
                accuracy_difference=summaries['fisher']['accuracy'] - summaries['naive']['accuracy'])


def compare_shared_predictions(fits: dict) -> dict:
    """Compare predictions and distances only where both methods are qualified.

    Parameters
    ----------
    fits : dict
        Results for methods sharing the identical fitted Fisher prototypes.

    Returns
    -------
    dict
        Disagreement counts with explicit denominators and d0 distance errors.
        Nonconverged prototypes exclude the trial from paired summaries.
    """
    reference = fits['fisher']
    result = {}
    for method in ('d0', 'euclidean_fisher'):
        candidate = fits[method]
        valid = ((reference['predictions'] >= 0) & (candidate['predictions'] >= 0)
                 & reference['all_fits_converged'])
        result[method] = dict(
            paired_count=int(valid.sum()),
            disagreements=int(np.sum(reference['predictions'][valid] != candidate['predictions'][valid])))
    valid = (reference['qualified_distances'] & fits['d0']['qualified_distances']
             & reference['all_fits_converged'])
    absolute = np.abs(fits['d0']['distances'][valid] - reference['distances'][valid])
    nonzero = reference['distances'][valid] > 1e-7
    relative = absolute[nonzero] / reference['distances'][valid][nonzero]
    result['d0'].update(distance_count=int(valid.sum()), relative_count=int(nonzero.sum()),
                        maximum_absolute_error=float(absolute.max()) if absolute.size else None,
                        maximum_relative_error=float(relative.max()) if relative.size else None)
    return result


def evaluate_components(train: dict, test: dict) -> dict:
    """Fit and evaluate separate covariance-only and mean-only classifiers.

    Parameters
    ----------
    train, test : dict
        Gaussian means, covariances and labels for independent balanced splits.

    Returns
    -------
    dict
        Prototypes, distances, predictions, fit diagnostics and timings by method.
        Nonconverged covariance fits are flagged, with last-iterate scores retained.
    """
    train = {key: np.asarray(train[key]) for key in ('means', 'covariances', 'labels')}
    test = {key: np.asarray(test[key]) for key in ('means', 'covariances', 'labels')}
    results = {}
    for method in ('covariance_only', 'mean_only'):
        start = perf_counter()
        prototypes, diagnostics = [], []
        for label in (0, 1):
            selected = train['labels'] == label
            if method == 'covariance_only':
                fit = covariance_mean(train['covariances'][selected])
                prototypes.append(fit.pop('matrix'))
                diagnostics.append(fit)
            else:
                prototypes.append(train['means'][selected].mean(axis=0))
                diagnostics.append(dict(converged=True))
        train_seconds = perf_counter() - start
        start = perf_counter()
        if method == 'mean_only':
            distances = np.linalg.norm(test['means'][:, None] - np.array(prototypes)[None], axis=2)
        else:
            distances = np.array([[covariance_distance(prototype, matrix) for prototype in prototypes]
                                  for matrix in test['covariances']])
        predictions = np.argmin(distances, axis=1)
        predict_seconds = perf_counter() - start
        confusion = np.zeros((2, 2), dtype=int)
        np.add.at(confusion, (test['labels'], predictions), 1)
        results[method] = dict(prototypes=prototypes, fit_details=diagnostics,
                               distances=distances, predictions=predictions, confusion=confusion,
                               accuracy=float(np.mean(predictions == test['labels'])),
                               all_fits_converged=all(fit['converged'] for fit in diagnostics),
                               prediction_coverage=1., train_seconds=train_seconds,
                               predict_seconds=predict_seconds)
    return results


def evaluate_triangles(train: dict, test: dict) -> dict:
    """Fit each directed triangle classifier on training data and predict held-out labels.

    Parameters
    ----------
    train, test : dict
        Means, SPD covariances and class labels on the fixed shared splits.

    Returns
    -------
    dict
        Squared divergences, prototypes, predictions, timings and fit diagnostics.
        Failed fits/predictions remain explicit and count as classification errors.
    """
    results = {}
    means, covariances = np.asarray(train['means']), np.asarray(train['covariances'])
    labels, targets = np.asarray(train['labels']), np.asarray(test['labels'])
    for kind in ('c', 'perp'):
        start = perf_counter()
        prototypes, diagnostics = [], []
        for label in (0, 1):
            try:
                selected = labels == label
                fit = triangle_center(means[selected], covariances[selected], kind=kind)
                prototypes.append((fit.pop('mean'), fit.pop('covariance')))
                diagnostics.append(fit)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                prototypes.append(None)
                diagnostics.append(dict(converged=False, error=str(error)))
        train_seconds = perf_counter() - start
        start = perf_counter()
        objective = TriangleObjective(test['means'], test['covariances'], kind)
        divergences = np.full((len(targets), 2), np.nan)
        prediction_errors = []
        for label, prototype in enumerate(prototypes):
            if prototype is not None:
                try:
                    divergences[:, label] = objective.evaluate(*prototype)[0]
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    prediction_errors.append(dict(label=label, error=str(error)))
        qualified = np.isfinite(divergences).all(axis=1)
        predictions = np.full(len(targets), -1)
        predictions[qualified] = np.argmin(divergences[qualified], axis=1)
        predict_seconds = perf_counter() - start
        confusion = np.zeros((2, 3), dtype=int)
        np.add.at(confusion, (targets, np.where(predictions < 0, 2, predictions)), 1)
        results[f'triangle_{kind}'] = dict(prototypes=prototypes, fit_details=diagnostics,
            squared_divergences=divergences, predictions=predictions, confusion=confusion,
            accuracy=float(np.mean(predictions == targets)),
            all_fits_converged=all(fit['converged'] for fit in diagnostics),
            prediction_coverage=float(qualified.mean()), prediction_errors=prediction_errors,
            train_seconds=train_seconds, predict_seconds=predict_seconds)
    return results
