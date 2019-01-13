import os
import glob
import numpy as np
from datetime import datetime
from config import ENVIRONMENT, RESULTS_SCORES, RESULTS_WEIGHTS


def save_results(weights, scores, timestamp=datetime.now().strftime('%Y%m%d%H%M%S')):
    """
    Save agent weights and scores as numpy arrays

    :param weights: agent weights
    :type weights: np.ndarray
    :param scores: agent scores
    :type scores: np.ndarray
    :param timestamp: run's datetime
    :type timestamp: str
    """

    assert len(weights) == len(scores)
    np.save(RESULTS_SCORES + '/{}-{}'.format(ENVIRONMENT.name, timestamp), scores)
    np.save(RESULTS_WEIGHTS + '/{}-{}'.format(ENVIRONMENT.name, timestamp), weights)
    print("Results saved.")


def get_results(timestamp=None):
    """
    Get numpy arrays with saved agents and scores.

    Provide datetime, to get the results from a specific run.
    If no datetime is provided, then the most recent results are returned.

    :param timestamp: datetime of a specific run (e.g. '20181227205531')
    :type timestamp: str

    :return: agent weights and scores
    """

    if timestamp:
        assert isinstance(timestamp, str)
        assert len(timestamp) == 14

        weights = np.load(os.path.join(RESULTS_WEIGHTS, ENVIRONMENT.name + '-' + timestamp + '.npy'))
        scores = np.load(os.path.join(RESULTS_SCORES, ENVIRONMENT.name + '-' + timestamp + '.npy'))

    else:
        weights_pattern = os.path.join(RESULTS_WEIGHTS, ENVIRONMENT.name + '*.npy')
        scores_pattern = os.path.join(RESULTS_SCORES, ENVIRONMENT.name + '*.npy')

        weights_npy = max(glob.iglob(weights_pattern), key=os.path.getctime)
        scores_npy = max(glob.iglob(scores_pattern), key=os.path.getctime)

        # check if they refer to the same run
        assert os.path.split(weights_npy)[-1] == os.path.split(scores_npy)[-1]

        weights = np.load(weights_npy)
        scores = np.load(scores_npy)

    return weights, scores


def get_best_agent(mean=True, timestamp=None):
    """
    Get weights of the agent which achieved the highest score in the last generation

    :param mean: use mean of the best agents as weights, otherwise use one of the best agents
    :type mean: bool
    :param timestamp: datetime of a specific run (see also :func:`get_results`)
    :type timestamp: str

    :return: weights of the best performing agent
    """
    weights, scores = get_results(timestamp)
    if mean:
        last_scores = scores[-1]
        best_scores = np.argsort(-last_scores)[:int(0.5 * len(last_scores))]
        best_weights = weights[-1][best_scores]
        weights_final = np.empty(best_weights[0].shape, dtype=np.ndarray)
        for i in range(len(best_weights[0])):
            weights_final[i] = np.mean(best_weights[:, i])
    else:
        weights_final = weights[-1][np.argmax(scores[-1])]
    return weights_final


def print_scores(generation, scores):
    scores = {
        'generation': generation,
        'scores': {
            'min': round(np.min(scores), 2),
            'mean': round(np.mean(scores), 2),
            'max': round(np.max(scores), 2),
            'std': round(np.std(scores), 2)
        }
    }
    print(scores)
