import os
import glob
import numpy as np
from datetime import datetime
from config import ENVIRONMENT, RESULTS_SCORES, RESULTS_WEIGHTS


def save_results(weights, scores):
    """
    Save agent weights and scores as numpy arrays

    :param weights: agent weights
    :type weights: np.ndarray
    :param scores: agent scores
    :type scores: np.ndarray
    """

    assert len(weights) == len(scores)
    np.save(RESULTS_SCORES + '/{}-{}'.format(ENVIRONMENT.name, datetime.now().strftime('%Y%m%d%H%M%S')), scores)
    np.save(RESULTS_WEIGHTS + '/{}-{}'.format(ENVIRONMENT.name, datetime.now().strftime('%Y%m%d%H%M%S')), weights)


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


def get_best_agent(timestamp=None):
    """
    Get weights of the agent which achieved the highest score in the last generation

    :param timestamp: datetime of a specific run (see also :func:`get_results`)
    :type timestamp: str

    :return: weights of the best performing agent
    """

    weights, scores = get_results(timestamp)
    return weights[-1][np.argmax(scores[-1])]
