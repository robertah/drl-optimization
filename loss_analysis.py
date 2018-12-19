import numpy as np


def interpolate(initial_weights, final_weights, objective, n_steps):
    """
    interpolation method to evaluate the path from initial weights to final weights
    that the network undergo during optimization (according to arXiv:1412.6544).
    We compute J(theta) for theta = (1-a)*wi + a*wf.
    :param initial_weights: the starting weights of the interpolation evaluation
            (for instance initial weights of the network)
    :param final_weights: the final weights of the interpolation evaluation
            (for instance final weights of the network
    :param objective: objective function J
    :param n_steps: number of interpolation steps
    :return: a numpy array containing J(theta) for each a (array of length n_steps)
    """

    alphas = np.linspace(0, 1, n_steps).reshape(n_steps, 1)
    initial_weights = initial_weights.reshape(1, initial_weights.shape[0])
    final_weights = initial_weights.reshape(1, final_weights.shape[0])

    # now each row has (1-alpha_i)w1, (1-alpha_i)w2 ...
    weighted_initial_weights = np.dot((1-alphas), initial_weights)

    # now each row has (alpha_i)w1, (alpha_i)w2 ...
    weighted_final_weights = np.dot(alphas, final_weights)

    new_weights = weighted_initial_weights + weighted_final_weights
    #v_objective = np.vectorize(objective)
    objective_values = np.apply_along_axis(objective, axis=1, arr=new_weights)

    return objective_values

