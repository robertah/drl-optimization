import numpy as np
from population import Agent
from config import ENVIRONMENT


def perturbate_weights(weights, n=5, noise_scale=1, render=False):
    """
    Randomly perturbate final weights, to verify the optimality condition

    :param weights: weights of the agent
    :type weights: array of weights and bias of neural network's layers
    :param n: number of perturbations
    :type n: int
    :param noise_scale: gaussian noise scale
    :type noise_scale: int or float
    :param render: animate the agent
    :type render: bool
    """
    for i in range(n):
        noise = np.random.normal(scale=noise_scale)
        agent = Agent(environment_config=ENVIRONMENT, weights=weights + noise)
        score = agent.run_agent(render=render)
        print("{} - score: {}".format(i, score))
