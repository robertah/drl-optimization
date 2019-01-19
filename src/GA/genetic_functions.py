from math import exp

import numpy as np

from config import RANDOM_SEED

if RANDOM_SEED:
    np.random.seed(RANDOM_SEED)


def crossover_function(agents, rewards):
    '''
    :param agents: a list of keras neural networks (parents agents)
    :param rewards: list (or array) of rewards associated to the performance of the
          corresponding model
    :return: the child weights computed by the weighted average
          of the parents w.r.t. the reward
    '''
    rewards = np.array(rewards)
    num_layers = len(agents[0].model.get_weights())
    normalized_rewards = rewards / np.sum(rewards)
    child_model = []
    for i in range(num_layers):
        new_layer = np.zeros_like(agents[0].model.get_weights()[i])
        for j, parent_agent in enumerate(agents):
            layer = parent_agent.model.get_weights()[i] * normalized_rewards[j]
            new_layer = new_layer + layer
        child_model.append(new_layer)
    return child_model


def generate_population(child_model, num_children, agents, noise):
    """
    :param child_model: model from which building the new population
    :param num_children: number of children to generate
    :param: list of agents
    """
    for child in range(num_children):
        new_child = []
        for layer in child_model:
            # new_layer = np.random.normal(layer, GA_MUTATION_NOISE)
            new_layer = np.random.normal(layer, noise)
            new_child.append(new_layer)

        agents[child].model.set_weights(new_child)
    return agents
