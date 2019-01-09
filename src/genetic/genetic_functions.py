import numpy as np

from config import RANDOM_SEED, GA_MUTATION_NOISE

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


def generate_population(child_model, num_children, agents):
    """
    :param child_model: model from which building the new population
    :param num_children: number of children to generate
    :param: list of agents
    """
    for child in range(num_children):
        new_child = []
        for layer in child_model:
            new_layer = np.random.normal(layer, GA_MUTATION_NOISE)
            new_child.append(new_layer)
        # Ho fatto questa piccola modifica per avere direttamente una lista di agenti che è quello che poi ci servirebbe piuttosto che una lista di modelli ma non sono sicuro che funzioni
        agents[child].model.set_weights(new_child)
    return agents


def crossover_function_2(agents, rewards):
    '''
    :param agents: a list of keras neural networks (parents agents)
    :param rewards: list (or array) of rewards associated to the performance of the
          corresponding model
    :return: the best two agents
    '''
    sorted_indeces = np.argsort(-np.array(rewards))
    best_agent = agents[sorted_indeces[0]]

    second_best = agents[sorted_indeces[1]]
    return best_agent, second_best


def generate_population_2(child_model1, child_model2, num_children, agents):
    '''
    :param child_model1: model from which building the new population
    :param child_model2: model from which building the new population
    :param num_children: number of children to generate
    :param agents: old agents
    :return
    '''

    for child in range(int(num_children / 2) - 1):
        new_child = []
        for layer in child_model1.model.get_weights():
            new_layer = np.random.normal(layer, GA_MUTATION_NOISE)
            new_child.append(new_layer)
        agents[child].model.set_weights(new_child)

    for child in range(int(num_children / 2) - 1, num_children - 2):
        new_child = []
        for layer in child_model2.model.get_weights():
            new_layer = np.random.normal(layer, GA_MUTATION_NOISE)
            new_child.append(new_layer)
        agents[child].model.set_weights(new_child)

    agents[-2].model.set_weights(child_model1.model.get_weights())
    agents[-1].model.set_weights(child_model2.model.get_weights())
    return agents


def terminate(scores, score_threshold=2000, perc_threshold=0.95, n_consecutive=5):
    """
    Check if conditions to terminate genetic algorithm are satified:
    terminate genetic, if ``perc_threshold`` of total agents have scored the ``score_threshold``, in the last
    ``n_consecutive`` runs.

    :param scores: array of scores updated to current generation
    :type scores: np.ndarray of shape (n_generations, n_agents)
    :param score_threshold: maximum score
    :type score_threshold: int or float
    :param perc_threshold: percentage of agents that should reach the ``score_threshold``
    :type perc_threshold: float between 0 and 1
    :param n_consecutive: number of last consecutive generations
    :type n_consecutive: int
    :return:
    """
    assert 0 < perc_threshold <= 1

    if scores.shape[0] >= n_consecutive:
        n_agents = scores.shape[1]
        last_runs = scores[-n_consecutive:]
        max_scores_count = [np.count_nonzero(s == score_threshold) for s in last_runs]
        if all(msc > n_agents*perc_threshold for msc in max_scores_count):
            return True
        else:
            return False
    return False
