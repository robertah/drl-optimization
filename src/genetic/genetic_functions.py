import numpy as np

from config import RANDOM_SEED, GA_MUTATION_NOISE

if RANDOM_SEED:
    np.random.seed(RANDOM_SEED)


def crossover_function(agents, rewards):
    '''
    agents: a list of keras neural networks (parents agents)
    rewards: list (or array) of rewards associated to the performance of the
          corresponding model
    return: the child weights computed by the weighted average
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
    child_model: model from which building the new population
    num_children: number of children to generate
    scale_noise: variance of the gaussian noise to apply
    agents: list of agents
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
    models: a list of keras neural networks (parents agents)
    rewars: list (or array) of rewards associated to the performance of the
          corresponding model
    return: the best two agents
    '''
    sorted_indeces = np.argsort(-np.array(rewards))
    best_agent = agents[sorted_indeces[0]]

    second_best = agents[sorted_indeces[1]]
    return best_agent, second_best


def generate_population_2(child_model1, child_model2, num_children, agents):
    '''
    child_model: model from which building the new population
    num_children: number of children to generate
    scale_noise: variance of the gaussian noise to apply
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
