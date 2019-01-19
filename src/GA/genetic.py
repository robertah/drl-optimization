from config import ENVIRONMENT
from population import Agent
from utils import save_results
from .genetic_functions import *


def run_agent_es(n_agents=50, n_generations=100, save=True):
    """
    implmentation of the evolutionary strategy (ES) algorithm

    :param n_agents: number of agent per generation
    :param n_generations: number of generations
    :param save: save weights and scores at the end of training
    :return:
    """
    n_weights = len(Agent().model.get_weights())

    agents_weights = np.empty((n_generations, n_agents, n_weights), dtype=np.ndarray)
    scores = np.empty((n_generations, n_agents), dtype=float)

    children = np.empty((n_generations, n_weights), dtype=np.ndarray)

    # initialize agents
    agents = [Agent() for _ in range(n_agents)]

    for i in range(n_generations):
        agents_weights[i] = np.array([a.model.get_weights() for a in agents], dtype=np.ndarray)

        for j, agent in enumerate(agents):  # TODO parallelize
            scores[i][j] = agent.run_agent()

        child = crossover_function(agents, scores[i])
        children[i] = np.array(child, dtype=np.ndarray).reshape(n_weights)
        agents = generate_population(child, n_agents, agents, noise=0.1)

    if save:
        save_results(agents_weights, scores)

    return agents_weights, scores, children
