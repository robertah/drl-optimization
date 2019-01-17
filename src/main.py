import os

import numpy as np
import tensorflow as tf

from GA import GeneticAlgorithm
from config import GA, RANDOM_SEED
from population import Population

if RANDOM_SEED:
    np.random.seed(RANDOM_SEED)
    tf.random.set_random_seed(RANDOM_SEED)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# agents_weights, scores, children = run_agent_genetic(n_agents=50, n_generations=20, save=True)


# weights = get_best_agent(mean=True, timestamp='20190103183514')
# perturbate_weights(weights)
#
ga = GeneticAlgorithm(best=GA.best, elite=GA.elite, noise_prob=GA.noise_prob)

agents = Population(optimizer=ga)
agents.evolve(save=True)


# weights = utils.get_best_agent()

# agent = Agent(ENVIRONMENT, weights)
# score = agent.run_agent(render=True)
# analysis.perturbate_weights(weights)


#

#
# from TD3 import Agent
#
# agent = Agent()
# agent.train()

