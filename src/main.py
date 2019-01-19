import os

import numpy as np
import tensorflow as tf

from GA import GeneticAlgorithm
from ES import EvolutionStrategies
from config import RANDOM_SEED, ENVIRONMENT, ALGORITHM
from population import Population

import TD3

if RANDOM_SEED:
    np.random.seed(RANDOM_SEED)
    tf.random.set_random_seed(RANDOM_SEED)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


if 'CartPole' in ENVIRONMENT.name:
    if 'dqn' in ALGORITHM:
        pass
    elif 'ga' in ALGORITHM:
        ga = GeneticAlgorithm()
        agents = Population(optimizer=ga)
        agents.evolve(save=True)
    elif 'es' in ALGORITHM:
        es = EvolutionStrategies()
        agents = Population(optimizer=es)
        agents.evolve(save=True)
    else:
        print("Please, check that configurations are set correctly.")

elif 'BipedalWalker' in ENVIRONMENT.name:
    if 'td3' in ALGORITHM:
        agent = TD3.Agent()
        agent.train()
    elif 'ga' in ALGORITHM:
        ga = GeneticAlgorithm()
        agents = Population(optimizer=ga)
        agents.evolve(save=True)
    else:
        print("Please, check that configurations are set correctly.")
else:
    print("Please, check that configurations are set correctly.")


