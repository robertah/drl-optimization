import os
import yaml
from .environment_config import EnvironmentConfig
from .population_config import PopulationConfig
from .optimizers_config import *
from .logger_config import Logger

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', '..'))

with open(os.path.join(ROOT_DIR, 'config.yml')) as c:
    config = yaml.load(c)

with open(os.path.join(ROOT_DIR, 'src', 'config', 'models.yml')) as m:
    models = yaml.load(m)

RANDOM_SEED = config['random_seed']
ENVIRONMENT = EnvironmentConfig(config, models, RANDOM_SEED)
ALGORITHM = config['algorithm']

RESULTS_SCORES = os.path.join(ROOT_DIR, config['results']['path'], config['results']['scores'])
RESULTS_WEIGHTS = os.path.join(ROOT_DIR, config['results']['path'], config['results']['weights'])
RESULTS_TRAINING = os.path.join(ROOT_DIR, config['results']['path'], config['results']['training'])

VISUALIZATION_WEIGHTS = os.path.join(ROOT_DIR, config['visualization']['path'], config['visualization']['weights'])

LOGGER = Logger(os.path.join(ROOT_DIR, config['logger']))

# A2C = ActorCriticConfig(config)
DDPG_Config = DDPGConfig(config, RESULTS_TRAINING)
TD3_Config = TD3Config(config, RESULTS_TRAINING)

# EVOLUTIONARY ALGORITHMS

POPULATION = PopulationConfig(config)
GA = GeneticAlgorithmConfig(config)
ES = EvolutionStrategiesConfig(config)
CMA_ES = CMAEvolutionStrategiesConfig(config)


